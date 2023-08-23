#include "partition_dpu.h"

#include <cassert>
#include <iostream>

#include "dpuext/api.h"
#include "generator/generator.h"

#include "umq/bitops.h"
#include "umq/kernels.h"

#include "partition.h"
#include "partitioner.h"

// This implementation uses NR_DPUS temporary buffers for capturing the outputs from the
// DPUs. Note that we can't use xfer to copy from each DPU directly into a destination
// partition buffer so we allocate these NR_DPUS temporary buffers and then we can rank
// transfer from the DPUs, although at the cost of additional copy within the host (from
// these temporary buffers to the target partition buffers). An alternative is to add
// permenant NR_DPUS buffers in each iteration and create small partitioned record
// batches using arrow::Buffer::SliceBuffer. This will create much smaller record
// batches but we can probably move multiple batches to a DPU using several transfer
// calls in the join phase. Need to validate what will be faster.

using namespace dpu;

namespace upmemeval {

namespace partition {

arrow::Status PartitionDpu::Prepare() {
  timers_ = std::make_shared<timer::Timers>(system_.ranks().size());

  if (batches_.empty()) {
    return arrow::Status::UnknownError("Partitioning requires at least one input batch");
  }
  if (nr_partitions_ < 2) {
    return arrow::Status::UnknownError(
        "Partitioning requires target partitions number of at least 2");
  }

  try {
    system_.load(PARTITION_DPU_BINARY);
  } catch (DpuError& e) {
    return arrow::Status::UnknownError(e.what());
  }
  return arrow::Status::OK();
}

arrow::Result<arrow::RecordBatchVector> PartitionDpu::Run() {
  auto pool = arrow::internal::GetCpuThreadPool();

  // Create timers
  auto allocationTimer = timers_->New("allocation");
  auto partitionTimer = timers_->New("partition");
  auto takeTimer = timers_->New("take");
  auto loadingTimer = timers_->New("load");
  auto dispatchTimer = timers_->New("dispatch");
  auto waitTimer = timers_->New("wait");

  auto batch_rows = batches_[0]->num_rows();
  auto nr_dpus = system_.dpus().size();
  auto buffer_size =
      batch_rows * sizeof(uint32_t);  // For a single column in a single batch
  auto partition_key_column = schema_->GetFieldIndex(partition_key_);
  if (partition_key_column < 0) {
    return arrow::Status::UnknownError("Partition key not found");
  }

  Partitioner partitioner{pool, schema_};

  // Allocation phase
  allocationTimer->Start();
  // XXX 2x+8K factor for parititons that are slightly larger than expected
  // ARROW_RETURN_NOT_OK(
  //     partitioner.AllocatePartitions(nr_partitions_, batch_rows * 2 + (8 << 10)));
  ARROW_RETURN_NOT_OK(partitioner.AllocatePartitions(
      nr_partitions_, (1.0 * batch_rows * batches_.size() / nr_partitions_) * 2));
  ARROW_RETURN_NOT_OK(partitioner.AllocateBuffers(nr_dpus, buffer_size));
  allocationTimer->Stop();

  // Process all batches (parallel on NR_DPUS)
  assert(batches_.size() % nr_dpus == 0);
  for (size_t i = 0; i < batches_.size(); i += nr_dpus) {
    // Run partition kernel on the partition key column
    partitionTimer->Start();
    ARROW_ASSIGN_OR_RAISE(auto metadata, partitioner.PartitionKernel(
                                             system_, batches_, i, partition_key_column));
    partitionTimer->Stop();

    // Wait until all previous buffers are processed before loading again
    waitTimer->Start();
    ARROW_RETURN_NOT_OK(partitioner.WaitForBackgroundTasks());
    waitTimer->Stop();

    // Load output buffers from DPU and process them in the background
    loadingTimer->Start();
    ARROW_RETURN_NOT_OK(partitioner.LoadBuffers(system_));
    loadingTimer->Stop();

    dispatchTimer->Start();
    ARROW_RETURN_NOT_OK(
        partitioner.BackgroundProcessBuffers(metadata, partition_key_column));
    dispatchTimer->Stop();

    for (int column_index = 0; column_index < schema_->num_fields(); ++column_index) {
      if (column_index == partition_key_column) continue;
      // Run take kernel on columns other than the partition key column
      takeTimer->Start();
      ARROW_RETURN_NOT_OK(partitioner.TakeKernel(system_, batches_, i, column_index));
      takeTimer->Stop();

      // Wait until all previous buffers are processed before loading again
      waitTimer->Start();
      ARROW_RETURN_NOT_OK(partitioner.WaitForBackgroundTasks());
      waitTimer->Stop();

      // Load output buffers from DPU and process them in the background
      loadingTimer->Start();
      ARROW_RETURN_NOT_OK(partitioner.LoadBuffers(system_));
      loadingTimer->Stop();

      dispatchTimer->Start();
      ARROW_RETURN_NOT_OK(partitioner.BackgroundProcessBuffers(metadata, column_index));
      dispatchTimer->Stop();
    }
  }
  // Wait until all buffers are processed
  waitTimer->Start();
  ARROW_RETURN_NOT_OK(partitioner.WaitForBackgroundTasks());
  waitTimer->Stop();

  // printf("partitions size %lu\n", partitioner.partitions().size());
  return partition::ToRecordBatches(partitioner.partitions());
}

}  // namespace partition
}  // namespace upmemeval
