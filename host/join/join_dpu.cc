#include "join_dpu.h"
#include <iostream>

// #include <gperftools/profiler.h>

#include "dpuext/api.h"
#include "generator/generator.h"

#include "umq/bitops.h"
#include "umq/kernels.h"
#include "umq/log.h"

using namespace dpu;

namespace upmemeval {

namespace join {

#define ASYNC_TIMER(timer_name, code) \
  start_async_timer(timer_name);      \
  code;                               \
  stop_async_timer(timer_name);

#define TIME_AND_CHECK(timer_name, code) \
  ASYNC_TIMER(timer_name, ARROW_RETURN_NOT_OK(code));

inline void JoinDpu::create_async_timers(__attribute((unused)) const std::vector<std::string>& names) {
#ifdef ACTIVATE_JOIN_TIMERS
  for (const auto& name : names) {
    timers_->New(name);
  }
#endif
}

inline void JoinDpu::start_async_timer(__attribute((unused)) const std::string& name) {
#ifdef ACTIVATE_JOIN_TIMERS
  auto timer = timers_->Get(name);
  system_.async().call([timer](__attribute((unused)) DpuSet& set,
                                uint32_t rank_id) -> void { timer->Start(rank_id); });
#endif
}

inline void JoinDpu::stop_async_timer(__attribute((unused)) const std::string& name) {
#ifdef ACTIVATE_JOIN_TIMERS
  auto timer = timers_->Get(name);
  system_.async().call([timer](__attribute((unused)) DpuSet& set,
                                uint32_t rank_id) -> void { timer->Stop(rank_id); });
#endif
}

/**
 * @brief Get data buffer lengths for data organized in batches
 */
static inline auto get_buffer_lengths(
    size_t n_partitions, const arrow::RecordBatchVector& batches,
    const std::vector<std::vector<uint32_t>>& metadata) {
  std::vector<std::vector<uint32_t>> buffer_lengths(n_partitions);
  for (size_t i = 0; i < n_partitions; i++) {
    uint32_t buffer_length = 0;
    for (size_t j = 0; j < batches.size(); j++) {
      buffer_length += metadata[j][i] - (i == 0 ? 0 : metadata[j][i - 1]);
    }
    buffer_lengths[i] = {buffer_length};
  }
  return buffer_lengths;
}

/**
 * @brief Get data buffer lengths for data organized in partitions
 */
static inline auto get_buffer_lengths(int key_column_index,
                                      arrow::RecordBatchVector& partitions) {
  std::vector<std::vector<uint32_t>> buffer_lengths(partitions.size());
  for (size_t i = 0; i < partitions.size(); i++) {
    auto buffer_length =
        arrow_data_buffer_length(partitions, i, key_column_index).ValueOrDie();
    buffer_lengths[i] = {buffer_length};
  }
  return buffer_lengths;
}

arrow::Result<std::shared_ptr<partition::Partitioner>> JoinDpu::do_partition(
    int k_column_index,
    const std::vector<uint32_t>& dpu_offset, std::vector<std::vector<uint32_t>>& metadata,
    uint32_t buffer_length, bool is_left) {
  using namespace partition;

  auto pool = arrow::internal::GetCpuThreadPool();
  auto schema = is_left ? left_schema_ : right_schema_;
  auto partitioner = std::make_shared<partition::Partitioner>(pool, schema);

  auto& batches = is_left ? left_batches_ : right_batches_;
  auto n_partitions = batches.size();
  auto nr_dpus = system_.dpus().size();
  auto batch_rows = batches[0]->num_rows();

  TIME_AND_CHECK("partitionAllocation",
                 partitioner->AllocatePartitions(n_partitions,
                                                  // TODO: check if overkill
                                                 batch_rows * 1.5));

  // allocate offset vectors for each partition, used by all threads
  // only used for left table
  TIME_AND_CHECK("partitionAllocation",
                  partitioner->AllocateOffsets(n_partitions, nr_dpus));
  TIME_AND_CHECK("partitionAllocation",
                  partitioner->GenerateRandomShifts(n_partitions, nr_dpus));

  // Process all batches (parallel on NR_DPUS)
  assert(batches.size() % nr_dpus == 0);
  for (size_t i = 0; i < batches.size(); i += nr_dpus) {
    // Run partition kernel on the partition key column
    TIME_AND_CHECK("partitionKernel", partitioner->PartitionKernel(
                                          system_, batches, i, k_column_index,
                                          n_partitions, dpu_offset, metadata));

    // (scatter/gather only) Each thread acquires a slot in each partition
    // in a round-robin fashion to write its output to.
    TIME_AND_CHECK("partitionOffsets",
                    partitioner->GetOffsets(system_, metadata, i, dpu_offset));
    // Load output buffers from DPU and process them in the background
    TIME_AND_CHECK(
        "partitionLoad",
        partitioner->LoadPartitions(system_, metadata, i, k_column_index, buffer_length));

    // Process all value columns
    for (int column_index = 0; column_index < left_schema_->num_fields();
         ++column_index) {
      if (column_index == k_column_index) continue;
      // Run take kernel on value columns
      TIME_AND_CHECK("partitionTake",
                     partitioner->TakeKernel(system_, batches, i, column_index));

      // Load output buffers from DPU and process them in the background
      TIME_AND_CHECK(
          "partitionLoad",
          partitioner->LoadPartitions(system_, metadata, i, column_index, buffer_length));
    }
  }

  return partitioner;
};

arrow::Status JoinDpu::Prepare() {
  timers_ = std::make_shared<timer::Timers>(system_.ranks().size());
  create_async_timers({"outer", "full", "build", "probe", "take", "partitionAllocation",
                       "partitionKernel", "partitionOffsets", "partitionLoad",
                       "partitionTake"});

  try {
    system_.load(JOIN_DPU_BINARY);
  } catch (DpuError& e) {
    return arrow::Status::UnknownError(e.what());
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> JoinDpu::Run() {

  start_async_timer("outer");
  auto result = Run_internal();
  stop_async_timer("outer");

  system_.async().sync();
  return result;
}

arrow::Result<std::shared_ptr<arrow::Table>> JoinDpu::Run_internal() {
  using std::as_const;

  start_async_timer("full");

  auto n_partitions = left_batches_.size();
  auto pk_column_index = right_schema_->GetFieldIndex("pk");
  assert(pk_column_index >= 0);
  auto fk_column_index = left_schema_->GetFieldIndex("fk");
  assert(fk_column_index >= 0);

  // Get data buffer length of a right batch
  // Assumes that buffers are of equal size
  ARROW_ASSIGN_OR_RAISE(uint32_t right_buffer_length,
                        arrow_data_buffer_length(right_batches_, 0UL, pk_column_index));

  // Get data buffer length of a left batch
  // Assumes that buffers are of equal size
  ARROW_ASSIGN_OR_RAISE(uint32_t left_buffer_length,
                        arrow_data_buffer_length(left_batches_, 0UL, fk_column_index));

  // Process all batches (parallel on NR_DPUS)
  auto nr_dpus = system_.dpus().size();
  assert(n_partitions % nr_dpus == 0);

  // create dpu offset auxiliary vector
  // each entry contains the sum of number of dpus in previous ranks
  std::vector<uint32_t> dpu_offset(system_.ranks().size(), 0);
  for (size_t i = 0; i < system_.ranks().size() - 1; ++i) {
    dpu_offset[i + 1] = dpu_offset[i] + (system_.ranks()[i])->dpus().size();
  }

  /* Asynchronous Left and Right Partitions */
  auto [left_partitions_,
        right_partitions_] = [this, &dpu_offset = as_const(dpu_offset),
                                       n_partitions, fk_column_index, pk_column_index,
                                       left_buffer_length, right_buffer_length] {
    // vector of metadata for each partition, used asynchronously by the left and right
    // partitioners
    std::vector<std::vector<uint32_t>> metadata(n_partitions);
    for (size_t i = 0; i < n_partitions; i++) {
      metadata[i].resize(n_partitions);
    }

    // do left partition
    auto left_partitioner =
        do_partition(fk_column_index, dpu_offset, metadata, left_buffer_length, true)
            .ValueOrDie();

    // do right partition
    auto right_partitioner =
        do_partition(pk_column_index, dpu_offset, metadata, right_buffer_length, false)
            .ValueOrDie();

    system_.async().sync();

    auto left_partitions =
        partition::ToRecordBatches(std::move(left_partitioner)->partitions())
            .ValueOrDie();
    auto right_partitions_ =
        partition::ToRecordBatches(std::move(right_partitioner)->partitions())
            .ValueOrDie();
    /* At this point both tables are partitioned. */

    return std::make_tuple(left_partitions, right_partitions_);
  }();

  assert(right_partitions_.size() == n_partitions);
  assert(left_partitions_.size() == n_partitions);
  
  // Get data buffer length of right batches
  // DOES NOT Assume that buffers are of equal size
  auto buffer_partitioned_length_param_right =
      get_buffer_lengths(pk_column_index, right_partitions_);

  // Get data buffer length of left batches
  // DOES NOT assume that buffers are of equal size
  auto buffer_partitioned_length_param_left =
      get_buffer_lengths(fk_column_index, left_partitions_);

  // Declare persistent variables
  std::vector<int32_t> kernel_param_build{KernelHashBuild};
  std::vector<int32_t> kernel_param_probe{KernelHashProbe};
  std::vector<int32_t> kernel_param_take{KernelTake};
  std::vector<arrow::RecordBatchVector> new_batches(system_.ranks().size());

  for (size_t batches_offset = 0; batches_offset < n_partitions; batches_offset += nr_dpus) {
    /* Hash Build Phase */
    ARROW_RETURN_NOT_OK(([this, &kernel_param_build = as_const(kernel_param_build),
                          &buffer_partitioned_length_param_right =
                              as_const(buffer_partitioned_length_param_right),
                          &right_partitions = as_const(right_partitions_), batches_offset,
                          pk_column_index] {
      start_async_timer("build");

      // Set kernel to hash build
      system_.async().copy("kernel", 0, kernel_param_build, sizeof(int32_t));
      // Copy buffer length
      system_.async().copy_from("buffer_length", buffer_partitioned_length_param_right,
                                batches_offset);
      // Copy input data buffers (primary key column of right batches)
      ARROW_RETURN_NOT_OK(arrow_copy_to_dpus(system_, "buffer", right_partitions,
                                             batches_offset, pk_column_index, true));
      // Execute DPU program asynchronously
      system_.async().exec();

      stop_async_timer("build");

      return arrow::Status::OK();
    }()));

    /* Hash Probe Phase */
    ARROW_RETURN_NOT_OK(([this, &kernel_param_probe = as_const(kernel_param_probe),
                          &buffer_partitioned_length_param_left =
                              as_const(buffer_partitioned_length_param_left),
                          &left_partitions_ = as_const(left_partitions_), batches_offset,
                          fk_column_index] {
      start_async_timer("probe");

      // Set kernel to hash probe
      system_.async().copy("kernel", 0, kernel_param_probe, sizeof(int32_t));
      // Copy buffer length
      system_.async().copy_from("buffer_length", buffer_partitioned_length_param_left,
                                batches_offset);
      // Copy input data buffers (foreign key column of left batches)
      ARROW_RETURN_NOT_OK(arrow_copy_to_dpus(system_, "buffer", left_partitions_,
                                             batches_offset, fk_column_index, true));
      // Execute DPU program asynchronously
      system_.async().exec();

      stop_async_timer("probe");

      return arrow::Status::OK();
    }()));

    /* Take Phase */
    ARROW_RETURN_NOT_OK(([this, &kernel_param_take = as_const(kernel_param_take),
                          &buffer_partitioned_length_param_left =
                              as_const(buffer_partitioned_length_param_left),
                          &left_partitions = left_partitions_,
                          &right_partitions =
                              as_const(right_partitions_),
                          &dpu_offset = as_const(dpu_offset), &new_batches,
                          pk_column_index, batches_offset, nr_dpus] {
      start_async_timer("take");

      // Fill output record batches with left table
      system_.async().call([&dpu_offset, &new_batches, batches_offset, &left_partitions](
                               DpuSet& set, unsigned rank_id) -> void {
        new_batches[rank_id].reserve(new_batches[rank_id].size() + set.dpus().size());
        for (size_t batch_offset = batches_offset + dpu_offset[rank_id];
             batch_offset < batches_offset + dpu_offset[rank_id] + set.dpus().size(); ++batch_offset) {
          new_batches[rank_id].push_back(std::move(left_partitions[batch_offset]));
        }
      });

      // Take right table value columns
      for (int column_index = 0; column_index < right_schema_->num_fields();
           ++column_index) {
        if (column_index == pk_column_index) {
          continue;
        }

        // Send information and column to DPUs
        system_.async().copy("kernel", 0, kernel_param_take, sizeof(int32_t));
        system_.async().copy_from("buffer_length", buffer_partitioned_length_param_left,
                                  batches_offset);
        ARROW_RETURN_NOT_OK(arrow_copy_to_dpus(system_, "buffer",
                                               right_partitions, batches_offset,
                                               column_index, true));

        // Execute DPU take asynchronously
        system_.async().exec();

        // Grab and write results to new batches
        system_.async().call([this, &dpu_offset, &buffer_partitioned_length_param_left,
                              &new_batches, column_index, batches_offset,
                              nr_dpus](DpuSet& set, unsigned rank_id) -> void {
          uint32_t result_offset = set.dpus().size() * batches_offset / nr_dpus;
          // Grab results
          auto results =
              arrow_copy_from_dpus(set, "output_buffer", arrow::uint32(),
                                   buffer_partitioned_length_param_left.begin() +
                                       dpu_offset[rank_id] + batches_offset)
                  .ValueOrDie();

          // Note: should be thread safe because different ranks access different indexes
          for (size_t result_index = 0; result_index < results.size(); ++result_index) {
            auto new_column = std::move(results[result_index]);
            auto batch = std::move(new_batches[rank_id][result_offset + result_index]);
            auto field = right_schema_->field(column_index);
            new_batches[rank_id][result_offset + result_index] =
                std::move(batch)->AddColumn(batch->num_columns(), field, std::move(new_column)).ValueOrDie();
          }
        });
      }

      stop_async_timer("take");

      return arrow::Status::OK();
    }()));
  }

  arrow::RecordBatchVector record_batches;
  record_batches.reserve(n_partitions);

  system_.async().sync();

  /* Build record batches */
  ARROW_RETURN_NOT_OK(([this, &new_batches, &record_batches,
                        n_partitions] {
    uint32_t c_partitions = 0, c_iter = 0;

    while (c_partitions < n_partitions) {
      uint32_t rank_id = 0;
      for (auto const& rank : system_.ranks()) {
        for (size_t d = 0; d < rank->dpus().size(); ++d) {
          record_batches.push_back(
              std::move(new_batches[rank_id][c_iter * rank->dpus().size() + d]));
          ++c_partitions;
        }
        rank_id++;
      }
      ++c_iter;
    }

    return arrow::Status::OK();
  }()));

  stop_async_timer("full");

  return ::arrow::Table::FromRecordBatches(std::move(record_batches)).ValueOrDie();
}

}  // namespace join
}  // namespace upmemeval
