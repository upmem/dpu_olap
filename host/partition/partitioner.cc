#include "partitioner.h"
#include "memory_utils/memcpy.h"
#include "umq/bitops.h"
#include "umq/kernels.h"

#include <iostream>

namespace upmemeval {
namespace partition {

Partitioner::Partitioner(::arrow::internal::ThreadPool* thread_pool,
                         std::shared_ptr<::arrow::Schema> schema)
    : thread_pool_(thread_pool), schema_(schema) {}

Partitioner::~Partitioner() {}

::arrow::Status Partitioner::AllocatePartitions(uint32_t nr_partitions,
                                                int64_t partition_rows) {
  const auto partition_length = partition_rows;
  for (size_t i = 0; i < nr_partitions; ++i) {
    auto partition = std::make_shared<Partition>(schema_, partition_length);
    ARROW_RETURN_NOT_OK(partition->Allocate());
    partitions_.push_back(partition);
  }
  return ::arrow::Status::OK();
}

::arrow::Result<::arrow::BufferVector> Partitioner::AllocateBuffers(size_t nr_buffers,
                                                                    size_t buffer_size) {
  ::arrow::BufferVector buffers;
  for (size_t i = 0; i < nr_buffers; ++i) {
    ARROW_ASSIGN_OR_RAISE(auto maybe_buffer, arrow::AllocateBuffer(buffer_size));
    std::shared_ptr<arrow::Buffer> buffer = std::move(maybe_buffer);
    buffers.push_back(buffer);
  }
  return buffers;
}

::arrow::Status Partitioner::AllocateOffsets(size_t nr_partitions, size_t nr_dpus) {
  for (size_t i = 0; i < nr_partitions; ++i) {
    std::vector<size_t> offset(nr_dpus);
    offsets_.push_back(offset);
  }
  return ::arrow::Status::OK();
}

::arrow::Status Partitioner::GenerateRandomShifts(size_t nr_partitions, size_t nr_dpus) {
  std::srand(42);
  output_shifts_.resize(nr_dpus);
  std::generate(output_shifts_.begin(), output_shifts_.end(), [nr_partitions]() {
    return std::rand() % nr_partitions;
  });
  return ::arrow::Status::OK();
}

::arrow::Status Partitioner::GenerateZeroShifts(size_t nr_dpus) {
  output_shifts_.resize(nr_dpus);
  std::fill(output_shifts_.begin(), output_shifts_.end(), 0);
  return ::arrow::Status::OK();
}

::arrow::Result<std::vector<std::vector<uint32_t>>> Partitioner::PartitionKernel(
    dpu::DpuSet& system_, arrow::RecordBatchVector& batches, size_t batches_offset,
    size_t column_index) {
  auto nr_dpus = system_.dpus().size();
  auto nr_partitions = partitions_.size();

  // Get data buffer length of a right batch
  // Assumes that buffers are of equal size
  ARROW_ASSIGN_OR_RAISE(uint32_t buffer_length,
                        dpu::arrow_data_buffer_length(batches, 0UL, column_index));

  // Set input parameters for the partition kernel
  kernel_partition_inputs_t partition_kernel_params;
  partition_kernel_params.nr_partitions = nr_partitions;
  partition_kernel_params.with_selection_indices_array =
      schema_->num_fields() > 1 ? 1 : 0;
  partition_kernel_params.buffer_length = buffer_length;
  //TODO set the appropriate value here
  partition_kernel_params.output_shift = 0;

  // Set kernel to partition kernel
  std::vector<int32_t> kernel_param{KernelPartition};
  std::vector<uint32_t> buffer_length_param{buffer_length};
  system_.copy("kernel", 0, kernel_param, sizeof(int32_t));

  // Set partition kernel parameters
  std::vector<kernel_partition_inputs_t> input_params{partition_kernel_params};
  system_.copy("INPUT", 0, input_params, sizeof(kernel_partition_inputs_t));
  system_.copy("buffer_length", 0, buffer_length_param, sizeof(uint32_t));

  // Copy input data buffers
  ARROW_RETURN_NOT_OK(
      dpu::arrow_copy_to_dpus(system_, "buffer", batches, batches_offset, column_index));
  // Execute DPU program synchroniously
#if ENABLE_LOG
  system_.exec_with_log(std::cout);
#else
  system_.exec();
#endif
  // Get output parameters
  std::vector<std::vector<kernel_partition_outputs_t>> output_params(nr_dpus);
  for (auto& x : output_params) {
    x.resize(1);
  }
  system_.copy(output_params, "OUTPUT");

  // Copy from DPUs the metadata
  std::vector<std::vector<uint32_t>> partitions_metadata(nr_dpus);
  for (auto& x : partitions_metadata) {
    x.resize(nr_partitions);
  }
  system_.copy(partitions_metadata, DPU_MRAM_HEAP_POINTER_NAME,
              output_params.front().front().partitions_metadata_offset);

  return partitions_metadata;
}

::arrow::Status Partitioner::PartitionKernel(dpu::DpuSet& system_,
                                             arrow::RecordBatchVector& batches,
                                             size_t batches_offset, size_t column_index,
                                             size_t nr_partitions,
                                             const std::vector<uint32_t>& dpu_offset,
                                             std::vector<std::vector<uint32_t>>& metadata) {
  // Get data buffer length of a right batch
  // Assumes that buffers are of equal size
  ARROW_ASSIGN_OR_RAISE(uint32_t buffer_length,
                        dpu::arrow_data_buffer_length(batches, 0UL, column_index));

  system_.async().call([nr_columns = this->schema_->num_fields(), nr_partitions,
                        buffer_length, &dpu_offset,
                        &output_shifts = this->output_shifts_](
                           dpu::DpuSet& set,
                           __attribute((unused)) unsigned rank_id) -> void {
    auto rank_offset = dpu_offset[rank_id];
    std::vector<std::vector<kernel_partition_inputs_t>> input_params(set.dpus().size());
    for (size_t each_dpu = 0; each_dpu < set.dpus().size(); ++each_dpu) {
      input_params[each_dpu].resize(1);
      // Set input parameters for the partition kernel
      input_params[each_dpu][0].nr_partitions = nr_partitions;
      input_params[each_dpu][0].with_selection_indices_array = nr_columns > 1 ? 1 : 0;
      input_params[each_dpu][0].buffer_length = buffer_length;
      // TODO set the appropriate value here
      input_params[each_dpu][0].output_shift = output_shifts[each_dpu + rank_offset];
    }

    // Set kernel to partition kernel
    std::vector<int32_t> kernel_param{KernelPartition};
    std::vector<uint32_t> buffer_length_param{buffer_length};
    set.copy("kernel", 0, kernel_param, sizeof(int32_t));

    // Set partition kernel parameters
    set.copy("INPUT", 0, input_params, sizeof(kernel_partition_inputs_t));
    set.copy("buffer_length", 0, buffer_length_param, sizeof(uint32_t));
  });

  // Copy input data buffers
  ARROW_RETURN_NOT_OK(dpu::arrow_copy_to_dpus(system_, "buffer", batches, batches_offset,
                                              column_index, true));
  // Execute DPU program synchroniously
#if ENABLE_LOG
  system_.exec_with_log(std::cout);
#else
  system_.async().exec();
#endif
  // Get output parameters
  system_.async().call([&dpu_offset, &metadata, nr_partitions, batches_offset](
                          dpu::DpuSet& set, unsigned rank_id) -> void {
    std::vector<std::vector<kernel_partition_outputs_t>> output_params(set.dpus().size());
    for (auto& x : output_params) {
      x.resize(1);
    }
    set.copy(output_params, "OUTPUT");

    // Copy the metadata from DPUs
    dpu::DpuSymbol sym(0, 1 << 16);
    set.copy(metadata, nr_partitions * sizeof(uint32_t), sym,
             output_params.front().front().partitions_metadata_offset,
             batches_offset + dpu_offset[rank_id]);
  });

  return arrow::Status::OK();
}

::arrow::Status Partitioner::TakeKernel(dpu::DpuSet& system_,
                                        arrow::RecordBatchVector& batches,
                                        size_t batches_offset, size_t column_index) {
  system_.async().call(
      [](dpu::DpuSet& set, __attribute((unused)) unsigned rank_id) -> void {
        // Set kernel to take kernel
        std::vector<int32_t> kernel_param{KernelTake};
        set.copy("kernel", 0, kernel_param, sizeof(int32_t));
      });

  // Copy input data buffers
  ARROW_RETURN_NOT_OK(dpu::arrow_copy_to_dpus(system_, "buffer", batches, batches_offset,
                                              column_index, true));

  // Execute DPU program synchroniously
#if ENABLE_LOG
  system_.exec_with_log(std::cout);
#else
  system_.async().exec();
#endif

  return ::arrow::Status::OK();
}

::arrow::Status Partitioner::LoadBuffers(dpu::DpuSet& system_) {
  // Transfer partitioned buffers from DPUs to the temporary buffers
  return dpu::arrow_copy_from_dpus(system_, "output_buffer", buffers_,
                                   buffers_[0]->size());
}

::arrow::Status Partitioner::LoadBuffers(dpu::DpuSet& system_, size_t batch_index,
                                         size_t column_index, size_t buffer_length) {
  // Transfer partitioned buffers from DPUs to the temporary buffers
  return dpu::arrow_copy_from_dpus(system_, "output_buffer", partitions_, batch_index,
                                   column_index, buffer_length, true);
}

::arrow::Status Partitioner::LoadBuffers(dpu::DpuSet& system_,
                                         ::arrow::BufferVector& buffers,
                                         size_t batch_index, int column_index) {
  // Transfer partitioned buffers from DPUs to a permanent buffer
  return dpu::arrow_copy_from_dpus_offset(
      system_, "output_buffer", buffers, buffers[0]->size(),
      (batch_index * schema_->num_fields()) + column_index, schema_->num_fields(), true);
}

::arrow::Status Partitioner::LoadBuffersWithLog(dpu::DpuSet& system_) {
  // Transfer partitioned buffers from DPUs to the temporary buffers
  auto return_status =
      dpu::arrow_copy_from_dpus(system_, "output_buffer", buffers_, buffers_[0]->size());
  for (auto& buffer : buffers_) {
    std::cout << "Buffer copied back: " << buffer->size() / sizeof(uint32_t) << std::endl;
    uint32_t* data = reinterpret_cast<uint32_t*>(buffer->mutable_data());
    uint32_t max_value = 0;
    for (size_t i = 0; i < buffer->size() / sizeof(uint32_t); ++i) {
      if (data[i] > max_value) {
        max_value = data[i];
      }
    }
    std::cout << "Max value: " << max_value << std::endl;
  }
  return return_status;
}

::arrow::Status Partitioner::BackgroundProcessBuffers(
    const std::vector<std::vector<uint32_t>>& partitions_metadata, size_t column_index) {
  // Submit memcpy tasks to the thread pool
  auto block_size = 64;
  auto byte_width = partitions_[0]->GetByteWidth(column_index);

  // TODO: changed num_threads to 1 because we do this NR_DPUs times
  // Profiling needed
  auto num_threads = 1;  // thread_pool_->GetCapacity();
  for (size_t dpu_idx = 0; dpu_idx < partitions_metadata.size(); ++dpu_idx) {
    auto src = buffers_[dpu_idx]->mutable_data();
    for (size_t partition_idx = 0; partition_idx < partitions_.size(); ++partition_idx) {
      auto partition = partitions_[partition_idx];
      auto src_offset =
          (partition_idx == 0) ? 0 : partitions_metadata[dpu_idx][partition_idx - 1];
      auto to_write = partitions_metadata[dpu_idx][partition_idx] - src_offset;

      auto dst = partitions_[partition_idx]->Write(column_index, to_write);

      if (to_write * byte_width < kMemcopyThreshold) {
        std::memcpy(dst, src + src_offset * byte_width, to_write * byte_width);
      } else {
        auto futures = parallel_memcopy(thread_pool_, dst, src + src_offset * byte_width,
                                        to_write * byte_width, block_size, num_threads);
        futures_.insert(std::end(futures_), std::begin(futures), std::end(futures));
      }
    }
  }
  return ::arrow::Status::OK();
}

::arrow::Status Partitioner::GetOffsets(
    dpu::DpuSet& system_, const std::vector<std::vector<uint32_t>>& partitions_metadata,
    size_t batches_offset, const std::vector<uint32_t>& dpu_offset) {
  system_.async().call([&partitions_ = this->partitions_, &partitions_metadata,
                        batches_offset, &dpu_offset, &offsets = this->offsets_,
                        &mutex_pool = this->mutex_pool](dpu::DpuSet& rank,
                                                        unsigned rank_id) -> void {
    auto rank_offset = dpu_offset[rank_id];
    for (size_t partition_idx = 0; partition_idx < partitions_.size(); ++partition_idx) {
      const std::lock_guard<std::mutex> lock(mutex_pool[partition_idx % mutex_pool_size]);
      auto partition = partitions_[partition_idx];
      auto to_write_total = 0U;

      for (size_t i = 0; i < rank.dpus().size(); ++i) {
        auto dpu_idx = rank_offset + i;
        auto batch_idx = dpu_idx + batches_offset;
        auto src_offset =
            (partition_idx == 0) ? 0 : partitions_metadata[batch_idx][partition_idx - 1];
        auto to_write = partitions_metadata[batch_idx][partition_idx] - src_offset;
        offsets[partition_idx][dpu_idx] = to_write_total;
        to_write_total += to_write;
      }

      auto slot_for_this_rank = partition->PrepareWrite(0, to_write_total);
      
      for (size_t i = 0; i < rank.dpus().size(); ++i) {
        auto dpu_idx = rank_offset + i;
        offsets[partition_idx][dpu_idx] += slot_for_this_rank;
      }
    }
  });
  return ::arrow::Status::OK();
};

/**
 * @brief Structure used to provide context to the sg_xfer function
 */
// extern "C" {
typedef struct sg_xfer_context {
  const std::vector<std::vector<uint32_t>>* metadata;
  const std::vector<std::vector<size_t>>* offsets;
  const Partitioner* partitioner;
  size_t batches_offset;
  const std::vector<uint32_t>* output_shifts;
  size_t column_index;
} sg_xfer_context_temp;

bool Partitioner::get_block(struct ::sg_block_info* out, uint32_t dpu_index,
                            uint32_t block_index, void* args) {
  auto ctx = static_cast<sg_xfer_context*>(args);
  auto nr_partitions = ctx->partitioner->partitions_.size();

  if (block_index >= nr_partitions) {
    return false;
  }

  auto batch_idx = dpu_index + ctx->batches_offset;
  auto partition_idx = (block_index + (*ctx->output_shifts)[dpu_index]) % nr_partitions;
  auto src_offset =
      (partition_idx == 0) ? 0 : (*ctx->metadata)[batch_idx][partition_idx - 1];
  auto length = (*ctx->metadata)[batch_idx][partition_idx] - src_offset;
  auto byte_width = ctx->partitioner->partitions_[0]->GetByteWidth(ctx->column_index);
  auto dst = ctx->partitioner->partitions_[partition_idx]->UnsafeWrite(ctx->column_index);

  out->length = length * byte_width;
  out->addr = dst + (*ctx->offsets)[partition_idx][dpu_index] * byte_width;

  return true;
}

::arrow::Status Partitioner::LoadPartitions(
    dpu::DpuSet& system_, const std::vector<std::vector<uint32_t>>& partitions_metadata,
    size_t batches_offset, size_t column_index, uint32_t buffer_length) {
  sg_xfer_context sc_args = {
      .metadata = &partitions_metadata,
      .offsets = &offsets_,
      .partitioner = &*this,
      .batches_offset = batches_offset,
      .output_shifts = &output_shifts_,
      .column_index = column_index,
  };
  get_block_t get_block_info = {
      .f = get_block,
      .args = &sc_args,
      .args_size = sizeof(sc_args),
  };

  auto byte_width = partitions_[0]->GetByteWidth(column_index);
  auto transfer_buffer_size = ROUND_UP_TO_MULTIPLE_OF_8(buffer_length * byte_width);
  DPU_ASSERT(dpu_push_sg_xfer(system_.unsafe(), DPU_XFER_FROM_DPU, "output_buffer", 0,
                              transfer_buffer_size, &get_block_info,
                              static_cast<dpu_sg_xfer_flags_t>(
                                  DPU_SG_XFER_ASYNC | DPU_SG_XFER_DISABLE_LENGTH_CHECK)));

  return ::arrow::Status::OK();
};

::arrow::Status Partitioner::WaitForBackgroundTasks() {
  // Wait for the previous memcpy tasks to finish
  for (auto& fut : futures_) {
    ARROW_RETURN_NOT_OK(fut.status());
  }
  futures_.clear();
  return ::arrow::Status::OK();
}

}  // namespace partition
}  // namespace upmemeval
