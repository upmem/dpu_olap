#pragma once

#include <mutex>
#include <arrow/api.h>
#include <arrow/util/thread_pool.h>
#include "dpuext/api.h"
#include "partition.h"

namespace upmemeval {
namespace partition {

constexpr size_t mutex_pool_size = 32;
class Partitioner {
 private:
  ::arrow::internal::ThreadPool* thread_pool_;
  std::shared_ptr<::arrow::Schema> schema_;             // Schema of the dataset
  std::vector<std::shared_ptr<Partition>> partitions_;  // Target partitions
  ::arrow::BufferVector buffers_;  // buffers for capturing the outputs from the DPUs
  std::vector<::arrow::Future<void*>> futures_;
  std::vector<std::vector<size_t>> offsets_;  // (scatter/gather only) offset vectors for
                                             // each partition, used by all threads
  std::array<std::mutex, mutex_pool_size>
      mutex_pool;  // mutex pool for concurrent access to vectors in offsets
  std::vector<uint32_t>
      output_shifts_;  // (scatter/gather only) random output shifts. Needed for an
                       // efficient DPU->Host scatter/gather

 public:
  Partitioner(::arrow::internal::ThreadPool* thread_pool,
              std::shared_ptr<::arrow::Schema> schema);
  ~Partitioner();

  const std::vector<std::shared_ptr<Partition>>& partitions() { return partitions_; }

  /**
   * @brief Allocate target partition and their owned buffers
   *
   * @param nr_partitions
   * @param partition_rows
   * @return ::arrow::Status
   */
  ::arrow::Status AllocatePartitions(uint32_t nr_partitions, int64_t partition_rows);

  /**
   * @brief Allocate permanent buffers to capture results from DPU
   *
   * @param nr_buffers
   * @param buffer_size
   * @return ::arrow::Result<::arrow::BufferVector>
   */
  ::arrow::Result<::arrow::BufferVector> AllocateBuffers(size_t nr_buffers,
                                                         size_t buffer_size);

  /**
   * @brief Allocate buffer for offsets used in scatter/gather
   *
   * @param nr_partitions
   * @param nr_dpus
   * @return ::arrow::Status
   */
  ::arrow::Status AllocateOffsets(size_t nr_partitions, size_t nr_dpus);

  /**
   * @brief Generate random shifts for scatter/gather
   *
   * @param nr_partitions
   * @param nr_dpus
   * @return ::arrow::Status
   */
  ::arrow::Status GenerateRandomShifts(size_t nr_partitions, size_t nr_dpus);

  /**
   * @brief Generate zero shifts for scatter/gather
   *
   * @param nr_partitions
   * @param nr_dpus
   * @return ::arrow::Status
   */
  ::arrow::Status GenerateZeroShifts(size_t nr_dpus);

  /**
   * @brief Run the partition DPU kernel on the partition key column of the input batches
   *
   * @return ::arrow::Result<std::vector<std::vector<uint32_t>>> partitions metadata
   */
  ::arrow::Result<std::vector<std::vector<uint32_t>>> PartitionKernel(
      dpu::DpuSet& system_, arrow::RecordBatchVector& batches, size_t batches_offset,
      size_t column_index);

  /**
   * @brief Run the partition DPU kernel on the partition key column of the input batches
   *
   * @return :arrow::Status
   */
  ::arrow::Status PartitionKernel(dpu::DpuSet& system_, arrow::RecordBatchVector& batches,
                                  size_t batches_offset, size_t column_index,
                                  size_t nr_partitions,
                                  const std::vector<uint32_t>& dpu_offset,
                                  std::vector<std::vector<uint32_t>>& metadata);

  /**
   * @brief Run the take DPU kernel on a non-key column of the input batches
   *
   * @param partitions_metadata
   * @return ::arrow::Status
   */
  ::arrow::Status TakeKernel(dpu::DpuSet& system_, arrow::RecordBatchVector& batches,
                             size_t batches_offset, size_t column_index);

  /**
   * @brief Load kernel results into the local buffers
   *
   * @return ::arrow::Status
   */
  ::arrow::Status LoadBuffers(dpu::DpuSet& system_);
  ::arrow::Status LoadBuffers(dpu::DpuSet& system_, size_t batch_index,
                              size_t column_index, size_t buffer_length);
  ::arrow::Status LoadBuffers(dpu::DpuSet& system_, ::arrow::BufferVector& buffers,
                              size_t batch_index, int column_index);
  ::arrow::Status LoadBuffersWithLog(dpu::DpuSet& system_);

  /**
   * @brief Process loaded buffers and move their content into the target partitions
   *
   * @param partitions_metadata
   * @return ::arrow::Status
   */
  ::arrow::Status BackgroundProcessBuffers(
      const std::vector<std::vector<uint32_t>>& partitions_metadata, size_t column_index);

  /**
   * @brief Get offsets from metadata.
   * Uses the metadata to get the starting positions where each DPU should start writing
   * each partition. The calculation is run concurrently for each rank. Each rank acquires
   * offsets in a round-robin fashion.
   *
   * @param system_ [in] The DPU set
   * @param partitions_metadata [in] Metadata received from the DPUs
   * @param batches_offset [in] Current batch * number of DPUs
   * @param offsets [out] The offsets to write to
   * @param dpu_offset [in] The index of the first DPU in each rank
   * @return ::arrow::Status
   */
  ::arrow::Status GetOffsets(
      dpu::DpuSet& system_, const std::vector<std::vector<uint32_t>>& partitions_metadata,
      size_t batches_offset, const std::vector<uint32_t>& dpu_offset);

  /**
   * @brief Directly load the buffers into the target partitions
   *
   * @param system_
   * @return ::arrow::Status
   */
  ::arrow::Status LoadPartitions(
      dpu::DpuSet& system_, const std::vector<std::vector<uint32_t>>& partitions_metadata,
      size_t batches_offset, size_t column_index, uint32_t buffer_length);

  /**
   * @brief Wait until all pending background tasks complete
   *
   * @return ::arrow::Status
   */
  ::arrow::Status WaitForBackgroundTasks();

  /**
   * @brief Get block information for the sg_xfer function
   *
   * @param out [out] The block information if it exists
   * @param dpu_index [in] The index of the DPU
   * @param block_index [in] The index of the block (partition)
   * @param args [in] The context for the function
   * @return true The block exists
   * @return false The block does not exist
   */
  static bool get_block(struct sg_block_info* out, uint32_t dpu_index,
                        uint32_t block_index, void* args);
};

}  // namespace partition
}  // namespace upmemeval

// extern "C" {
// bool get_block_simple(__attribute__((unused)) struct ::sg_block_info *out, __attribute__((unused)) uint32_t dpu_index,
//                 __attribute__((unused)) uint32_t block_index, __attribute__((unused)) void *args);
// }