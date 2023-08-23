#pragma once

#include <arrow/api.h>
#include <stdint.h>
#include <string>

#include "partition/partition.h"

namespace dpu {

class DpuSet;

/**
 * @brief Get the length of the data buffer of an arrow array from a record batch vector.
 *
 * @param batches record batch vector
 * @param batch_index
 * @param column_index
 * @return arrow::Result<uint32_t>
 */
arrow::Result<uint32_t> arrow_data_buffer_length(arrow::RecordBatchVector& batches,
                                                 uint64_t batch_index, int column_index);

/**
 * @brief Copy data buffers from arrow data to DPUs
 *
 * Note that this function only copies a single data buffer from a single column and
 * assumes the data type is a non-nullable fixed size binary. Future work should use the
 * Arrow C Data Interface to copy full structures to and from DPUs.
 *
 * Note that it is assumed that there are at least #DPUs batches starting from the
 * specified batches_offset.
 *
 * @param system_ The DPU set
 * @param symbol The symbol in the DPU binary to copy to
 * @param batches The batches to copy from (only copies from #DPUs batches)
 * @param batches_offset The offset in the batches vector to start the copy from
 * @param column_index The column in the batch to copy
 * @param async true if the transfert should be asynchronous
 * @return arrow::Status
 */
arrow::Status arrow_copy_to_dpus(DpuSet& system_, std::string symbol,
                                 const arrow::RecordBatchVector& batches, size_t batches_offset,
                                 size_t column_index, bool async = false);

/**
 * @brief Same as above, but performs a scatter/gather transfer to the DPUs
 * 
 * @param system_ The DPU set
 * @param symbol The symbol in the DPU binary to copy to
 * @param buffers The batches to copy from (only copies from #DPUs batches)
 * @param batches_offset The offset in the batches vector to start the copy from
 * @param column_index The column in the batch to copy
 * @param metadata The metadata to use for the scatter/gather transfer
 * @param async true if the transfert should be asynchronous
 * @return arrow::Status 
 */
arrow::Status arrow_copy_to_dpus(DpuSet& system_, std::string symbol,
                                 const arrow::RecordBatchVector& buffers, size_t batches_offset,
                                 size_t column_index,
                                 const std::vector<std::vector<uint32_t>>& metadata,
                                 bool async = false);

/**
 * @brief Get array vector from DPUs
 *
 * @param system The DPU set
 * @param symbol The symbol in the DPU binary to copy from
 * @param dtype The Array data type
 * @param length The number of elements to copy (per array)
 * @param async true if the transfert should be asynchronous
 * @return arrow::Result<arrow::ArrayVector>
 */
arrow::Result<arrow::ArrayVector> arrow_copy_from_dpus(
    DpuSet& system, std::string symbol, std::shared_ptr<arrow::DataType> dtype,
    uint32_t length, bool async = false);

/**
 * @brief Copy buffers from DPUs to a vector of arrow buffers
 * 
 * @param system The DPU set
 * @param symbol The symbol in the DPU binary to copy from
 * @param buffers The destinations buffers to copy to
 * @param buffer_size The size of each buffer (all buffers must be of the same length)
 * @param async true if the transfert should be asynchronous
 * @return arrow::Status 
 */
arrow::Status arrow_copy_from_dpus(DpuSet& system,
                                   std::string symbol,
                                   arrow::BufferVector buffers,
                                   uint64_t buffer_size,
                                   bool async = false);

arrow::Status arrow_copy_from_dpus_offset(DpuSet& system,
                                   std::string symbol,
                                   arrow::BufferVector& buffers,
                                   uint64_t buffer_size,
                                   size_t host_offset,
                                   size_t nr_columns,
                                   bool async = false);

/**
 * @brief Copy buffers from DPUs to a vector of arrow buffers
 * 
 * @param system The DPU set
 * @param symbol The symbol in the DPU binary to copy from
 * @param dtype The Array data type
 * @param length The number of elements to copy (per dpu)
 * @param async true if the transfert should be asynchronous
 * @return arrow::Result<arrow::ArrayVector> 
 */
arrow::Result<arrow::ArrayVector> arrow_copy_from_dpus(
    DpuSet& system, std::string symbol, std::shared_ptr<arrow::DataType> dtype,
    std::vector<std::vector<uint32_t>>::const_iterator length, bool async = false);

/**
 * @brief Copy buffers from DPUs to a vector of Partitions
 * 
 * @param system The DPU set
 * @param symbol The symbol in the DPU binary to copy from
 * @param partitions The partitions to copy to
 * @param batch_index the starting batch index
 * @param column_index the column to transfer
 * @param buffer_length the length of buffers
 * @param async true if the transfert should be asynchronous
 * @return arrow::Status 
 */
arrow::Status arrow_copy_from_dpus(
    DpuSet& system, std::string symbol,
    std::vector<std::shared_ptr<upmemeval::partition::Partition>> partitions,
    size_t batch_index, size_t column_index, size_t buffer_length, bool async);

}  // namespace dpu
