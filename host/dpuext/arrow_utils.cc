#include "arrow_utils.h"

#include <arrow/compute/api_scalar.h>
#include <dpuext/status.h>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <dpuext/dpuext.hpp>
#include <iostream>
#include <string>
#include <arrow/compute/api.h>

#include "dpu.h"
#include "umq/bitops.h"

namespace dpu {

#define MAX_NR_DPU 2048
uint8_t *BUFFERS[MAX_NR_DPU][MAX_NR_DPU+1];
uint32_t LENGTHS[MAX_NR_DPU][MAX_NR_DPU+1];
uint8_t NIL_BUFFER[MAX_NR_DPU][8 * sizeof(uint8_t)];

const size_t ARROW_DATA_BUFFER_INDEX = 1;

arrow::Result<uint32_t> arrow_data_buffer_length(arrow::RecordBatchVector& batches,
                                                 uint64_t batch_index,
                                                 int32_t column_index) {
  if (batch_index > batches.size() - 1) {
    return arrow::Status::Invalid("Invalid batch index");
  }

  if (column_index > batches[batch_index]->num_columns() - 1) {
    return arrow::Status::Invalid("Invalid column index");
  }

  // XXX: narrow cast
  return static_cast<uint32_t>(
      batches[batch_index]->column(column_index)->data()->length);
}

int64_t get_byte_width(std::shared_ptr<arrow::DataType> dtype) {
  assert(dtype->layout().buffers[ARROW_DATA_BUFFER_INDEX].kind ==
         arrow::DataTypeLayout::FIXED_WIDTH);
  return dtype->layout().buffers[ARROW_DATA_BUFFER_INDEX].byte_width;
}

arrow::Status arrow_copy_to_dpus(DpuSet& system_, std::string symbol,
                                 const arrow::RecordBatchVector& batches, size_t batches_offset,
                                 size_t column_index, bool async) {
  // Detect buffer size in bytes
  auto dtype = batches[batches_offset]->column(column_index)->type();
  auto transfer_size = 0L;

  // Rank transfer buffers
  auto nr_dpus = system_.dpus().size();
  for (size_t i = 0; i < nr_dpus; ++i) {
    assert((i + batches_offset) < batches.size());  // ASSUMPTION: #batches mod #dpus is 0
    auto batch_index = i + batches_offset;
    auto column = batches[batch_index]->column(column_index);
    auto buffer_size = batches[batch_index]->num_rows() * get_byte_width(dtype);
    if (buffer_size > transfer_size) {
      transfer_size = buffer_size;
    }
    auto values_buffer =
        column->data()->GetMutableValues<uint32_t>(ARROW_DATA_BUFFER_INDEX);
    DPU_RETURN_NOT_OK(dpu_prepare_xfer(system_.dpus()[i]->unsafe(), values_buffer));
  }
  transfer_size = ROUND_UP_TO_MULTIPLE_OF_8(transfer_size);
  DPU_RETURN_NOT_OK(dpu_push_xfer(system_.unsafe(), DPU_XFER_TO_DPU, symbol.c_str(), 0,
                                  transfer_size, async ? DPU_XFER_ASYNC : DPU_XFER_DEFAULT));

  return arrow::Status::OK();
}

/**
 * @brief Structure used to provide context to the sg_xfer function
 */
typedef struct sg_xfer_context {
  const std::vector<std::vector<uint32_t>>* metadata;
  size_t batches_offset;
  const arrow::RecordBatchVector* buffers;
  size_t column_index;
} sg_xfer_context;

static bool get_block(struct ::sg_block_info *out, uint32_t dpu_index,
               uint32_t block_index, void *args) {
  auto ctx = static_cast<sg_xfer_context *>(args);
  auto nr_partitions = ctx->metadata->size();

  if (block_index >= nr_partitions) {
    return false;
  }

  auto partition_idx = dpu_index + ctx->batches_offset;
  auto from = (partition_idx == 0) ? 0 : (*ctx->metadata)[block_index][partition_idx - 1];
  auto length = (*ctx->metadata)[block_index][partition_idx] - from;
  auto byte_width = sizeof(uint32_t);
  auto src = (*ctx->buffers)[block_index]->column(ctx->column_index)->data()->GetMutableValues<uint32_t>(ARROW_DATA_BUFFER_INDEX);

  out->length = length * byte_width;
  out->addr = reinterpret_cast<uint8_t *>(src + from);

  return true;
}

arrow::Status arrow_copy_to_dpus(DpuSet& system_, std::string symbol,
                                 const arrow::RecordBatchVector& buffers, size_t batches_offset,
                                 size_t column_index,
                                 const std::vector<std::vector<uint32_t>>& metadata,
                                 bool async) {
  sg_xfer_context sc_args = {
      .metadata = &metadata,
      .batches_offset = batches_offset,
      .buffers = &buffers,
      .column_index = column_index,
  };
  get_block_t get_block_info = {
      .f = get_block,
      .args = &sc_args,
      .args_size = sizeof(sc_args),
  };

  auto byte_width = sizeof(uint32_t);
  auto nr_dpus = system_.dpus().size();
  auto max_length = 0UL;
  for (size_t dpu_idx = 0; dpu_idx < nr_dpus; ++dpu_idx) {
    size_t total_length = 0UL;
    auto partition_idx = dpu_idx + batches_offset;
    for (size_t batch_idx = 0; batch_idx < metadata.size(); ++batch_idx) {
      auto from = (partition_idx == 0) ? 0 : metadata[batch_idx][partition_idx - 1];
      auto length = metadata[batch_idx][partition_idx] - from;
      total_length += length;
    }
    max_length = std::max(max_length, total_length);
  }
  max_length = ROUND_UP_TO_MULTIPLE_OF_8(max_length * byte_width);

  DPU_ASSERT(dpu_push_sg_xfer(system_.unsafe(), DPU_XFER_TO_DPU, symbol.c_str(), 0, max_length,
                   &get_block_info,
                   static_cast<dpu_sg_xfer_flags_t>(
                       DPU_SG_XFER_DISABLE_LENGTH_CHECK |
                       (async ? DPU_SG_XFER_ASYNC : DPU_SG_XFER_DEFAULT))));

  return arrow::Status::OK();
}

arrow::Status arrow_copy_from_dpus(DpuSet& system_,
                                   std::string symbol,
                                   arrow::BufferVector buffers,
                                   uint64_t buffer_size,
                                   bool async) {
  // Rank transfer buffers
  for (size_t i = 0; i < system_.dpus().size(); ++i) {
      auto dpu = system_.dpus()[i];
      auto buffer = buffers[i];
      // Prepare transfer to buffer
      DPU_RETURN_NOT_OK(dpu_prepare_xfer(dpu->unsafe(), buffer->mutable_data()));
  }
  // Push from DPUs
  auto transfer_buffer_size = ROUND_UP_TO_MULTIPLE_OF_8(buffer_size);  // 8 bytes alignment
  DPU_RETURN_NOT_OK(dpu_push_xfer(system_.unsafe(), DPU_XFER_FROM_DPU, symbol.c_str(), 0,
                                  transfer_buffer_size, async ? DPU_XFER_ASYNC : DPU_XFER_DEFAULT));
  
  return arrow::Status::OK();
}

arrow::Status arrow_copy_from_dpus_offset(DpuSet& system_,
                                          std::string symbol,
                                          arrow::BufferVector& buffers,
                                          uint64_t buffer_size,
                                          size_t host_offset,
                                          size_t nr_columns,
                                          bool async) {
  // Rank transfer buffers
  for (size_t i = 0; i < system_.dpus().size(); ++i) {
      auto dpu = system_.dpus()[i];
      auto buffer = buffers[i * nr_columns + host_offset];
      // Prepare transfer to buffer
      DPU_RETURN_NOT_OK(dpu_prepare_xfer(dpu->unsafe(), buffer->mutable_data()));
  }
  // Push from DPUs
  auto transfer_buffer_size = ROUND_UP_TO_MULTIPLE_OF_8(buffer_size);  // 8 bytes alignment
  DPU_RETURN_NOT_OK(dpu_push_xfer(system_.unsafe(), DPU_XFER_FROM_DPU, symbol.c_str(), 0,
                                  transfer_buffer_size, async ? DPU_XFER_ASYNC : DPU_XFER_DEFAULT));
  
  return arrow::Status::OK();
}

arrow::Result<arrow::ArrayVector> arrow_copy_from_dpus(
    DpuSet& system_, std::string symbol, std::shared_ptr<arrow::DataType> dtype,
    uint32_t length, bool async) {
  // Allocate buffers
  auto buffer_size = length * get_byte_width(dtype);
  arrow::BufferVector buffers;
  buffers.reserve(system_.dpus().size());
  for (size_t i = 0; i < system_.dpus().size(); ++i) {
    // Allocate buffer
    ARROW_ASSIGN_OR_RAISE(auto maybe_buffer, arrow::AllocateBuffer(buffer_size));
    std::shared_ptr<arrow::Buffer> buffer = std::move(maybe_buffer);
    buffers.push_back(buffer);
  }
  
  ARROW_RETURN_NOT_OK(arrow_copy_from_dpus(system_, symbol, buffers, buffer_size, async));

  // Build result
  arrow::ArrayVector out;
  out.reserve(system_.dpus().size());
  for (auto& buffer : buffers) {
    auto array = arrow::MakeArray(arrow::ArrayData::Make(dtype, length, {nullptr, buffer}));
    out.push_back(std::move(array));
  }
  return out;
}

arrow::Result<arrow::ArrayVector> arrow_copy_from_dpus(
    DpuSet& system_, std::string symbol, std::shared_ptr<arrow::DataType> dtype,
    std::vector<std::vector<uint32_t>>::const_iterator length, bool async) {
  // Allocate buffers
  uint32_t max_buffer_size =
      std::max_element(length, length + system_.dpus().size(),
                       [](const std::vector<uint32_t>& a,
                          const std::vector<uint32_t>& b) { return a[0] < b[0]; })[0][0] *
      get_byte_width(dtype);
  arrow::BufferVector buffers;
  buffers.reserve(system_.dpus().size());
  for (size_t i = 0; i < system_.dpus().size(); ++i) {
    // Allocate buffer
    ARROW_ASSIGN_OR_RAISE(auto maybe_buffer, arrow::AllocateBuffer(max_buffer_size));
    std::shared_ptr<arrow::Buffer> buffer = std::move(maybe_buffer);
    buffers.push_back(buffer);
  }
  
  ARROW_RETURN_NOT_OK(arrow_copy_from_dpus(system_, symbol, buffers, max_buffer_size, async));

  // Build result
  arrow::ArrayVector out;
  out.reserve(system_.dpus().size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    auto array = arrow::MakeArray(arrow::ArrayData::Make(dtype, length[i][0], {nullptr, std::move(buffers[i])}));
    out.push_back(std::move(array));
  }
  return out;
}

arrow::Status arrow_copy_from_dpus(
    DpuSet& system_, std::string symbol,
    std::vector<std::shared_ptr<upmemeval::partition::Partition>> partitions,
    size_t batch_index, size_t column_index, size_t buffer_length, bool async) {
  auto byte_width = partitions[0]->GetByteWidth(column_index);

  for (size_t partition_idx_unshifted = 0; partition_idx_unshifted < system_.dpus().size();
       partition_idx_unshifted++) {
    auto partition_idx = partition_idx_unshifted + batch_index;
    auto partition = partitions[partition_idx];
    auto dpu = system_.dpus()[partition_idx_unshifted];
    auto buffer = partition->Write(column_index, buffer_length);
    // Prepare transfer to buffer
    DPU_RETURN_NOT_OK(dpu_prepare_xfer(dpu->unsafe(), buffer));
  }

  auto transfer_buffer_size = ROUND_UP_TO_MULTIPLE_OF_8(buffer_length * byte_width);  // 8 bytes alignment
  DPU_RETURN_NOT_OK(dpu_push_xfer(system_.unsafe(), DPU_XFER_FROM_DPU, symbol.c_str(), 0,
                                  transfer_buffer_size, async ? DPU_XFER_ASYNC : DPU_XFER_DEFAULT));

  return arrow::Status::OK();
}

}  // namespace dpu
