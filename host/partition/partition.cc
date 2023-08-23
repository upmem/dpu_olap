#include "partition.h"
#include <stdexcept>

namespace upmemeval {
namespace partition {

::arrow::Status Partition::Allocate() {
  for (int column_index = 0; column_index < schema_->num_fields(); ++column_index) {
    auto size = partition_length_ * GetByteWidth(column_index);
    // TODO: change to resizable buffer?
    ARROW_ASSIGN_OR_RAISE(auto maybe_buffer, ::arrow::AllocateBuffer(size));
    std::shared_ptr<arrow::Buffer> buffer = std::move(maybe_buffer);
    buffers_.push_back(buffer);
    next_write_index_.push_back(std::make_unique<std::atomic<size_t>>(0));
  }
  return ::arrow::Status::OK();
}

uint8_t* Partition::Write(size_t column_index, size_t size) {
  size_t start_index = next_write_index_[column_index]->fetch_add(size);
  if (start_index > partition_length_) {
    throw std::runtime_error("target partition size is too small");
  }
  return buffers_[column_index]->mutable_data() +
         start_index * GetByteWidth(column_index);
}

size_t Partition::PrepareWrite(size_t column_index, size_t size) {
  size_t start_index = next_write_index_[column_index]->fetch_add(size);
  if (start_index > partition_length_) {
    throw std::runtime_error("target partition size is too small");
  }
  return start_index;
}

uint8_t* Partition::UnsafeWrite(size_t column_index, size_t size, size_t start_index) {
  if (start_index + size > partition_length_) {
    throw std::runtime_error("target partition size is too small");
  }
  return buffers_[column_index]->mutable_data() +
         start_index * GetByteWidth(column_index);
}

uint8_t* Partition::UnsafeWrite(size_t column_index) {
  return buffers_[column_index]->mutable_data();
}

::arrow::Result<::arrow::RecordBatchVector> ToRecordBatches(
    const std::vector<std::shared_ptr<Partition>>& partitions) {
  ::arrow::RecordBatchVector record_batches;
  record_batches.reserve(partitions.size());
  // int i = 0;
  for (auto& partition : partitions) {
    ::arrow::ArrayDataVector data_vector;
    auto fields = partition->schema()->fields();
    // printf("partition %d rows %lu\n", i++, partition->num_rows());
    for (size_t field_index = 0; field_index < fields.size(); ++field_index) {
      auto& field = fields[field_index];
      auto data_buffer = partition->buffers()[field_index];
      data_vector.push_back(::arrow::ArrayData::Make(field->type(), partition->num_rows(),
                                                     {nullptr, data_buffer}));
    }
    record_batches.push_back(::arrow::RecordBatch::Make(
        partition->schema(), partition->num_rows(), data_vector));
  }
  return record_batches;
}

::arrow::Result<std::shared_ptr<::arrow::Table>> ToTable(
    const std::vector<std::shared_ptr<Partition>>& partitions) {
  ARROW_ASSIGN_OR_RAISE(auto rb, partition::ToRecordBatches(partitions));
  return ::arrow::Table::FromRecordBatches(rb);
}

}  // namespace partition
}  // namespace upmemeval
