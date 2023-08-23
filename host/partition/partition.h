#pragma once

#include <arrow/api.h>
namespace upmemeval {
namespace partition {

class Partition {
 private:
  // XXX: a relaxing assumption of single buffer per column
  std::vector<std::shared_ptr<arrow::Buffer>> buffers_;
  std::vector<std::unique_ptr<std::atomic<size_t>>> next_write_index_;
  std::shared_ptr<::arrow::Schema> schema_;
  size_t partition_length_;

 public:
  Partition(std::shared_ptr<::arrow::Schema> schema, size_t partition_length)
      : schema_(schema), partition_length_(partition_length) {}
  Partition(const Partition&) = delete;
  ~Partition() = default;

  std::vector<std::shared_ptr<arrow::Buffer>> buffers() { return buffers_; }

  std::shared_ptr<::arrow::Schema> schema() { return schema_; }

  int64_t num_rows() { return static_cast<int64_t>(*next_write_index_[0]); }

  ::arrow::Status Allocate();

  uint8_t* Write(size_t column_index, size_t size);
  size_t PrepareWrite(size_t column_index, size_t size);
  uint8_t* UnsafeWrite(size_t column_index, size_t size, size_t start_index);
  uint8_t* UnsafeWrite(size_t column_index);

  size_t GetByteWidth(size_t column_index) {
    const size_t ARROW_DATA_BUFFER_INDEX = 1;
    auto dtype = schema_->field(column_index)->type();
    assert(dtype->layout().buffers[ARROW_DATA_BUFFER_INDEX].kind ==
           arrow::DataTypeLayout::FIXED_WIDTH);
    return dtype->layout().buffers[ARROW_DATA_BUFFER_INDEX].byte_width;
  }
};

::arrow::Result<::arrow::RecordBatchVector> ToRecordBatches(
    const std::vector<std::shared_ptr<Partition>>& partitions);

::arrow::Result<std::shared_ptr<::arrow::Table>> ToTable(
    const std::vector<std::shared_ptr<Partition>>& partitions);

}  // namespace partition
}  // namespace upmemeval
