#pragma once

#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/testing/random.h>
#include <arrow/util/async_generator.h>

namespace upmemeval {

namespace generator {

arrow::RecordBatchVector MakeRandomRecordBatches(
    ::arrow::random::RandomArrayGenerator& g,
    const std::shared_ptr<arrow::Schema>& schema, int num_batches, int batch_size);

arrow::Result<arrow::ArrayVector> MakeIndexColumn(int num_batches, int batch_size);

arrow::RecordBatchVector AddColumn(const std::string& name,
                                   const arrow::RecordBatchVector& batches,
                                   arrow::ArrayVector indexColumn);

arrow::Result<arrow::ArrayVector> MakeForeignKeyColumn(
    ::arrow::random::RandomArrayGenerator& g, uint32_t pk_batch_size, int32_t num_batches,
    int32_t batch_size);

std::vector<arrow::compute::ExecBatch> ToExecBatches(arrow::RecordBatchVector batches);

arrow::AsyncGenerator<arrow::util::optional<arrow::compute::ExecBatch>>
MakeAsyncGenerator(std::vector<arrow::compute::ExecBatch> batches, bool parallel);

template <typename T>
std::shared_ptr<::arrow::Array> ArrayOf(std::vector<typename T::c_type> vector) {
  ::arrow::NumericBuilder<T> builder;
  auto status = builder.AppendValues(std::move(vector));
  if (!status.ok()) {
    abort();
  }
  return builder.Finish().ValueOrDie();
}

std::shared_ptr<::arrow::RecordBatch> RecordBatchOf(
    std::vector<std::string> names, std::vector<std::shared_ptr<arrow::Array>> data);

}  // namespace generator
}  // namespace upmemeval
