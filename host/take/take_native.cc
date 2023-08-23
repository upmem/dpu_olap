#include "take_native.h"

#include <iostream>
#include <memory>

#include <arrow/status.h>

#include "generator/generator.h"

namespace upmemeval {

namespace take {

namespace cp = ::arrow::compute;

arrow::Status TakeNative::Prepare() { return arrow::Status::OK(); }

arrow::Result<std::shared_ptr<::arrow::Table>> TakeNative::Run() {
  auto pool = arrow::internal::GetCpuThreadPool();
  std::vector<arrow::Future<>> futures;

  ::arrow::RecordBatchVector result_batches(batches_.size());
  for(size_t i = 0; i < batches_.size(); ++i) {
    auto& batch = batches_[i];
    auto& indices = indices_batches_[i];
    ARROW_ASSIGN_OR_RAISE(auto future, pool->Submit([i, &batch, &indices, &result_batches, this]() {
      ARROW_ASSIGN_OR_RAISE(auto taken, cp::Take(batch->column(0), indices->column(0), cp::TakeOptions::NoBoundsCheck()));
      auto as_array = taken.make_array();
      result_batches[i] = ::arrow::RecordBatch::Make(schema_, as_array->length(), {as_array});
      return arrow::Status::OK();
    }));
    futures.push_back(future);
  }
  for (auto& fut : futures) {
    auto _ = fut.status();
  }
  return ::arrow::Table::FromRecordBatches(result_batches);
}

}  // namespace take
}  // namespace upmemeval
