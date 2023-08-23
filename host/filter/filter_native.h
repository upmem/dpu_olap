#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>

#include "generator/generator.h"
#include "timer/timer.h"

namespace upmemeval {

namespace filter {

namespace cp = ::arrow::compute;

class FilterNative {
 public:
  FilterNative(std::shared_ptr<arrow::Schema> schema, arrow::RecordBatchVector batches)
      : schema_(schema), batches_(generator::ToExecBatches(batches)) {}

  arrow::Status Prepare();

  arrow::Result<std::shared_ptr<arrow::Table>> GetResult();
  arrow::Result<uint64_t> Run();

  std::shared_ptr<timer::Timers> Timers() { return nullptr; }

 private:
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<arrow::compute::ExecBatch> batches_;
  cp::ExecContext* context_;
  std::shared_ptr<cp::ExecPlan> plan_;
  arrow::AsyncGenerator<arrow::util::optional<cp::ExecBatch>> sink_;
};

}  // namespace filter
}  // namespace upmemeval
