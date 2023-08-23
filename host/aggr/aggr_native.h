#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>

#include "generator/generator.h"
#include "timer/timer.h"

namespace upmemeval {

namespace aggr {

namespace cp = ::arrow::compute;

template<typename ArrayType>
class AggrNative {
 public:
  AggrNative(std::shared_ptr<arrow::Schema> schema, arrow::RecordBatchVector batches, std::string fn)
      : schema_(schema), batches_(generator::ToExecBatches(batches)), fn_(fn) {}

  arrow::Status Prepare();

  std::shared_ptr<timer::Timers> Timers() { return nullptr; }
 
  arrow::Result<typename ArrayType::value_type> Run();

 private:
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<arrow::compute::ExecBatch> batches_;
  std::string fn_;
  cp::ExecContext* context_;
  std::shared_ptr<cp::ExecPlan> plan_;
  arrow::AsyncGenerator<arrow::util::optional<cp::ExecBatch>> sink_;
};

}  // namespace filter
}  // namespace upmemeval
