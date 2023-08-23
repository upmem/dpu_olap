#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/exec.h>
#include <memory>
#include <vector>

#include "generator/generator.h"
#include "timer/timer.h"

namespace upmemeval {
namespace join {
namespace cp = ::arrow::compute;

class JoinNative {
 public:
  JoinNative(std::shared_ptr<arrow::Schema> left_schema,
             std::shared_ptr<arrow::Schema> right_schema,
             arrow::RecordBatchVector left_batches,
             arrow::RecordBatchVector right_batches, bool partitioned = false)
      : left_schema_(left_schema),
        right_schema_(right_schema),
        left_batches_(generator::ToExecBatches(left_batches)),
        right_batches_(generator::ToExecBatches(right_batches)),
        partitioned_(partitioned) {
    assert(!partitioned || left_batches_.size() == right_batches_.size());
  }

  arrow::Status Prepare();

  arrow::Result<std::shared_ptr<arrow::Table>> Run();

  std::shared_ptr<timer::Timers> Timers() { return nullptr; }

 private:
  arrow::Status prepare_plan(std::vector<arrow::compute::ExecBatch> left,
                             std::vector<arrow::compute::ExecBatch> right);
  arrow::Result<std::shared_ptr<arrow::Table>> run_plan();

  std::shared_ptr<arrow::Schema> left_schema_;
  std::shared_ptr<arrow::Schema> right_schema_;
  std::vector<arrow::compute::ExecBatch> left_batches_;
  std::vector<arrow::compute::ExecBatch> right_batches_;
  std::shared_ptr<arrow::Schema> output_schema_;
  cp::ExecContext* context_;
  std::shared_ptr<cp::ExecPlan> plan_;
  arrow::AsyncGenerator<arrow::util::optional<cp::ExecBatch>> sink_;
  bool partitioned_;
};

}  // namespace join
}  // namespace upmemeval
