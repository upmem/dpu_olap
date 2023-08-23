#include "filter_native.h"

#include <arrow/compute/exec/exec_plan.h>
#include <arrow/status.h>
#include <iostream>

#include "generator/generator.h"

namespace upmemeval {

namespace filter {

namespace cp = ::arrow::compute;

arrow::Result<std::shared_ptr<arrow::Table>> ExecutePlan(
    std::shared_ptr<cp::ExecPlan> plan,
    std::shared_ptr<arrow::RecordBatchReader> sink_reader) {
  // start the ExecPlan
  ARROW_RETURN_NOT_OK(plan->StartProducing());

  // collect sink_reader into a Table
  ARROW_ASSIGN_OR_RAISE(auto response_table,
                        arrow::Table::FromRecordBatchReader(sink_reader.get()));

  // stop producing
  plan->StopProducing();
  // plan mark finished
  auto future = plan->finished();
  auto status = future.status();
  if (status.ok()) {
    return response_table;
  }
  return future.status();
}

arrow::Status FilterNative::Prepare() {
  // Input generator
  auto data = generator::MakeAsyncGenerator(batches_, true);

  // Multi threaded execution context
  // NOTE: set_use_threads is unused, but still all threads are used because we create a
  // context with an executor
  context_ = cp::default_exec_context();
  *context_ =
      cp::ExecContext(arrow::default_memory_pool(), arrow::internal::GetCpuThreadPool());
  context_->set_use_threads(true);

  // Create plan
  ARROW_ASSIGN_OR_RAISE(plan_, cp::ExecPlan::Make(context_));

  // Source node
  auto source_node_options = cp::SourceNodeOptions{schema_, data};
  ARROW_ASSIGN_OR_RAISE(cp::ExecNode * source,
                        cp::MakeExecNode("source", plan_.get(), {}, source_node_options));

  // Filter node
  // This filter removes all rows where the value of the "v" column is greater than or
  // equals 1<<30.
  cp::Expression filter_opt = cp::less(cp::field_ref("v"), cp::literal(1 << 30));
  ARROW_ASSIGN_OR_RAISE(cp::ExecNode * filter,
                        cp::MakeExecNode("filter", plan_.get(), {source},
                                         cp::FilterNodeOptions{filter_opt}));

  // Sink node
  ARROW_RETURN_NOT_OK(
      cp::MakeExecNode("sink", plan_.get(), {filter}, cp::SinkNodeOptions{&sink_}));

  // validate the ExecPlan
  ARROW_RETURN_NOT_OK(plan_->Validate());

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> FilterNative::GetResult() {
  auto sink_reader =
      cp::MakeGeneratorReader(schema_, std::move(sink_), context_->memory_pool());
  return ExecutePlan(plan_, sink_reader);
}

arrow::Result<uint64_t> FilterNative::Run() {
  auto sink_reader =
      cp::MakeGeneratorReader(schema_, std::move(sink_), context_->memory_pool());
  return ExecutePlan(plan_, sink_reader)->get()->num_rows();
}

}  // namespace filter
}  // namespace upmemeval
