#include "aggr_native.h"

#include <iostream>
#include <memory>

#include <arrow/compute/exec/exec_plan.h>
#include <arrow/status.h>

#include "generator/generator.h"

namespace upmemeval {

namespace aggr {

namespace cp = ::arrow::compute;

arrow::Result<std::shared_ptr<::arrow::Table>> ExecutePlanAndCollectAsTable(
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

template<class ArrayType>
arrow::Status AggrNative<ArrayType>::Prepare() {
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
  // // This filter removes all rows where the value of the "v" column is greater than or
  // // equals 1<<30.
  // cp::Expression filter_opt = cp::less(cp::field_ref("v"), cp::literal(1 << 30));
  // ARROW_ASSIGN_OR_RAISE(cp::ExecNode * filter,
  //                       cp::MakeExecNode("filter", plan_.get(), {source},
  //                                        cp::FilterNodeOptions{filter_opt}));

  // Aggregate node
  auto aggregate_options = cp::AggregateNodeOptions{/*aggregates=*/{{fn_, nullptr}},
                                                    /*targets=*/{"v"},
                                                    /*names=*/{fn_ + "(v)"}};
  ARROW_ASSIGN_OR_RAISE(
      cp::ExecNode * aggregate,
      cp::MakeExecNode("aggregate", plan_.get(), {source}, aggregate_options));

  // Sink node
  ARROW_RETURN_NOT_OK(
      cp::MakeExecNode("sink", plan_.get(), {aggregate}, cp::SinkNodeOptions{&sink_}));

  // validate the ExecPlan
  ARROW_RETURN_NOT_OK(plan_->Validate());

  return arrow::Status::OK();
}

template<typename ArrayType>
arrow::Result<typename ArrayType::value_type> AggrNative<ArrayType>::Run() {
  auto sink_reader =
      cp::MakeGeneratorReader(schema_, std::move(sink_), context_->memory_pool());
  ARROW_ASSIGN_OR_RAISE(auto result, ExecutePlanAndCollectAsTable(plan_, sink_reader));
  auto array = std::dynamic_pointer_cast<::arrow::UInt64Array>(result->column(0)->chunk(0));
  return array->Value(0);
}

// explicit instantiations
template class AggrNative<::arrow::UInt64Array>;
template class AggrNative<::arrow::DoubleArray>;

}  // namespace aggr
}  // namespace upmemeval
