#include "join_native.h"
#include <arrow/compute/exec/exec_plan.h>
#include <arrow/compute/exec/options.h>
#include <arrow/pretty_print.h>
#include <iostream>
#include "generator/generator.h"

namespace upmemeval {
namespace join {

static bool g_verbose = false;
namespace cp = ::arrow::compute;

arrow::Status JoinNative::prepare_plan(
    std::vector<arrow::compute::ExecBatch> left_batches,
    std::vector<arrow::compute::ExecBatch> right_batches) {
  ARROW_ASSIGN_OR_RAISE(plan_, cp::ExecPlan::Make(context_));

  auto left_data = generator::MakeAsyncGenerator(left_batches, true);
  auto right_data = generator::MakeAsyncGenerator(right_batches, true);

  auto left_source_options = cp::SourceNodeOptions{left_schema_, left_data};
  auto right_source_options = cp::SourceNodeOptions{right_schema_, right_data};

  ARROW_ASSIGN_OR_RAISE(cp::ExecNode * left_source,
                        cp::MakeExecNode("source", plan_.get(), {}, left_source_options));
  ARROW_ASSIGN_OR_RAISE(
      cp::ExecNode * right_source,
      cp::MakeExecNode("source", plan_.get(), {}, right_source_options));

  arrow::compute::HashJoinNodeOptions join_opts{arrow::compute::JoinType::INNER,
                                                /*in_left_keys=*/{"fk"},
                                                /*in_right_keys=*/{"pk"},
                                                /*filter*/ arrow::compute::literal(true),
                                                /*output_suffix_for_left*/ "_l",
                                                /*output_suffix_for_right*/ "_r"};

  ARROW_ASSIGN_OR_RAISE(
      auto hashjoin,
      cp::MakeExecNode("hashjoin", plan_.get(), {left_source, right_source}, join_opts));
  output_schema_ = hashjoin->output_schema();
  if (g_verbose) {
    std::cout << "Output schema:\n";
    ARROW_RETURN_NOT_OK(arrow::PrettyPrint(*output_schema_, 1, &std::cout));
    std::cout << "\n---\n";
  }

  ARROW_RETURN_NOT_OK(
      cp::MakeExecNode("sink", plan_.get(), {hashjoin}, cp::SinkNodeOptions{&sink_}));

  // validate the ExecPlan
  ARROW_RETURN_NOT_OK(plan_->Validate());
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> JoinNative::run_plan() {
  std::shared_ptr<arrow::RecordBatchReader> sink_reader =
      cp::MakeGeneratorReader(output_schema_, std::move(sink_), context_->memory_pool());

  // start the ExecPlan
  ARROW_RETURN_NOT_OK(plan_->StartProducing());

  // collect sink_reader into a Table
  ARROW_ASSIGN_OR_RAISE(auto response_table,
                        arrow::Table::FromRecordBatchReader(sink_reader.get()));

  if (g_verbose)
    std::cout << "Results Rows: " << response_table->num_rows()
              << " Columns: " << response_table->num_columns() << std::endl;

  plan_->StopProducing();
  auto future = plan_->finished();
  auto status = future.status();
  if (status.ok()) {
    return response_table->RemoveColumn(response_table->schema()->GetFieldIndex("pk"))
        .ValueOrDie();
  }
  return future.status();
}

arrow::Status JoinNative::Prepare() {
  // XXX : see context comment used in native_filter context
  context_ = cp::default_exec_context();
  *context_ =
      cp::ExecContext(arrow::default_memory_pool(), arrow::internal::GetCpuThreadPool());
  context_->set_use_threads(true);

  if (!partitioned_) {
    return prepare_plan(left_batches_, right_batches_);
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> JoinNative::Run() {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  tables.reserve(left_batches_.size());

  if (!partitioned_) {
    return run_plan();
  }

  for (size_t i = 0; i < left_batches_.size(); i++) {
    auto left = left_batches_[i];
    auto right = right_batches_[i];
    ARROW_RETURN_NOT_OK(prepare_plan({left}, {right}));
    ARROW_ASSIGN_OR_RAISE(auto table, run_plan());
    tables.push_back(table);
  }

  return arrow::ConcatenateTables(tables);
}

}  // namespace join
}  // namespace upmemeval
