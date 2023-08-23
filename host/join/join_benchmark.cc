#include <arrow/api.h>
#include <benchmark/benchmark.h>
#include <iostream>

#include "dpuext/api.h"
#include "generator/generator.h"
#include "timer/timer.h"

#include "umq/cflags.h"
#include "umq/kernels.h"

#include "system/system.h"

#include "join_dpu.h"
#include "join_native.h"

using namespace dpu;

namespace upmemeval {

using namespace generator;
using namespace timer;

namespace join {

template <class T>
void BM_Join(benchmark::State& state, T& joiner) {
  uint64_t total_rows = 0;
  auto nr_dpus = state.range(3);
  double nr_ranks = (nr_dpus + 63) / 64.0;
  for (auto _ : state) {
    auto prepared = joiner.Prepare();
    if (!prepared.ok()) {
      auto err = std::string("Prepare failed with error: ") + prepared.ToString();
      state.SkipWithError(err.c_str());
      break;
    }

    arrow::Result<std::shared_ptr<arrow::Table>> result = joiner.Run();
    if (!result.ok()) {
      auto err = std::string("Run failed with error: ") + result.status().ToString();
      state.SkipWithError(err.c_str());
      break;
    }

    total_rows += result->get()->num_rows();

    // Add timers
    auto timers = joiner.Timers();
    if (timers != nullptr) {
      for (auto& r : timers->get()) {
        auto name = r.first;
        auto timer = r.second;
        auto& counter = state.counters[name];
        counter.value +=
            std::chrono::duration_cast<std::chrono::milliseconds>(timer->Result())
                .count() /
            nr_ranks;
      }
    }
  }

  std::cout << "Total Rows: " << total_rows << " Iterations: " << state.iterations()
            << std::endl;
}

class PartitionedBatchGeneratorFixture : public benchmark::Fixture {
 public:
  PartitionedBatchGeneratorFixture() : rng_(42) {}

  void SetUp(::benchmark::State& state) override {
    auto num_batches = state.range(0);
    auto left_batch_size = state.range(1);
    auto right_batch_size = state.range(2);
    assert(num_batches > 0);
    assert(left_batch_size > 0);
    assert(right_batch_size > 0);

    if (right_batches_.size() != static_cast<size_t>(num_batches) ||
        right_batches_[0]->num_rows() != right_batch_size) {
      auto schema =
          arrow::schema({arrow::field("x", arrow::uint32(), /*nullable=*/false)});
      auto right_batches =
          generator::MakeRandomRecordBatches(rng_, schema, num_batches, right_batch_size);
      auto right_pk_column = generator::MakeIndexColumn(num_batches, right_batch_size);
      right_batches_ =
          generator::AddColumn("pk", right_batches, right_pk_column.ValueOrDie());
      right_schema_ = right_batches_[0]->schema();
    }

    std::cout << right_batches_.size() << " right batches generated" << std::endl;

    if (left_batches_.size() != static_cast<size_t>(num_batches) ||
        left_batches_[0]->num_rows() != left_batch_size) {
      auto schema =
          arrow::schema({arrow::field("y", arrow::uint32(), /*nullable=*/false)});
      auto left_batches =
          generator::MakeRandomRecordBatches(rng_, schema, num_batches, left_batch_size);
      auto left_fk_column = generator::MakeForeignKeyColumn(rng_, right_batch_size,
                                                            num_batches, left_batch_size);
      left_batches_ =
          generator::AddColumn("fk", left_batches, left_fk_column.ValueOrDie());
      left_schema_ = left_batches_[0]->schema();
    }

    std::cout << left_batches_.size() << " left batches generated" << std::endl;
  }

  void TearDown(::benchmark::State&) override {
    left_batches_.clear();
    right_batches_.clear();
  }

  int64_t total_items() {
    auto left_batch_total_size = left_batches_.size() * left_batches_[0]->num_rows() *
                                 left_batches_[0]->num_columns();
    auto right_batch_total_size = right_batches_.size() * right_batches_[0]->num_rows() *
                                  right_batches_[0]->num_columns();
    return left_batch_total_size + right_batch_total_size;
  }

  int64_t total_bytes() {
    // XXX assumes all columns are of the same uint32_t type
    return total_items() * sizeof(uint32_t);
  }

 protected:
  arrow::random::RandomArrayGenerator rng_;
  std::shared_ptr<arrow::Schema> left_schema_;
  std::shared_ptr<arrow::Schema> right_schema_;
  arrow::RecordBatchVector left_batches_;
  arrow::RecordBatchVector right_batches_;
};

BENCHMARK_DEFINE_F(PartitionedBatchGeneratorFixture, BM_JoinNative)
(benchmark::State& state) {
  bool partitioned = state.range(3);
  JoinNative joiner{left_schema_, right_schema_, left_batches_, right_batches_,
                    partitioned};
  BM_Join<>(state, joiner);
  state.SetItemsProcessed(total_items());
  state.SetBytesProcessed(total_bytes());
}

BENCHMARK_DEFINE_F(PartitionedBatchGeneratorFixture, BM_JoinDpu)
(benchmark::State& state) {
  auto nr_dpus = state.range(3);
  auto system_ = DpuSet::allocate(nr_dpus, "nrJobsPerRank=256,sgXferEnable=true");
  std::cout << "Allocated DPUs: " << system_.dpus().size() << std::endl;
  JoinDpu joiner{system_, left_schema_, right_schema_, left_batches_, right_batches_};
  BM_Join<>(state, joiner);
  state.SetItemsProcessed(total_items());
  state.SetBytesProcessed(total_bytes());
#if ENABLE_PERF
  state.counters["cycles"] = cycles_count(system_);
#endif
}

BENCHMARK_REGISTER_F(PartitionedBatchGeneratorFixture, BM_JoinNative)
    ->ArgNames({"Batches", "L-Batch-Size", "R-Batch-Size", "Partitioned"})
    ->Args({variables::scale_factor() << 5, 64 << 10, 64 << 10, true})
    ->Args({variables::scale_factor() << 5, 64 << 10, 64 << 10, false})
    // ->Args({variables::max_dpus(), 2 << 20, 2 << 20, true})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(PartitionedBatchGeneratorFixture, BM_JoinDpu)
    ->ArgNames({"Batches", "L-Batch-Size", "R-Batch-Size", "DPUs"})
    ->Args({variables::scale_factor(), 2 << 20, 2 << 20,
            std::min(variables::max_dpus(), variables::scale_factor())})
    // ->Args({variables::scale_factor() << 5, 64 << 10, 64 << 10,
    // std::min(variables::max_dpus(), variables::scale_factor() << 5)})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace join
}  // namespace upmemeval
