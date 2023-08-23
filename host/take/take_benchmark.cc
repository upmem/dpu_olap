#include <benchmark/benchmark.h>
#include <iostream>

#include "generator/generator.h"

#include "dpuext/api.h"
#include "take_dpu.h"
#include "take_native.h"

#include "system/system.h"

using namespace dpu;
using namespace benchmark;

namespace upmemeval {

using namespace generator;

namespace take {

template <typename T>
void BM_Take(benchmark::State& state, T& taker) {
  auto num_batches = state.range(0);
  auto batch_size = state.range(1);

  for (auto _ : state) {
    auto prepared = taker.Prepare();
    if (!prepared.ok()) {
      auto err = std::string("Prepare failed with error: ") + prepared.ToString();
      state.SkipWithError(err.c_str());
      break;
    }

    auto result = taker.Run();
    if (!result.ok()) {
      auto err = std::string("Run failed with error: ") + result.status().ToString();
      state.SkipWithError(err.c_str());
      break;
    }
    // Add timers
    auto timers = taker.Timers();
    if (timers != nullptr) {
      for (auto& r : timers->get()) {
        auto name = r.first;
        auto timer = r.second;
        auto& counter = state.counters[name];
        counter.value +=
            std::chrono::duration_cast<std::chrono::milliseconds>(timer->Result())
                .count();
      }
    }
  }

  auto totalProcessed = num_batches * batch_size * state.iterations();
  state.SetBytesProcessed(totalProcessed * sizeof(uint32_t));
  state.SetItemsProcessed(totalProcessed);
}

class BatchGeneratorFixture : public benchmark::Fixture {
 public:
  BatchGeneratorFixture() : rng_(42) {}
  void SetUp(::benchmark::State& state) override {
    int num_batches = state.range(0);
    if (num_batches == 0) {
      schema_.reset();
      batches_.clear();
      if (!arrow::SetCpuThreadPoolCapacity(variables::max_threads()).ok()) {
        state.SkipWithError("Failed to reset thread count");
      }
      state.SkipWithError("not an error :)");
      return;
    }
    int batch_size = state.range(1);
    int indices_batch_size = state.range(2);
    auto nr_threads = state.range(3);

    if (!arrow::SetCpuThreadPoolCapacity(nr_threads).ok()) {
      state.SkipWithError("Set CPU threads failed");
    }

    if (num_batches == static_cast<int>(batches_.size()) &&
        batch_size == batches_[0]->num_rows())
      return;

    schema_ = arrow::schema({arrow::field("v", arrow::uint32(), /*nullable=*/false)});
    batches_ = generator::MakeRandomRecordBatches(rng_, schema_, num_batches, batch_size);
    std::cout << "generated " << batches_.size() << " batches of " << batch_size
              << " rows" << std::endl;

    auto indices_metadata = arrow::key_value_metadata(
        {{"min", std::to_string(0)}, {"max", std::to_string(batch_size - 1)}});
    auto indices_schema = arrow::schema(
        {arrow::field("i", arrow::uint32(), /*nullable=*/false, indices_metadata)});
    indices_batches_ = generator::MakeRandomRecordBatches(
        rng_, indices_schema, num_batches, indices_batch_size);
    std::cout << "generated " << indices_batches_.size() << " index batches of "
              << indices_batch_size << " rows" << std::endl;
  }

  void TearDown(::benchmark::State&) override {}

 protected:
  arrow::random::RandomArrayGenerator rng_;
  std::shared_ptr<arrow::Schema> schema_;
  arrow::RecordBatchVector batches_;
  arrow::RecordBatchVector indices_batches_;
};

BENCHMARK_DEFINE_F(BatchGeneratorFixture, BM_TakeNative)(benchmark::State& state) {
  if (state.range(0) == 0) {
    state.SkipWithError("skip");
    return;
  }
  TakeNative taker{schema_, batches_, indices_batches_};
  BM_Take(state, taker);
}

BENCHMARK_DEFINE_F(BatchGeneratorFixture, BM_TakeDpu)(benchmark::State& state) {
  if (state.range(0) == 0) {
    state.SkipWithError("skip");
    return;
  }
  auto nr_dpus = state.range(3);
  auto system = DpuSet::allocate(nr_dpus);
  std::cout << "Allocated DPUs: " << system.dpus().size() << std::endl;
  TakeDpu taker{system, batches_, indices_batches_};
  BM_Take(state, taker);
#if ENABLE_PERF
  state.counters["cycles"] = cycles_count(system);
#endif
}

BENCHMARK_REGISTER_F(BatchGeneratorFixture, BM_TakeNative)
    ->ArgNames({"Batches", "Batch-Size", "Indices-Batch-Size", "Threads"})
    // ->ArgsProduct({{variables::max_dpus()},
    //                {8 << 20},
    //                {1 << 20},
    //                benchmark::CreateDenseRange(2, variables::max_threads(), 2)})
    ->ArgsProduct({{variables::scale_factor()},
                   {4 << 20},
                   {4 << 20 >> 3},
                   {variables::max_threads_no_ht(), variables::max_threads()}})
    ->Args({0, 0, 0, 0})
    ->ArgsProduct({{variables::scale_factor() << 6},
                   {64 << 10},
                   {64 << 10 >> 3},
                   {variables::max_threads_no_ht(), variables::max_threads()}})
    ->Args({0, 0, 0, 0})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

// TODO: in the current implementation the number of batches must be a multiple of NR_DPUS
BENCHMARK_REGISTER_F(BatchGeneratorFixture, BM_TakeDpu)
    ->ArgNames({"Batches", "Batch-Size", "Indices-Batch-Size", "DPUs"})
    ->Args({variables::scale_factor(), 4 << 20, 4 << 20 >> 3, std::min(variables::max_dpus(), variables::scale_factor())})
    ->Args({0, 0, 0, 0})
    ->Args({variables::scale_factor() << 6, 64 << 10, 64 << 10 >> 3, std::min(variables::max_dpus(), variables::scale_factor() << 6)})
    ->Args({0, 0, 0, 0})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

}  // namespace take
}  // namespace upmemeval
