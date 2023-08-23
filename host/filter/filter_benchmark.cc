#include <benchmark/benchmark.h>
#include <iostream>

#include "generator/generator.h"

#include "dpuext/api.h"
#include "filter_dpu.h"
#include "filter_native.h"

#include "system/system.h"

using namespace dpu;
using namespace benchmark;

namespace upmemeval {

using namespace generator;

namespace filter {

template <class T>
void BM_Filter(benchmark::State& state, T& filter) {
  auto num_batches = state.range(0);
  auto batch_size = state.range(1);
  auto nr_dpus = state.range(2);
  double nr_ranks = (nr_dpus + 63) / 64.0;

  double totalResults = 0.0;

  for (auto _ : state) {
    // Create filter
    // state.PauseTiming();
    // TODO: move to outside the measurement loop
    auto prepared = filter.Prepare();
    if (!prepared.ok()) {
      auto err = std::string("Prepare failed with error: ") + prepared.ToString();
      state.SkipWithError(err.c_str());
      break;
    }
    // state.ResumeTiming();

    // Run filter
    auto result = filter.Run();
    if (!result.ok()) {
      auto err = std::string("Run failed with error: ") + result.status().ToString();
      state.SkipWithError(err.c_str());
      break;
    }
    totalResults += *result;

    // Add timers
    auto timers = filter.Timers();
    if (timers != nullptr) {
      for (auto& r : timers->get()) {
        auto name = r.first;
        auto timer = r.second;
        auto& counter = state.counters[name];
        counter.value +=
            std::chrono::duration_cast<std::chrono::milliseconds>(timer->Result())
                .count() / nr_ranks;
      }
    }
  }

  auto totalProcessed = num_batches * batch_size * state.iterations();
  state.SetBytesProcessed(totalProcessed * sizeof(uint32_t));
  state.SetItemsProcessed(totalProcessed);
  state.counters["Processed (AVG)"] =
      benchmark::Counter(totalProcessed, benchmark::Counter::kAvgIterations);
  state.counters["Results (AVG)"] =
      benchmark::Counter(totalResults, benchmark::Counter::kAvgIterations);
}

class BatchGeneratorFixture : public benchmark::Fixture {
 public:
  BatchGeneratorFixture() : rng_(42) {}
  void SetUp(::benchmark::State& state) override {
    int num_batches = state.range(0);
    if(num_batches == 0) {
      schema_.reset();
      batches_.clear();
      if (!arrow::SetCpuThreadPoolCapacity(variables::max_threads()).ok()) {
        state.SkipWithError("Failed to reset thread count");
      }
      state.SkipWithError("not an error :)");
      return;
    }
    int batch_size = state.range(1);
    auto nr_threads = state.range(2);

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
  }

  void TearDown(::benchmark::State&) override {
  }

 protected:
  arrow::random::RandomArrayGenerator rng_;
  std::shared_ptr<arrow::Schema> schema_;
  arrow::RecordBatchVector batches_;
};


BENCHMARK_DEFINE_F(BatchGeneratorFixture, BM_FilterNative)(benchmark::State& state) {
  if (state.range(0) == 0) {
    state.SkipWithError("skip");
    return;
  }
  FilterNative filter{schema_, batches_};
  BM_Filter<>(state, filter);
}

BENCHMARK_DEFINE_F(BatchGeneratorFixture, BM_FilterDpu)(benchmark::State& state) {
  if(state.range(0) == 0) {
    state.SkipWithError("skip");
    return;
  }
  auto nr_dpus = state.range(2);
  auto system = DpuSet::allocate(nr_dpus);
  std::cout << "Allocated DPUs: " << system.dpus().size() << std::endl;
  FilterDpu filter{system, batches_};
  BM_Filter<>(state, filter);
#if ENABLE_PERF
  state.counters["cycles"] = cycles_count(system);
#endif
}

BENCHMARK_REGISTER_F(BatchGeneratorFixture, BM_FilterNative)
    ->ArgNames({"Batches", "Batch-Size", "Threads"})
    // ->Args({variables::scale_factor(), 8 << 20, variables::max_threads_no_ht()})
    ->Args({variables::scale_factor() << 7, 64 << 10, variables::max_threads_no_ht()})
    ->Args({0, 0, 0})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

// TODO: in the current implementation the number of batches must be a multiple of NR_DPUS
BENCHMARK_REGISTER_F(BatchGeneratorFixture, BM_FilterDpu)
    ->ArgNames({"Batches", "Batch-Size", "DPUs"})
    // ->Args({variables::scale_factor(), 8 << 20, std::min(variables::max_dpus(), variables::scale_factor())})
    ->Args({variables::scale_factor() << 7, 64 << 10, std::min(variables::max_dpus(), variables::scale_factor() << 7)})
    ->Args({0, 0, 0})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

}  // namespace filter
}  // namespace upmemeval
