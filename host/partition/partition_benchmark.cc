#include <arrow/api.h>
#include <arrow/util/thread_pool.h>
#include <benchmark/benchmark.h>
#include <iostream>

#include "dpuext/api.h"
#include "timer/timer.h"

#include "generator/generator.h"
#include "partition.h"
#include "partition_dpu.h"
#include "partitioner.h"

#include "system/system.h"

#include "umq/kernels.h"

using namespace dpu;

namespace upmemeval {
namespace partition {

using namespace generator;

using namespace timer;

class BatchGeneratorFixture : public benchmark::Fixture {
 public:
  BatchGeneratorFixture() : rng_(42) {}
  void SetUp(::benchmark::State& state) override {
    if (!arrow::SetCpuThreadPoolCapacity(variables::max_threads_no_ht()).ok()) {
      state.SkipWithError("Set CPU threads failed");
      return;
    }

    int num_batches = state.range(0);
    int batch_size = state.range(1);
    assert(num_batches > 0);
    if (static_cast<size_t>(num_batches) == batches_.size() &&
        batch_size == batches_[0]->num_rows())
      return;
    // schema_ = arrow::schema({arrow::field("pk", arrow::uint32(), /*nullable=*/false),
    //                          arrow::field("v1", arrow::uint32(), /*nullable=*/false),
    //                          arrow::field("v2", arrow::uint32(), /*nullable=*/false),
    //                          arrow::field("v3", arrow::uint32(), /*nullable=*/false)});
    schema_ = arrow::schema({arrow::field("pk", arrow::uint32(), /*nullable=*/false),
                             arrow::field("v1", arrow::uint32(), /*nullable=*/false)});
    batches_ = generator::MakeRandomRecordBatches(rng_, schema_, num_batches, batch_size);
    // auto schema = arrow::schema({arrow::field("v", arrow::uint32(),
    // /*nullable=*/false)}); auto batches = generator::MakeRandomRecordBatches(rng_,
    // schema, num_batches, batch_size); auto pk_column =
    // generator::MakeIndexColumn(num_batches, batch_size); batches_ =
    // generator::AddColumn("pk", batches, pk_column.ValueOrDie()); schema_ =
    // batches_[0]->schema();
  }

  void TearDown(::benchmark::State&) override { batches_.clear(); }

 protected:
  arrow::random::RandomArrayGenerator rng_;
  std::shared_ptr<arrow::Schema> schema_;
  arrow::RecordBatchVector batches_;
};

// Turned off because standalone partition is not implemented
BENCHMARK_DEFINE_F(BatchGeneratorFixture, DISABLED_BM_PartitionDpu)(benchmark::State& state) {
  // Allocate DPUs
  auto nr_dpus = state.range(2);
  auto system = DpuSet::allocate(nr_dpus);
  std::cout << "Allocated DPUs: " << system.dpus().size() << std::endl;

  auto nr_partitions = static_cast<uint64_t>(state.range(3));
  // TODO: current implementation assumes preserving number of batches
  // auto nr_partitions = batches_.size();

  PartitionDpu partition_dpu{system, schema_, batches_, nr_partitions, "pk"};

  // Benchmark loop
  for (auto _ : state) {
    auto prepared = partition_dpu.Prepare();
    if (!prepared.ok()) {
      auto err = std::string("Prepare failed with error: ") + prepared.ToString();
      state.SkipWithError(err.c_str());
      break;
    }

    auto result = partition_dpu.Run();
    if (!result.ok()) {
      auto err = std::string("Run failed with error: ") + result.status().ToString();
      state.SkipWithError(err.c_str());
      break;
    }

    // for(auto& rb : result.ValueOrDie()) {
    //   std::cout << rb->num_rows() << std::endl;
    // }

    // Add timers
    auto timers = partition_dpu.Timers();
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

#if ENABLE_PERF
  state.counters["cycles"] = cycles_count(system);
#endif
}

// Turned off because standalone partition is not implemented
BENCHMARK_REGISTER_F(BatchGeneratorFixture, DISABLED_BM_PartitionDpu)
    ->ArgNames({"Batches", "Batch-Size", "DPUs", "Partitions"})
    ->Args({variables::scale_factor(), 2 << 20,
            std::min(variables::max_dpus(), variables::scale_factor()),
            variables::scale_factor()})
    // ->Args({variables::scale_factor() << 5, 64 << 10, std::min(variables::max_dpus(),
    // variables::scale_factor()), variables::scale_factor()})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace partition
}  // namespace upmemeval
