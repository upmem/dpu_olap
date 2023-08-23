#include <arrow/api.h>
#include <arrow/util/thread_pool.h>
#include <benchmark/benchmark.h>
#include <iostream>

#include "memcpy.h"

namespace upmemeval {

void BM_parallel_memcpy(benchmark::State& state) {
  auto buffer_size = state.range(0);
  auto threads = state.range(1);

  int* source = new int[buffer_size];
  int* dest = new int[buffer_size];
  auto block_size = 64;

  auto pool = ::arrow::internal::ThreadPool::Make(threads);

  for (auto _ : state) {
    state.PauseTiming();
    pool->get()->WaitForIdle();
    state.ResumeTiming();

    auto futures = parallel_memcopy(pool->get(), reinterpret_cast<uint8_t*>(dest),
                                    reinterpret_cast<const uint8_t*>(source),
                                    buffer_size * sizeof(int), block_size, threads);
    for (auto& fut : futures) {
      auto _ = fut.status();
    }
  }
  state.SetBytesProcessed(buffer_size * sizeof(int) * state.iterations());

  delete[] source;
  delete[] dest;
}

void BM_memcpy(benchmark::State& state) {
  auto buffer_size = state.range(0);
  auto block_size = state.range(1);

  int* source = new int[buffer_size];
  int* dest = new int[buffer_size];
  auto block_bytes = sizeof(int) * block_size;

  for (auto _ : state) {
    for (int i = 0; i < buffer_size; i += block_size) {
      benchmark::DoNotOptimize(memcpy(&dest[i], &source[i], block_bytes));
    }
  }
  state.SetBytesProcessed(buffer_size * sizeof(int) * state.iterations());

  delete[] source;
  delete[] dest;
}

BENCHMARK(BM_parallel_memcpy)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond)
    ->ArgsProduct({{2 << 10, 500 << 10, 1 << 20, 2 << 20}, {2, 4, 8, 16}});

BENCHMARK(BM_memcpy)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond)
    ->Ranges({{2 << 20, 2 << 20}, {8, 2 << 20}});

}  // namespace upmemeval
