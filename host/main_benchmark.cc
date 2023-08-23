#include <benchmark/benchmark.h>
#include <string>

#include "system/system.h"

int main(int argc, char** argv) {
  benchmark::AddCustomContext("NR_DPUS", std::to_string(variables::max_dpus()));
  benchmark::AddCustomContext("SF", std::to_string(variables::scale_factor()));
  benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
