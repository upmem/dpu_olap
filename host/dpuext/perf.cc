#include "perf.h"

#include "dpuext.hpp"

namespace dpu {

uint32_t cycles_count(DpuSet& system) {
  // Print number of cycles for DPU 0
  std::vector<std::vector<uint32_t>> nb_cycles(system.dpus().size());
  for (auto& v : nb_cycles) {
    v.resize(1);
  }
  system.copy(nb_cycles, "nb_cycles");
  // std::cout << "#cycles (DPU 0): " << nb_cycles.front().front()
  //           << std::endl;
  return nb_cycles.front().front();
}

}  // namespace dpu
