#pragma once

#include <cstdint>

namespace dpu {

class DpuSet;

/**
 * @brief Get number of cycles (DPU 0)
 * 
 * @param system 
 * @return uint32_t 
 */
uint32_t cycles_count(dpu::DpuSet& system);

}  // namespace dpu
