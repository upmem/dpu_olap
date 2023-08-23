#include <cstdlib>
#include <string>
#include <thread>

namespace variables {

static inline int __getenv_int(std::string name, int fallback = 0) {
  if (const char* env_value = std::getenv(name.c_str())) {
    return std::stoi(env_value);
  }
  return fallback;
}

static inline int max_dpus() { return __getenv_int("NR_DPUS", NR_DPUS); }

static inline int scale_factor() { return __getenv_int("SF", max_dpus()); }

static inline int max_threads() { return __getenv_int("MAX_THREADS", std::thread::hardware_concurrency()); }

static inline int max_threads_no_ht() { return __getenv_int("MAX_THREADS", std::thread::hardware_concurrency() / 2); }

}  // namespace variables
