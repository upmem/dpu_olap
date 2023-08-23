#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "umq/cflags.h"
#include "umq/kernels.h"
#include "umq/log.h"

#if ENABLE_PERF
#include <perfcounter.h>
#endif


#include "kernels/filter.h"

// Buffer size
// Note that if logging is enabled then 1MiB is reserved for the printf buffer so this
// must be lowered to 7MiB or less
#define BUFFER_LENGTH (8 << 20)                              // 8MiB items
#define BUFFER_SIZE_IN_BYTES (BUFFER_LENGTH << T_SIZE_LOG2)  // 32MiB

// Buffers
__mram_noinit T buffer[BUFFER_LENGTH];
__mram_noinit T output_buffer[BUFFER_LENGTH];
__host uint32_t buffer_length = 0;
__host uint32_t output_buffer_length;

// Performance counter
__host uint32_t nb_cycles = 0;

// Barrier shared with all tasklets
BARRIER_INIT(tasklets_barrier, NR_TASKLETS);

int main() {
  uint32_t tasklet_id = me();

#if ENABLE_PERF
  if (tasklet_id == 0) {
    perfcounter_config(COUNT_CYCLES, true);
  }
#endif

  int result = kernel_filter(tasklet_id, &tasklets_barrier, buffer, buffer_length,
                             output_buffer, &output_buffer_length);

#if ENABLE_PERF
  nb_cycles = perfcounter_get();
#endif

  return result;
}
