#include <alloc.h>
#include <assert.h>
#include <barrier.h>
#include <defs.h>
#include "mram_alloc.h"

#include "umq/bitops.h"
#include "umq/cflags.h"
#include "umq/kernels.h"
#include "umq/log.h"

#include "kernels/common.h"
#include "kernels/take.h"

#if ENABLE_PERF
#include <perfcounter.h>
#endif

#define BUFFER_LENGTH (4 << 20)                              // 4MiB items
#define BUFFER_SIZE_IN_BYTES (BUFFER_LENGTH << T_SIZE_LOG2)  // 16MiB

// Inputs
__host enum Kernel kernel = KernelTake;
__host dpu_take_input INPUT;
__mram_noinit T buffer[BUFFER_LENGTH];
__mram_noinit uint32_t selection_indices_vector[BUFFER_LENGTH];

// Outputs
__mram_noinit T output_buffer[BUFFER_LENGTH];

// Synchronization
BARRIER_INIT(barrier, NR_TASKLETS);

// Performance counter
__host uint32_t nb_cycles = 0;

int main() {
  uint32_t tasklet_id = me();

#if ENABLE_PERF
  if (tasklet_id == 0) {
    perfcounter_config(COUNT_CYCLES, true);
  }
#endif
  trace("Tasklet %d main: running kernel %d\n", tasklet_id, kernel);

  // Reset heap
  if (tasklet_id == 0) {
    mem_reset();
  }
  barrier_wait(&barrier);

  int result;
  switch (kernel) {
    case KernelTake:
      result = kernel_take(tasklet_id, buffer, output_buffer, selection_indices_vector, INPUT.selection_indices_vector_length);
      break;
    default:
      log("Unknown kernel");
      result = 1;
      break;
  }

#if ENABLE_PERF
  nb_cycles = perfcounter_get();
#endif

  return result;
}
