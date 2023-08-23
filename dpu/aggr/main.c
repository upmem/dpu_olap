#include <string.h>
#include <alloc.h>
#include <assert.h>
#include <barrier.h>
#include <defs.h>
#include "mram_alloc.h"

#include "umq/bitops.h"
#include "umq/cflags.h"
#include "umq/kernels.h"
#include "umq/log.h"

#include "kernels/aggr.h"
#include "kernels/common.h"

#if ENABLE_PERF
#include <perfcounter.h>
#endif

#define BUFFER_LENGTH (4 << 20)                              // 4MiB items
#define BUFFER_SIZE_IN_BYTES (BUFFER_LENGTH << T_SIZE_LOG2)  // 16MiB

// Inputs
__host enum Kernel kernel = KernelAggregate;
__host dpu_aggr_input INPUT;
__mram_noinit T buffer[BUFFER_LENGTH];

// Output
__host dpu_aggr_output OUTPUT;

// Synchronization
BARRIER_INIT(barrier, NR_TASKLETS);

// Performance counter
__host uint32_t nb_cycles = 0;

// Sum aggregator
typedef struct {
  uint64_t sum;
} SumAggregatorState;

SumAggregatorState states[NR_TASKLETS];

void sum(void* state, T* items, int items_len) {
  uint64_t sum = 0;
  for (int i = 0; i < items_len; ++i) {
    sum += items[i];
  }
  // log("%d block sum: %lu\n", me(), sum);
  ((SumAggregatorState*)state)->sum += sum;
}

// Main

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
    case KernelAggregate:
      switch (INPUT.aggregator_type) {
        case AggrSum: {
          if (tasklet_id == 0) {
            memset(states, 0, sizeof(states));
            OUTPUT.sum_result = 0;
          }
          barrier_wait(&barrier);
          result = kernel_aggr(tasklet_id, buffer, INPUT.buffer_length, sum, &states[tasklet_id]);
          barrier_wait(&barrier);
          if (tasklet_id == 0) {
            for(int i = 0; i < NR_TASKLETS; ++i) {
              OUTPUT.sum_result += states[i].sum;
              log("adding %lu to aggregate sum: %lu\n", states[i].sum, OUTPUT.sum_result);
            }
          }
        } break;
        default:
          log("Unknown aggregator");
          result = 1;
          break;
      }
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
