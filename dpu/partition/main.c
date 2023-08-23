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
#include "kernels/partition.h"
#include "kernels/take.h"

#if ENABLE_PERF
#include <perfcounter.h>
#endif

#define BUFFER_LENGTH (4 << 20)                              // 4MiB items
#define BUFFER_SIZE_IN_BYTES (BUFFER_LENGTH << T_SIZE_LOG2)  // 16MiB

// Inputs
__host enum Kernel kernel = KernelUnspecified;
__host kernel_partition_inputs_t INPUT;
__mram_noinit T buffer[BUFFER_LENGTH];

// Outputs
__host kernel_partition_outputs_t OUTPUT;
__mram_noinit T output_buffer[BUFFER_LENGTH];

// Internal state
__mram_noinit uint32_t selection_indices_vector[BUFFER_LENGTH];  // for non-key columns

// histogram for each partition
__host uint32_t* histogram_wram = 0;

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
    assert("nr_partitions must be at least 2" && INPUT.nr_partitions >= 2);
    assert("nr_partitions must be a power of two" &&
           INPUT.nr_partitions == ROUND_UP_TO_POWER_OF_2(INPUT.nr_partitions));
  }
  barrier_wait(&barrier);

  int result;
  switch (kernel) {
    case KernelPartition:
      result = kernel_partition(tasklet_id, &barrier, INPUT.nr_partitions,
                                &histogram_wram,
                                selection_indices_vector, buffer,
                                output_buffer, INPUT.buffer_length, 0);

      // update meta-data for the host to know where to retrieve the 
      // histogram
      OUTPUT.partitions_metadata_offset = (uintptr_t)histogram_wram;
      // XXX: this shouldn't be here but ok for now
      OUTPUT.output_buffer_length = INPUT.buffer_length;

      break;
    case KernelTake:
      // Assumes that `selection_indices_vector` is already populated (i.e., via
      // kernel_hash_probe in the partitioning kernel).
      // Note: the current implementation assumes that there are no skipped rows from the
      // left relation (hence, no filter kernel).
      result = kernel_take(tasklet_id, buffer, output_buffer, selection_indices_vector,
                           INPUT.buffer_length);
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
