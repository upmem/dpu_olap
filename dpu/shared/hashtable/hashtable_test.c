#include <alloc.h>
#include <assert.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <perfcounter.h>
#include <stdio.h>

#include "hashtable.h"
#include "mram_alloc.h"

BARRIER_INIT(barrier, NR_TASKLETS);
MUTEX_INIT(writer_mutex);
hash_table_t hashtable;

__host uint32_t nb_cycles;

#define HT_CAPACITY (2 << 20)
#define HT_ITEMS (HT_CAPACITY >> 1)

int main() {
  uint32_t tasklet_id = me();

  // One time initialization
  if (tasklet_id == 0) {
    perfcounter_config(COUNT_CYCLES, true);
    mem_reset();  // Reset the heap
    mram_reset();
    ht_init(&hashtable, writer_mutex, HT_CAPACITY);
  }
  barrier_wait(&barrier);

  uint32_t to_add_in_tasklet = HT_ITEMS / NR_TASKLETS;
  uint32_t tasklet_start = tasklet_id * to_add_in_tasklet;
  uint32_t tasklet_end = tasklet_start + to_add_in_tasklet;
  for (uint32_t i = tasklet_start; i < tasklet_end; ++i) {
    bool ok = ht_put(&hashtable, i, i);
    assert(ok);
  }

  barrier_wait(&barrier);

  // ht_value_t value;
  // for (uint32_t i = 0; i < HT_ITEMS; ++i) {
  //   assert(ht_get(&hashtable, i, &value));
  //   assert(value == i);
  // }

  if (tasklet_id == 0) {
    nb_cycles = perfcounter_get();
    printf("NR_TASKLETS=%d, nb_cycles=%d\n", NR_TASKLETS, nb_cycles);
#if HT_ENABLE_STATS
    printf("#SLOW_PATH=%d, #ITEMS=%d, #DISTANCE=%d\n", hashtable.stats_total_slowpath,
           hashtable.stats_total_items, hashtable.stats_total_distance);
#endif
  }

  return 0;
}
