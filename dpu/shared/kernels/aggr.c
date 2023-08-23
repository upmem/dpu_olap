#include "aggr.h"

#include <alloc.h>
#include <memmram_utils.h>

#include "mram_alloc.h"
#include "mram_ra.h"

#include "umq/cflags.h"
#include "umq/log.h"

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

int kernel_aggr(uint32_t tasklet_id, __mram_ptr T* buffer, uint32_t buffer_len, aggregator_fn_t fn, void* state){
  trace("Tasklet %d kernel_aggr\n", tasklet_id);

  // Initialize a local WRAM cache for tasklet input
  T* cache = (T*)mem_alloc(BLOCK_SIZE_IN_BYTES);

  // Scan blocks from input
  trace("Tasklet %d kernel_aggr: scanning blocks\n", tasklet_id);
  for (unsigned int block_offset = tasklet_id << BLOCK_LENGTH_LOG2;
       block_offset < buffer_len; block_offset += BLOCK_LENGTH * NR_TASKLETS) {
    // Load block from indices vector
    mram_read(&buffer[block_offset], cache, BLOCK_SIZE_IN_BYTES);
    fn(state, cache, MIN(buffer_len - block_offset, BLOCK_LENGTH));
  }

  trace("Tasklet %d kernel_aggr: done\n", tasklet_id);
  return 0;
}
