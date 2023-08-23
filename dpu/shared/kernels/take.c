#include "take.h"

#include <alloc.h>
#include <memmram_utils.h>

#include "mram_alloc.h"
#include "mram_ra.h"

#include "umq/cflags.h"
#include "umq/log.h"

int kernel_take(uint32_t tasklet_id, __mram_ptr T* buffer, __mram_ptr T* output_buffer,
                __mram_ptr uint32_t* selection_indices_vector,
                uint32_t selection_indices_vector_length) {
  trace("Tasklet %d kernel_take\n", tasklet_id);

  // Initialize a local WRAM cache for tasklet input
  uint32_t* indices_cache = (uint32_t*)mem_alloc(BLOCK_LENGTH * sizeof(uint32_t));
  T* output_cache = (T*)mem_alloc(BLOCK_SIZE_IN_BYTES);
  void* ra_cache = mem_alloc(MRAM_CACHE_SIZE);

  // Scan blocks from selection_indices_vector
  trace("Tasklet %d kernel_take: scanning blocks\n", tasklet_id);
  for (unsigned int block_offset = tasklet_id << BLOCK_LENGTH_LOG2;
       block_offset < selection_indices_vector_length; block_offset += BLOCK_LENGTH * NR_TASKLETS) {
    // Load block from indices vector
    mram_read(&selection_indices_vector[block_offset], indices_cache,
              BLOCK_SIZE_IN_BYTES);
    // Build output block
    uint32_t max = BLOCK_LENGTH;
    if(selection_indices_vector_length >= block_offset && (selection_indices_vector_length - block_offset) < max)
      max = selection_indices_vector_length - block_offset;
    for (unsigned int i = 0; i < max; ++i) {
    /*for (size_t i = 0; i < BLOCK_LENGTH && block_offset + i < selection_indices_vector_length; ++i) {*/
      uint32_t index = indices_cache[i];
      // TODO: Add bound check for 0 <= index <= length of buffer?
      T item = mram_load(item, ra_cache, &buffer[index]);
      output_cache[i] = item;
    }
    // Write output block
    mram_write(output_cache, &output_buffer[block_offset], BLOCK_SIZE_IN_BYTES);
  }

  trace("Tasklet %d kernel_take: done\n", tasklet_id);

  return 0;
}
