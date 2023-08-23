#include "filter.h"

#include <alloc.h>
// #include <assert.h>
#include <handshake.h>
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>

#include "umq/log.h"

#define FILTER_BLOCK_LENGTH_LOG2 7
#define FILTER_BLOCK_LENGTH (1 << FILTER_BLOCK_LENGTH_LOG2)
#define FILTER_BLOCK_SIZE_IN_BYTES (FILTER_BLOCK_LENGTH << T_SIZE_LOG2)

// Array for communication between adjacent tasklets
struct message_t {
    uint32_t p_count;
    T write_carry;
};
struct message_t message[NR_TASKLETS];
uint32_t message_partial_count;
#define WRAM_BUFFER_LENGTH FILTER_BLOCK_LENGTH*NR_TASKLETS

bool predicate(const T item) { return item < (1 << 30); }

// Handshake with adjacent tasklets
struct message_t handshake_sync(uint32_t l_count, T last_entry, uint32_t tasklet_id) {
  uint32_t p_count = 0;
  T write_carry = 0;
  // Wait and read message
  if (tasklet_id != 0) {
    handshake_wait_for(tasklet_id - 1);
    p_count = message[tasklet_id-1].p_count;
    write_carry = message[tasklet_id-1].write_carry;
  } else {
    write_carry = message[NR_TASKLETS-1].write_carry;
  }
  if (l_count > 0) {
    message[tasklet_id].write_carry = last_entry;
  } else {
    if (tasklet_id>0) {
      message[tasklet_id].write_carry = message[tasklet_id-1].write_carry;
    } else {
      message[tasklet_id].write_carry = message[NR_TASKLETS-1].write_carry;
    }
  }

  // Write message and notify
  message[tasklet_id].p_count = p_count + l_count;
  if (tasklet_id < NR_TASKLETS - 1) {
    handshake_notify();
  }
  return (struct message_t){p_count, write_carry};
}

int kernel_filter(uint32_t tasklet_id, barrier_t* tasklets_barrier, __mram_ptr T* buffer,
                  uint32_t buffer_length, __mram_ptr T* output_buffer,
                  uint32_t* output_buffer_length) {
  if (0 == tasklet_id) {
    // Initialize shared variable
    message_partial_count = 0;
    message[NR_TASKLETS-1].p_count = 0;
    // clean-up the heap (see
    // https://sdk.upmem.com/2021.3.0/031_DPURuntimeService_Memory.html?highlight=mem_reset)
    mem_reset();
  }
  barrier_wait(tasklets_barrier);

  // Initialize a local WRAM cache for tasklet input
  T* input_cache = (T*)mem_alloc(FILTER_BLOCK_SIZE_IN_BYTES);

  // Initialize a local WRAM cache for tasklet output
  T* output_cache = (T*)mem_alloc(FILTER_BLOCK_SIZE_IN_BYTES);
  T* output_cache_iter = output_cache+1;

  // Iterate over the blocks of the buffer that are assigned to this tasklet
  unsigned int tasklet_start_index = tasklet_id << FILTER_BLOCK_LENGTH_LOG2;
  T write_carry;

  for (unsigned int block_offset = tasklet_start_index; block_offset < buffer_length;
       block_offset += FILTER_BLOCK_LENGTH * NR_TASKLETS) {
    mram_read(&buffer[block_offset], input_cache, FILTER_BLOCK_SIZE_IN_BYTES);
    output_cache_iter = output_cache;

    // Scan block
#pragma unroll 8
    for (unsigned int i = 0; i < FILTER_BLOCK_LENGTH && block_offset + i < buffer_length; ++i) {
      T item = input_cache[i];
      if (predicate(item)) {
        // Write to output cache
        *output_cache_iter = item;
        ++output_cache_iter;
      }
    }

    // How many items passed the filter predicate
    int written = output_cache_iter - output_cache;

    // Sync with adjacent tasklets
    struct message_t msg = handshake_sync(written, output_cache_iter[-1], tasklet_id);
    uint32_t p_count = msg.p_count;
    write_carry = msg.write_carry;

    // Write output_cache to mram at correct address;
    // Eventually write what was given by previous tasklet.
    // Eventually do not write last entry and let next tasklet write it.
    int write_length = written;
    if (((message_partial_count+p_count+written) & 0x1) == 1) {
      write_length -= 1;
    }
    if (((message_partial_count+p_count) & 0x1) == 1 && write_length>0) {
      // Shift output_cache to the right to make room for the "write_carry" value
      #pragma unroll(32)
      for (int32_t i = written-1; i>=0; i--) {
        output_cache[i+1] = output_cache[i];
      }
      output_cache[0] = write_carry;
      write_length+=1;
      // assert((write_length & 0x1) == 0);
      // assert((message_partial_count + p_count & 0x1) == 1);
      mram_write(output_cache,
                 (__mram_ptr void*)(&output_buffer[message_partial_count + p_count-1]),
                 (write_length)*sizeof(T));
    } else if (write_length > 0) {
      // assert((write_length & 0x1) == 0);
      // assert((message_partial_count + p_count & 0x1) == 0);
      mram_write(output_cache,
                 (__mram_ptr void*)(&output_buffer[message_partial_count + p_count]),
                 (write_length)*sizeof(T));
    }

    // Wait until all tasklets finish their current copy
    barrier_wait(tasklets_barrier);

    // Total count in this DPU
    if (tasklet_id == NR_TASKLETS - 1) {
      message_partial_count += p_count + written;
    }
    barrier_wait(tasklets_barrier);
  }

  // Empty sync in the end
  uint32_t remainder = ((buffer_length % (NR_TASKLETS*FILTER_BLOCK_LENGTH)) + (FILTER_BLOCK_LENGTH-1)) / FILTER_BLOCK_LENGTH;
  if (remainder != 0 && tasklet_id >= remainder) {
    struct message_t msg = handshake_sync(0, 0, tasklet_id);
    uint32_t p_count = msg.p_count;
    T write_carry = msg.write_carry;
    if (tasklet_id == remainder && (p_count & 0x1) == 1) {
        output_cache[0] = write_carry;
        mram_write(output_cache,
                   (__mram_ptr void*)(&output_buffer[message_partial_count + p_count - 1]),
                   2*sizeof(T));
    }
    barrier_wait(tasklets_barrier);
    if (tasklet_id == NR_TASKLETS - 1) {
      message_partial_count += p_count;
    }
  } else if (tasklet_id == NR_TASKLETS -1 && (message_partial_count & 0x1) == 1) {
    if (output_cache_iter == output_cache) {
      mram_write(&write_carry,
                 (__mram_ptr void*)(&output_buffer[message_partial_count-1]),
                 2*sizeof(T));
    } else {
      mram_write(&output_cache_iter[-1],
                 (__mram_ptr void*)(&output_buffer[message_partial_count-1]),
                 2*sizeof(T));
    }
  }

  barrier_wait(tasklets_barrier);
  if (tasklet_id == NR_TASKLETS - 1) {
    *output_buffer_length = message_partial_count;
  }

  return 0;
}
