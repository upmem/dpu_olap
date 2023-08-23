#include "partition.h"

#include <alloc.h>
#include <barrier.h>
#include <memmram_utils.h>

#include "mram_alloc.h"
#include "mram_ra.h"
#include "mutex_pool.h"

#include "umq/bitops.h"
#include "umq/cflags.h"
#include "umq/log.h"
#include <stddef.h>
#include <stdint.h>
#include <defs.h>
#include <stdio.h>
#include <perfcounter.h>

uint32_t wang_hash_uint32(uint32_t key) {
  key += ~(key << 15);
  key ^= (key >> 10);
  key += (key << 3);
  key ^= (key >> 6);
  key += ~(key << 11);
  key ^= (key >> 16);
  return key;
}

// __attribute__((unused)) inline uint32_t jenkins_hash_uint32(uint32_t a) {
//   a = (a + 0x7ed55d16) + (a << 12);
//   a = (a ^ 0xc761c23c) ^ (a >> 19);
//   a = (a + 0x165667b1) + (a << 5);
//   a = (a + 0xd3a2646c) ^ (a << 9);
//   a = (a + 0xfd7046c5) + (a << 3);
//   a = (a ^ 0xb55a4f09) ^ (a >> 16);
//   return a;
// }

// __attribute__((unused)) inline uint32_t mul_hash_uint32(uint32_t key) {
//   return key * 2654435769U;
// }

#if USE_RADIX_PARTITIONING
#define BUCKET_SHIFT(n) (1 + __builtin_clz(n))
#define BUCKET_OF(x) (wang_hash_uint32(x) >> __bucket_shift)
#else
#define BUCKET_OF(x) (wang_hash_uint32(x) % nr_partitions)
#endif

/**
 * @brief Build histogram
 * Use a global histogram for all tasklets
 *
 */

/*#define NEW_PREFIX_SUM*/
#ifdef NEW_PREFIX_SUM
uint32_t aux[NR_TASKLETS] = {0};
#endif
MUTEX_POOL_INIT(histogram_mutexes, 16);
MUTEX_POOL_INIT(output_mutexes, 8);

uint32_t output_shift;
uint32_t nb_elem_input;

void build_histogram(uint32_t tasklet_id, uint32_t nr_partitions,
                     uint32_t* histogram, T *input_cache, __mram_ptr T* buffer,
                     uint32_t buffer_length) {

#if USE_RADIX_PARTITIONING
  uint32_t __bucket_shift = BUCKET_SHIFT(nr_partitions);
#endif

  // Scan blocks
  for (unsigned int block_offset = tasklet_id << BLOCK_LENGTH_LOG2;
       block_offset < buffer_length; block_offset += BLOCK_LENGTH * NR_TASKLETS) {
    // Load block
    mram_read(&buffer[block_offset], input_cache, BLOCK_SIZE_IN_BYTES);
    // Process block
    uint32_t max = BLOCK_LENGTH;
    if(buffer_length >= block_offset && (buffer_length - block_offset) < max)
      max = buffer_length - block_offset;
    for (unsigned int i = 0; i < max; ++i) {
      T item = input_cache[i];
      size_t bucket = BUCKET_OF(item);
      mutex_pool_lock(&histogram_mutexes, bucket);
      histogram[bucket+1]++;
      mutex_pool_unlock(&histogram_mutexes, bucket);
    }
  }
}

void prefix_sum(uint32_t tasklet_id, uint32_t *histogram, uint32_t nr_partitions, __attribute((unused)) barrier_t* barrier) {

#ifdef NEW_PREFIX_SUM

  // parallel prefix sum implementation

  if(tasklet_id != NR_TASKLETS - 1)
    aux[tasklet_id + 1] = 0;
  else
    aux[0] = 0;

  int nr_elem_per_tasklets = (nr_partitions + 1) / NR_TASKLETS;
  if(tasklet_id != NR_TASKLETS - 1) {
    for(uint32_t i = tasklet_id * nr_elem_per_tasklets; i < (tasklet_id + 1) * nr_elem_per_tasklets; ++i)
      aux[tasklet_id + 1] += histogram[i];
  }
  barrier_wait(barrier);

  if(tasklet_id == 0) {
    for(int i = 2; i < NR_TASKLETS; i++) {
      aux[i] += aux[i - 1];
    }
  }
  barrier_wait(barrier);
  
  histogram[tasklet_id * nr_elem_per_tasklets] += aux[me()];
  uint32_t max = (me() == NR_TASKLETS - 1) ? nr_partitions + 1 : (tasklet_id + 1) * nr_elem_per_tasklets;
  for(uint32_t i = tasklet_id * nr_elem_per_tasklets + 1; i < max; ++i) {
    histogram[i] += histogram[i - 1];
    // make sure that the histogram value does not exceed 2 ^31
    // this would break the fact that we use the MSB as a flag later in
    // the algorithm
    assert(histogram[i] < (1U << 31));
  }
#else

  if(tasklet_id == 0) {
    for(uint32_t i = 1; i < nr_partitions + 1; i ++) {
      histogram[i] += histogram[i-1]; 
      assert(histogram[i] < (1U << 31));
    }
  }
#endif
}

static uint32_t output_offset_shift(uint32_t offset) {

    if(offset < output_shift) {
      return (uint32_t)((int)offset - output_shift + nb_elem_input);
    }
    else {
      return (uint32_t)((int)offset - output_shift);
    }
}


/**
 * @struct output_cache
 * @brief WRAM cache for values that need to go in output 
 * vector in MRAM
 **/
struct output_cache {
  T *cache_item;
  uint32_t *cache_indice;
};
struct output_cache cache;

/**
 * @brief Write a new output in the output vector at the correct
 * position depending on the partition (bucket).
 * Handles WRAM cache, i.e., do no systematically write in MRAM
 * but can also cache the value in WRAM first.
 **/
void write_new_output(
    uint32_t *histogram, 
    __mram_ptr T* output_buffer, 
    __mram_ptr uint32_t* selection_indices_vector, 
    T item, size_t bucket, uint32_t indice) {

  uint32_t offset = 0;
  mutex_pool_lock(&histogram_mutexes, bucket);
  // check if the cache contains an element already
  // use the MSB of histogram value for this
  offset = histogram[bucket];
  T item_cached = 0;
  uint32_t indice_cached = 0;
  if(offset & (1U << 31)) {
    // there is a value in the cache
    // store the two values (cached + current) in MRAM
    offset &= (~(1U << 31));
    histogram[bucket] = offset + 2;
    item_cached = cache.cache_item[bucket];
    indice_cached = cache.cache_indice[bucket];
    mutex_pool_unlock(&histogram_mutexes, bucket);

    uint32_t shifted_offset = output_offset_shift(offset);
    // if an element is cached, it means 
    // the MRAM address should be aligned on 8 bytes
    assert((shifted_offset & 1) == 0);

    // here we are ready to perform the DMA transfer
    // no need to lock as the transfer is aligned (address and size)
    __dma_aligned uint64_t items = item_cached | ((uint64_t)item << 32);
    __dma_aligned uint64_t indices = indice_cached | ((uint64_t)indice << 32);
    mram_write(&items, &output_buffer[shifted_offset], 8);
    mram_write(&indices, &selection_indices_vector[shifted_offset], 8);
  }
  else {

    // no value is currently cached

    // two cases:
    // 1. The MRAM address is unaligned on 8 bytes
    //    a) take a lock to prevent issue on the frontier
    //    b) write the 4 bytes (after this the next offset is aligned on 8 bytes)
    //    c) release the lock
    // 2. The MRAM address is aligned on 8 bytes
    //    a) take the histogram lock
    //    b) store the value in the cache
    //    c) release the histogram lock

    uint32_t shifted_offset = output_offset_shift(offset);
    if(shifted_offset & 1) {
      // unaligned offset (case 1)
      offset = histogram[bucket]++;
      mutex_pool_unlock(&histogram_mutexes, bucket);
      mutex_pool_lock(&output_mutexes, shifted_offset >> 1);
      output_buffer[shifted_offset] = item;
      selection_indices_vector[shifted_offset] = indice;
      mutex_pool_unlock(&output_mutexes, shifted_offset >> 1);
    }
    else {
      // aligned offset (case 2)
      // store value in the cache
      cache.cache_item[bucket] = item;
      cache.cache_indice[bucket] = indice;
      histogram[bucket] |= (1U << 31);
      mutex_pool_unlock(&histogram_mutexes, bucket);
    }
  }
}

/**
 * @brief empty the cache of output values
 **/
void empty_output_cache(
    uint32_t *histogram, 
    __mram_ptr T* output_buffer, 
    __mram_ptr uint32_t* selection_indices_vector, 
    size_t bucket) {

  // check if there is a cache value for this partition
  // if there is one, write it
  // Note: no need to take a lock for the histogram,
  // as each tasklet handles the cached value for
  // different partitions
  uint32_t offset = 0;
  offset = histogram[bucket];
  if(offset & (1U << 31)) {
    // a value is cached
    // we need to clean the histogram value and increment it
    // as the host reads the histogram after this kernel execution
    offset &= (~(1U << 31));
    histogram[bucket] = offset + 1;
    uint32_t shifted_offset = output_offset_shift(offset);
    mutex_pool_lock(&output_mutexes, shifted_offset >> 1);
    output_buffer[shifted_offset] = cache.cache_item[bucket];
    selection_indices_vector[shifted_offset] = cache.cache_indice[bucket];
    mutex_pool_unlock(&output_mutexes, shifted_offset >> 1);
  }
}


void partition_array(uint32_t tasklet_id, uint32_t nr_partitions,
                     __mram_ptr uint32_t* selection_indices_vector,
                     uint32_t* histogram, T *input_cache, __mram_ptr T* buffer,
                     __mram_ptr T* output_buffer, uint32_t buffer_length, barrier_t * barrier) {
#if USE_RADIX_PARTITIONING
  uint32_t __bucket_shift = BUCKET_SHIFT(nr_partitions);
#endif

  // Scan blocks
  for (unsigned int block_offset = tasklet_id << BLOCK_LENGTH_LOG2;
       block_offset < buffer_length; block_offset += BLOCK_LENGTH * NR_TASKLETS) {
    // Load block
    mram_read(&buffer[block_offset], input_cache, BLOCK_SIZE_IN_BYTES);
    uint32_t max = BLOCK_LENGTH;
    if(buffer_length >= block_offset && (buffer_length - block_offset) < max)
      max = buffer_length - block_offset;
    for (unsigned int i = 0; i < max; ++i) {
      T item = input_cache[i];
      size_t bucket = BUCKET_OF(item);
      write_new_output(histogram, output_buffer, 
          selection_indices_vector, item, bucket, block_offset + i);
    }
  }
  barrier_wait(barrier);
  for(unsigned i = tasklet_id; i < nr_partitions; i += NR_TASKLETS) {
    empty_output_cache(histogram, output_buffer, selection_indices_vector, i);
  }
}

int kernel_partition(uint32_t tasklet_id, barrier_t* barrier, uint32_t nr_partitions,
                     uint32_t **histogram_wram,
                     __mram_ptr uint32_t* selection_indices_vector,
                     __mram_ptr T* buffer,
                     __mram_ptr T* output_buffer, uint32_t buffer_length, uint32_t o_shift) {

  if(tasklet_id == 0) {
    *histogram_wram = (uint32_t*)mem_alloc((nr_partitions + 1) * sizeof(uint32_t));
    for(uint32_t i = 0; i < nr_partitions + 1; ++i)
      (*histogram_wram)[i] = 0;
    /*perfcounter_config(COUNT_CYCLES, true);*/
    cache.cache_item = (T*)mem_alloc(nr_partitions * sizeof(T));
    cache.cache_indice = (uint32_t*)mem_alloc(nr_partitions * sizeof(uint32_t));
  }
  barrier_wait(barrier);

  T* input_cache = (T*)mem_alloc(BLOCK_SIZE_IN_BYTES);
  /*printf("build histogram: cache %p\n", input_cache);*/
  build_histogram(tasklet_id, nr_partitions, *histogram_wram, input_cache, buffer, buffer_length);

  barrier_wait(barrier);
  /*if(me() == 0)*/
    /*printf("cycles hist %lu\n", perfcounter_get());*/

  /*printf("prefix\n");*/
  prefix_sum(tasklet_id, *histogram_wram, nr_partitions, barrier);

  barrier_wait(barrier);
  if(me() == 0) {
    // setting the output shift based on the number
    // of elements in the histogram prefix sum
    output_shift = (*histogram_wram)[o_shift];
    nb_elem_input = buffer_length;
  }
  barrier_wait(barrier);
  /*if(me() == 0)*/
    /*printf("cycles prefix %lu\n", perfcounter_get());*/

  /*printf("partition\n");*/
  partition_array(tasklet_id, nr_partitions, selection_indices_vector, *histogram_wram, input_cache, buffer,
                  output_buffer, buffer_length, barrier);
  barrier_wait(barrier);
  /*if(me() == 0)*/
    /*printf("cycles part %lu\n", perfcounter_get());*/
  return 0;
}
