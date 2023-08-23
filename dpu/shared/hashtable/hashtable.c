#include <alloc.h>
#include <assert.h>
#include <mram.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hashtable.h"
#include "mram_alloc.h"
#include "mram_ra.h"
#include "../kernels/mutex_pool.h"

#include "umq/cflags.h"
#include "umq/log.h"

#define ht_index_t uint32_t

#define __ht_used(flag, i) (flag[i >> 5] >> (i & 0x1fU) & 1U)
#define __ht_set_used(flag, i) (flag[i >> 5] |= 1U << (i & 0x1fU))
#define __ht_set_unused(flag, i) (flag[i >> 5] &= ~(1U << (i & 0x1fU)))
#define __ht_used_size(m) ((m) < 32 ? 1 : (m) >> 5)

#define __ht_hash_eq(a, b) ((a) == (b))
#ifdef HT_64BIT_KEYS
#define __ht_hash_func(key) (ht_index_t)((key) >> 33 ^ (key) ^ (key) << 11)
#else

#if HT_USE_WANG_HASH
static inline ht_index_t __ht_hash_func(ht_key_t key) {
  key += ~(key << 15);
  key ^= (key >> 10);
  key += (key << 3);
  key ^= (key >> 6);
  key += ~(key << 11);
  key ^= (key >> 16);
  return key;
}
#else
#define __ht_hash_func(key) (ht_index_t)(key)
#endif
#endif

#if HT_ENABLE_MULTILOCK
//#define NR_MUTEXES 16
//#define FOR_ALL_MUTEXES(f)                                                            \
//  f(0x00) f(0x01) f(0x02) f(0x03) f(0x04) f(0x05) f(0x06) f(0x07) f(0x08) f(0x09)     \
//      f(0x0A) f(0x0B) f(0x0C) f(0x0D) f(0x0E) f(0x0F) f(0x10) f(0x11) f(0x12) f(0x13) \
//          f(0x14) f(0x15) f(0x16) f(0x17) f(0x18) f(0x19) f(0x1A) f(0x1B) f(0x1C)     \
//              f(0x1D) f(0x1E) f(0x1F)
//#define MUTEX_FOR(i) ((i >> 6) & (NR_MUTEXES - 1))
//#define WRITER_MUTEX(i) MUTEX_INIT(writer_mutex_##i);
//#define REGISTER_WRITER_MUTEX(i) __MUTEXES[i] = writer_mutex_##i;
//#define WRITER_MUTEX_LOCK(i) mutex_lock(__MUTEXES[MUTEX_FOR(i)]);
//#define WRITER_MUTEX_UNLOCK(i) mutex_unlock(__MUTEXES[MUTEX_FOR(i)]);
//FOR_ALL_MUTEXES(WRITER_MUTEX)
//mutex_id_t __MUTEXES[NR_MUTEXES];
MUTEX_POOL_INIT(ht_mutex_pool, 16);
#else
#define WRITER_MUTEX_LOCK(...) mutex_lock(table->writer_mutex);
#define WRITER_MUTEX_UNLOCK(...) mutex_unlock(table->writer_mutex);
#endif

bool ht_init(hash_table_t* table, mutex_id_t mutex, size_t capacity) {
#if HT_ENABLE_STATS
  size_t allocated_before = (size_t)__mram_next();
#endif

#if HT_ENABLE_MULTILOCK
  /*FOR_ALL_MUTEXES(REGISTER_WRITER_MUTEX);*/
#endif

  table->writer_mutex = mutex;
  table->n_buckets = capacity;
  table->items = (__mram_ptr ht_entry_t*)mram_alloc(sizeof(ht_entry_t) * capacity);

  size_t bitmap_size = sizeof(uint32_t) * __ht_used_size(capacity);
  table->used = (__mram_ptr uint32_t*)mram_calloc(bitmap_size);

#if HT_ENABLE_STATS
  table->stats_allocated_bytes = ((size_t)__mram_next()) - allocated_before;
  table->stats_total_distance = 0;
  table->stats_total_items = 0;
  table->stats_total_slowpath = 0;
#endif

  return true;
}

bool ht_put(hash_table_t* table, ht_key_t key, ht_value_t value) {
  size_t used_index;
  size_t used_shift;
  uint32_t used_bitmap;

  __dma_aligned uint64_t cache;
  __dma_aligned ht_entry_t entry;
  __dma_aligned ht_entry_t item = {key, value};

  ht_index_t k = __ht_hash_func(key);
  ht_index_t mask = table->n_buckets - 1;
  ht_index_t i = k & mask;
  ht_index_t last = i;

  // Fast Path
  used_index = i >> 5;
  used_shift = i & 0x1fU;
  /*WRITER_MUTEX_LOCK(i);*/
  mutex_pool_lock(&ht_mutex_pool, i >> 6);
  used_bitmap = mram_load32(&cache, (size_t)&table->used[used_index]);
  bool used = (used_bitmap >> used_shift) & 1U;
  if (!used) {
    // Found a cell
    used_bitmap |= (1U << used_shift);
    mram_modify32(&cache, (size_t)&table->used[used_index], used_bitmap);
    /*WRITER_MUTEX_UNLOCK(i);*/
    mutex_pool_unlock(&ht_mutex_pool, i >> 6);
    mram_write(&item, &table->items[i], sizeof(ht_entry_t));
#if HT_ENABLE_STATS
    mutex_lock(table->writer_mutex);
    table->stats_total_items += 1;
    mutex_unlock(table->writer_mutex);
#endif
    return true;
  }

#if HT_ENABLE_STATS
  uint32_t distance = 0;
  mutex_lock(table->writer_mutex);
  table->stats_total_slowpath += 1;
  mutex_unlock(table->writer_mutex);
#endif

  mram_read(table->items + i, &entry, sizeof(ht_entry_t));
  while (used && !__ht_hash_eq(key, entry.key)) {
    /*WRITER_MUTEX_UNLOCK(i);*/
    mutex_pool_unlock(&ht_mutex_pool, i >> 6);
    i = (i + 1U) & mask;
    if (i == last) {
      return false;
    };
#if HT_ENABLE_STATS
    ++distance;
#endif
    used_index = i >> 5;
    used_shift = i & 0x1fU;
    /*WRITER_MUTEX_LOCK(i);*/
    mutex_pool_lock(&ht_mutex_pool, i >> 6);
    used_bitmap = mram_load32(&cache, (size_t)&table->used[used_index]);
    used = (used_bitmap >> used_shift) & 1U;
    mram_read(table->items + i, &entry, sizeof(ht_entry_t));
  }

  // Found a cell
  used_bitmap |= (1U << used_shift);
  mram_modify32(&cache, (size_t)&table->used[used_index], used_bitmap);
  mram_write(&item, &table->items[i], sizeof(ht_entry_t));
  /*WRITER_MUTEX_UNLOCK(i);*/
  mutex_pool_unlock(&ht_mutex_pool, i >> 6);
#if HT_ENABLE_STATS
  mutex_lock(table->writer_mutex);
  table->stats_total_items += 1;
  table->stats_total_distance += distance;
  mutex_unlock(table->writer_mutex);
#endif
  return true;
}

bool ht_get(hash_table_t* table, ht_key_t key, ht_value_t* value) {
  __dma_aligned uint64_t cache;
  __dma_aligned ht_entry_t entry;

  ht_index_t k = __ht_hash_func(key);
  ht_index_t mask = table->n_buckets - 1;
  ht_index_t i = k & mask;
  ht_index_t last = i;

  do {
    size_t used_index = i >> 5;
    size_t used_shift = i & 0x1fU;
    uint32_t used_bitmap = mram_load(used_bitmap, &cache, &table->used[used_index]);
    bool used = (used_bitmap >> used_shift) & 1U;
    if (used) {
      mram_read(table->items + i, &entry, sizeof(ht_entry_t));
      if (entry.key == key) {
        *value = entry.value;
        return true;
      }
    }
    i = (i + 1U) & mask;
  } while (i != last);

  return false;
}

ht_value_t ht_get_unsafe(hash_table_t* table, ht_key_t key) {
  __dma_aligned ht_entry_t entry;

  ht_index_t k = __ht_hash_func(key);
  ht_index_t mask = table->n_buckets - 1;
  ht_index_t i = k & mask;
  ht_index_t last = i;

  mram_read(table->items + i, &entry, sizeof(ht_entry_t));
  while (entry.key != key) {
    i = (i + 1U) & mask;
    if (i == last) {
      trace("key %d not found", key);
      assert(false);
    }
    mram_read(table->items + i, &entry, sizeof(ht_entry_t));
  }
  return entry.value;
}
