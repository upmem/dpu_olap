#ifndef _HASHMAP_H_
#define _HASHMAP_H_

#include <defs.h>
#include <mutex.h>
#include <seqread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "umq/cflags.h"

#define ht_value_t uint32_t
#ifdef HT_64BIT_KEYS
#define ht_key_t uint64_t
#else
#define ht_key_t uint32_t
#endif

typedef struct {
  ht_key_t key;
  ht_value_t value;
} ht_entry_t;

/**
 * @brief Hash table.
 *
 * This hash table is designed to be used in a SQL JOIN operation and thus assumes that
 * there are no write operations after the first read and there are no delete operations.
 */
typedef struct {
  // Linear items array (MRAM)
  __mram_ptr ht_entry_t* items;
  // Linear bitmap of used items (MRAM)
  __mram_ptr uint32_t* used;
  // Number of elements that this hashtable can hold  (must be a power of 2)
  uint32_t n_buckets;
  // A writer mutex (it is assumed that there are no writes during reads)
  mutex_id_t writer_mutex;
#if HT_ENABLE_STATS
  // Number of allocated bytes
  size_t stats_allocated_bytes;
  // Used to calculate average distance from exact location
  uint32_t stats_total_distance;
  uint32_t stats_total_items;
  // Number of times ht_put entered the slow path
  uint32_t stats_total_slowpath;
#endif
} hash_table_t;

/**
 * @brief ht_init allocates a new hash table
 *
 * @param mutex a mutex previously obtained using MUTEX_INIT
 * @param capacity the capacity of the hash table. This must be a power of 2 that
 * ensures a load factor < 0.75. The allocated capacity in bytes is
 * capacity*(sizeof(ht_key_t)+sizeof(ht_value_t)) + capacity>>5.
 * @return hash_table_t* a newly allocated hash table
 */
bool ht_init(hash_table_t* table, mutex_id_t mutex, size_t capacity);

/**
 * @brief Put a new key-value pair in the hash table
 *
 * @param table
 * @param key
 * @param value
 * @return true if the element was added to the hash table
 * @return false if the hash table is full
 */
bool ht_put(hash_table_t* table, ht_key_t key, ht_value_t value);

/**
 * @brief Get a value by key from the hash table
 *
 * @param table hash table
 * @param key a lookup key
 * @param value [out] the value associated with the key
 * @return true if the key was found hash in the hash table
 * @return false if the key was not found in the hash table
 */
bool ht_get(hash_table_t* table, ht_key_t key, ht_value_t* value);

/**
 * @brief Get a value by key from the hash table
 *
 * Only use this method if the key is gauranteed to exist in the table
 *
 * @param table
 * @param key
 * @return ht_value_t
 */
ht_value_t ht_get_unsafe(hash_table_t* table, ht_key_t key);

static inline size_t ht_allocated_bytes(__attribute__((unused)) hash_table_t* table) {
#if HT_ENABLE_STATS
  return table->stats_allocated_bytes;
#else
  return 0;
#endif
}

#endif
