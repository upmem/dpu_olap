#ifndef _MRAM_RA_H
#define _MRAM_RA_H

#include <mram.h>

#define mram_load(T, cache, address) \
  _Generic((T), uint32_t : mram_load32, uint64_t : mram_load64)(cache, (size_t)address)
#define mram_store(cache, address, value) \
  _Generic((value), uint32_t              \
           : mram_store32, uint64_t       \
           : mram_store64)(cache, (size_t)address, value)
#define mram_modify(cache, address, value) \
  _Generic((value), uint32_t               \
           : mram_modify32, uint64_t       \
           : mram_store64)(cache, (size_t)address, value)

/**
 * @file mram_ra.h
 * @brief Provides random access reader and writer to MRAM
 */

/**
 * @brief Read 4B from MRAM
 *
 * @param cache 8B cache
 * @param address 4B aligned MRAM addressed
 * @return uint32_t
 */
static __used uint32_t mram_load32(void* cache, size_t address) {
  // Load 8B
  size_t base = address & ~7;
  mram_read((__mram_ptr void const*)base, cache, 8);
  // Extract 4B
  uint32_t* cache_32 = (uint32_t*)cache;
  return cache_32[(address & 7) != 0];
}

/**
 * @brief Store 4B to MRAM
 *
 * @param cache 8B cache
 * @param address 4B aligned MRAM addressed
 * @param value The value to write
 */
static __used void mram_store32(void* cache, size_t address, uint32_t value) {
  // Load 8B
  size_t base = address & ~7;
  mram_read((__mram_ptr void const*)base, cache, 8);
  // Modify 4B
  uint32_t* cache_32 = (uint32_t*)cache;
  cache_32[(address & 7) != 0] = value;
  // Write back 8B
  mram_write(cache, (__mram_ptr void*)base, 8);
}

/**
 * @brief Store 4B to MRAM as part of a read-modify-write operation
 *
 * The 8B cache must be populated by a previous call to load4B with the same
 * address.
 *
 * @param cache 8B cache
 * @param address 4B aligned MRAM addressed
 * @param value The value to write
 */
static __used void mram_modify32(void* cache, size_t address, uint32_t value) {
  // Modify 4B
  uint32_t* cache_32 = (uint32_t*)cache;
  cache_32[(address & 7) != 0] = value;
  // Write back 8B
  size_t base = address & ~7;
  mram_write(cache, (__mram_ptr void*)base, 8);
}

/**
 * @brief Read 8B from MRAM
 *
 * @param cache 8B cache
 * @param address 8B aligned MRAM addressed
 * @return uint64_t
 */
static __used uint64_t mram_load64(void* cache, size_t address) {
  mram_read((__mram_ptr void const*)(address), cache, 8);
  uint64_t* cache_64 = (uint64_t*)cache;
  return *cache_64;
}

/**
 * @brief Write 8B to MRAM
 *
 * @param cache 8B cache
 * @param address 8B aligned MRAM addressed
 * @param value The value to write
 */
static __used void mram_store64(void* cache, size_t address, uint64_t value) {
  uint64_t* cache_64 = (uint64_t*)cache;
  *cache_64 = value;
  mram_write(cache_64, (__mram_ptr void*)(address), 8);
}

#endif
