#ifndef _MRAM_ALLOC_H
#define _MRAM_ALLOC_H

/**
 * @file mram_alloc.h
 * @brief Provides a way to manage MRAM allocation.
 *
 * @internal The heap is situated after the different kernel structures, local and global
 * variables. It can grow until reaching the end of the WRAM. A reboot of the DPU reset
 * the Heap. The current heap pointer can be accessed at the address defined by
 * __HEAP_POINTER__.
 */

#include <attributes.h>
#include <stddef.h>

/**
 * @fn mram_alloc
 * @brief Allocates a buffer of the given size in the MRAM heap.
 *
 * The allocated buffer is aligned on 64 bits, in order to ensure compatibility
 * with the maximum buffer alignment constraint.
 *
 * Not thread safe.
 *
 * @param size the allocated buffer's size, in bytes
 * @throws a fault if there is no memory left
 * @return The allocated buffer address.
 */
__mram_ptr void* mram_alloc(size_t size);

/**
 * @fn mram_calloc
 * @brief Allocate and zero out a buffer of the given size in the MRAM heap
 *
 * The difference in mram_alloc and mram_calloc is that mram_alloc does not set the memory
 * to zero where as mram_calloc sets allocated memory to zero.
 *
 * The allocated buffer is aligned on 64 bits, in order to ensure compatibility
 * with the maximum buffer alignment constraint.
 *
 * Not thread safe.
 *
 * @param size the allocated buffer's size, in bytes
 * @throws a fault if there is no memory left
 * @return The allocated buffer address.
 */
__mram_ptr void* mram_calloc(size_t size);

/**
 * @fn mram_reset
 * @brief Resets the MRAM heap.
 *
 * Every allocated buffer becomes invalid, since subsequent allocations restart from the
 * beginning of the MRAM heap.
 *
 * Not thread safe.
 *
 * @return The MRAM heap initial address.
 */
__mram_ptr void* mram_reset(void);

__mram_ptr void* __mram_next();

#endif /* _MRAM_ALLOC_H */
