#include "mram_alloc.h"
#include <assert.h>
#include <defs.h>
#include <mram.h>
#include <string.h>

__mram_ptr void* __next = DPU_MRAM_HEAP_POINTER;

__dma_aligned uint8_t zeros[512];

__mram_ptr void* __mram_next() { return __next; }

__mram_ptr void* mram_alloc(size_t size) {
  size_t pointer = (size_t)__next;
  if (size != 0) {
    pointer = (pointer + 7) & ~7;
    __next = (__mram_ptr void*)(pointer + size);
  }
  return (__mram_ptr void*)pointer;
}

__mram_ptr void* mram_calloc(size_t size) {
  __mram_ptr void* ptr = mram_alloc(size);
  for (size_t i = 0; i < size; i += sizeof(zeros)) {
    mram_write(zeros, ptr + i, sizeof(zeros));
  }
  return ptr;
}

__mram_ptr void* mram_reset(void) {
  __next = DPU_MRAM_HEAP_POINTER;
  return __next;
}
