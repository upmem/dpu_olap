#pragma once

#include <barrier.h>
#include <mram.h>
#include <stdint.h>

#include "common.h"

int kernel_filter(uint32_t tasklet_id, barrier_t* tasklets_barrier, __mram_ptr T* buffer,
                  uint32_t buffer_length, __mram_ptr T* output_buffer,
                  uint32_t* output_buffer_length_x);
