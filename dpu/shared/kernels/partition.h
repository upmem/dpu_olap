#pragma once

#include <barrier.h>
#include <mram.h>
#include <stdint.h>

#include "common.h"

/**
 * @brief Partitioning kernel
 * @return int
 */
int kernel_partition(uint32_t tasklet_id, barrier_t* barrier, uint32_t nr_partitions,
                     uint32_t **histogram_wram,
                     __mram_ptr uint32_t* selection_indices_vector,
                     __mram_ptr T* buffer,
                     __mram_ptr T* output_buffer, uint32_t buffer_length, uint32_t output_shift);
