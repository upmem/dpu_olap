#pragma once

#include <mram.h>
#include <stdint.h>

#include "common.h"

/**
 * @brief Take kernel
 *
 * Process the `buffer` input array and output a reordered array
 * using a selection indices vector.
 *
 * @param tasklet_id
 * @param buffer Input buffer
 * @param output_buffer Output buffer
 * @param selection_indices_vector Selection indices vector
 * @param selection_indices_vector_length Length of selection indices vector
 * @return int
 */
int kernel_take(uint32_t tasklet_id, __mram_ptr T* buffer, __mram_ptr T* output_buffer,
                __mram_ptr uint32_t* selection_indices_vector,
                uint32_t selection_indices_vector_length);
