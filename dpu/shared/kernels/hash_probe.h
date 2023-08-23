#pragma once

#include <mram.h>
#include <stdint.h>

#include "common.h"
#include "hashtable/hashtable.h"

/**
 * @brief Hash Probe kernel
 *
 * Process the `buffer` input array (left relation) and populate a selection indices
 * vector. Assumes that `hashtable` is already populated (i.e., via kernel_hash_build).
 *
 * Currently it is assumed that there is always a match in the joined relation.
 *
 * @param tasklet_id
 * @param buffer
 * @param buffer_length
 * @param hashtable
 * @param selection_indices_vector
 * @return int
 */
int kernel_hash_probe(uint32_t tasklet_id, __mram_ptr T* buffer, uint32_t buffer_length,
                      hash_table_t* hashtable,
                      __mram_ptr uint32_t* selection_indices_vector);
