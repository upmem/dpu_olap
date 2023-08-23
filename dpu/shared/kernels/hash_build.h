#pragma once

#include <mram.h>
#include <stdint.h>

#include "common.h"
#include "hashtable/hashtable.h"

/**
 * @brief Hash Build Kernel
 *
 * Process the `buffer` input array (right relation) and populate `hashtable`
 *
 * @param tasklet_id
 * @param buffer
 * @param buffer_length
 * @param hashtable
 * @return int
 */
int kernel_hash_build(uint32_t tasklet_id, __mram_ptr T* buffer, uint32_t buffer_length,
                      hash_table_t* hashtable);
