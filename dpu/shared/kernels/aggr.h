#pragma once

#include <mram.h>
#include <stdint.h>

#include "common.h"


typedef void (*aggregator_fn_t)(void* state, T* items, int items_len);


/**
 * 
 * @brief Aggregate kernel
 *
 * Process the `buffer` input array and output an aggregation
 *
 * @param tasklet_id
 * @param buffer Input buffer
 * @param buffer_len
 * @param fn 
 * @param state
 * @return int
 */
int kernel_aggr(uint32_t tasklet_id, __mram_ptr T* buffer, uint32_t buffer_len, aggregator_fn_t fn, void* state);
