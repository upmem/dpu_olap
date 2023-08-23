#pragma once

#define T uint32_t
#define T_SIZE_LOG2 2

#define BLOCK_LENGTH_LOG2 8
#define BLOCK_LENGTH (1 << BLOCK_LENGTH_LOG2)
#define BLOCK_SIZE_IN_BYTES (BLOCK_LENGTH << T_SIZE_LOG2)
