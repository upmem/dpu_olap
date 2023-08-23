#ifndef _UMQ_BITOPS_H_
#define _UMQ_BITOPS_H_

#define ROUND_UP_TO_MULTIPLE_OF_8(x) ((x + 7) & (-8))

#define ROUND_UP_TO_POWER_OF_2(x) ((x > 2) ? ((1ULL << 32) >> __builtin_clz(x - 1)) : x)

#endif
