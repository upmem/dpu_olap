#ifndef _UMQ_CFLAGS_H_
#define _UMQ_CFLAGS_H_

#ifndef ENABLE_PERF
#define ENABLE_PERF 1
#endif

#ifndef ENABLE_LOG
#define ENABLE_LOG 0
#endif

#ifndef ENABLE_TRACE
#define ENABLE_TRACE 0
#endif

#ifndef HT_ENABLE_STATS
#define HT_ENABLE_STATS 0
#endif

#ifndef HT_ENABLE_MULTILOCK
#define HT_ENABLE_MULTILOCK 1
#endif

#ifndef HT_USE_WANG_HASH
#define HT_USE_WANG_HASH 1
#endif

#ifndef USE_RADIX_PARTITIONING
#define USE_RADIX_PARTITIONING 1
#endif

#endif
