#ifndef _UMQ_LOG_H_
#define _UMQ_LOG_H_

#include "cflags.h"

#if ENABLE_LOG
#include <stdio.h>
#define log printf
#else
#define log(fmt, ...)
#endif

#if ENABLE_TRACE
#define trace log
#else
#define trace(fmt, ...)
#endif

#endif
