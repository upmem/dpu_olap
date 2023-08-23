#pragma once

#include <dpuext/dpuext.hpp>
#include <dpu_error.h>
#include <arrow/result.h>

#define DPU_RETURN_NOT_OK(statement)                                                                 \
  do                                                                                                 \
  {                                                                                                  \
    dpu_error_t __error = (statement);                                                               \
    ARROW_RETURN_IF(__error != DPU_OK, arrow::Status::UnknownError(dpu_error_to_string(__error))); \
  } while (0)
