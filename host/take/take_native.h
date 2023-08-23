#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include "timer/timer.h"

namespace upmemeval {

namespace take {

namespace cp = ::arrow::compute;

class TakeNative {
 public:
  TakeNative(std::shared_ptr<arrow::Schema> schema, arrow::RecordBatchVector batches, arrow::RecordBatchVector indices_batches)
      : schema_(schema), batches_(batches), indices_batches_(indices_batches) {}

  arrow::Status Prepare();

  std::shared_ptr<timer::Timers> Timers() { return nullptr; }
 
  arrow::Result<std::shared_ptr<::arrow::Table>> Run();

 private:
  std::shared_ptr<arrow::Schema> schema_;
  arrow::RecordBatchVector batches_;
  arrow::RecordBatchVector indices_batches_;
};

}  // namespace filter
}  // namespace upmemeval
