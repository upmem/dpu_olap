#pragma once

#include <arrow/api.h>
#include "timer/timer.h"

namespace dpu {
class DpuSet;
}

namespace upmemeval {

namespace take {

class TakeDpu {
 public:
  TakeDpu(dpu::DpuSet& system, arrow::RecordBatchVector batches,
          arrow::RecordBatchVector indices_batches);

  arrow::Status Prepare();

  arrow::Result<std::shared_ptr<::arrow::Table>> Run();

  std::shared_ptr<timer::Timers> Timers() { return timers_; }

 private:
  dpu::DpuSet& system_;
  arrow::RecordBatchVector batches_;
  arrow::RecordBatchVector indices_batches_;
  std::shared_ptr<timer::Timers> timers_;
};

}  // namespace take
}  // namespace upmemeval
