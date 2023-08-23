#pragma once

#include <arrow/api.h>
#include "timer/timer.h"

namespace dpu {
  class DpuSet;
}

namespace upmemeval {

namespace aggr {

class SumDpu {
 public:
  SumDpu(dpu::DpuSet& system, arrow::RecordBatchVector batches);

  arrow::Status Prepare();

  arrow::Result<uint64_t> Run();

  std::shared_ptr<timer::Timers> Timers() { return timers_; }

 private:
  dpu::DpuSet& system_;
  arrow::RecordBatchVector batches_;
  std::shared_ptr<timer::Timers> timers_;
};

}  // namespace filter
}  // namespace upmemeval
