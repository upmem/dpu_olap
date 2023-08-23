#pragma once

#include <arrow/api.h>
#include "timer/timer.h"

namespace dpu {
class DpuSet;
}

namespace upmemeval {
namespace partition {

class PartitionDpu {
 public:
  PartitionDpu(dpu::DpuSet& system_, std::shared_ptr<arrow::Schema> schema,
               arrow::RecordBatchVector batches, uint64_t nr_partitions,
               const std::string& partition_key)
      : system_(system_),
        schema_(schema),
        batches_(std::move(batches)),
        nr_partitions_(nr_partitions),
        partition_key_(partition_key) {}

  arrow::Status Prepare();

  arrow::Result<arrow::RecordBatchVector> Run();

  std::shared_ptr<timer::Timers> Timers() { return timers_; }

 private:
  dpu::DpuSet& system_;
  std::shared_ptr<arrow::Schema> schema_;
  arrow::RecordBatchVector batches_;
  uint64_t nr_partitions_;
  std::string partition_key_;

  std::shared_ptr<timer::Timers> timers_;
};

}  // namespace partition
}  // namespace upmemeval
