#pragma once

#include <arrow/api.h>
#include "timer/timer.h"

#include "partition/partitioner.h"
namespace dpu {
class DpuSet;
}

namespace upmemeval {
namespace join {

class JoinDpu {
 public:
  JoinDpu(dpu::DpuSet& system_, std::shared_ptr<arrow::Schema> left_schema,
          std::shared_ptr<arrow::Schema> right_schema,
          arrow::RecordBatchVector left_batches, arrow::RecordBatchVector right_batches)
      : system_(system_),
        left_schema_(left_schema),
        right_schema_(right_schema),
        left_batches_(std::move(left_batches)),
        right_batches_(std::move(right_batches)) {}

  arrow::Status Prepare();

  arrow::Result<std::shared_ptr<arrow::Table>> Run();

  std::shared_ptr<timer::Timers> Timers() { return timers_; }

 private:
  dpu::DpuSet& system_;
  std::shared_ptr<arrow::Schema> left_schema_;
  std::shared_ptr<arrow::Schema> right_schema_;
  arrow::RecordBatchVector left_batches_;
  arrow::RecordBatchVector right_batches_;
  std::shared_ptr<timer::Timers> timers_;

  arrow::Result<std::shared_ptr<partition::Partitioner>> do_partition(
      int k_column_index,
      const std::vector<uint32_t>& dpu_offset,
      std::vector<std::vector<uint32_t>>& metadata, uint32_t buffer_length, bool is_left);
  arrow::Result<std::shared_ptr<arrow::Table>> Run_internal();

  void create_async_timers(const std::vector<std::string>& names);
  void start_async_timer(const std::string& name);
  void stop_async_timer(const std::string& name);
};

}  // namespace join
}  // namespace upmemeval
