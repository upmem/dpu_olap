#include "aggr_dpu.h"

#include <iostream>

#include "dpuext/api.h"
#include "generator/generator.h"

#include "umq/bitops.h"
#include "umq/kernels.h"

using namespace dpu;

namespace upmemeval {

namespace aggr {

SumDpu::SumDpu(DpuSet& system, arrow::RecordBatchVector batches)
    : system_(system), batches_(std::move(batches)) {}

arrow::Status SumDpu::Prepare() {
  timers_ = std::make_shared<timer::Timers>(system_.ranks().size());

  try {
    system_.load(AGGREGATE_DPU_BINARY);
  } catch (DpuError& e) {
    return arrow::Status::UnknownError(e.what());
  }
  return arrow::Status::OK();
}

arrow::Result<uint64_t> SumDpu::Run() {
  uint64_t result = 0;

  auto copyToDpuTimer = timers_->New("copy-to-dpu");
  auto copyFromDpuTimer = timers_->New("copy-from-dpu");
  auto dpuWorkTimer = timers_->New("dpu-work");
  auto buildResultTimer = timers_->New("build-result");

  // Copy buffer length
  // XXX: It's assumed that all buffers are of the same length
  ARROW_ASSIGN_OR_RAISE(uint32_t buffer_length,
                        arrow_data_buffer_length(batches_, 0UL, 0));

  dpu_aggr_input input;
  input.aggregator_type = AggrSum;
  input.buffer_length = buffer_length;

  // Set kernel to partition kernel
  std::vector<int32_t> kernel_param{KernelAggregate};
  system_.copy("kernel", 0, kernel_param, sizeof(int32_t));

  // Set partition kernel parameters
  std::vector<dpu_aggr_input> input_params{input};
  system_.copy("INPUT", 0, input_params, sizeof(dpu_aggr_input));

  auto nr_dpus = system_.dpus().size();
  assert(batches_.size() % nr_dpus == 0);

  for (size_t i = 0; i < batches_.size(); i += nr_dpus) {
    copyToDpuTimer->Start();
    // Copy data buffers
    ARROW_RETURN_NOT_OK(arrow_copy_to_dpus(system_, "buffer", batches_,
                                           /*batches_offset=*/i, /*column_index=*/0));
    copyToDpuTimer->Stop();

    // Execute DPU program synchroniously
    dpuWorkTimer->Start();
    system_.exec_with_log(std::cout);
    dpuWorkTimer->Stop();

    // Get output
    copyFromDpuTimer->Start();
    std::vector<std::vector<dpu_aggr_output>> outputs(nr_dpus);
    for (auto& x : outputs) {
      x.resize(1);
    }
    system_.copy(outputs, "OUTPUT");
    copyFromDpuTimer->Stop();

    // Calculate total result
    buildResultTimer->Start();
    for (auto& x : outputs) {
      result += x.front().sum_result;
    }
    buildResultTimer->Stop();
  }

  return result;
}

}  // namespace aggr
}  // namespace upmemeval
