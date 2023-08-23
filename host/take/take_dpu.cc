#include "take_dpu.h"

#include <iostream>

#include "dpuext/api.h"
#include "generator/generator.h"

#include "umq/bitops.h"
#include "umq/kernels.h"

using namespace dpu;

namespace upmemeval {

namespace take {

TakeDpu::TakeDpu(DpuSet& system, arrow::RecordBatchVector batches,
                 arrow::RecordBatchVector indices_batches)
    : system_(system),
      batches_(std::move(batches)),
      indices_batches_(std::move(indices_batches)) {}

arrow::Status TakeDpu::Prepare() {
  timers_ = std::make_shared<timer::Timers>(system_.ranks().size());

  try {
    system_.load(TAKE_DPU_BINARY);
  } catch (DpuError& e) {
    return arrow::Status::UnknownError(e.what());
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<::arrow::Table>> TakeDpu::Run() {
  // Holds the result array
  arrow::RecordBatchVector results;

  auto copyToDpuTimer = timers_->New("copy-to-dpu");
  auto copyFromDpuTimer = timers_->New("copy-from-dpu");
  auto dpuWorkTimer = timers_->New("dpu-work");
  auto buildResultTimer = timers_->New("build-result");

  // XXX: It's assumed that all buffers are of the same length
  ARROW_ASSIGN_OR_RAISE(uint32_t buffer_length,
                        arrow_data_buffer_length(batches_, 0UL, 0));
  ARROW_ASSIGN_OR_RAISE(uint32_t indices_buffer_length,
                        arrow_data_buffer_length(indices_batches_, 0UL, 0));

  dpu_take_input input;
  input.buffer_length = buffer_length;
  input.selection_indices_vector_length = indices_buffer_length;

  // Set kernel to take kernel
  std::vector<int32_t> kernel_param{KernelTake};
  system_.copy("kernel", 0, kernel_param, sizeof(int32_t));

  // Set kernel parameters
  std::vector<dpu_take_input> input_params{input};
  system_.copy("INPUT", 0, input_params, sizeof(dpu_take_input));

  auto nr_dpus = system_.dpus().size();
  assert(batches_.size() % nr_dpus == 0);
  for (size_t i = 0; i < batches_.size(); i += nr_dpus) {
    copyToDpuTimer->Start();
    // Copy data buffers
    ARROW_RETURN_NOT_OK(arrow_copy_to_dpus(system_, "buffer", batches_,
                                           /*batches_offset=*/i, /*column_index=*/0));
    ARROW_RETURN_NOT_OK(arrow_copy_to_dpus(system_, "selection_indices_vector",
                                           indices_batches_,
                                           /*batches_offset=*/i, /*column_index=*/0));
    copyToDpuTimer->Stop();

    // Execute DPU program synchroniously
    dpuWorkTimer->Start();
    system_.exec_with_log(std::cout);
    dpuWorkTimer->Stop();

    // Read output buffers from DPUs
    copyFromDpuTimer->Start();
    arrow::BufferVector buffers;
    auto output_buffer_size = indices_buffer_length * sizeof(uint32_t);
    for (size_t i = 0; i < nr_dpus; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto maybe_buffer, arrow::AllocateBuffer(output_buffer_size));
      buffers.push_back(std::move(maybe_buffer));
      DPU_ASSERT(
          dpu_prepare_xfer(system_.dpus()[i]->unsafe(), buffers[i]->mutable_data()));
    }
    DPU_ASSERT(dpu_push_xfer(system_.unsafe(), DPU_XFER_FROM_DPU, "output_buffer", 0,
                             ROUND_UP_TO_MULTIPLE_OF_8(output_buffer_size),
                             DPU_XFER_DEFAULT));
    copyFromDpuTimer->Stop();

    buildResultTimer->Start();
    for (size_t i = 0; i < nr_dpus; ++i) {
      auto array_data = arrow::ArrayData::Make(arrow::uint32(), indices_buffer_length,
                                               {nullptr, buffers[i]});
      auto array = arrow::MakeArray(array_data);
      results.push_back(::arrow::RecordBatch::Make(batches_[0]->schema(), array->length(), {array}));
    }
    buildResultTimer->Stop();

  }
  return arrow::Table::FromRecordBatches(results);
}

}  // namespace take
}  // namespace upmemeval
