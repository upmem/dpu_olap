#include "filter_dpu.h"
#include <arrow/status.h>

#include <iostream>
#include <mutex>
#include <map>

#include "dpuext/api.h"
#include "generator/generator.h"

#include "umq/bitops.h"
#include "umq/kernels.h"

using namespace dpu;

namespace upmemeval {

namespace filter {

FilterDpu::FilterDpu(DpuSet& system, arrow::RecordBatchVector batches)
    : system_(system), batches_(std::move(batches)) {}

arrow::Status FilterDpu::Prepare() {
  timers_ = std::make_shared<timer::Timers>(system_.ranks().size());

  try {
    system_.load(FILTER_DPU_BINARY);
  } catch (DpuError& e) {
    return arrow::Status::UnknownError(e.what());
  }
  return arrow::Status::OK();
}

arrow::Status arrow_assign_or_raise_func(arrow::BufferVector & buffers, unsigned int buffer_size) {

  ARROW_ASSIGN_OR_RAISE(auto maybe_buffer, arrow::AllocateBuffer(buffer_size));
  buffers.push_back(std::move(maybe_buffer));
  return arrow::Status::OK();
}

struct FilteringPostProcessCallbackData { 
  uint32_t batch_offset;
  std::map<uint32_t, std::shared_ptr<arrow::Array>> *chunks;
  std::mutex *mutex;

  std::shared_ptr<timer::Timer> buildResultTimer;
  std::shared_ptr<timer::Timer> copyFromDpuTimer;
};

dpu_error_t FilteringPostProcessCallback(struct dpu_set_t rank, 
    __attribute((unused)) uint32_t rank_id, void * args) {

  FilteringPostProcessCallbackData * data = (FilteringPostProcessCallbackData*)args;

  data->copyFromDpuTimer->Start(rank_id);

  // Get output buffers lengths
  uint64_t output_length_results[64] = {0};
  struct dpu_set_t dpu;
  uint32_t each_dpu;
  DPU_FOREACH(rank, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &output_length_results[each_dpu]));
  }
  DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "output_buffer_length", 0, 4, DPU_XFER_DEFAULT));

  // Read output buffers from DPUs
  unsigned int max_buffer_size = 0;
  DPU_FOREACH(rank, dpu, each_dpu) {
    auto output_buffer_length = output_length_results[each_dpu];
    auto output_buffer_size = output_buffer_length << 2;
    auto transfer_buffer_size = ROUND_UP_TO_MULTIPLE_OF_8(output_buffer_size);
    if (transfer_buffer_size > max_buffer_size) {
      max_buffer_size = transfer_buffer_size;
    }
  }

  arrow::BufferVector buffers;
  DPU_FOREACH(rank, dpu, each_dpu) {
    const arrow::Status s = arrow_assign_or_raise_func(buffers, max_buffer_size);
    if(!s.ok()) return DPU_ERR_INTERNAL;
    DPU_ASSERT(dpu_prepare_xfer(dpu, buffers[each_dpu]->mutable_data()));
  }
  DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "output_buffer", 0, max_buffer_size, DPU_XFER_DEFAULT));

  data->copyFromDpuTimer->Stop(rank_id);

  data->buildResultTimer->Start(rank_id);

  DPU_FOREACH(rank, dpu, each_dpu) {
    auto output_buffer_length = output_length_results[each_dpu];
    auto array_data = arrow::ArrayData::Make(arrow::uint32(), output_buffer_length, {nullptr, buffers[each_dpu]});
    auto array = arrow::MakeArray(array_data);
    // need to take a mutex to update the global array
    const std::lock_guard<std::mutex> lock(*data->mutex);
    data->chunks->insert(std::make_pair(data->batch_offset + rank_id * 64 + each_dpu, array));
  }

  data->buildResultTimer->Stop(rank_id);

  return DPU_OK;
}


arrow::Result<std::shared_ptr<arrow::ChunkedArray>> FilterDpu::GetResult() {

  // in asynchronous mode, use one timer per rank
  std::shared_ptr<timer::Timer> copyToDpuTimer = timers_->New("copy-to-dpu");
  std::shared_ptr<timer::Timer> copyFromDpuTimer = timers_->New("copy-from-dpu");
  std::shared_ptr<timer::Timer> dpuWorkTimer = timers_->New("dpu-work");
  std::shared_ptr<timer::Timer> buildResultTimer = timers_->New("build-result");

  // Copy buffer length
  // XXX: It's assumed that all buffers are of the same length
  ARROW_ASSIGN_OR_RAISE(uint32_t buffer_length,
                        arrow_data_buffer_length(batches_, 0UL, 0));
  std::vector<uint32_t> buffer_length_param{buffer_length};
  system_.async().copy("buffer_length", 0, buffer_length_param, sizeof(uint32_t));

  // Holds the result array
  arrow::ArrayVector chunks;
  std::map<uint32_t, std::shared_ptr<arrow::Array>> chunks_map;
  std::mutex chunks_mutex;
  std::vector<FilteringPostProcessCallbackData> data(batches_.size(), 
      {0, &chunks_map, &chunks_mutex, buildResultTimer, copyFromDpuTimer});

  auto nr_dpus = system_.dpus().size();
  assert(batches_.size() % nr_dpus == 0);
  for (size_t i = 0; i < batches_.size(); i += nr_dpus) {

    // use a callback in order to introduce timers
    system_.async().call([this, &copyToDpuTimer, i](DpuSet &set, uint32_t rank_id)->void {

        copyToDpuTimer->Start(rank_id);
        arrow::Status status = arrow_copy_to_dpus(set, "buffer", batches_,
              /*batches_offset=*/i, /*column_index=*/0, /*async*/false);
        if(!status.ok()) throw std::exception();
        copyToDpuTimer->Stop(rank_id);
    });

    // Execute DPU program asynchronously
    // Excecute in a callback in order to introduce timers
    system_.async().call([&dpuWorkTimer](DpuSet& set, uint32_t rank_id)->void {

        dpuWorkTimer->Start(rank_id);
        set.exec();
        dpuWorkTimer->Stop(rank_id);
    });

    // Display logs from DPUs
    // system_.log(std::cout);

    /* Gather results from all DPUs */
    //use a callback to perform the transfer + post-processing
    data[i].batch_offset = i * nr_dpus;
    DPU_ASSERT(dpu_callback(system_.unsafe(), FilteringPostProcessCallback, &data[i], DPU_CALLBACK_ASYNC));
  }

  // synchronization, wait for all DPUS
  // to finish their jobs
  system_.async().sync();

  for(auto const &e : chunks_map) {
    chunks.push_back(std::move(e.second));
  }

  ARROW_ASSIGN_OR_RAISE(auto result, arrow::ChunkedArray::Make(chunks));

  return result;
}

arrow::Result<uint64_t> FilterDpu::Run() {
  return GetResult()->get()->length();
}

}  // namespace filter
}  // namespace upmemeval
