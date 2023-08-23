#include "generator.h"

#include <arrow/testing/random.h>

namespace upmemeval {

namespace generator {

using namespace arrow;
using namespace arrow::compute;

template <typename Fn, typename From,
          typename To = decltype(std::declval<Fn>()(std::declval<From>()))>
std::vector<To> MapVector(Fn&& map, const std::vector<From>& source) {
  std::vector<To> out;
  out.reserve(source.size());
  std::transform(source.begin(), source.end(), std::back_inserter(out),
                 std::forward<Fn>(map));
  return out;
}

arrow::RecordBatchVector MakeRandomRecordBatches(
    ::arrow::random::RandomArrayGenerator& g,
    const std::shared_ptr<arrow::Schema>& schema, int num_batches, int batch_size) {
  RecordBatchVector out(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    out[i] = g.BatchOf(schema->fields(), batch_size);
  }
  return out;
}

arrow::RecordBatchVector AddColumn(const std::string& name,
                                   const arrow::RecordBatchVector& batches,
                                   arrow::ArrayVector indexColumn) {
  arrow::RecordBatchVector out;
  out.reserve(batches.size());

  assert(batches.size() == indexColumn.size());
  for (size_t i = 0; i < batches.size(); ++i) {
    assert(batches[i]->num_rows() == indexColumn[i]->length());
    out.push_back(batches[i]->AddColumn(0, name, indexColumn[i]).ValueOrDie());
  }
  return out;
}

arrow::Result<arrow::ArrayVector> MakeForeignKeyColumn(
    ::arrow::random::RandomArrayGenerator& g, uint32_t pk_batch_size, int32_t num_batches,
    int32_t batch_size) {
  arrow::ArrayVector out(num_batches);
  for (uint32_t i = 0; i < (uint32_t)num_batches; ++i) {
    auto metadata =
        arrow::key_value_metadata({{"min", std::to_string(i * pk_batch_size)},
                                   {"max", std::to_string((i + 1) * pk_batch_size - 1)}});
    out[i] = g.ArrayOf(*arrow::field("fk", arrow::uint32(), false, metadata), batch_size);
  }
  return out;
}

arrow::Result<arrow::ArrayVector> MakeIndexColumn(int num_batches, int batch_size) {
  uint32_t value = 0;
  arrow::ArrayVector out(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    arrow::UInt32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(batch_size));
    for (int j = 0; j < batch_size; ++j) {
      builder.UnsafeAppend(value++);
    }
    ARROW_ASSIGN_OR_RAISE(out[i], builder.Finish());
  }
  return out;
}

std::vector<arrow::compute::ExecBatch> ToExecBatches(arrow::RecordBatchVector batches) {
  return MapVector([](std::shared_ptr<RecordBatch> batch) { return ExecBatch(*batch); },
                   batches);
}

arrow::AsyncGenerator<arrow::util::optional<arrow::compute::ExecBatch>>
MakeAsyncGenerator(std::vector<arrow::compute::ExecBatch> batches, bool parallel) {
  auto opt_batches = MapVector(
      [](arrow::compute::ExecBatch batch) {
        return util::make_optional(std::move(batch));
      },
      batches);

  AsyncGenerator<util::optional<ExecBatch>> gen;

  if (parallel) {
    // emulate batches completing initial decode-after-scan on a cpu thread
    gen = MakeBackgroundGenerator(MakeVectorIterator(std::move(opt_batches)),
                                  ::arrow::internal::GetCpuThreadPool())
              .ValueOrDie();

    // ensure that callbacks are not executed immediately on a background thread
    gen = MakeTransferredGenerator(std::move(gen), ::arrow::internal::GetCpuThreadPool());
  } else {
    gen = MakeVectorGenerator(std::move(opt_batches));
  }

  return gen;
}

std::shared_ptr<::arrow::RecordBatch> RecordBatchOf(
    std::vector<std::string> names, std::vector<std::shared_ptr<arrow::Array>> data) {
  ::arrow::FieldVector fields(names.size());
  for (size_t i = 0; i < fields.size(); ++i) {
    fields[i] = ::arrow::field(names[i], data[i]->type());
  }
  return ::arrow::RecordBatch::Make(arrow::schema(fields), data[0]->length(), data);
}

}  // namespace generator
}  // namespace upmemeval
