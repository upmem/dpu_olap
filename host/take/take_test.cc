#include <iostream>
#include <string>

#include <arrow/api.h>

#include <gtest/gtest.h>

#include "dpuext/api.h"

#include "generator/generator.h"

#include "take/take_dpu.h"
#include "take/take_native.h"

using namespace upmemeval::generator;
using namespace upmemeval::take;

template <typename T>
::arrow::Result<std::shared_ptr<::arrow::Table>> do_take(T&& t) {
  ARROW_RETURN_NOT_OK(t.Prepare());
  return t.Run();
}

TEST(TakeTest, SimpleTestNative) {
  auto rb = RecordBatchOf({"v"}, {ArrayOf<::arrow::UInt32Type>({0, 2, 3, 8, 9})});
  auto indices = RecordBatchOf({"i"}, {ArrayOf<::arrow::UInt32Type>({0, 1, 4})});
  auto result = do_take(TakeNative{rb->schema(), {rb}, {indices}}).ValueOrDie();

  auto array =
      std::dynamic_pointer_cast<::arrow::UInt32Array>(result->column(0)->chunk(0));
  EXPECT_EQ(array->length(), 3);
  EXPECT_EQ(array->Value(0), 0);
  EXPECT_EQ(array->Value(1), 2);
  EXPECT_EQ(array->Value(2), 9);
}

TEST(TakeTest, SimpleTestDpu) {
  auto system = dpu::DpuSet::allocate(1);
  auto rb = RecordBatchOf({"v"}, {ArrayOf<::arrow::UInt32Type>({0, 2, 3, 8, 9})});
  auto indices = RecordBatchOf({"i"}, {ArrayOf<::arrow::UInt32Type>({0, 1, 4})});
  auto result = do_take(TakeDpu{system, {rb}, {indices}}).ValueOrDie();

  auto array =
      std::dynamic_pointer_cast<::arrow::UInt32Array>(result->column(0)->chunk(0));
  EXPECT_EQ(array->length(), 3);
  EXPECT_EQ(array->Value(0), 0);
  EXPECT_EQ(array->Value(1), 2);
  EXPECT_EQ(array->Value(2), 9);
}

TEST(TakeTest, LargeTest) {
  auto system = dpu::DpuSet::allocate(16);
  auto num_batches = 128;
  auto batch_size = 64 << 10;
  auto indices_batch_size = 8 << 10;

  arrow::random::RandomArrayGenerator rng(42);
  auto schema = arrow::schema({arrow::field("v", arrow::uint32(), /*nullable=*/false)});
  auto batches = MakeRandomRecordBatches(rng, schema, num_batches, batch_size);

  auto indices_metadata = arrow::key_value_metadata(
      {{"min", std::to_string(0)}, {"max", std::to_string(batch_size - 1)}});
  auto indices_schema = arrow::schema(
      {arrow::field("i", arrow::uint32(), /*nullable=*/false, indices_metadata)});
  auto indices_batches = MakeRandomRecordBatches(
      rng, indices_schema, num_batches, indices_batch_size);

  auto result_native = do_take(TakeNative{schema, batches, indices_batches}).ValueOrDie();
  auto result_dpu = do_take(TakeDpu{system, batches, indices_batches}).ValueOrDie();

  EXPECT_TRUE(result_native->Equals(*result_dpu));
}
