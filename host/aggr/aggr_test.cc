#include <iostream>
#include <string>

#include <arrow/api.h>

#include <gtest/gtest.h>

#include "dpuext/api.h"

#include "generator/generator.h"

#include "aggr/aggr_dpu.h"
#include "aggr/aggr_native.h"

using namespace upmemeval::generator;
using namespace upmemeval::aggr;

template <typename A>
::arrow::Result<uint64_t> sum(A&& a) {
  ARROW_RETURN_NOT_OK(a.Prepare());
  return a.Run();
}

TEST(SumTest, SimpleTestNative) {
  auto rb = RecordBatchOf({"v"}, {ArrayOf<::arrow::UInt32Type>({0, 2, 3, 8, 9})});
  auto result = sum(AggrNative<::arrow::UInt64Array>{rb->schema(), {rb}, "sum"}).ValueOrDie();
  EXPECT_EQ(result, 22);
}

TEST(SumTest, SimpleTestDpu) {
  auto system = dpu::DpuSet::allocate(1);
  auto rb = RecordBatchOf({"v"}, {ArrayOf<::arrow::UInt32Type>({0, 2, 3, 8, 9})});
  auto result = sum(SumDpu{system, {rb}}).ValueOrDie();
  EXPECT_EQ(result, 22);
}

TEST(SumTest, LargeTest) {
  auto system = dpu::DpuSet::allocate(16);
  auto num_batches = 128;
  auto batch_size = 64 << 10;

  arrow::random::RandomArrayGenerator rng(42);
  auto schema = arrow::schema({arrow::field("v", arrow::uint32(), /*nullable=*/false)});
  auto batches = MakeRandomRecordBatches(rng, schema, num_batches, batch_size);

  auto result_native = sum(AggrNative<::arrow::UInt64Array>{schema, batches, "sum"}).ValueOrDie();
  auto result_dpu = sum(SumDpu{system, batches}).ValueOrDie();
  EXPECT_EQ(result_native, result_dpu);
}
