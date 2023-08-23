#include <iostream>
#include <string>

#include <arrow/api.h>

#include <gtest/gtest.h>

#include "dpuext/api.h"

#include "generator/generator.h"

#include "filter/filter_dpu.h"
#include "filter/filter_native.h"

using namespace upmemeval::generator;
using namespace upmemeval::filter;

template <typename F>
::arrow::Result<uint64_t> filter(F&& f) {
  ARROW_RETURN_NOT_OK(f.Prepare());
  return f.Run();
}

TEST(FilterTest, SimpleTest) {
  auto rb = RecordBatchOf({"v"}, {ArrayOf<::arrow::UInt32Type>({0, 2, 3, 8, 9})});

  auto system = dpu::DpuSet::allocate(1);
  auto dpuEnabled = filter(FilterDpu{system, {rb}}).ValueOrDie();
  auto native = filter(FilterNative{rb->schema(), {rb}}).ValueOrDie();
  EXPECT_EQ(native, dpuEnabled);
}

TEST(FilterTest,  ResultTest) {
  auto input_array_builder = arrow::NumericBuilder<::arrow::UInt32Type>();
  for (int i=0; i<4096; i++) {
    switch (i) {
    case 5:
    case 8:
    case 9:
    case 100:
    case 270:
      EXPECT_TRUE(input_array_builder.Append(i).ok());
      break;
    default:
      EXPECT_TRUE(input_array_builder.Append(i+(1<<30)).ok());
    }
  }
  auto input_array = input_array_builder.Finish().ValueOrDie();
  auto rb = RecordBatchOf({"v"}, {input_array});

  auto system = dpu::DpuSet::allocate(1);
  auto dpuFilter = FilterDpu{system, {rb}};
  EXPECT_TRUE(dpuFilter.Prepare().ok());
  auto dpuResult = dpuFilter.GetResult().ValueOrDie();
  auto nativeFilter = FilterNative{rb->schema(), {rb}};
  EXPECT_TRUE(nativeFilter.Prepare().ok());
  auto nativeResult = nativeFilter.GetResult().ValueOrDie()->column(0);
  // std::cout << "dpu result:" << dpuResult->ToString() << "\n";
  // std::cout << "native result:" << nativeResult->ToString() << "\n";
  EXPECT_TRUE(dpuResult->Equals(nativeResult));
}

TEST(FilterTest,  LongerTest) {
  auto schema = arrow::schema({arrow::field("v", arrow::uint32(), /*nullable=*/false)});
  auto rng = arrow::random::RandomArrayGenerator(42);
  auto batches = MakeRandomRecordBatches(rng, schema, 1, 1<<16);

  auto system = dpu::DpuSet::allocate(1);
  auto dpuFilter = FilterDpu{system, {batches}};
  EXPECT_TRUE(dpuFilter.Prepare().ok());
  auto dpuResult = dpuFilter.GetResult().ValueOrDie();
  auto nativeFilter = FilterNative{schema, {batches}};
  EXPECT_TRUE(nativeFilter.Prepare().ok());
  auto nativeResult = nativeFilter.GetResult().ValueOrDie()->column(0);
  // std::cout << "dpu result: (" << dpuResult->length() << ")" << dpuResult->ToString() << "\n";
  // std::cout << "native result: (" << nativeResult->length() << ")" << nativeResult->ToString() << "\n";
  EXPECT_TRUE(dpuResult->Equals(nativeResult));
}
