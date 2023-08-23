#include <algorithm>
#include <iostream>
#include <string>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/pretty_print.h>

#include <gtest/gtest.h>

#include "dpuext/api.h"

#include "generator/generator.h"

#include "partition/partition_dpu.h"

namespace upmemeval {
using namespace upmemeval::generator;
using namespace upmemeval::partition;

TEST(PartitionTest, SimpleTest) {
  GTEST_SKIP() << "Skipping PartitionTest.SimpleTest (manually disabled)";

  auto system_ = dpu::DpuSet::allocate(1);
  auto nr_partitions = 2;

  auto data = {
      ArrayOf<::arrow::UInt32Type>({0, 2, 3, 8}),
      ArrayOf<::arrow::UInt32Type>({100, 101, 102, 103}),
  };
  auto batch = RecordBatchOf({"pk", "x"}, data);

  auto partition_dpu =
      PartitionDpu(system_, batch->schema(), {batch}, nr_partitions, "pk");
  EXPECT_TRUE(partition_dpu.Prepare().ok());

  auto partitioned_batches = partition_dpu.Run().ValueOrDie();
  EXPECT_EQ(partitioned_batches.size(), nr_partitions);
  EXPECT_EQ(partitioned_batches[0]->num_rows(), 3);
  EXPECT_EQ(partitioned_batches[1]->num_rows(), 1);

  int64_t num_rows = 0;
  for (auto& batch : partitioned_batches) {
    EXPECT_GT(batch->num_rows(), 0);
    num_rows += batch->num_rows();
  }
  EXPECT_EQ(num_rows, batch->num_rows());

  auto partitioned_table = ::arrow::Table::FromRecordBatches(partitioned_batches);
  EXPECT_TRUE(::arrow::compute::Sum(partitioned_table->get()->GetColumnByName("pk"))
                  .ValueOrDie()
                  .scalar()
                  ->Equals(arrow::UInt64Scalar(13)));
  EXPECT_TRUE(::arrow::compute::Sum(partitioned_table->get()->GetColumnByName("x"))
                  .ValueOrDie()
                  .scalar()
                  ->Equals(arrow::UInt64Scalar(406)));
}

TEST(PartitionTest, LargeTest) {
  GTEST_SKIP() << "Skipping PartitionTest.LargeTest (manually disabled)";
  
  auto system_ = dpu::DpuSet::allocate(16);
  auto nr_partitions = 32;
  auto num_batches = 128;
  auto batch_size = 64 << 10;

  auto approax_target_batch_size = (1.0 * num_batches * batch_size) / nr_partitions;

  arrow::random::RandomArrayGenerator rng(42);
  auto schema = arrow::schema({arrow::field("pk", arrow::uint32(), /*nullable=*/false),
                               arrow::field("x", arrow::uint32(), /*nullable=*/false),
                               arrow::field("y", arrow::uint32(), /*nullable=*/false)});
  auto batches = generator::MakeRandomRecordBatches(rng, schema, num_batches, batch_size);

  auto partition_dpu = PartitionDpu(system_, schema, batches, nr_partitions, "pk");
  EXPECT_TRUE(partition_dpu.Prepare().ok());
  auto partitioned_batches = partition_dpu.Run().ValueOrDie();
  EXPECT_EQ(partitioned_batches.size(), nr_partitions);

  int64_t num_rows = 0;
  for (auto& batch : partitioned_batches) {
    EXPECT_EQ(batch->num_columns(), schema->num_fields());
    EXPECT_GT(batch->num_rows(), 0);
    // Check that actual batch size is +-1% from expectation
    EXPECT_LE(std::abs(batch->num_rows() - approax_target_batch_size) /
                  approax_target_batch_size,
              0.1);
    num_rows += batch->num_rows();
  }
  EXPECT_EQ(num_rows, num_batches * batch_size);
}

}  // namespace upmemeval
