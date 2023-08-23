#include <iostream>
#include <list>
#include <string>

#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <gtest/gtest.h>

#include "dpuext/api.h"

#include "generator/generator.h"

#include "join/join_dpu.h"
#include "join/join_native.h"

namespace upmemeval {
using namespace upmemeval::generator;
using namespace upmemeval::join;

template <typename J>
::arrow::Result<std::shared_ptr<arrow::Table>> do_join(J&& joiner) {
  ARROW_RETURN_NOT_OK(joiner.Prepare());
  return joiner.Run();
}

::arrow::Result<std::shared_ptr<arrow::Table>> do_sort(
    std::shared_ptr<arrow::Table> table, std::list<std::string> sort_keys) {
  auto sortOptions = arrow::compute::SortOptions::Defaults();
  for (auto& sort_key : sort_keys) {
    sortOptions.sort_keys.push_back(arrow::compute::SortKey{arrow::FieldRef(sort_key)});
  }
  auto indices = arrow::compute::SortIndices(table, sortOptions).ValueOrDie();
  ARROW_ASSIGN_OR_RAISE(
      auto result,
      arrow::compute::Take(table, indices, arrow::compute::TakeOptions::Defaults()));
  return result.table();
}

TEST(JoinTest, SimpleTest) {
  arrow::RecordBatchVector left_vector(2);
  arrow::RecordBatchVector right_vector(2);

  auto left_data1 = {
      ArrayOf<::arrow::UInt32Type>({0, 2, 3, 8, 9}),
      ArrayOf<::arrow::UInt32Type>({100, 102, 103, 108, 109}),
  };
  left_vector[0] = RecordBatchOf({"fk", "v_l"}, left_data1);
  auto left_data2 = {
      ArrayOf<::arrow::UInt32Type>({10, 12, 13, 18, 19}),
      ArrayOf<::arrow::UInt32Type>({110, 112, 113, 118, 119}),
  };
  left_vector[1] = RecordBatchOf({"fk", "v_l"}, left_data2);

  auto right_data1 = {
      ArrayOf<::arrow::UInt32Type>({3, 8, 9, 0, 12}),
      ArrayOf<::arrow::UInt32Type>({53, 58, 59, 50, 62}),
  };
  right_vector[0] = RecordBatchOf({"pk", "v_r"}, right_data1);
  auto right_data2 = {
      ArrayOf<::arrow::UInt32Type>({13, 18, 19, 10, 2}),
      ArrayOf<::arrow::UInt32Type>({63, 68, 69, 60, 52}),
  };
  right_vector[1] = RecordBatchOf({"pk", "v_r"}, right_data2);

  auto system_ = dpu::DpuSet::allocate(2, "sgXferEnable=true");

  auto native = do_join(JoinNative{left_vector[0]->schema(), right_vector[0]->schema(),
                                   left_vector, right_vector})
                    .ValueOrDie();
//   std::cout << "Native: " << native->ToString() << std::endl;
  auto dpuEnabled = do_join(JoinDpu{system_, left_vector[0]->schema(),
                                    right_vector[0]->schema(), left_vector, right_vector})
                        .ValueOrDie();

  auto sorted_dpu = do_sort(dpuEnabled, {"v_l", "fk"}).ValueOrDie();
  auto sorted_native = do_sort(native, {"v_l", "fk"}).ValueOrDie();

  EXPECT_TRUE(sorted_native->Equals(*sorted_dpu));
}

TEST(JoinTest, LargeTest) {
  arrow::random::RandomArrayGenerator rng(42);
  auto num_batches = 128;
  auto batch_size = 64 << 10;
  auto system_ = dpu::DpuSet::allocate(128, "nrJobsPerRank=256,sgXferEnable=true");

  auto right_schema =
      arrow::schema({arrow::field("x", arrow::uint32(), /*nullable=*/false)});
  auto right_batches =
      generator::MakeRandomRecordBatches(rng, right_schema, num_batches, batch_size);
  auto right_pk_column = generator::MakeIndexColumn(num_batches, batch_size);
  right_batches = generator::AddColumn("pk", right_batches, right_pk_column.ValueOrDie());
  right_schema = right_batches[0]->schema();

  auto left_schema =
      arrow::schema({arrow::field("y", arrow::uint32(), /*nullable=*/false)});
  auto left_batches =
      generator::MakeRandomRecordBatches(rng, left_schema, num_batches, batch_size);
  auto left_fk_column =
      generator::MakeForeignKeyColumn(rng, batch_size, num_batches, batch_size);
  left_batches = generator::AddColumn("fk", left_batches, left_fk_column.ValueOrDie());
  left_schema = left_batches[0]->schema();

  auto native =
      do_join(JoinNative{left_schema, right_schema, left_batches, right_batches})
          .ValueOrDie();
  auto sorted_native = do_sort(native, {"fk", "y"}).ValueOrDie();

  auto dpuEnabled =
      do_join(JoinDpu{system_, left_schema, right_schema, left_batches, right_batches})
          .ValueOrDie();
  auto sorted_dpu = do_sort(dpuEnabled, {"fk", "y"}).ValueOrDie();

  EXPECT_EQ(sorted_native->num_rows(), num_batches * batch_size);
  EXPECT_EQ(sorted_dpu->num_rows(), num_batches * batch_size);
  for (auto& field_name : sorted_native->schema()->field_names()) {
    EXPECT_TRUE(sorted_native->GetColumnByName(field_name)
                    ->Equals(sorted_dpu->GetColumnByName(field_name)));
  }
}

}  // namespace upmemeval
