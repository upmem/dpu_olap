#ifndef _UMQ_KERNELS_H_
#define _UMQ_KERNELS_H_

#include <stdint.h>

#define FILTER_DPU_BINARY "kernel-filter"
#define PARTITION_DPU_BINARY "kernel-partition"
#define JOIN_DPU_BINARY "kernel-join"
#define TAKE_DPU_BINARY "kernel-take"
#define AGGREGATE_DPU_BINARY "kernel-aggr"

enum Kernel {
  KernelUnspecified = 0,
  KernelFilter,
  KernelTake,
  KernelHashBuild,
  KernelHashProbe,
  KernelPartition,
  KernelAggregate,
};

enum AggregatorType {
  AggrUnspecified = 0,
  AggrSum,
};

typedef struct {
  uint32_t nr_partitions;  // Must be a power of two
  uint32_t with_selection_indices_array;
  uint32_t buffer_length;
  uint32_t output_shift;
} kernel_partition_inputs_t __attribute__((aligned(8)));

typedef struct {
  uint32_t partitions_metadata_offset;
  uint32_t output_buffer_length;
} kernel_partition_outputs_t __attribute__((aligned(8)));

typedef struct {
  uint32_t buffer_length;
  uint32_t selection_indices_vector_length;
} dpu_take_input __attribute__((aligned(8)));

typedef struct {
  uint32_t buffer_length;
  int32_t aggregator_type;
} dpu_aggr_input __attribute__((aligned(8)));

typedef struct {
  uint64_t sum_result;
} dpu_aggr_output __attribute__((aligned(8)));

#endif
