#!/usr/bin/env bash
set +x
set +e

sudo cpupower frequency-set --governor performance

REPEATS=1
OUT_DIR=../../../results/16-8/2048
PATH=~/.local/bin/:$PATH cmake --build ./build --config Release --target all --

cd ./build/bin/Release;
mkdir -p $OUT_DIR
echo "UPMEM Enabled (2048)"

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
do
NR_DPUS=2048 SF=$i numactl --interleave=all ./upmem-query-host --benchmark_filter=BM_FilterDpu --benchmark_out=$OUT_DIR/filter_dpu_$i.json --benchmark_out_format=json --benchmark_repetitions=$REPEATS;
done

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
do
NR_DPUS=2048 SF=$i numactl --interleave=all ./upmem-query-host --benchmark_filter=BM_JoinDpu --benchmark_out=$OUT_DIR/join_dpu_$i.json --benchmark_out_format=json --benchmark_repetitions=$REPEATS;
done

for i in 2 4 8 16 32 64 128 256 512 1024 2048 4096
do
NR_DPUS=2048 SF=$i numactl --interleave=all ./upmem-query-host --benchmark_filter=BM_Partition --benchmark_out=$OUT_DIR/partition_dpu_$i.json --benchmark_out_format=json --benchmark_repetitions=$REPEATS;
done

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
do
NR_DPUS=2048 SF=$i numactl --interleave=all ./upmem-query-host --benchmark_filter=BM_SumDpu --benchmark_out=$OUT_DIR/sum_dpu_$i.json --benchmark_out_format=json --benchmark_repetitions=$REPEATS;
done

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
do
NR_DPUS=2048 SF=$i numactl --interleave=all ./upmem-query-host --benchmark_filter=BM_TakeDpu --benchmark_out=$OUT_DIR/take_dpu_$i.json --benchmark_out_format=json --benchmark_repetitions=$REPEATS;
done
