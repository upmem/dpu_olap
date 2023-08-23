#!/usr/bin/env bash
set +x
set +e

sudo cpupower frequency-set --governor performance

mkdir -p reports
PATH=~/.local/bin/:$PATH cmake --build ./build --config Release --target all --
cd ./build/bin/Release;

echo "Baseline"

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048
do
SF=$i ./upmem-query-host --benchmark_filter=BM_JoinNative --benchmark_out=../../../reports/join_native_$i.json --benchmark_out_format=json --benchmark_repetitions=3;
done

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048
do
SF=$i ./upmem-query-host --benchmark_filter=BM_FilterNative --benchmark_out=../../../reports/filter_native_$i.json --benchmark_out_format=json --benchmark_repetitions=3;
done

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048
do
SF=$i ./upmem-query-host --benchmark_filter=BM_SumNative --benchmark_out=../../../reports/sum_native_$i.json --benchmark_out_format=json --benchmark_repetitions=3;
done

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048
do
SF=$i ./upmem-query-host --benchmark_filter=BM_TakeNative --benchmark_out=../../../reports/take_native_$i.json --benchmark_out_format=json --benchmark_repetitions=3;
done
