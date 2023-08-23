#!/usr/bin/env bash
set +x
set +e

sudo cpupower frequency-set --governor performance

mkdir -p results/baseline
source ./.venv/bin/activate
cd ./baseline;

pip install -r requirements.txt

echo "Running Python Baselines"

echo "Partitioned Join using datafusion"
./join.py datafusion --partition_size=2097152 > ../results/baseline/join_datafusion-partitioned.csv
echo "Partitioned Join using polars"
./join.py polars --partition_size=2097152 > ../results/baseline/join_polars-partitioned.csv

echo "Join using datafusion"
./join.py datafusion > ../results/baseline/join_datafusion.csv
echo "Join using polars"
./join.py polars > ../results/baseline/join_polars.csv
echo "Join using arrow"
./join.py arrow > ../results/baseline/join_arrow.csv

echo "Filter using datafusion"
./filter.py datafusion > ../results/baseline/filter_datafusion.csv
echo "Filter using polars"
./filter.py polars > ../results/baseline/filter_polars.csv
echo "Filter using arrow"
./filter.py arrow > ../results/baseline/filter_arrow.csv

echo "Sum using polars"
./sum.py polars > ../results/baseline/sum_polars.csv
echo "Sum using arrow"
./sum.py arrow > ../results/baseline/sum_arrow.csv

echo "Take using arrow"
./take.py arrow > ../results/baseline/take_arrow.csv
