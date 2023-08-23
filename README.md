UPMEM-PIM Evaluation for SQL Query Acceleration
===============================================

This repo includes a benchmark of compute operators Filter and Join.
The benchmark uses Apache Arrow Streaming Execution Engine as baseline and UPMEM-PIM implementations as target.

Contributors
------------

- [Roee Shlomo](roee88@gmail.com)
- [Julien Legriel](jlegriel@upmem.com)
- [Aphélie Moisson](amoisson@upmem.com)
- [Sylvan Brocard](sbrocard@upmem.com)

Dependencies
------------

- Install [UPMEM SDK](https://sdk.upmem.com/)
- Install [CMake](https://cmake.org/download/) 3.16 or higher

CPM will automatically download the following dependencies:

- [Google Benchmark](https://github.com/google/benchmark)
- [GoogleTest](https://github.com/google/googletest)
- [Apache Arrow](https://arrow.apache.org/install/) 8.0.0

If you already have these dependencies installed, CPM will use them instead of downloading them (this can be disabled by setting `CPM_USE_LOCAL_PACKAGES` to `OFF`).

Build
-----

```bash
cmake -B./build -G "Ninja Multi-Config"
cmake --build ./build --config Release --target all --
```

or

```bash
make build
```

Run
---

```bash
cd ./build/bin/Release/
./upmem-query-host
```

Add `--help` for additional usage instructions.

or

```bash
make run
```

Note: Don't try to run all the benckmarks at once. Select one, for example:

```bash
./upmem-query-host --benchmark_filter=BM_JoinDpu
```

Add `--benchmark_list_tests` to list all the benchmarks.

The default number of DPUs and scale factor are 2048. To change them, prefix the command with `NR_DPUS=1024 SF=1024` (for example).

UPMEM Cloud environment
-----------------------

### Install numactl

```bash
sudo apt install numactl
```

### Run baseline benchmarks

**Only** if you're on a baseline machine setup run the following:

```bash
nohup ./scripts/run-python-baselines.sh &
```

### Run UPMEM benchmarks with 2048 DPUs

First edit `./scripts/run-upmem-2048.sh` and change `OUT_DIR` to reflect the current machine setup. **For example**, if the setup is 12-12 then replace `OUT_DIR=../../../results/16-8/2048` with `OUT_DIR=../../../results/12-12/2048`

Then run the script:

```bash
nohup ./scripts/run-upmem-2048.sh &
```

### Run UPMEM benchmarks with scaling number of DPUs up to 2048

First edit `./scripts/run-upmem-scale.sh` and change `OUT_DIR` to reflect the current machine setup. **For example**, if the setup is 12-12 then replace `OUT_DIR=../../../results/16-8/scale` with `OUT_DIR=../../../results/12-12/scale`

Then run the script:

```bash
nohup ./scripts/run-upmem-scale.sh &
```

### Gather results

To gather the results locally run from your own local machine:

```bash
scp -r upmemcloud3:~/workspace/upmem-query/results ./results/
```

Current status
--------------

Standalone partition is non-functional at the moment due to changes to support partition in the join operation.
The corresponding benchmark and tests are disabled.
