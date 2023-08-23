#!/usr/bin/env python
import time
import argparse
import resource
from multiprocessing import Process

import datafusion
import polars as pl
import pyarrow as pa
import numpy as np


def data(n_batches, batch_size):
    left_batches = []
    for i in range(n_batches):
        fk = np.random.randint(i*batch_size, (i+1)*batch_size, size=batch_size)
        x = np.random.randint(64 << 10, size=batch_size)
        batch = pa.record_batch([fk, x], names=["fk", "x"])
        left_batches.append(batch)

    right_batches = []
    for i in range(n_batches):
        pk = np.arange(i*batch_size, (i+1)*batch_size)
        y = np.random.randint(64 << 10, size=batch_size)
        batch = pa.record_batch([pk, y], names=["pk", "y"])
        right_batches.append(batch)

    return left_batches, right_batches


def with_datafusion(left_batches, right_batches):

    ctx = datafusion.SessionContext()
    left = ctx.create_dataframe([left_batches])
    right = ctx.create_dataframe([right_batches])
    result_batches = left.join(right, join_keys=(
        ["fk"], ["pk"]), how="inner").collect()
    return pa.Table.from_batches(result_batches)


def with_polars(left_batches, right_batches):
    left_table = pa.Table.from_batches(left_batches)
    left = pl.from_arrow(left_table)
    right_table = pa.Table.from_batches(right_batches)
    right = pl.from_arrow(right_table)
    result = left.join(right, left_on="fk", right_on="pk", how="inner")
    return result.to_arrow()


def with_arrow(left_batches, right_batches):
    left = pa.Table.from_batches(left_batches)
    right = pa.Table.from_batches(right_batches)
    return left.join(right, keys="fk", right_keys="pk", join_type="inner")


def with_noop(*args):
    return []


def run(sf, n_batches, batch_size, partition_size, engine):
    left_batches, right_batches = data(n_batches, batch_size)

    results_len = 0

    st_real = time.time()
    st_cpu = time.process_time()
    if partition_size is not None:
        assert(partition_size % batch_size == 0)
        factor = int(partition_size / batch_size)
        for l, r in zip(zip(*[iter(left_batches)]*factor), zip(*[iter(right_batches)]*factor)):
            result = engine(list(l), list(r))
            results_len += len(result)
    else:
        result = engine(left_batches, right_batches)
        results_len = len(result)
    et_cpu = time.process_time()
    et_real = time.time()
    ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print("{},{},{},{},{},{},{}".format(sf,
                                        n_batches,
                                        batch_size,
                                        results_len,
                                        int((et_real-st_real)*1000),
                                        int((et_cpu-st_cpu)*1000),
                                        ram_usage))


def main():
    engines = {
        "polars": with_polars,
        "datafusion": with_datafusion,
        "arrow": with_arrow,
        "noop": with_noop,
    }
    parser = argparse.ArgumentParser(description='Benchmark Join.')
    parser.add_argument('engine', type=str, choices=engines.keys(),
                        help='Engine to use for benchmarking')
    parser.add_argument('--partition_size', dest='partition_size', type=int, default=None,
                        help='If specified, assumes that batches are partitioned in a given partition size')
    parser.add_argument('--repeats', dest='repeats', type=int, default=3,
                        help='number of times to repeat each run (default: 3)')
    parser.add_argument('--batches', dest='batches', type=int, default=32,
                        help='number of batches with scale factor of 1 (default: 32)')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64 << 10,
                        help='number of rows in each batch (default: 65536)')
    args = parser.parse_args()
    engine = engines[args.engine]

    print("SF,Batches,Batch-Size,Results,Real-Time(ms),CPU-Time(ms),Memory(KiB)")
    for sf in (2**p for p in range(0, 12)):
        for _ in range(args.repeats):
            p = Process(target=run,
                        args=(sf, sf * args.batches, args.batch_size, args.partition_size, engine))
            p.start()
            p.join()


if __name__ == "__main__":
    main()
