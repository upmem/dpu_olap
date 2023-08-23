#!/usr/bin/env python
import time
import argparse
import resource

from multiprocessing import Process

# import datafusion
# from datafusion import functions as datafusion_functions
import polars as pl
import pyarrow as pa
import pyarrow.compute as cp
import numpy as np


def data(n_batches, batch_size):
    batches = []
    for i in range(n_batches):
        v = np.random.randint((1 << 32) + 1, size=batch_size)
        batch = pa.record_batch([v], names=["v"])
        batches.append(batch)
    return batches


# def with_datafusion(batches):
#     ctx = datafusion.SessionContext()
#     df = ctx.create_dataframe([batches])
#     return df.select(datafusion_functions.sum(datafusion.col("v"))).collect()


def with_polars(batches):
    table = pa.Table.from_batches(batches)
    df = pl.from_arrow(table)
    return df.select([pl.col("v").sum()])["v"][0]


def with_arrow(batches):
    table = pa.Table.from_batches(batches)
    return cp.sum(table.column("v"))


def run(sf, n_batches, batch_size, engine):
    batches = data(n_batches, batch_size)

    st_real = time.time()
    st_cpu = time.process_time()
    result = engine(batches)
    et_cpu = time.process_time()
    et_real = time.time()
    ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print("{},{},{},{},{},{},{}".format(sf,
                                        n_batches,
                                        batch_size,
                                        result,
                                        int((et_real-st_real)*1000),
                                        int((et_cpu-st_cpu)*1000),
                                        ram_usage))


def main():
    engines = {
        "polars": with_polars,
        # "datafusion": with_datafusion,
        "arrow": with_arrow
    }
    parser = argparse.ArgumentParser(description='Benchmark Sum.')
    parser.add_argument('engine', type=str, choices=engines.keys(),
                        help='Engine to use for benchmarking')
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
                        args=(sf, sf * args.batches, args.batch_size, engine))
            p.start()
            p.join()


if __name__ == "__main__":
    main()
