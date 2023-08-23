#!/usr/bin/env python
import time
import argparse
import resource

from multiprocessing import Process

import pyarrow as pa
import pyarrow.compute as cp
import numpy as np


def data(n_batches, batch_size, indices_size):
    batches = []
    for i in range(n_batches):
        v = np.random.randint((1 << 32) + 1, size=batch_size)
        batch = pa.record_batch([v], names=["v"])
        batches.append(batch)
    return batches, [np.random.randint(batch_size, size=indices_size) for _ in range(n_batches)]

def with_arrow(batches, indices):
    results = []
    for batch, indexes in zip(batches, indices):
        results.append(cp.take(batch, indexes))
    return results

def run(sf, n_batches, batch_size, indices_size, engine):
    batches, indices = data(n_batches, batch_size, indices_size)

    st_real = time.time()
    st_cpu = time.process_time()
    result = engine(batches, indices)
    et_cpu = time.process_time()
    et_real = time.time()
    ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print("{},{},{},{},{},{},{}".format(sf,
                                        n_batches,
                                        batch_size,
                                        sum([len(x) for x in result]),
                                        int((et_real-st_real)*1000),
                                        int((et_cpu-st_cpu)*1000),
                                        ram_usage))


def main():
    engines = {
        "arrow": with_arrow
    }
    parser = argparse.ArgumentParser(description='Benchmark Take.')
    parser.add_argument('engine', type=str, choices=engines.keys(),
                        help='Engine to use for benchmarking')
    parser.add_argument('--repeats', dest='repeats', type=int, default=3,
                        help='number of times to repeat each run (default: 3)')
    parser.add_argument('--batches', dest='batches', type=int, default=1,
                        help='number of batches with scale factor of 1 (default: 1)')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4 << 20,
                        help='number of rows in each batch (default: 4MiB)')
    parser.add_argument('--indices_size', dest='indices_size', type=int, default=4 << 20 >> 3,
                        help='number of rows in each batch')
    args = parser.parse_args()
    engine = engines[args.engine]

    print("SF,Batches,Batch-Size,Results,Real-Time(ms),CPU-Time(ms),Memory(KiB)")
    for sf in (2**p for p in range(0, 12)):
        for _ in range(args.repeats):
            p = Process(target=run,
                        args=(sf, sf * args.batches, args.batch_size, args.indices_size, engine))
            p.start()
            p.join()


if __name__ == "__main__":
    main()
