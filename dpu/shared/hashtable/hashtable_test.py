#!/usr/bin/env python3

import time 
from dpu import DpuSet 
from dpu import ALLOCATE_ALL

with DpuSet(1, binary = "./build/dpu/join/Release/upmem-query-dpu-join-hashmap-test") as dpus:

    print("DPU allocated") 
    start = time.monotonic() 
    dpus.exec() 
    end = time.monotonic() 

    nb_cycles =[bytearray(4) for _ in dpus] 
    dpus.copy(nb_cycles, 'nb_cycles') 
    
    print("DPU cycles:", int.from_bytes(nb_cycles[0], 'little')) 
    print("Host elapsed time:", "{:.2e}".format(end - start), "secs.")
    
    dpus.log()
    
if __name__ == "__main__":
    pass
