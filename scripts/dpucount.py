#!/usr/bin/env python3

from dpu import DpuSet 
from dpu import ALLOCATE_ALL

    
if __name__ == "__main__":
  with DpuSet(ALLOCATE_ALL, profile="backend=hw") as dpus:
    print("{} DPUs allocated".format(len(dpus.dpus()))) 
