#!/usr/bin/env python3
import os
import sys
import json
import csv
import math

def main(input_dir):
  json_files = [json_filepath for json_filepath in os.listdir(input_dir) if json_filepath.endswith('.json')]
  for json_file in json_files:
    json_file = os.path.join(input_dir, json_file) 
    with open(json_file, "r") as f:
      try:
        data = json.load(f)    
      except Exception as e:
        print("failed to load {}: {}".format(json_file, e))
        continue
    with open(json_file.replace(".json", ".csv"), 'w', newline='') as csvfile:
        fieldnames = set()
        for benchmark in data['benchmarks']:
          if benchmark.get("error_occurred", False):
            continue
          fieldnames = fieldnames.union(benchmark.keys())
          fieldnames.add("operator")
          name_parts = benchmark["name"].split("/")
          fieldnames = fieldnames.union(set([part.split(":")[0] for part in name_parts if ":" in part]))
          
        writer  = csv.DictWriter(csvfile, fieldnames=list(fieldnames))
        writer.writeheader()
        for benchmark in data['benchmarks']:
          if benchmark.get("error_occurred", False):
            continue
          name_parts = benchmark["name"].split("/")
          benchmark["operator"] = name_parts[1]
          benchmark = {**benchmark, **{part.split(":")[0]: part.split(":")[1] for part in name_parts if ":" in part}} 
          for k, v in benchmark.items():
            if type(v) is float and math.isnan(v):
              benchmark[k] = 0.0
          writer.writerow(benchmark)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: ./parse_results.py <Input Dir>")
    sys.exit(1)
  main(sys.argv[1])
