import os

import argparse

args = argparse.ArgumentParser()
args.add_argument("--dir", type=str, default="logdirmb1")

args = args.parse_args()

file_name = os.listdir(args.dir)
t = 1
infs = []
for fname in file_name:
    with open(os.path.join(args.dir, fname), "rt") as f:
        lines = list(f.readlines())
        if t > 0:
            title = [c.strip() for c in lines[-2].strip().split("|")] + ["bag length", "num train", "dim"]
            print(",".join(title))
            t -= 1
        data_list = [c.strip() for c in lines[-1].strip().split("|")] + fname.split("-")[-3:]
        print(",".join(data_list))
