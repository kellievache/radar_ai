#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:19:58 2026

@author: vachek
"""


#!/usr/bin/env python3
import argparse, sys, os

# If stack_to_netcdf lives in a file inside your project, make sure it's importable:
# sys.path.insert(0, "/nfs/pancake/u5/projects/vachek/automate_qc")
# from your_module import stack_to_netcdf

# If it's in the same dir as this CLI, use relative:
# from stack_to_netcdf import stack_to_netcdf

# ----- Replace the import below with the correct location of the function -----
from CreateNetCDF_FromPRISM_BIL_9999_parallel import stack_to_netcdf  # <- make sure this resolves in your env

def parse_years(text: str):
    """
    Accept '2023' or '2015,2019' or '2015:2019' (inclusive range).
    """
    text = text.strip()
    if ":" in text:
        a, b = text.split(":")
        a, b = int(a), int(b)
        return tuple(range(a, b + 1))
    if "," in text:
        return tuple(int(x) for x in text.split(",") if x.strip())
    return (int(text),)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--years", required=True,
                    help="Single year (2023), comma list (2015,2019) or inclusive range (2015:2019)")
    ap.add_argument("--var_name", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--include_text", required=True)
    args = ap.parse_args()

    years = parse_years(args.years)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)

    stack_to_netcdf(
        args.root,
        years=years,
        var_name=args.var_name,
        out_path=args.out_path,
        include_text=args.include_text
    )

if __name__ == "__main__":
    main()
