#!/usr/bin/env python3

import argparse
import re
import sys

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H-bonding Pairs Sanity Check")
    parser.add_argument("file", type=str, help="Path to H bond count file")
    parser.add_argument("-p", type=float, default=7.0, help="Microstate pH value (default: 7.0)")
    args = parser.parse_args()

    print("H-bonding Pairs Sanity Check...")

    # Read in the H bond count file
    if args.file.endswith(".txt"):
        with open(args.file) as f:
            first_line = f.readline()
        match = re.search(r"Total microstate count:\s*(\d+)", first_line)
        if not match:
            print(f"Error: Could not parse total microstate count from first line of file '{args.file}'.")
            sys.exit(1)
        total_microstates = int(match.group(1))
        df = pd.read_csv(args.file, sep=r"\s+", comment="#")  # ignore comment lines
        # Get donor and acceptor pairs and their counts
        donor_acceptor_pairs = df[["Donor_ConfID", "Acceptor_ConfID"]].values
        counts = df["Count"].values
        fractions = counts / total_microstates
    elif args.file.endswith(".csv"):
        df = pd.read_csv(args.file, comment="#")
        # Get donor and acceptor pairs and their occ
        donor_acceptor_pairs = df[["donor", "acceptor"]].values
        for c in df.columns:
            if c.startswith("occ") or c.endswith("ms_occ"):
                occ_col = c
                break
        fractions = df[occ_col].values
    else:
        print("Unsupported file format. Please provide a .txt or .csv file.")
        sys.exit(1)

    passed = True
    # Check against hah.txt to see if any pair is not in hah.txt
    hbond_pairs = set()
    fname_hah = "step2_out_hah.txt"
    with open(fname_hah) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                donor = parts[0].strip()
                acceptor = parts[1].strip()
                hbond_pairs.add((donor, acceptor))
    for d, a in donor_acceptor_pairs:
        if (str(d), str(a)) not in hbond_pairs:
            print(f"Warning: Donor-Acceptor pair ({d}, {a}) not found in {fname_hah}")
            passed = False

    # Check against fort.38 to see if the donor - acceptor occupancy is within the bounds
    fname_fort38 = "fort.38"
    df_fort38 = pd.read_csv(fname_fort38, sep=r"\s+")
    fort38_dict = {}
    for _, row in df_fort38.iterrows():
        confid = row[df_fort38.columns[0]]
        frac = row[f"{float(args.p):.1f}"]
        fort38_dict[confid] = frac
    for (d, a), frac in zip(donor_acceptor_pairs, fractions):
        if (str(d), str(a)) not in hbond_pairs:
            continue
        d_frac = fort38_dict.get(d, None)
        if d_frac is None and d[3:5] != "BK":
            print(f"Warning: Donor {d} not found in fort.38")
            passed = False
            continue
        a_frac = fort38_dict.get(a, None)
        if a_frac is None and a[3:5] != "BK":
            print(f"Warning: Acceptor {a} not found in fort.38")
            passed = False
            continue
        if d_frac is not None and a_frac is not None:
            frac = round(frac, 3)
            occ_max = min(d_frac, a_frac)
            occ_min = max(0.0, d_frac + a_frac - 1.0)
            if frac > round(occ_max + 1e-3, 3):
                print(f"Warning: {occ_col} {frac:.3f} for pair ({d}, {a}) exceeds max bound {occ_max:.3f} obtained from {d_frac:.3f} (donor) and {a_frac:.3f} (acceptor)")
                passed = False
            if frac < round(occ_min - 1e-3, 3):
                print(f"Warning: {occ_col} {frac:.3f} for pair ({d}, {a}) below min bound {occ_min:.3f} obtained from {d_frac:.3f} (donor) and {a_frac:.3f} (acceptor)")
                passed = False
    
    if passed:
        print("Sanity check passed: All donor-acceptor pairs are valid and occupancies are within bounds.")
    else:
        print("Sanity check failed: Issues found with donor-acceptor pairs or occupancies. Either the hbonding network or fort.38 file may be incorrect.")
