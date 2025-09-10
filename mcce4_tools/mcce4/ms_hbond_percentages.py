#!/usr/bin/env python3

"""
Created on Mar 13 09:00:00 2025

@author: Gehan Ranepura
"""
import argparse
import collections
import os
from pathlib import Path
import sys


def get_hah_file_uniq_bounds(hah_file) -> set:
    unique_bonds = set()
    with open(hah_file) as file:
        for line in file:
            parts = line.strip().split()
            # Ensure we have at least donor and acceptor
            if len(parts) >= 2:
                donor, acceptor = parts[:2]
                hbond = (donor, acceptor)
                unique_bonds.add(hbond)

    return unique_bonds


def get_all_uniq_hbonds(input_dir):
    # Dictionary to count occurrences of each hydrogen bond pair
    hbond_counts = collections.Counter()
    # Number of PDBs processed (i.e., number of .txt files)
    pdb_count = 0

    # Iterate through all .txt files in the directory (excluding those starting with "blocking_file")
    for entry in os.scandir(input_dir):
        if (entry.is_file()
            and entry.name.endswith("hah.txt")
            and not entry.name.endswith("blocking.txt")):
            pdb_count += 1
            unique_bonds = get_hah_file_uniq_bounds(entry.path)
            for bond in unique_bonds:
                hbond_counts[bond] += 1  # Count uniq bonds per PDB
    
    return pdb_count, hbond_counts


def write_pct_table(input_dir, pdb_count, hbond_counts):
    """
    Calculate percentages and write results to a file
    """
    output_file = f"{Path(input_dir).stem}_percentages.txt"

    with open(output_file, "w") as out:
        out.write(f"{'Donor':<20}{'Acceptor':<20}{'PDB Count':<15}{'Percentage of PDBs (%)'}\n")
        out.write("=" * 75 + "\n")
        for (donor, acceptor), count in sorted(hbond_counts.items(),
                                               key=lambda x: x[1],
                                               reverse=True):
            percentage = count / pdb_count
            out.write(f"{donor:<20}{acceptor:<20}{count:<15}{percentage:.3%}\n")

    print(f"Hydrogen bond percentages saved to {output_file}")

    return

def cli_parser():
    parser = argparse.ArgumentParser(
        prog="ms_hbond_percentages",
        description="Compute hydrogen bond percentages from text files."
    )
    parser.add_argument(
        "dir",
        nargs="?",
        type=str,
        default="ms_pdb_output_hbonds",
        help="Directory containing hbond connection files (default: pdb_output_mc_hbonds)"
    )

    return parser


def cli(argv=None):
    p = cli_parser()
    args = p.parse_args()

    # Validate directory existence
    if not Path(args.dir).is_dir():
        sys.exit(f"Error: Path {args.dir!s} not found or is not a directory.")

    pdb_count, hbond_counts = get_all_uniq_hbonds(args.dir)
    if pdb_count == 0:
        sys.exit("No valid PDB files found. Exiting.")

    print(f"Number of PDBs = {pdb_count}")
    write_pct_table(args.dir, pdb_count, hbond_counts)

    return


if __name__ == "__main__":
    cli(sys.argv[:1])

