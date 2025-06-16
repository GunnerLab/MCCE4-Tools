#!/usr/bin/env python
"""
Module: collect_ms_pdbs_hbonds.py

Description

Command-line tool to run MCCE_bin.detect_hbonds.detect_hbonds function over a collection
of pdbs in step2_out.pdb format with options for excluding backbone atoms and defining
an output directory.

Usage with all default arguments:
  > collect_ms-pdb_hbonds.py

Options default values:
  -input_dir       : ms_pdb_output
  -output_dir      : ms_pdb_output_hbonds
  --include_bk     : False
  --no_empty_files : False
"""
import argparse
from pathlib import Path
import shutil
import sys

from mcce4 import detect_hbonds


def collect_hbs(args: argparse.Namespace):
    pdbs_dir = Path(args.input_dir).resolve()
    if not pdbs_dir.is_dir():
        sys.exit(f"Error: The directory {pdbs_dir!s} does not exist.")

    # Delete & recreate the existing output directory or create it
    out_dir = Path(args.output_dir).resolve()
    if out_dir.exists():
        print(f"Output directory {out_dir!s} already exists. Deleting it and recreating.")
        shutil.rmtree(out_dir)  # remove dir & its files

    out_dir.mkdir()
    print(f"Created output directory: {out_dir!s}")

    pdb_files = list(pdbs_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"No pdb files found in {pdbs_dir!s}.")
        sys.exit()

    for pdb in pdb_files:
        print(f"Processing {pdb!s}...")
        result = detect_hbonds.detect_hbonds(
            pdb, args.include_bk, args.no_empty_files, out_dir
        )
        if result == (0, 0):
            print("WARNING! No hbonds found in", str(pdb))

    print("Processing over!")


def collect_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run mcce4.detect_hbonds.detect_hbonds function over a collection of pdbs in step2_out.pdb format."
    )
    p.add_argument(
        "-input_dir",
        type=str,
        default="ms_pdb_output",
        help="Directory containing pdb files. (default: %(default)s)",
    )
    p.add_argument(
        "-output_dir",
        type=str,
        default="ms_pdb_output_hbonds",
        help="Directory to save the output files. (default: %(default)s)",
    )
    p.add_argument(
        "--include_bk",
        action="store_true",
        default=False,
        help="Include backbone atoms? (default: %(default)s)",
    )
    p.add_argument(
        "--no_empty_files",
        action="store_true",
        default=False,
        help="Don't create an empty file when no H-bonds are found. (default: %(default)s)",
    )
    return p

def cli(argv=None):
    p = collect_parser()
    args = p.parse_args(argv)
    collect_hbs(args)

    return


if __name__ == "__main__":
    cli(sys.argv[:1])
