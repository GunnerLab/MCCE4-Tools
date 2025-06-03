#!/usr/bin/env python

"""

Module: clear_mcce_outputs.py

Purpose:
Delete all MCCE outputs from the current folder, except for run.prm*, the
original pdb and prot.pdb, as well as any non-MCCE files.

Usage: Run the tool in the folder of interest

"""

from pathlib import Path
import shutil
import sys


MCCE_OUTPUTS = [
    "acc.atm",
    "acc.res",
    "debug.log",
    "energies",
    "entropy.out",
    "err.log",
    "extra.tpl",
    "fort.38",
    "head1.lst",
    "head2.lst",
    "head3.lst",
    "mc_out",
    "ms_out",
    "name.txt",
    "new.tpl",
    "null",
    "param",
    "pK.out",
    "progress.log",
    "respair.lst",
    "rot_stat",
    "run.log",
    "run.prm.record",
    "step0_out.pdb",
    "step1_out.pdb",
    "step2_out.pdb",
    "sum_crg.out",
    "vdw0.lst",
]


def delete_mcce_outputs(mcce_dir: str):
    """Delete all MCCE output files or folders from a MCCE run folder.
    Kept: run.prm, original pdb and prot.pdb, as well as other non-MCCE files.
    """
    folder = Path(mcce_dir)

    for fname in MCCE_OUTPUTS:
        fp = folder.joinpath(fname)
        if fp.is_dir():
            shutil.rmtree(fp)
            print(f"  Deleted folder: {fname}")
        else:
            if fp.exists():
                fp.unlink()
                print(f"  Deleted file: {fname}")

    return


def cli():
    folder = Path.cwd()
    proceed = input(f"Delete all MCCE outputs from {folder.name}? (y/n) ")
    if proceed.lower() == "n":
        sys.exit()
    elif proceed.lower() == "y":
        delete_mcce_outputs(folder)
    else:
        sys.exit(f"Unrecognized answer: {proceed}.")

    return


if __name__ == "__main__":
    cli()
