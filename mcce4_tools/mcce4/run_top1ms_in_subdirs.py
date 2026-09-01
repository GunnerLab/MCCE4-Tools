#!/usr/bin/env python3
"""
Module: run_top1ms_in_subdirs.py

Use case:
Get the top 1 charge ms in all completed mcce runs in the multiple
simulation subfolders of the current folder without producing pdbs.

Details:
The program looks for the msout files 2 levels deep from
the current folder (*/*/ms_out/pH7*eH0*ms.txt).

The tool is preset to output the top1ms .tsv and summary files without
producing any  pdbs.

"""
from collections import defaultdict
from pathlib import Path

from mcce4 import topn_cms_to_pdbs as tcms


def main():
    # create arg_dict
    args = {
        "mcce_dir": ".",
        "ph": 7,
        "eh": 0,
        "n_top": 1,
        "pdb_format": False,
        "residue_kinds": "ASP,GLU,ARG,HIS,LYS,CYS,TYR,SEC,NTR,CTR",
        "min_occ": 0,
        "wet": False,
        "reduced_ms_rows": True,
        "no_pdbs": True,
    }

    current_dir = Path.cwd()

    # get list of msout files as Path objects:
    msoutfiles_list = list(current_dir.glob("*/*/ms_out/pH7*eH0*ms.txt"))
    if not msoutfiles_list:
        print("Could not list any msoutfile paths from the current dir subfolders.")
        return

    # extract subdir/protdir:
    prot_dirs = [fp.parent.parent for fp in msoutfiles_list]

    # get a 'done' list: look for /topms_ph7eh0_top1/top1_ms.tsv
    done_dir_names = defaultdict(list)
    for dp in prot_dirs:  # e.g.: Path("runs/4LZT"):
        if (dp.joinpath("topms_ph7eh0_top1/top1_ms.tsv").exists()
            or dp.joinpath("topms_ph7.00eh0.00_top1/top1_ms.tsv").exists()):
            done_dir_names[dp.parent.name].append(dp.name)

    for dp in prot_dirs:
        if dp.name in done_dir_names[dp.parent.name]:
            print(f"  Skipping {dp!s}: found 'top1_ms.tsv'.")
            continue

        # update arg dict:
        args.update({"mcce_dir": str(dp)})

        pipeline = tcms.TopNCmsPipeline(args)
        pipeline.run()

    print(f"\nCompleted 'top1ms' in {current_dir!s}.")

    return
