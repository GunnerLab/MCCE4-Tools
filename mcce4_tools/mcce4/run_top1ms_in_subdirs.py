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
producing any pdbs.

Example of expected folder structure:
......................................

Top folder: mcce_sims_param_e

cd mcce_sims_param_e; tree -L 2 --dirsfirst
.
├── kinases_e8
│   └── <prot dirs>
├── kinases_e20
│   └── <prot dirs>
├── lysozymes_e8
│   └── <prot dirs>
├── lysozymes_e20
│   └── <prot dirs>
└── experiment_info.txt
"""
from pathlib import Path
# from shutil import which
# from subprocess import Popen

from mcce4 import topn_cms_to_pdbs as tcms

## TODO: send ms_top2pdbs + arg as a subprocess => using args_min
# # Note: if calling the 'topcms' pipeline, all args are required!
# # vs calling ms_top2pdbs w/ Popen, args dict setup would be args_min,
# # => only non-default options:
# args_min = {
#     "mcce_dir": ".",  # needed: updated in loop
#     "n_top": 1,
#     "reduced_ms_rows": True,
#     "no_pdbs": True,
# }
## e.g.: mcce_benchmark.batch_submit.batch_run:
##  Popen(
##     f"../{job_script}",
##     cwd=f"./{entry.name}",
##     close_fds=True,
##     stdout=open(f"./{entry.name}/run.log", "a"),
##     stderr=open(f"./{entry.name}/err.log", "a"),
## )
##  the cmd full path would be:
## mstop1_tool = which("ms_top1ms_subdirs")
##    # print("mstop1_tool = ", mstop1_tool)

args_full = {
    "mcce_dir": ".",
    "ph": 7,
    "eh": 0,
    "n_top": 1,
    "residue_kinds": "ASP,GLU,ARG,HIS,LYS,CYS,TYR,SEC,NTR,CTR",
    "min_occ": 0,
    "wet": False,
    "reduced_ms_rows": True,
    "no_pdbs": True,
    "pdb_format": False,
}


topms1 = "topms_ph7.00eH0.00_top1/top1_ms.tsv".lower()
msout1 = "ms_out/pH7.00eH0.00ms.txt"
topms2 = "topms_ph7eH0_top1/top1_ms.tsv".lower()
msout2 = "ms_out/pH7eH0ms.txt"


def main(args: dict = args_full):
    current_dir = Path.cwd()

    for dp in current_dir.glob("*/*/ms_out"):  # subdirs/pdb_dirs/ms_out
        print(f"CHECKING {dp.parent.name}...")
        # check for final output:
        tsvs = [dp.parent.joinpath(topms1), dp.parent.joinpath(topms2)]
        fps = [tsv for tsv in tsvs if tsv.exists()]
        if fps:
            print(f"  SKIPPING {dp.parent.name}: found 'top1_ms.tsv' in {fps[0].parent.name}")
            continue

        # check for the expected msout file:
        if (dp.parent.joinpath(msout1).exists()
            or dp.parent.joinpath(msout2).exists()
            ):
            # update arg dict for current prot:
            args.update({"mcce_dir": str(dp.parent)})
            print(f"  PROCESSING {dp.parent!s}")
            pipeline = tcms.TopNCmsPipeline(args)
            pipeline.run()
        else:
            print("  No msout file with needed ph/eh found.")

    print(f"\nCOMPLETED 'top1ms' in {current_dir!s}\n")

    return
