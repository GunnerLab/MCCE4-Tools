#!/usr/bin/env python3

"""
Module: postrun.py

Provides basic diagnostics on sum_crg.out and pK.out data:
   - Non-canonically charged residues;
   - Residues without curve fit or high chi^2 (>= 3).
   - Residues out-of-bounds, < first point, or > last point.
Output:
  - Empty file run_dir/'postrun.ok' is created when no issues were found.
  - Sample contents in run_dir/'postrun.bad' file (when all categories have data):

    WARNING       :: Neutral ARG found in 'non canonical' list (@ point 7).
    PROTDIR :: non canonical :: ARG+A0005_, ASP-A0018_
    PROTDIR :: chi2 >= 3.0   :: LYS+A0033_, HIS+A0055_
    PROTDIR :: no curve      :: TYR-A0023_
    PROTDIR :: out_of_bounds: 1  :: [('TYR-A0053_', '>14.0')]
"""
import argparse
from collections import defaultdict
import logging
import operator
import os
from pathlib import Path
import sys
from typing import Tuple

import pandas as pd

from mcce4.constants import CANONICAL
import mcce4.io_utils as mciou


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# default high chi^2:
HI_CHI = 3.0


def get_titr_info(df: pd.DataFrame) -> tuple:
    """Return the titr type and df column name to use as titr point.
    """
    # determine which titr point (column) to use:
    cols = df.columns.to_list()
    titr_type = cols[0].lower()
    if titr_type == "eh":
        # treated like single point titration;
        # assumed: eh titration run at at pH7:
        return titr_type, cols[1]

    titr_col = ""
    # other cases:
    try:
        # idx will be 8 if titration starts at 0 and includes ph 7,
        # else, ph7 can be in a different column:
        idx = cols.index("7")
        titr_col = cols[idx]
    except ValueError:
        pass

    return titr_type, titr_col


def get_noncanonical(df: pd.DataFrame, titr_col: str = "") -> Tuple[list, bool]:
    """Return the list of res with non-canonical charge in df of sum_crg.out,
    along with a boolean flag indicating presence of non-canonical ARGs.
    """
    # compare res charges with those in the canonical dict;
    # add to list if not within 20% of their canonical values:
    is_arg = False
    out = []
    if titr_col:
        for idx in df.index:
            res = df.loc[idx, df.columns[0]]
            crg = float(df.loc[idx, titr_col])
            stdcrg = CANONICAL.get(res[:3])
            if stdcrg is None:
                continue

            if res[:3] in ["TYR", "CYS"]:
                # 0 crg
                if crg <= -0.2:
                    out.append(res)
            else:
                op1 = operator.le if res[3] == "+" else operator.ge
                if op1(round(crg, 1), stdcrg*0.8):
                    out.append(res)
                    if res[:3] == "ARG":
                        is_arg = True

    return out, is_arg


def get_bad_pks(pko: Path) -> Tuple[list, list, list]:
    """Get bad data in pK.out with path 'pko'.
    Return a 2-tuple of lists for high chi-squared values
    and residues with no curve fit, and those out-of-bounds.
    """
    chi = []
    curve = []
    oob = []
    with open(pko) as fin:
        for j, line in enumerate(fin.readlines()):
            if j == 0:
               continue
            if "too sharp" in line:
                curve.append(line[:10])
                continue
            cols = line.split()
            if len(cols) < 15:
                oob.append((cols[0], cols[1]))
            if float(cols[3]) >= HI_CHI:
                chi.append(cols[0])

    return chi, curve, oob


def get_postrun_report(run_dir: str, summary: dict = None) -> None:
    """Write a post step 4 run report into empty file 'postrun.ok'
    if the 'run_dir' folder was processed and found acceptable,
    or into file 'postrun.bad' which flags problematic conditions.

    If summary (dict) is not None, it is updated with the count and 
    list of residues in each categories for each protein (key).

    If any of the input files is missing, the 'postrun.bad' file
    contains a single line with the missing name(s), otherwise it
    has a line for any of these cases:
     - A warning if any ARG found with non-canonical charge;
     - A list of residues with non_canonical charges, with count;
     - A list of residues with chi squared values > HI_CHI=3 threshold, with count;
     - A list of residues with no curve fit, with count;
     - A line stating a 'new.tpl' file was found.
     """
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        logger.critical("Not found: " + str(run_dir))
        sys.exit(1)

    # check needed files:
    missing = []
    sumcrg = run_dir.joinpath("sum_crg.out")
    if not sumcrg.exists():
        logger.critical("Not found: " + str(sumcrg))
        missing.append(sumcrg.name)

    pko = run_dir.joinpath("pK.out")
    if not pko.exists():
        logger.critical("Not found: " + str(pko))
        missing.append(pko.name)

    dname = run_dir.name
    # add dirname to each line to retain info if reports are collated:
    dirname = f"{dname} :: "

    if missing:  # report
        if summary is not None:
            summary["missing_files"].append(dname)

        out_fp = run_dir.joinpath("postrun.bad")
        out_fp.write_text(dirname + f"{'missing':18s}:: {missing}\n")
        return

    df = mciou.mcfile2df(sumcrg)
    titr_type, titr_col = get_titr_info(df)

    nc_out = []
    if titr_col:
        # Get non-canonically charged residues:
        nc_out, is_arg = get_noncanonical(df, titr_col=titr_col)

    # Get hi-chi,  no-curve, or out-of-bounds pkas from pK.out:
    chi, curve, oob = get_bad_pks(pko)

    # Prep output:
    if not (nc_out or chi or curve or oob):
        out_fp = run_dir.joinpath("postrun.ok")
        out_fp.touch(exist_ok=True)
        if summary is not None:
            summary["postrun_ok"].append(dname)
        return

    if summary is not None:
        summary["postrun_bad"].append(dname)

    outtxt = ""
    if nc_out:
        if titr_type == "eh":
            outtxt += f"Non-canonical residues @ Eh point {float(titr_col):.1f}:\n"
        else:
            outtxt += f"Non-canonical residues @ pH point 7:\n"
        if is_arg:
            outtxt += dirname + f"{'WARNING':18s}:: Neutral ARG found in 'non_canonical' list.\n"
            if summary is not None:
                summary["non_canonical_arg"].append(dname)
        
        cat_count = f"non_canonical: {len(nc_out)}"
        outtxt += dirname + f"{cat_count:18s}:: " + ", ".join(nc for nc in nc_out) + "\n"
        if summary is not None:
            summary["non_canonical"].append((dname, len(nc_out)))
        
    if chi:
        cat_count = f"chi2 >= {HI_CHI}: {len(chi)}"
        outtxt += dirname + f"{cat_count:18s}:: " + ", ".join(hc for hc in chi) + "\n"
        if summary is not None:
            summary["hi_chi"].append((dname, len(chi)))

    if curve:
        cat_count = f"no_curve: {len(curve)}"
        outtxt += dirname + f"{cat_count:18s}:: " + ", ".join(c for c in curve) + "\n"
        if summary is not None:
            summary["no_curve"].append((dname, len(curve)))

    if oob:
        cat_count = f"out_of_bounds: {len(oob)}"
        outtxt += dirname + f"{cat_count:18s}:: {oob}\n"
        if summary is not None:
            summary["out_of_bounds"].append((dname, len(oob)))
    
    # finally:
    if run_dir.joinpath("new.tpl").exists():
        outtxt += dirname + f"{'new_tpl':18s}:: True\n"
        if summary is not None:
            summary["new_tpl"].append(dname)
    outtxt += "\n"

    out_fp = run_dir.joinpath("postrun.bad")
    out_fp.write_text(outtxt)

    return


def get_bench_postrun_reports(run_dir) -> dict:
    """Iterate over run_dir/runs subfolder to create a postrun report,
    then collate all the bad ones into a single file.
    Return a summarizing dict.
    """
    bench_dir = Path(run_dir)
    summary = defaultdict(list)
    for d in bench_dir.joinpath("runs").iterdir():
        if not d.is_dir():
            continue
        get_postrun_report(d, summary=summary)

    # collate with cat command:
    mciou.subprocess_run(f"cat runs/*/postrun.bad > all_postrun.bad")

    return summary


def write_summary(summary: dict, smry_fp: Path):
    txt = f"Post run summary in {str(smry_fp.parent.resolve())}:\n"

    if summary.get("postrun_bad") is not None and summary["postrun_bad"]:
        txt += f"postrun_bad: {len(summary['postrun_bad']): >3d}\n"
    if summary.get("postrun_ok") is not None and summary["postrun_ok"]:
        txt += f"postrun_ok : {len(summary['postrun_ok']): >3d}\n"
    if summary.get("missing_files") is not None and summary["missing_files"]:
        txt += "missing pK or sum_crg:\n" + "\n".join(d for d in summary["missing_files"])
    txt += "....................\n\n"
    if summary.get("non_canonical") is not None and summary["non_canonical"]:
        txt += f"non_canonical    : {len(summary['non_canonical']): >3d}\n"
    if summary.get("non_canonical_arg") is not None and summary["non_canonical_arg"]:
        txt += f"non_canonical_arg: {len(summary['non_canonical_arg']): >3d}\n"
    if summary.get("non_canonical") is not None and summary["non_canonical"]:
        sorted_nc = sorted(summary["non_canonical"], key=lambda x:x[1], reverse=True)
        df = pd.DataFrame(sorted_nc, columns=["non_canonical", "count"])
        df.set_index("non_canonical", inplace=True)
        txt += "\n" + df.to_string() + "\n"
        txt += "....................\n\n"
    if summary.get("hi_chi") is not None and summary["hi_chi"]:
        txt += f"hi_chi           : {len(summary['hi_chi']): >3d}\n"
        sorted_hichi = sorted(summary["hi_chi"], key=lambda x:x[1], reverse=True)
        df = pd.DataFrame(sorted_hichi, columns=["hi_chi", "count"])
        df.set_index("hi_chi", inplace=True)
        txt += "\n" + df.to_string() + "\n"
        txt += "....................\n\n"
    if summary.get("new_tpl") is not None and summary["new_tpl"]:
        txt += f"new_tpl          : {len(summary['new_tpl']): >3d}\n"
        txt += "\n" + "\n".join(nt for nt in summary['new_tpl']) + "\n"
        txt += "....................\n\n"

    txt += "Details for residues in 'all_postrun.bad` file.\n"
    smry_fp.write_text(txt)

    return


def pr_cli(argv = None):

    p = argparse.ArgumentParser(
        prog="postrun",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Report issues here:
        https://github.com/GunnerLab/MCCE4-Tools/issues""",
    )
    p.add_argument(
        "-run_dir",
        type=str,
        default=".",
        help="Path to a mcce run directory; default: %(default)s.",
    )
    p.add_argument(
        "--is_benchmark",
        default=False,
        action="store_true",
        help="The given run_dir is a benchmark dir; default: %(default)s.",
    )

    args = p.parse_args(argv)

    here = args.run_dir == "."
    smry_fp = None
    if args.is_benchmark:
        if not here:
            rundir = Path(args.run_dir).resolve()
            os.chdir(rundir)
        else:
            rundir = Path.cwd()
        # iterate over run_dir/runs subfolders & collate
        summary = get_bench_postrun_reports(rundir)

        if here:
            if summary:
                smry_fp = Path("postrun_summary")
            rpt_fp = Path("all_postrun.bad")
        else:
            if summary:
                smry_fp = rundir.joinpath("postrun_summary")
            rpt_fp = rundir.joinpath("all_postrun.bad")
        
        if smry_fp is not None:
            write_summary(dict(summary), smry_fp)
    else:
        get_postrun_report(args.run_dir)
        if here:
            rpt_fp = Path("postrun.bad")
        else:
            rpt_fp = Path(args.run_dir).joinpath("postrun.bad")

    if rpt_fp.exists():
        logger.info(f"Postrun over: bad runs reported in {str(rpt_fp)}:\n")
        print(rpt_fp.read_text())
    else:
        logger.info("Postrun over: no bad runs detected.")

    return
