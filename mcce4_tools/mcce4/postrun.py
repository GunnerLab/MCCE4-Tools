#!/usr/bin/env python3

"""
Module: postrun.py

Provides basic diagnostics on sum_crg.out and pK.out data
in a tab-separated file 'postrun.bad':
   - Non-canonically charged residues;
   - Residues without curve fit or high chi^2 (>= 3).
   - Residues out-of-bounds, < first point, or > last point.
Output:
  - Empty file run_dir/'postrun.ok' is created when no issues were found.
  - Sample contents in run_dir/'postrun.bad' file:
    run     category        count   value
    1OTS    non_canonical   8       NTR+A0017_, CYS-A0085_, GLU-A0203_, GLU-A0414_, NTR+B0018_, GLU-B0203_, LYS+B0216_, GLU-B0414_
    1OTS    out_of_bounds   152     Likely due to short (1 pts) titration; not listed
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

from mcce4 import CLI_EPILOG
from mcce4.constants import CANONICAL
import mcce4.io_utils as mciou


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# default high chi^2:
HI_CHI = 3.0


def get_titr_info(df: pd.DataFrame) -> tuple:
    """Return a 3-tuple:
       the titr type, column name to use as titr point, number of titr cols.
    """
    # determine which titr point (column) to use:
    cols = df.columns.to_list()
    titr_type = cols[0].lower()
    if titr_type == "eh":
        # treated like single point titration;
        # assumed: eh titration run at at pH7:
        return titr_type, cols[1], len(cols) - 1

    titr_col = ""
    # other cases:
    try:
        # idx will be 8 if titration starts at 0 and includes ph 7,
        # else, ph7 can be in a different column:
        idx = cols.index("7")
        titr_col = cols[idx]
    except ValueError:
        pass

    return titr_type, titr_col, len(cols) - 1


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
            # pK.out is a subset of pK_extended.out wiith only 3 cols
            if len(cols) < 3:
                oob.append((cols[0], cols[1]))
            try:
                if float(cols[3]) >= HI_CHI:
                    chi.append(cols[0])
            except IndexError:
                pass

    return chi, curve, oob


def get_postrun_report(run_dir: str, summary: dict = None) -> list:
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

    postrun_lst = []
    dname = run_dir.name
    print(f"Processing {dname}...")

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

    if missing:  # report
        postrun_lst.append([dname, "missing", len(missing), ", ".join(ms for ms in missing)])
        if summary is not None:
            summary["missing_files"].append(dname)
    else:
        df = mciou.mcfile2df(sumcrg)
        titr_type, titr_col, titr_points = get_titr_info(df)
        too_many_msg = f"Likely due to short ({titr_points} pts) titration; not listed"

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
            return []

        if summary is not None:
            summary["postrun_bad"].append(dname)

        if nc_out:
            n_nc = len(nc_out)
            no_list = n_nc > 20 and titr_points < 7
            if no_list:
                value = too_many_msg
                no_list = 0
            else:
                value =  ", ".join(nc for nc in nc_out)
            postrun_lst.append([dname, "non_canonical", n_nc, value])
            if summary is not None:
                summary["non_canonical"].append((dname, len(nc_out)))
        
            if is_arg:
                postrun_lst.append([dname, "non_canonical_arg", 1,
                                    "One or more ARG in non_canonical list"])
                if summary is not None:
                    summary["non_canonical_arg"].append(dname)
        
        if chi:
            postrun_lst.append([dname, "hi_chi", len(chi), ", ".join(hc for hc in chi)])
            if summary is not None:
                summary["hi_chi"].append((dname, len(chi)))

        if curve:
            postrun_lst.append([dname, "no_curve", len(curve), ", ".join(c for c in curve)])
            if summary is not None:
                summary["no_curve"].append((dname, len(curve)))

        if oob:
            n_oob = len(oob)
            no_list = n_oob > 20 and titr_points < 7
            cat_count = f"{dname}\tout_of_bounds\t{n_oob}\t"
            if no_list:
                value = too_many_msg
                no_list = 0
            else:
                value = ", ".join(f"({tp[0]},{tp[1]})" for tp in oob)
            postrun_lst.append([dname, "out_of_bounds", n_oob, value])
            if summary is not None:
                summary["out_of_bounds"].append((dname, n_oob))
    
        # finally:
        if run_dir.joinpath("new.tpl").exists():
            postrun_lst.append([dname, "new_tpl", 1, "new.tpl file found"])
            if summary is not None:
                summary["new_tpl"].append(dname)

    out_fp = run_dir.joinpath("postrun.bad")
    df = pd.DataFrame(postrun_lst, columns=["run", "category", "count", "value"])
    df.to_csv(out_fp, sep="\t", index=False)

    return postrun_lst


def get_postrun_reports(run_dir) -> dict:
    """Iterate over run_dir[/runs] subfolders that hold a step2_out.pdb
    file to create a postrun report, then collate all the bad ones into a single file.
    Note: step2_out.pdb is used as a sentinel to identify a mcce run folder.
    Return a summarizing dict.
    """
    summary = defaultdict(list)
    if Path(run_dir).joinpath("runs").is_dir():
        bench_dir = Path(run_dir).joinpath("runs")
    else:
        bench_dir = Path(run_dir)

    pr_lst = []
    for d in bench_dir.iterdir():
        if not d.is_dir():
            continue
        if d.joinpath("step2_out.pdb").exists():
            pr_data = get_postrun_report(d, summary=summary)
            if pr_data:
                pr_lst.extend(pr_data)

    out_fp = bench_dir.joinpath("all_postrun.bad")
    df = pd.DataFrame(pr_lst, columns=["run", "category", "count", "value"])
    df.to_csv(out_fp, sep="\t", index=False)

    return summary


def write_summary(summary: dict, smry_fp: Path):
    txt = f"Post run summary in {str(smry_fp.parent.resolve())}:\n"

    if summary.get("postrun_bad") is not None and summary["postrun_bad"]:
        txt += f"postrun_bad: {len(summary['postrun_bad']): >3d}\n"
    if summary.get("postrun_ok") is not None and summary["postrun_ok"]:
        txt += f"postrun_ok : {len(summary['postrun_ok']): >3d}\n"
    if summary.get("missing_files") is not None and summary["missing_files"]:
        txt += "missing pK or sum_crg:\n" + "\n".join(d for d in summary["missing_files"])
    txt += "\n....................\n\n"
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
        epilog=CLI_EPILOG
    )
    p.add_argument(
        "-run-dir",
        type=str,
        default=".",
        help="Path to a mcce run directory; default: %(default)s",
    )
    p.add_argument(
        "--do-subfolders",
        default=False,
        action="store_true",
        help="If given, the program will check subfolders for mcce runs; default: %(default)s",
    )

    args = p.parse_args(argv)

    here = args.run_dir == "."
    smry_fp = None
    if args.do_subfolders:
        if not here:
            rundir = Path(args.run_dir).resolve()
            os.chdir(rundir)
        else:
            rundir = Path.cwd()
        # iterate over run_dir subfolders & collate
        summary = get_postrun_reports(rundir)

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
