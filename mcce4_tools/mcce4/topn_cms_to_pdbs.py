#!/usr/bin/env python

"""
Module: topn_cms_to_pdbs.py

Codebase for tool `ms_top2pdbs`.
Processing functions to obtain tautomeric, topN protonation microstates data.

Description:
  ms_top2pdbs :: TopN protonation microstates to pdb & pqr files.

  This tool outputs:
  - The listing of the topN tautomeric protonation microstates, along with
    their related properties: mean energy (E), net charge (sum_crg), count,
    and occupancy (occ);
  - A summary file identifying ionizable residues with non canonical charge,
    and which of them do not change their charges over the topN set.
  - The <topN> files of each charge state in mcce-pdb & pqr and pdb formats.

Usage examples:
* If called inside a mcce output folder (not a requirement) & using the defaults;
  mcce_dir=., ph=7, eh=0, n_top=5, min_occ=0 and residue_kinds=ionizable residues:
  > ms_top2pdbs

* Otherwise:
  > ms_top2pdbs path/to/mcce_dir
  > ms_top2pdbs path/to/mcce_dir -eh 30
  > ms_top2pdbs path/to/mcce_dir -min_occ 0.002
  > ms_top2pdbs path/to/mcce_dir -ph 4 n_top 10

  # Residues: comma-separated; order & case insensitive:
  > ms_top2pdbs -residue_kinds _CL,his,GLU

INPUT FILES: head3.lst, step2_out.pdb, and the 'msout file' in the ms_out sub-directory,
             e.g. ms_out/pH7eH0ms.txt at ph 7.

OUTPUT FOLDER: mcce_dir/topms_ph7eh0_top5 (7, 0 and 5 are the default pH, Eh and number
               of top states requested).

OUTPUT FILES:
 - topN_ms.tsv: Charge of all acid/base residues in topN tautomeric protonation microstates + neural
                His tautomer. The last rows contain the mean energy (E) of this microstate, total
                charge (sum_crg), count (size), and probability (occ) of this microstate in ensemble.
 - summary.txt: List non-charged Asp, Glu, Arg, Lys, charged His, Tyr, Cys and neutral His tautomers
                in topN microstates.

 At most top_n coordinate files in these formats:
  - topmsN.pdb output each microstate in pdb format; N=1, 2, ..., top_n
  - s2_topmsN.pdb output each microstate in MCCE step2_out.pdb format
  - s2_topmsN.pqr output in pqr format (position, charge, radius)
"""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import logging
from pathlib import Path
from shutil import rmtree as rmdir
from string import ascii_uppercase as uppers
import sys
from typing import TextIO, Union
import time

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"CRITICAL: Oops! Forgot to activate an appropriate environment?\n{e}")
    sys.exit(1)

from mcce4.io_utils import get_mcce_filepaths, parse_mcce_line, show_elapsed_time
from mcce4.msout_np import MSout_np
from mcce4.constants import CANONICAL, IONIZABLE_RES, ALL_RES


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s: %(message)s")


# entry pint name in MCCE_bin:
APP_NAME = "ms_top2pdbs"
logger = logging.getLogger(APP_NAME)


OUT_DIR = "topms"  # output dir prefix
S2PREF = "s2_"  # mcce pdb & pqr files prefix
MIN_OCC = 0.0  # occupancy threshold
N_TOP = 5  # default value for 'n_top' option
TER = ["NTR", "CTR"]
HIS0_tautomers = {0: "NE2", 1: "ND1", 2: 1}
neutral_tautomers = ["NE2", "ND1", "OD1", "OD2", "OE2", "OE1", "HO", "HXT"]


def get_input_pdb_name(mcce_dir: Path) -> str:
    parent = "prot.pdb"
    # try via prot.pdb as symlink:
    prot_fp = mcce_dir.joinpath("prot.pdb")
    if prot_fp.exists():
        if prot_fp.is_symlink():
            parent = prot_fp.readlink().name

    return parent


class Mcce2PDBConverter:
    """A class for converting MCCE PDB files to PDB file."""

    def __init__(self, mccepdb_path: Path, s2rename_dict: dict, output_dir: Path):
        """
        Attributes:
          - mccepdb_path (Path): The path to the MCCE PDB file.
          - s2rename_dict (dict): Dict for renaming step2_out.pdb TER residues;
          - output_dir (Path): Dir path for the converted file.
        """
        self.mccepdb_path = mccepdb_path
        self.rename_dict = s2rename_dict
        # values of chain_dict is the step2 line number (starting at 1)
        self.output_dir = output_dir

    def get_pdb_line(self, mcce_line: str) -> str:
        """Return the standard pdb coordinates line."""
        # [rec, seq, atm, alt, res, conf, x, y, z, rad, crg, hist]
        rec, seq, atm, alt, res, conf, x, y, z, *_ = parse_mcce_line(mcce_line)
        if not alt:
            alt = " "
        res_index = conf.split("_")[0]

        if rec == "ATOM":
            if res == "NTG":
                res = "GLY"
            else:
                # {'A0001': ['NTR', 'LYS'], 'A0129': ['LEU', 'CTR']}
                if self.rename_dict.get(res_index) is not None:
                    if res in TER:  # list has 2 items
                        ix = int(res == "NTR")
                        res = self.rename_dict[res_index][ix]

        res_index = f"{res_index[0]}{res_index[1:].lstrip('0'):>4}"
        align = "^" if len(atm) < 3 else ">"
        # occ=1, b-factor=0; last item atm[0] is element id
        pline = (
            f"{rec:<6s}{seq:>5s} {atm:{align}4s}{alt}{res:<3s} {res_index:>4s}{' ':4s}"
            f"{float(x):>8.3f}{float(y):>8.3f}{float(z):>8.3f}  1.00  0.00{' ':11s}{atm[0]:<2s}\n"
        )

        return pline

    def get_pdb_remark(self) -> Union[str, None]:
        """Read the remark file associated with the pdb if any."""
        remark_section = None
        remark_fp = self.mccepdb_path.with_suffix(".remark")
        if remark_fp.exists():
            remark_section = remark_fp.read_text()
            remark_fp.unlink()

        return remark_section

    def mcce_to_pdb(self, out_name: str):
        """
        Converts an MCCE PDB file to PDB file.
        Args:
            out_name (str): Converted PDB filename.
        """
        # lines from pdb w/step2 format;
        # assumed: only coordinates lines:
        with open(self.mccepdb_path) as mcfp:
            mcce_lines = mcfp.readlines()

        # read the remark file associated with the pdb if any:
        remark_section = self.get_pdb_remark()
        prev_chain = ""

        # output pdb file with remark section:
        out_fp = self.output_dir.joinpath(out_name)
        with open(out_fp, "w") as ofp:
            if remark_section is not None:
                ofp.writelines(remark_section)

            for i, mcce_line in enumerate(mcce_lines):
                if i == 0:
                    prev_chain = mcce_line[21]

                # exclude alt loc if it exists and not A:
                if mcce_line[16] not in [" ", "A"]:
                    continue
                new_line = self.get_pdb_line(mcce_line)
                ofp.write(new_line)
                if mcce_line[21] != prev_chain:
                    ofp.write("TER\n")
                    prev_chain = mcce_line[21]

            ofp.write("TER\n")

        return


def get_non_canonical_dict(df: pd.DataFrame) -> dict:
    """Return a dict with the non-canonical residues info, if any."""
    non_canonical = defaultdict(dict)
    vec_cols = df.columns[1 : df.columns.tolist().index("info")]
    for c in df[vec_cols]:
        res_lst = []
        cof_lst = []

        for i, resid in enumerate(df.residues):
            res = resid[:3]
            # get the crg to compare:
            charge = df.loc[i, c]
            try:
                crg = int(charge)
            except ValueError:
                # tautomer string
                if charge in neutral_tautomers:
                    crg = 0
                else:
                    crg = charge

            # CANONICAL :: dict(RES, int crg)
            defres = CANONICAL.get(res)
            if defres is None:
                # MCCE does not assume HIS to have a canonical charge, always output
                if res == "HIS":
                    res_lst.append((resid, crg))
                continue

            if crg != defres:
                if res in ALL_RES:
                    res_lst.append((resid, crg))
                else:
                    cof_lst.append((resid, crg))
        if cof_lst:
            non_canonical[c] = {"residues": res_lst, "cofactors": cof_lst}
        else:
            non_canonical[c] = {"residues": res_lst}

    return dict(non_canonical)


def non_canonical_text(nc_dict: dict) -> str:
    """Return the formated data from get_non_canonical_dict as string."""
    if nc_dict:
        all_res = ""
        # for wrapping lines:
        k_max = len(max(nc_dict))
        ends = ["; ", ";\n " + " " * k_max]
        n_cols = 9

        for k in nc_dict:
            if nc_dict[k].get("residues") is not None:
                res = ""
                for i, v in enumerate(nc_dict[k]["residues"], start=1):
                    end = ends[int(i % n_cols == 0)]
                    res = res + f"{v[0]}:{v[1]}{end}"
                all_res += f"{k: >{k_max}} {res}\n"
            else:
                all_res = "None"
    else:
        all_res = "None"

    return all_res


def write_summary_file(tsv_fp: Path, n_top: int, res_kinds: list = None):
    sumry = """
# Summary for top {} charge microstates:
# --------------------------------------
 * Canonical states in MCCE:  ASP, GLU, CTR: -1; ARG, LYS, NTR: 1; TYR, CYS: 0
 * Note: HIS has no canonical state in MCCE, hence is always included.
# --------------------------------------

# Residues with non-canonical charge:
{}

# Residues with differing charge:
{}

# Cofactors charge:
{}

"""
    changes = "None"
    df = pd.read_csv(tsv_fp, sep="\t")

    delta = df[df.crg_changes == True][df.columns[:-1]].set_index("residues")
    if delta.shape[0]:
        changes = delta.to_string(float_format=lambda x: "{:.0f}".format(x))
    nc_dict = get_non_canonical_dict(df)
    all_res = non_canonical_text(nc_dict)

    user_kinds = [rk for rk in res_kinds if rk not in IONIZABLE_RES]
    msk = df.residues.str[:3].isin(user_kinds)
    non_res_df = df[msk]
    if len(non_res_df):
        non_res = non_res_df.to_string(index=False)
    else:
        non_res = "None"

    sumry_fp = tsv_fp.parent.joinpath("summary.txt")
    sumry_fp.write_text(sumry.format(n_top, all_res, changes, non_res))

    return


def get_chain(df: pd.DataFrame, n_exclude_last_rows: int = 4) -> pd.Series:
    out = df.residues.str[3]
    out[-n_exclude_last_rows:] = ""

    return out


def get_changing_crg(df: pd.DataFrame, n_exclude_last_rows: int = 4) -> list:
    """List the residues charge changes in the crg columns of df."""
    changes = [len(set(vals)) > 1 for vals in df[df.columns[1:-2]].values]
    changes[-n_exclude_last_rows:] = [""] * n_exclude_last_rows

    return changes


def finalize_topN_df(top_df: pd.DataFrame) -> pd.DataFrame:
    """Sort residue rows by chain & seqnum
    - Add chain
    - Add crg_changes
    """
    try:
        top_df.drop(["idx"], axis=0, inplace=True)
    except KeyError:
        # Assume df indexed with 'residues' col
        top_df.drop(0, axis=0, inplace=True)

    msk = top_df["info"] == "totals"
    res_df = pd.DataFrame(
        sorted(
            top_df[~msk].values, key=lambda x: (uppers.index(x[0][-5]), int(x[0][-4:]))
        ),
        columns=top_df.columns,
    )
    if res_df.shape[0]:
        df = pd.concat([res_df, top_df[msk]], ignore_index=False)
        df["chain"] = get_chain(df)
        df["crg_changes"] = get_changing_crg(df)
    else:
        logger.critical("No res_df data!")
        print(f"finalize_topN_df: {top_df.shape = }\n", top_df, "\n")
        sys.exit(1)

    return df


def filter_df_residue_kinds(df: pd.DataFrame, residue_kinds: list = IONIZABLE_RES):
    """Display the content of df filtered by residue_kinds list."""
    tots = df["info"] == "totals"
    inkind = df["residues"].str[:3].isin(residue_kinds)
    new = pd.concat([df[inkind], df[tots]]).replace(pd.NA, " ")
    return new


def topNdf_to_tsv(
    output_dir: Path, top_df: pd.DataFrame, n_top: int, res_kinds: list = None
) -> Path:
    """Save topN_df to a tab-separated-values (.tsv) file in output_dir.
    Return its filepath.
    """
    if res_kinds is None:
        tsv_fp = output_dir.joinpath(f"top{n_top}_ms.tsv")
        top_df.to_csv(tsv_fp, sep="\t", index=False)
    else:
        sm_df = filter_df_residue_kinds(top_df, res_kinds)
        tsv_fp = output_dir.joinpath(f"top{n_top}_ms_reskinds.tsv")
        sm_df.to_csv(tsv_fp, sep="\t", index=False)

    return tsv_fp


def get_pdb_remark(remark_data: dict):
    """
    Return a REMARK 250 header to prepend the final pdb with.
    Args:
      remark_data (dict): Keys: INPDB, T, PH, E, SUM_CRG, COUNT, OCC;
      the values are assumed to be strings reflecting appropriate formats.

    > REMARK 250 is mandatory if other than X-ray, NMR, neutron, or electron study.
    [Ref]: https://www.wwpdb.org/documentation/file-format-content/format33/remarks1.html#REMARK%20250
    """

    R250 = """REMARK 250
REMARK 250 EXPERIMENTAL DETAILS
REMARK 250   EXPERIMENT TYPE               : MCCE simulation
REMARK 250   DATE OF DATA COLLECTION       : {DATE}
REMARK 250   REMARK: DATE OF DATA COLLECTION is the creation date of this pdb.
REMARK 250 EXPERIMENTAL CONDITIONS
REMARK 250   SIMULATION INPUT PDB          : {INPDB}
REMARK 250   TEMPERATURE                   : {T} (K)
REMARK 250   PH                            : {PH}
REMARK 250   EH                            : {EH}
REMARK 250 CHARGE MICROSTATE INFORMATION
REMARK 250   ENERGY                        : {E} (kcal/mol)
REMARK 250   NET CHARGE                    : {SUM_CRG}
REMARK 250   SIZE                          : {SIZE}
REMARK 250   OCCUPANCY                     : {OCC}
REMARK 250 REMARK:
REMARK 250  This pdb was created from a tautomeric charge microstate vector
REMARK 250  extracted by {TOOL}, a MCCE4 tool.
REMARK 250
"""
    return R250.format(
        TOOL=APP_NAME, DATE=datetime.today().strftime("%d-%b-%y"), **remark_data
    )


def confs_to_pdb(
    step2_fh: TextIO, selected_confids: dict, output_pdb: str, dry: bool = True
) -> None:
    """Read step2_out coordinate line for each conformer in list `selected_confs`
    and creates a pdb file in step2 format.

    Args:
    step2_fh (TextIO): File handle of 'step2_out.pdb'.
    selected_confids (dict): A microstate's dict of conformer ids.
        Note: dict is for matching confIDs, values are all 1 (i.e. True) & unused.
    output_pdb (str): Output pdb file_path.
    dry (bool, True): whether to output waters
    """
    with open(output_pdb, "w") as out:
        for line in step2_fh:
            if len(line) < 82:
                continue
            if line[80:82] == "BK":
                out.write(line)
                continue
            res = line[17:20]
            if res == "MEM":
                continue
            if dry and res == "HOH":
                continue
            # exclude dummies
            if not dry and line[80:82] == "DM":
                continue
            if line[16] not in [" ", "A"]:
                continue
            confID = res + line[80:82] + line[21:26] + "_" + line[27:30]
            # ATOM     14  CB  LYS A0001_001   1.180   5.987  12.487   2.000       0.000      01O000M000
            # 012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
            #          1         2         3         4         5         6         7         8

            if selected_confids.get(confID) is not None:
                out.write(line)

    return


def s2out2pqr(s2_fp: str):
    """Convert a MCCE step2_out.pdb file to a pqr file (s2.pqr)."""
    pqr_frmt = "{:6s} {:>5} {:^4} {:3} {:>5} {:>8} {:>8} {:>8} {:>6} {:>6}\n"
    #           rec, seqnum, atm, res, resnum, x, y, z, crg, rad

    s2_fp = Path(s2_fp)
    pqr_fp = s2_fp.with_suffix(".pqr")

    with open(pqr_fp, "w") as pqr:
        with open(s2_fp) as s2:
            for line in s2:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                try:
                    rec, seqnum, atm, _, res, confid, x, y, z, rad, crg, _ = (
                        parse_mcce_line(line)
                    )
                except ValueError:
                    logger.error(
                        "Could not parse step2 line into (rec, seqnum, atm, res, confid, x, y, z, rad, crg)."
                    )
                    return

                # resnum from confid: int(confid[1:-4])
                pqr.write(
                    pqr_frmt.format(
                        rec, seqnum, atm, res, int(confid[1:-4]), x, y, z, crg, rad
                    )
                )

    return


def get_confids_dict(
    top_ms: list, conf_ids: np.ndarray, fixed_iconfs: list, ms_idx: int
) -> dict:
    """Combine the free confids in a microstate with the fixed ones into a dict."""
    fixed_ids = conf_ids[fixed_iconfs]
    free_ids = conf_ids[top_ms[ms_idx][1], 1]
    # dict used for convenience of '.get' method for identifying conf to write in pdb;
    # 1 is a dummy value
    confids_dict = dict((cid, 1) for cid in free_ids)
    confids_dict.update((cid, 1) for cid in fixed_ids[:, 1])

    return confids_dict


def write_tcrgms_pdbs(
    out_dir: Path,
    tcrgms_df: pd.DataFrame,
    top_ms: list,
    conf_ids: np.ndarray,
    fixed_iconfs: list,
    remark_args: dict,
    name_pref: str = S2PREF,
    dry: bool = True,
) -> None:
    """
    Write the topN tcms pdbs from the cms data columns in tcrgms_df.
    Note: It is assumed that the dataframe is not 'finalized': it's last
          column (at index -1) is 'info' and first row of indices
          referring to the cms-associated ms.
    """
    step2_fh = open(out_dir.parent.joinpath("step2_out.pdb"))

    top_df = tcrgms_df.set_index("residues")
    # isolate the series of associated ms indices:
    ids = top_df.iloc[0][:-1]
    # split df viz "totals" rows:
    df_tots = top_df[top_df["info"] == "totals"]

    for i, c in enumerate(df_tots.columns[:-1]):
        idx = ids[c]
        confids_dict = get_confids_dict(top_ms, conf_ids, fixed_iconfs, idx)

        # populate a dict for remark data
        R250_d = remark_args.copy()
        R250_d["E"] = df_tots.loc["E", c]
        R250_d["SUM_CRG"] = df_tots.loc["sum_crg", c]
        R250_d["SIZE"] = df_tots.loc["size", c]
        R250_d["OCC"] = df_tots.loc["occ", c]

        # rename final step2_out with identifiers:
        # s2_topmsN.pdb
        pdb_out = out_dir.joinpath(f"{name_pref}topms{c}.pdb")
        # get the pdb header text with remarks:
        remark = get_pdb_remark(R250_d)
        # save remarks as pdb_out with new extension ".remark" for final pdbs:
        pdb_out.with_suffix(".remark").write_text(remark)
        confs_to_pdb(step2_fh, confids_dict, pdb_out, dry=dry)

        step2_fh.seek(0)

        # convert step2_out output pdb file to .pqr:
        s2out2pqr(pdb_out)

    step2_fh.close()

    return


def mccepdbs_to_pdbs(out_dir: Path, s2_rename_dict: dict, rm_prefix: str = None) -> None:
    """Convert MCCE formatted pdbs in `out_dir` to pdb format.

    Args:
    out_dir (Path): Folder path for mcce pdbs
    s2_rename_dict (dict): Dict for renaming step2_out.pdb TER residues;
    rm_prefix (str): If not None, the output file name won't have this prefix.
    """
    remove_prefix = rm_prefix is not None and rm_prefix
    glob_str = "*.pdb"
    if remove_prefix:
        glob_str = f"{rm_prefix}*.pdb"

    for fp in out_dir.glob(glob_str):
        converter = Mcce2PDBConverter(fp, s2_rename_dict, output_dir=out_dir)
        out_name = f"{fp.stem}.pdb"
        if remove_prefix:
            out_name = f"{fp.stem.removeprefix(rm_prefix)}.pdb"
        converter.mcce_to_pdb(out_name)

    # FIX: temp warning:
    logger.warning(
        "Final conversion to pdb is incomplete: contain atoms/groups renamed by name.txt.\n"
    )

    return


def get_output_dirname(ph: Union[int, float], eh: Union[int, float], n_top: int) -> str:
    """Use given ph to create a path with the required number format,
    to match the precision in the msout file name.
    """
    if type(ph) is int:
        return OUT_DIR + f"_ph{ph}eh{eh}_top{n_top}"

    return OUT_DIR + f"_ph{ph:.2f}eh{eh:.2f}_top{n_top}"


# KEEP?
def extend_residue_kinds(res_kinds: list) -> list:
    """Return the IONIZABLES_RES list augmented with new kinds in 'res_kinds'
    in the same order as in IONIZABLES_RES, i.e.:
    acid, base, polar, N term, C term, followed by user-provided cofactors or groups.
    """
    if not res_kinds:
        return IONIZABLE_RES
    if not isinstance(res_kinds, list):
        res_kinds = [res_kinds]
    userlst = [res.upper() for res in res_kinds]
    ioniz_set = set(IONIZABLE_RES)
    sym_diff = ioniz_set.symmetric_difference(userlst)
    new_res = sym_diff.difference(ioniz_set)
    if new_res:
        return IONIZABLE_RES + sorted(new_res)
    else:
        return IONIZABLE_RES


def sort_resoi_list(res_kinds: list) -> list:
    """Return the input res_kinds (residues of interest, resoi) list
    with ionizable residues in the same order as in mcce4.constants.IONIZABLE_RES,
    i.e.: acid, base, polar, N-ter, C-ter, followed by the other user 'residues'.
    """
    if not res_kinds:
        return IONIZABLE_RES

    if not isinstance(res_kinds, list):
        res_kinds = [res_kinds]

    userlst = [res.upper() for res in res_kinds]
    ioniz = deepcopy(IONIZABLE_RES)

    ioniz_set = set(ioniz)
    sym_diff = ioniz_set.symmetric_difference(userlst)
    new_res = sym_diff.difference(ioniz_set)
    removal = sym_diff.difference(new_res)
    if removal:
        for res in removal:
            ioniz.pop(ioniz.index(res))

    return ioniz + sorted(new_res)


class TopNCmsPipeline:
    def __init__(self, args: Union[Namespace, dict]):
        self.int_ph: bool = True
        self.inpdb: str = "prot.pdb"
        self.mcce_files: tuple = None
        self.outname: str = None
        self.output_dir: Path = None
        self.mso: MSout_np = None
        self.top_cms: list = None
        self.top_ms: dict = None
        self.top_df = None

        self.args = self._parse_and_validate_args(args)
        self.mcce_dir = Path(self.args.mcce_dir)
        self.inpdb = get_input_pdb_name(self.mcce_dir)
        self.dry = not self.args.wet
        self.min_occ = float(self.args.min_occ)
        self.residue_kinds = self._validate_res_kinds()

    def _parse_and_validate_args(self, args) -> Namespace:
        # Logic to handle Namespace or dict, validate pH/Eh format etc.
        if isinstance(args, dict):
            args = Namespace(**args)

        # to get the output name with same input ph format
        if "." in args.ph:
            self.int_ph = False
            args.ph = float(args.ph)
            args.eh = float(args.eh)
        else:
            args.ph = int(args.ph)
            args.eh = int(args.eh)

        return args

    def _validate_res_kinds(self) -> list:
        if isinstance(self.args.residue_kinds, str):
            residue_kinds = self.args.residue_kinds.split(",")
        else:
            residue_kinds = self.args.residue_kinds

        if set(residue_kinds).symmetric_difference(IONIZABLE_RES):
            return sort_resoi_list(residue_kinds)

        return residue_kinds

    def setup_environment(self, tool_prompt: bool = False):
        # Logic for creating/checking output directory based on self.args.overwrite
        self.outname = get_output_dirname(self.args.ph, self.args.eh, self.args.n_top)
        self.output_dir = self.mcce_dir.joinpath(self.outname)
        # ... directory handling logic ...
        if self.output_dir.exists():
            if tool_prompt:
                ans = input(
                    (
                        f"The {self.outname} output folder already exists in {self.mcce_dir!s}.\n"
                        "Would you like to proceed and rewrite this folder? y/n "
                    )
                )
                if ans.strip().lower() == "y":
                    self.args.overwrite = True
                else:
                    self.args.overwrite = False

            if not self.args.overwrite:
                logger.info(
                    "The %s output folder already exists in %s.\n & overwrite option is False. Exiting."
                    % (self.outname, str(self.mcce_dir))
                )
                sys.exit()
            rmdir(self.output_dir)

        self.output_dir.mkdir()
        return

    def display_options(self):
        phstr = f"{self.args.ph:.0f}" if self.int_ph else f"{self.args.ph:.2f}"
        ehstr = f"{self.args.eh:.0f}" if self.int_ph else f"{self.args.eh:.2f}"
        msg = f"""
        Input options:
        Run dir: {self.mcce_dir!s};
        pH point: {phstr};
        Eh point: {ehstr};
        Residue kinds: {self.residue_kinds};
        Top N crg ms to process: {self.args.n_top};
        Occupancy threshold: {self.min_occ:.2%};
        Keep waters? {self.args.wet};
        Overwrite existing files? {self.args.overwrite};
        Output folder: {self.output_dir}
        """
        logger.info(msg)

    def load_data(self):
        # Logic to get file paths and instantiate MSout_np
        start_time = time.time()
        # h3_fp, step2_fp, msout_fp
        self.mcce_files = get_mcce_filepaths(self.mcce_dir, self.args.ph, self.args.eh)
        # using default mc_load="all": load ms and cms data:
        self.mso = MSout_np(
            self.mcce_files[0],
            self.mcce_files[2],
            res_kinds=self.residue_kinds,
            with_tautomers=True,
        )
        logger.info(self.mso)
        show_elapsed_time(
            start_time,
            info="Loading msout file for charge ms and associated conf ms data",
        )

    def process_microstates(self):
        # Logic using self.mso to get unique and top N states
        start_time = time.time()
        self.mso.get_uniq_ms()
        self.top_cms, self.top_ms = self.mso.get_topN_data(
            N=self.args.n_top, min_occ=self.args.min_occ
        )
        if not len(self.top_cms):
            sys.exit(f"NO DATA: Occupancies below threshold {self.args.min_occ:.2%}")

        self.top_df = self.mso.top_cms_df(self.top_cms)  # Initial DF
        show_elapsed_time(
            start_time, info="Getting unique charge ms and associated conf ms data"
        )

    def write_mcce_pdbs(self):
        # Logic calling write_tcrgms_pdbs
        # format common keys for final pdb REMARK 250:
        remark_args = {
            "INPDB": self.inpdb,
            "T": f"{self.mso.T:.2f}",
            "PH": f"{self.mso.pH:.2f}",
            "EH": f"{self.mso.Eh:.2f}",
        }

        write_tcrgms_pdbs(
            self.output_dir,
            self.top_df,
            self.top_ms,
            self.mso.conf_ids,
            self.mso.fixed_iconfs,
            remark_args,
            name_pref=S2PREF,
            dry=self.dry,
        )

    def convert_pdbs(self):
        # Logic calling mccepdbs_to_pdbs
        mccepdbs_to_pdbs(self.output_dir, self.mso.get_ter_dict(), rm_prefix=S2PREF)

    def write_tsv_and_summary(self):
        # Finalize DF, write TSV, write summary
        final_df = finalize_topN_df(self.top_df)
        tsv_fp = topNdf_to_tsv(self.output_dir, final_df, self.args.n_top)
        write_summary_file(tsv_fp, self.args.n_top, res_kinds=self.residue_kinds)
        msg = "".join(["Summary:\n", tsv_fp.parent.joinpath("summary.txt").read_text()])
        logger.info(msg)

    def run(self, tool_prompt: bool = False):
        # Orchestrates the pipeline call sequence
        start_time = time.time()
        logger.info("Starting...")
        self.setup_environment(tool_prompt)
        self.display_options()
        self.load_data()
        self.process_microstates()
        out_time = time.time()
        self.write_mcce_pdbs()
        self.convert_pdbs()
        self.write_tsv_and_summary()
        show_elapsed_time(out_time, info="Writing all output files")
        show_elapsed_time(start_time, info="Entire pipeline")


DESC = """
Description:
  ms_top2pdbs :: TopN protonation microstates to pdb & pqr files.

  This tool outputs:
  - The listing of the topN tautomeric protonation microstates, along with
    their related properties: mean energy (E), net charge (sum_crg), count,
    and occupancy (occ);
  - A summary file identifying ionizable residues with non canonical charge,
    and which of them do not change their charges over the topN set.
  - The <topN> files of each charge state in mcce-pdb & pqr and pdb formats.

Usage examples:
* If called inside a mcce output folder (not a requirement) & using the defaults;
  mcce_dir=., ph=7, eh=0, n_top=5, min_occ=0 and residue_kinds=ionizable residues:
  > ms_top2pdbs

* Otherwise:
  > ms_top2pdbs path/to/mcce_dir
  > ms_top2pdbs path/to/mcce_dir -eh 30
  > ms_top2pdbs path/to/mcce_dir -min_occ 0.002
  > ms_top2pdbs path/to/mcce_dir -ph 4 n_top 10

  # Residues: comma-separated; order & case insensitive:
  > ms_top2pdbs -residue_kinds _CL,his,GLU

INPUT FILES: head3.lst, step2_out.pdb, and the 'msout file' in the ms_out sub-directory,
             e.g. ms_out/pH7eH0ms.txt at ph 7.

OUTPUT FOLDER: mcce_dir/topms_ph7eh0_top5 (7, 0 and 5 are the default pH, Eh and number
               of top states requested).

OUTPUT FILES:
 - topN_ms.tsv: Charge of all acid/base residues in topN tautomeric protonation microstates + neural
                His tautomer. The last rows contain the mean energy (E) of this microstate, total
                charge (sum_crg), count (size), and probability (occ) of this microstate in ensemble.
 - summary.txt: List non-charged Asp, Glu, Arg, Lys, charged His, Tyr, Cys and neutral His tautomers
                in topN microstates.

 At most top_n coordinate files in these formats:
  - topmsN.pdb output each microstate in pdb format; N=1, 2, ..., top_n
  - s2_topmsN.pdb output each microstate in MCCE step2_out.pdb format
  - s2_topmsN.pqr output in pqr format (position, charge, radius)
"""


def cli_parser() -> ArgumentParser:
    p = ArgumentParser(
        prog=f"{APP_NAME}",
        description=DESC,
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""Report issues & feature requests here:
        https://github.com/GunnerLab/MCCE4/issues
        """,
    )
    p.add_argument(
        "-mcce_dir",
        default="./",
        type=str,
        help="Path to a mcce run dir; Default: %(default)s.",
    )
    p.add_argument(
        "-ph",
        default="7",
        # parser will receive a string, no conversion: easier to determine if number in int or float later.
        type=str,
        help="pH point (e.g.: 7, 7.5), at which the charge microstates are retrieved; Default: %(default)s.",
    )
    p.add_argument(
        "-eh",
        default="0",
        # parser will receive a string, no conversion: easier to determine if number in int or float later.
        type=str,
        help="pH point (e.g.: 7, 7.5), at which the charge microstates are retrieved; Default: %(default)s.",
    )
    p.add_argument(
        "-n_top",
        default=N_TOP,
        type=int,
        help="Number of most favorable charge microstates to return; Default: %(default)s.",
    )
    p.add_argument(
        "-residue_kinds",
        nargs="?",
        type=str,
        default=IONIZABLE_RES,
        help="Filter mcce residues (including cofactors) with these kinds, e.g. ASP,GLU,HEM; Default: %(default)s.",
    )
    p.add_argument(
        "-min_occ",
        default=MIN_OCC,
        type=float,
        help="Output topN ms with occ >= min_occ; Default: %(default)s.",
    )
    p.add_argument(
        "--wet",
        default=False,
        action="store_true",
        help="Output files with waters; Default: %(default)s.",
    )
    p.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite existing output files; Default: %(default)s.",
    )

    return p


def cli(argv=None, tool_prompt=False):
    """Obtain the top N charge microstates from the mcce run dir
    given by the input pdb path and output their step2_out, pqr and pdb files, along
    a summary file, and a .tsv file of cms state vectors & totals.
    """
    cli_parser = cli_parser()
    args = cli_parser.parse_args(argv)
    pipeline = TopNCmsPipeline(args)
    pipeline.run(tool_prompt=tool_prompt)

    return
