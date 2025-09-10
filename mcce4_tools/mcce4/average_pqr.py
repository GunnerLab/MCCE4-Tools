#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Union

import pandas as pd

from mcce4.constants import aliphatic_groups, pqr_frmt
from mcce4.io_utils import parse_mcce_line


TER = ["NTR", "CTR"]


apqr_filename_frmt = "average_{titr_pt}.pqr"
# temp: to which confs were selected for pqr:
occdf_filename_frmt = "occ_df_{titr_pt}.tsv"


DEFAULT_RUNPRM = "run.prm.record"


class ENV:
    # Custom ENV for average_pqr purposes.
    def __init__(self, rundir_path: str, runprm_file: str = DEFAULT_RUNPRM) -> dict:
        self.rundir = Path(rundir_path)
        self.runprm: dict = {}
        self.conflist: dict = {}
        self.res_filer: list = None
        self.titr_bounds = None
        self.sumcrg_hdr = ""
        # populate self.runprm dict:
        self.load_runprm(runprm_file)
        # check MCCE_HOME from loading of runprm_file is accessible:
        if not self.rundir.joinpath(self.runprm["MCCE_HOME"]).exists():
            print(f"CRITICAL: {runprm_file!s} in {self.rundir!s} points to an",
                  f"inaccessible MCCE_HOME: {self.runprm['MCCE_HOME']!s}.\n",
                  "If this folder is a testing copy, modify the path to MCCE_HOME & rerun."
                  )
            sys.exit(1)

        return

    def load_runprm(self, runprm_file: str = DEFAULT_RUNPRM):
        fp = self.rundir.joinpath(runprm_file)
        if not fp.exists():
            raise FileNotFoundError(f"Not found: {runprm_file} in {self.rundir}")

        with open(fp) as fin:
            for line in fin:
                entry_str = line.strip().split("#")[0]
                fields = entry_str.split()
                if len(fields) > 1:
                    key_str = fields[-1]
                    if key_str[0] == "(" and key_str[-1] == ")":
                        key = key_str.strip("()").strip()
                        # inconsistent output in run.prm.record:
                        if key == "EPSILON_PROT":
                            value = str(round(float(fields[0]), 1))
                        else:
                            value = fields[0]
                        self.runprm[key] = value

        return

    def get_ftpl_conflist(self, fname: Union[str, Path]) -> tuple:
        """Return a 2-tuple: res, confs (w/o BK).
        """
        found = False
        confs = None
        res, confs_str = "", ""
        PREF = "CONFLIST, "    # e.g.: CONFLIST, CYS: CYSBK, CYS01, CYS-1
        with open(fname) as fh:
            for line in fh:
                if not line.startswith(PREF):
                    continue
                found = True
                res, confs_str = line.removeprefix(PREF).split(":")
                confs_str = confs_str.strip().split("#")[0]
                # get list and exclude BK:
                confs = [c.strip() for c in confs_str.split(",")][1:]
                break
        if not found:
            print(f"No CONFLIST in {fname!s}!")
            return None, None

        return res, confs

    def load_ftpl(self, res_filter: list = None):
        """
        Load ENV.param dict holding each res conflist for each res in
        res_filter, if any else for all res.
        Args:
         - res_filter (list): A list of res names; typically passed
                              in order to reduce the number of ftpl loaded,
                              e.g. to only load the res present in a protein.
        """
        if "FTPLDIR" in self.runprm:
            ftpldir = Path(self.runprm["FTPLDIR"])
        else:
            ftpldir = Path(self.runprm["MCCE_HOME"]).joinpath("param")

        RES_filter = None
        if not res_filter is None:
            # save as attribute:
            self.res_filer = res_filter
            # save in ucase for comparing:
            RES_filter = [res.upper() for res in res_filter]

        print(f"Reading parameters from {ftpldir!s}")
        self.param = defaultdict(list)
        for fp in ftpldir.glob("*.ftpl"):
            RES = fp.stem.upper()
            if RES_filter:
                if RES not in RES_filter:
                    continue
                res, confs = self.get_ftpl_conflist(fp)
                if res is not None:
                    self.conflist[res] = confs

                RES_filter.remove(RES)
                if not RES_filter:
                    break
            else:
                res, confs = self.get_ftpl_conflist(fp)
                if res is not None:
                    self.conflist[res] = confs

        return

    def renamed_res_to_dict(self) -> Union[dict, None]:
        """Capture the 'from' and 'to' residues from a name.txt file
        into a 'reduced' dictionary.
        Note: dict may be empty if list was used to filter ftpl conflist.
        """
        name_fp = Path(self.runprm["MCCE_HOME"]).joinpath("name.txt")
        res_grouped_dict = defaultdict(dict)
        #aliphatic_groups = "CT1, CT2, CT3, CT4, FAR, dgd, lhg, lmg, lmt, sqd".split(", ")

        # Regex to capture the 'from' and 'to' residues
        pattern = r".{5}(.{3}).{13}(.{3}).*"
        with open(name_fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue

                match = re.match(pattern, line)
                if match:
                    from_res = match.group(1)
                    if from_res.count("*") > 1:
                        continue

                    # if list was passed when loading ftpl conflist:
                    if self.res_filer is not None and from_res not in self.res_filer:
                        continue

                    to_res = match.group(2)
                    if to_res == "***":
                        continue

                    if from_res != to_res:
                        if to_res in aliphatic_groups:
                            # preset value to False (not ionizable)
                            res_grouped_dict[from_res].update({to_res:False})
                        else:
                            res_grouped_dict[from_res].update({to_res:None})

        if not res_grouped_dict:
            return None

        # further reduce, exclude moiety renaming entry:
        # discard if value has only 1 key and value is None:
        out = {}
        for k in res_grouped_dict:
            if len(res_grouped_dict[k]) == 1:
                if list(res_grouped_dict[k].values())[0] is not None:
                    out[k] = res_grouped_dict[k]
            else:
                out[k] = res_grouped_dict[k]

        return out

    def get_titr_bounds(self):
        """
        Populate self.titr_bounds
        ph       "ph" for pH titration, "eh" for eh titration       (TITR_TYPE)
        0.0      Initial pH                                         (TITR_PH0)
        1.0      pH interval                                        (TITR_PHD)
        0.0      Initial Eh                                         (TITR_EH0)
        30.0     Eh interval (in mV)                                (TITR_EHD)
        15       Number of titration points                         (TITR_STEPS)
        """
        prec = 0
        if self.runprm["TITR_TYPE"] == "ph":
            b1 = float(self.runprm["TITR_PH0"])
            step = float(self.runprm["TITR_PHD"])
        else:
            b1 = float(self.runprm["TITR_EH0"])
            step = float(self.runprm["TITR_EHD"])

        b2 = [b1 + i * step for i in range(int(self.runprm["TITR_STEPS"]))]
        bounds = round(b1, prec), round(b2[-1], prec)
        self.titr_bounds = bounds

        return bounds

    def __str__(self):
        out = f"rundir: {self.rundir}\nrunprm dict:\n"
        for k in self.runprm:
            out = out + f"{k} : {self.runprm[k]}\n"
        return out


def val_is_numeric(value: str) -> bool:
    """Return True if value is numeric or strictly boolean.
    """
    if value in ["True", "False"]:
        return True
    is_num = False
    try:
        num = str(int(float(value)))
        is_num = True
    except ValueError:
        pass
    return is_num


class AverPQR:
    """Class for processing fort.38 (the occupancy table) and step2_out.pdb
    in order to obtain pqr file with with occupancy-weighted charges
    assigned to the most occupied conformers in fort.38.
    """
    def __init__(self, mcce_dir: str, titr_pt: Union[str, int, float]):
        """
        Args:
         - mcce_dir (str): The mcce simulation folder.
         - titr_pt (str, int, float): The column name (titration point) from
           fort.38 that holds the occupancies; e.g.: '7.0' in a pH-titration,
           or '340.0' in a Eh-titration.
        """
        self.mcce_dir = Path(mcce_dir)
        self.s2_fp = self.mcce_dir.joinpath("step2_out.pdb")
        if not self.s2_fp.exists():
            raise FileNotFoundError(f"Not found: {self.s2_fp!s}")
        self.f38_fp = self.mcce_dir.joinpath("fort.38") 
        if not self.f38_fp.exists():
            raise FileNotFoundError(f"Not found: {self.f38_fp!s}")

        self.pqr_fp = None
        self.conformer_col = None
        self.occ_col = None
        self.uniq_resids = None
        self.uniq_res = None
        self.grouped_res = None

        self.env = ENV(self.mcce_dir)
        # cast to str if num:
        self.titr_pt = str(titr_pt) if val_is_numeric(titr_pt) else titr_pt
        self.df_38 = self.get_df38()

        uniq_res = self.uniq_res.copy()
        # Populate self.env.conflist dict with res conflist:
        self.env.load_ftpl(res_filter=uniq_res)

        # known grouped res (by name.txt):
        # key = main moiety, value = dict:: key = group_name, value = is_ionizable
        # e.g.: "HEM": {"PAA": True, "PDD": True, "FAR": False},
        self.grouped_res = self.get_renamed_res()


    def get_df38(self) -> pd.DataFrame:

        def extract_res(ro: pd.Series):
                """Extract the res id from confid including 'DM' if found."""
                if ro[self.conformer_col][3:5] == "DM":
                    return ro[self.conformer_col][:5]
                return ro[self.conformer_col][:3]

        df = pd.read_csv(self.f38_fp, sep=r"\s+")
        cols = df.columns.tolist()
        try:
            idx_titr = cols.index(self.titr_pt)
        except ValueError:
            sys.exit(f"Titration point {self.titr_pt!s} not in fort.38")

        self.occ_col = cols[idx_titr]
        self.conformer_col = cols[0]
        df["res"] = df.apply(extract_res, axis=1)
        df["confid"] = df[self.conformer_col].str[5:]
        df["resid"] = df[self.conformer_col].str[5:-4]

        self.uniq_resids = df["resid"].unique().tolist()
        self.uniq_res = df["res"].unique().tolist()

        return df

    def get_renamed_res(self) -> Union[dict, None]:
        """
        Update inital res_groups dict produced by env.renamed_res_to_dict()
        with ionizable flag using data in AverPQR.env.conflist dict.
        """
        res_groups = self.env.renamed_res_to_dict()
        if res_groups is None:
            return None

        for k in res_groups:
            for cid in res_groups[k]:
                # already set:
                if res_groups[k][cid] is not None:
                    continue

                is_ioniz = False
                if cid in self.env.conflist:
                    confs = self.env.conflist[cid]
                    for cf in confs:
                        is_ioniz = is_ioniz or ("+" in cf) or ("-" in cf)

                    res_groups[k][cid] = is_ioniz

        return res_groups

    def get_moiety_groups(self, res_list: list) -> tuple:
        """Assuming unique resnames in res_list, return the
        main moiety (the top key in grouped_res) and the groups resnames.
        """
        if self.grouped_res is None:
            return None, None

        mo = [k for k in res_list if k in self.grouped_res.keys()]
        if mo:
            mo = mo[0]
        else:
            return None, None

        grps = tuple(self.grouped_res[mo].keys())

        return mo, grps

    def get_occ_df(self) -> pd.DataFrame:
        occdf = self.df_38.copy()
        occdf["seq"] = occdf.index + 1
        occdf["keep"] = False
        occdf = occdf.rename(columns={self.conformer_col:"conformer", self.occ_col:"occ"})
        occdf = occdf[["conformer","seq","resid","confid","res","occ","keep"]]

        for ux, ur in enumerate(self.uniq_resids, start=1):
            df1 = occdf.loc[occdf["resid"]==ur]
            uconfs = df1["confid"].unique()
            if len(uconfs) != len(df1):
                term_group = set(df1["res"].unique().tolist()).intersection(TER)
                if term_group:
                    # res with multiple confids due to TER capping
                    #  (this is the step1.py default: 'no_ter' option is False)
                    # TODO:
                    # - Get which charged ter conf to use according to titr_pt
                    #   - if 'conformer_col == "eh": ph is 7 by default but it can be changed;
                    #   => need to read run.prm.record
                    #   => need to get reference pKas for all ionizable residues

                    # default charged ter conf at ph7:
                    is_ter = df1["conformer"].str.startswith(("NTR+1","CTR-1"))
                    terx = df1.loc[is_ter].index[0]
                    occdf.loc[terx, "keep"] = True
                    ter_cid = df1.loc[terx, "confid"]

                    in_ter = df1["res"].isin(TER)
                    resdf = df1.loc[~in_ter]
                    if len(resdf) == 1:
                        resx = resdf.index[0]
                    else:
                        resx = resdf.loc[resdf["confid"]==ter_cid].index[0]
                        # if resdf["res"].unique().tolist()[0] is CST.NEUTRAL_RES:
                        #     resx = resdf["occ"].idxmax()
                        # else:
                        #     resx = resdf["occ"].idxmin()

                    # update
                    occdf.loc[resx, "keep"] = True
                else:
                    # res with multiple confids
                    res_lst = df1["res"].unique().tolist()
                    mo_res, grp_res_tpl = self.get_moiety_groups(res_lst)
                    if mo_res is None:
                        print(f"No residue names for this resid {ur!r} were found in dict 'grouped_res',",
                               "which gathers known split residues (moiety, groups) from 'name.txt'.",
                               f"\nThese residues are: {res_lst}")
                        sys.exit(1)

                    # 2. get max occ for mo_res and each group
                    mo_ix = df1.loc[df1["res"] == mo_res]["occ"].idxmax()
                    occdf.loc[mo_ix, "keep"] = True

                    for gres in grp_res_tpl:
                        # need to check if group existsL
                        df2 = df1.loc[df1["res"] == gres]
                        if len(df2):
                            gr_ix = df2["occ"].idxmax()
                            occdf.loc[gr_ix, "keep"] = True
            else:
                ix = df1["occ"].idxmax()
                if "DM" in df1.loc[ix, "res"]:
                    continue
                else:
                    occdf.loc[ix, "keep"] = True
        # save:
        occ_fp = self.mcce_dir.joinpath(occdf_filename_frmt.format(titr_pt=self.titr_pt))
        occdf.to_csv(occ_fp, sep="\t")

        return occdf

    def get_dfs2(self) -> pd.DataFrame:
        # load step2_out.pdb in a df with header:
        hdr = "atype	seq	atm	res	confid	x	y	z	rad	q	hist".split()
        df = pd.read_csv(self.s2_fp, sep=r"\s+", header=None, names=hdr)
        df = df.drop("hist", axis=1)
        df["resid"] = df.confid.str[:-4]
        # create & initialize weighted q column:
        df["wq"] = df["q"] * 1

        return df

    def get_reduced_dfs2(self) -> pd.DataFrame:
        """Return a step2 df with aggregated weighted occ and fitered
        for the most occupied conformers to keep in the pqr file.
        """
        occ_df = self.get_occ_df()
        dfs2 = self.get_dfs2()
        for_keeps = []

        for _, ro in occ_df.iterrows():
            confid = ro["confid"]
            res = ro["res"]

            if "DM" in res:
                continue
            if ro["keep"]:
                for_keeps.append((confid, res))

            # update occ-weigthed charge:
            msk = (dfs2["confid"]==confid) & (dfs2["res"]==res)
            rodf = dfs2.loc[msk]
            if len(rodf):
                new_q = rodf["q"] * ro["occ"]
                dfs2.loc[msk, "wq"] = new_q
            else:
                print(f"Problem: No s2 lines found for {confid= } - {res= }")

        # sum wq over atoms, the final df is used to write pqr
        grp2 = dfs2.groupby(by=["seq","confid","res","atm","resid"])["wq"].sum()
        df2 = grp2.reset_index()
        df2.sort_values(by="seq", inplace=True)

        # add col for filtering
        df2["keep"] = False
        for tpl in for_keeps:
            keepconf, keepres = tpl
            mk = (df2["confid"] == keepconf) & (df2["res"] == keepres)
            df2.loc[mk, "keep"] = True

        # return filtered df for confs to keep
        return df2.loc[df2["keep"] == True]

    def s2_to_aver_pqr(self):
        df_keep = self.get_reduced_dfs2()
        self.pqr_fp = self.s2_fp.with_name(apqr_filename_frmt.format(titr_pt=self.titr_pt))

        lines_out = []
        with open(self.s2_fp) as s2:
            for line in s2:
                line  = line.strip()
                rec, seq, atm, alt, res, conf, x, y, z, rad, crg, _ = parse_mcce_line(line)
                resnum = str(int(conf[1:-4]))

                if conf.endswith("_000"):
                    # write pqr line for BK as is
                    lines_out.append(pqr_frmt.format(rec, seq, atm, res, resnum, x, y, z, crg, rad))
                else:
                    # confid res atm
                    msk = (df_keep["confid"]==conf) & (df_keep["res"]==res) & (df_keep["atm"]==atm)
                    ok_df = df_keep.loc[msk]
                    if ok_df.shape[0]:  # should be 1
                        q = f"{ok_df['wq'].tolist()[0]:.3f}"
                        lines_out.append(
                            pqr_frmt.format(rec, seq, atm, res, resnum, x, y, z, q, rad)
                        )

        with open(self.pqr_fp, "w") as pqr:
            pqr.writelines(lines_out)
        print(f"Created occ-weighted average charge pqr file: {self.pqr_fp!s}")

        return


def cli_parser():
    p = ArgumentParser(
        prog="get_aver_pqr",
        description="""Obtain an occupancy-weigthed charge on the most 
        occupied conformers in fort.38 together with step2_out.pdb data.
        """,
        usage="%(prog)s # minimal usage if pH=7.0, else:\n       %(prog)s -titr_pt <value>",
        add_help=True
    )
    p.add_argument(
        "-mcce_dir",
        type=str,
        default=".",
        help="Path of the mcce simulation folder; default: %(default)s"
    )
    p.add_argument(
        "-titr_pt",
        default=7.0,
        help="The titration point at which to fetch the occupancies in fort.38; default: %(default)s"
    )

    return p


def get_aver_pqr_cli(argv=None):
    """Cli function for the `get_aver_pqr` tool.
    """
    p = cli_parser()
    args = p.parse_args(argv)
    apqr = AverPQR(args.mcce_dir, args.titr_pt)
    apqr.s2_to_aver_pqr()
    print(f"Average pqr file creation over for {Path(args.mcce_dir).absolute().name!r}.")

    return


if __name__ == "__main__":
    get_aver_pqr_cli(sys.argv)
