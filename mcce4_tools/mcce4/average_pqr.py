#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
import logging
from pathlib import Path
import re
import sys
from typing import Tuple, Union

import pandas as pd

from mcce4.constants import aliphatic_groups, pqr_frmt, ACIDIC_RES, NEUTRAL_RES, CLI_EPILOG
from mcce4.io_utils import parse_mcce_line


logger = logging.getLogger("get_aver_pqr")
if "debug" in sys.argv:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


logmsg_no_replace = "Output file %s already exists; either rename/delete" + \
                   " it or use the '--replace' option."
logmsg_save_interm = "Saved intermediate file %s to obtain sumcrg from pqr file."


TER = ["NTR", "CTR", "NTG"]
# confs to keep for pqr:
CONF01_RES = ACIDIC_RES + ["CYS", "CYD","TYR", "CTR", "HOH"] + NEUTRAL_RES
DEFAULT_RUNPRM = "run.prm.record"
INTERM_FILNAME = "average_with_resids.pqr"


apqr_filename_frmt = "average_{titr_pt}.pqr"


class ENV:
    """Custom ENV for average_pqr purposes."""
    def __init__(self, rundir_path: str, runprm_file: str = DEFAULT_RUNPRM) -> dict:
        self.rundir = Path(rundir_path)
        self.runprm: dict = {}
        self.conflist: dict = {}
        self.res_filer: list = None

        # populate self.runprm dict:
        self.load_runprm(runprm_file)
        # check if MCCE_HOME in runprm_file is accessible:
        if not self.rundir.joinpath(self.runprm["MCCE_HOME"]).exists():
            logger.critical(("Problem: %s in %s points to an inaccessible MCCE_HOME: %s\n"
                             "If this folder is a copy, modify the path to MCCE_HOME & rerun."),
                            str(runprm_file), str(self.rundir), str(self.runprm["MCCE_HOME"]))
            sys.exit(1)

        return

    def load_runprm(self, runprm_file: str = DEFAULT_RUNPRM):
        fp = self.rundir.joinpath(runprm_file)
        if not fp.exists():
            logger.critical("File not found: %s in %s", str(runprm_file), str(self.rundir))
            sys.exit(1)

        with open(fp) as fh:
            for line in fh:
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

    def _get_ftpl_conflist(self, fname: Path) -> Tuple[str, list, dict]:
        """Return a 3-tuple: res, confs list (w/o BK), atom order for each conf as parsed
        from the CONFLIST and CONNECT lines of ftpl file fname.
        Used by load_ftpl.

        Args:
         - fname: the name of a residue ftpl file.
        Returns:
         - 3-tuple: res, confs list (w/o BK), conf_order(dict)

        Example of conf_order dict:
          {'NTR01': {'CA': 1, 'HA': 2, 'N': 3, 'H': 4, 'H2': 5},
           'NTR+1': {'CA': 1, 'HA': 2, 'N': 3, 'H': 4, 'H2': 5, 'H3': 6}}
        """
        found = False
        confs = None
        res, confs_str = "", ""
        conf_order = defaultdict(dict)
        ix = 0
        mx = 0
        prev_conf = ""
        CLST = "CONFLIST, "    # e.g.: CONFLIST, CYS: CYSBK, CYS01, CYS-1
        CNCT = "CONNECT, "     # e.g.: CONNECT, " O  ", ASPBK: sp2, " C  "
        cnct_lines = []
        with open(fname) as fh:
            for line in fh:
                if not line.startswith((CLST, CNCT)):
                    if line.startswith("CHARGE"):
                        break
                    continue
                found = True
                if line.startswith(CLST):
                    res, confs_str = line.removeprefix(CLST).split(":")
                    confs_str = confs_str.strip().split("#")[0]
                    # get list and exclude BK:
                    confs = [c.strip() for c in confs_str.split(",")][1:]

                if line.startswith(CNCT):
                    _, atm, conf, *_ = line.split(", ", maxsplit=3)
                    atm = atm.strip('"').strip()
                    conf = conf.split(":", maxsplit=1)[0].strip()
                    cnct_lines.append([conf, atm])

        if not found:
            logger.debug("No CONFLIST in %s", str(fname))
            return None, None, None

        for ix, (conf, atm) in enumerate(cnct_lines, start=1):
            if ix == 1:
                mx = 0
                prev_conf = conf
            if prev_conf != conf:
                mx = ix-1
            conf_order[conf].update({atm:ix-mx})
            prev_conf = conf
            
        return res, confs, dict(conf_order)

    def load_ftpl(self, res_filter: list = None):
        """
        Load ENV.conflist dict holding each res conflist for each res in
        res_filter if any, else for all res.
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
        if res_filter:
            # save as attribute:
            self.res_filer = res_filter
            # save in ucase for comparing:
            RES_filter = [res.upper() for res in res_filter]

        logger.debug("ENV.load_ftpl :: Reading parameters from %s", str(ftpldir))
        for fp in ftpldir.glob("*.ftpl"):
            RES = fp.stem.upper()
            if RES_filter:
                if RES not in RES_filter:
                    continue
                res, confs, conf_order = self._get_ftpl_conflist(fp)
                if res is not None:
                    self.conflist[res] = [confs, conf_order]

                RES_filter.remove(RES)
                if not RES_filter:
                    break
            else:
                # no filtering
                res, confs, conf_order = self._get_ftpl_conflist(fp)
                if res is not None:
                    self.conflist[res] = [confs, conf_order]

        return

    def renamed_res_to_dict(self) -> Union[dict, None]:
        """Capture the 'from' and 'to' residues from a name.txt file
        into a 'reduced' dictionary.
        Note: dict may be empty if list was used to filter ftpl conflist.
        """
        name_fp = Path(self.runprm["MCCE_HOME"]).joinpath("name.txt")
        res_grouped_dict = defaultdict(dict)

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

    def __str__(self):
        out = f"rundir: {self.rundir}\nrunprm dict:\n"
        for k in self.runprm:
            out = out + f"{k} : {self.runprm[k]}\n"
        return out


def check_file(file_fp: Path, replace: bool, do_exit: bool = True) -> Tuple[bool, bool]:
    """
    Check whether the file exists and emove it according to replace and perrhaps
    terminate the program according do_exit.
    Returns:
      A 2-tuple of booleans: (exists, removed) if do_exit = False.
    """
    if file_fp.exists():
        if replace:
            file_fp.unlink()
            return True, True
        else:
            if not do_exit:
                logger.warning("File %s not saved (replace is False).", str(file_fp))
                return True, False
            logger.error(logmsg_no_replace, str(file_fp))
            sys.exit(1)
    else:
        return False, False


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
    def __init__(self, mcce_dir: str, titr_pt: Union[str, int, float],
                 replace: bool,
                 debug: bool = False):
        """
        Args:
         - mcce_dir (str): The mcce simulation folder.
         - titr_pt (str, int, float): The column name (titration point) from
           fort.38 that holds the occupancies; e.g.: '7.0' in a pH-titration,
           or '340.0' in a Eh-titration.
        """
        self.mcce_dir = Path(mcce_dir)
        # files checks:
        self.s2_fp = self.mcce_dir.joinpath("step2_out.pdb")
        if not self.s2_fp.exists():
            logger.critical("File not found: %s", str(self.s2_fp))
            sys.exit(1)

        self.f38_fp = self.mcce_dir.joinpath("fort.38") 
        if not self.f38_fp.exists():
            logger.critical("File not found: %s", str(self.f38_fp))
            sys.exit(1)

        self.replace = replace
        self.debug = debug
        self.uniq_res = None
        # set by AverPQR.get_occ_df if titr pt found in fort.38:
        self.pqr_fp = None
        # file to verify res sum crg:
        self.pqr_with_resids_fp = None
        self.sumcrg_fp = None

        self.env = ENV(self.mcce_dir)
        # cast to str if num:
        self.titr_pt = str(titr_pt) if val_is_numeric(titr_pt) else titr_pt
        self.occ_df = self.get_occ_df()
        self.uniq_res = self.occ_df["res"].unique().tolist()

        # Populate self.env.conflist dict with protein res:
        self.env.load_ftpl(res_filter=self.uniq_res)

        # get known grouped res (by name.txt) + ionizable flag:
        # key = main moiety, value = dict:: key = group_name, value = is_ionizable
        # e.g.: "HEM": {"PAA": True, "PDD": True, "FAR": False},
        self.grouped_res = self.get_renamed_res()

        return

    def get_renamed_res(self) -> Union[dict, None]:
        """
        Update inital res_groups dict produced by env.renamed_res_to_dict()
        with ionizable flag using data in AverPQR.env.conflist dict.
        Ionizable res have multiple confs in fort.38; need to know if a renamed
        group is ionizable or not.
        """
        # renamed groups from name.txt:
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
        Note:
          res_list: list of confs returned when fort.38 is filtered by resid,
          e.g. A0003. If the residue has been renamed by name.txt, the list holds
          the main res (moiety), along with its associated group(s), e.g.:
          ["PAA", "PDD", "HEM"]
        """
        if self.grouped_res is None:
            return None, None
        # return the key (main res) if found:
        mo = [k for k in res_list if k in self.grouped_res.keys()]
        if mo:
            mo = mo[0]
        else:
            return None, None

        return mo, tuple(self.grouped_res[mo].keys())

    def get_occ_df(self) -> pd.DataFrame:
        """Get processed fort.38 in a pandas.DataFrame.
        Note:  The last column 'keep' is used by most_occ_to_pqr.
        """
        # fn for df.apply:
        def extract_res(ro: pd.Series):
            """Extract the res id from confid including 'DM' if found."""
            if ro["conformer"][3:5] == "DM":
                return ro["conformer"][:5]
            return ro["conformer"][:3]

        # save to file which confs were selected for pqr:
        occdf_fname_frmt = "occ_df_{titr_pt}.tsv"

        df = pd.read_csv(self.f38_fp, sep=r"\s+")
        cols = df.columns.tolist()
        try:
            idx_titr = cols.index(self.titr_pt)
            self.pqr_fp = self.s2_fp.with_name(apqr_filename_frmt.format(titr_pt=self.titr_pt))
        except ValueError:
            logger.critical("Titration point %s is not in fort.38", self.titr_pt)
            sys.exit(1)

        df = df.rename(columns={cols[0]:"conformer", cols[idx_titr]:"occ"})
        df["res"] = df.apply(extract_res, axis=1)
        df["confid"] = df["conformer"].str[5:]
        df["resid"] = df["conformer"].str[5:-4]

        uniq_resids = df["resid"].unique().tolist()
        df["keep"] = False
        df = df[["conformer","resid","confid","res","occ","keep"]]

        # update the 'keep' column:
        for ur in uniq_resids:
            df1 = df.loc[df["resid"]==ur]
            uconfs = df1["confid"].unique()
            if len(uconfs) != len(df1):
                # case of res with multiple confids
                term_group = set(df1["res"].unique().tolist()).intersection(TER)
                if term_group:
                    # res with multiple confids due to TER capping
                    #  (this is the step1.py default: 'no_ter' option is False)
                    # TODO:
                    # - Get which charged ter conf to use according to titr_pt
                    #   - if 'conformer_col == "eh": ph is 7 by default but it can be changed;
                    #   => need to read env.runprm PH0 key
                    #   => need to get reference pKas for all ionizable residues

                    # default: charged ter capping group at ph7:
                    is_ter = df1["conformer"].str.startswith(("NTR+1","CTR-1"))
                    terx = df1.loc[is_ter].index[0]
                    df.loc[terx, "keep"] = True

                    # now get the actual terminal residue
                    in_ter = df1["res"].isin(TER)
                    resdf = df1.loc[~in_ter]
                    if len(resdf) == 1:
                        resx = resdf.index[0]
                    else:
                        resx = resdf["occ"].idxmax()
                        # if resdf["res"].unique().tolist()[0] is in CST.NEUTRAL_RES:
                        #     resx = resdf["occ"].idxmax()
                        # else:
                        #     resx = resdf["occ"].idxmin()
                    # update
                    df.loc[resx, "keep"] = True
                else:
                    # non-terminal res with multiple confids
                    res_lst = df1["res"].unique().tolist()
                    # 1. check mo_res exists
                    mo_res, grp_res_tpl = self.get_moiety_groups(res_lst)
                    if mo_res is None:
                        logger.critical(f"No residue names for this resid {ur!r} were found in dict 'grouped_res',",
                               "which gathers known split residues (moiety, groups) from 'name.txt'.",
                               f"\nThese residues are: {res_lst}")
                        sys.exit(1)

                    # 2. get max occ for mo_res and each group
                    mo_ix = df1.loc[df1["res"] == mo_res]["occ"].idxmax()
                    df.loc[mo_ix, "keep"] = True
                    for gres in grp_res_tpl:
                        # flag the max occ group
                        df2 = df1.loc[df1["res"] == gres]
                        if len(df2):
                            gr_ix = df2["occ"].idxmax()
                            df.loc[gr_ix, "keep"] = True
            else:
                ix = df1["occ"].idxmax()
                if "DM" in df1.loc[ix, "res"]:
                    continue
                else:
                    df.loc[ix, "keep"] = True

        if self.debug:
            occ_fp = self.mcce_dir.joinpath(occdf_fname_frmt.format(titr_pt=self.titr_pt))
            exists, removed = check_file(occ_fp, self.replace, do_exit=False)
            if (exists, removed) != (True, False):
                df.to_csv(occ_fp, sep="\t")
                logger.info("Saved the occupancy dataframe as %s", str(occ_fp))

        return df

    def most_occ_to_pqr(self):
        """Create the pqr file of the most occ'd conformers.
        """
        df_keep = self.occ_df.loc[self.occ_df["keep"] == True]  # for most occ'd
        lines_out = []
        lines_conf = []
        with_conf_frmt = "{:6s} {:>5} {:>5} {:^4} {:3} {:>5} {:>8} {:>8} {:>8} {:>6} {:>6}\n"
        #                resid:  D1023 _005
        new_pair = None, None  # conf, res
        new_found = False
        with open(self.s2_fp) as s2:
            for line in s2:
                line = line.strip()
                rec, seq, atm, alt, res, conf, x, y, z, rad, crg, _ = parse_mcce_line(line)
                resnum = str(int(conf[1:-4]))

                if conf.endswith("_000") and (res[0] != "_"):
                    # write pqr line for BK as is
                    lines_out.append(pqr_frmt.format(rec, seq, atm, res, resnum, x, y, z, crg, rad))
                    if self.debug:
                        lines_conf.append(with_conf_frmt.format(rec,conf[:5],seq,atm,res,resnum,x,y,z,crg,rad))
                else:
                    pair = conf, res
                    if pair != new_pair:
                        new_pair = pair
                        # maybe get occ to keep until new pair
                        msk = (df_keep["confid"]==conf) & (df_keep["res"]==res)
                        ok_df = df_keep.loc[msk]
                        if self.debug:
                            print(f"ok_df: {new_pair= }:", ok_df, sep="\n")

                        if ok_df.shape[0]:
                            occ = ok_df["occ"].values[0]
                            new_found = True
                        else:
                            new_found = False

                    if (pair == new_pair) and new_found:
                        q = f"{float(crg)*occ:.3f}"
                        lines_out.append(pqr_frmt.format(rec, seq, atm, res, resnum, x, y, z, q, rad))
                        if self.debug:
                            lines_conf.append(with_conf_frmt.format(rec,conf[:5],seq,atm,res,resnum,x,y,z,q,rad))

        logger.debug(f"Number of lines to write: {len(lines_out):,}")
        with open(self.pqr_fp, "w") as pqr:
            pqr.writelines(lines_out)
        logger.info("Saved the occ-weighted charge pqr file: %s", str(self.pqr_fp))

        if self.debug:
            self.pqr_with_resids_fp = self.pqr_fp.with_name(INTERM_FILNAME)
            exists, removed = check_file(self.pqr_with_resids_fp, self.replace)
            if (exists, removed) == (True, False):
                return
            with open(self.pqr_with_resids_fp, "w") as pqr:
                pqr.writelines(lines_conf)
            logger.info(logmsg_save_interm, str(self.pqr_with_resids_fp))

        return

    def s2_to_aver_pqr(self):
        """Create the pqr file of the weighted aver crg over a residue confs.
        """
        logger.info("""Version 1 of the tool:
  Averaged coordinates with Boltzmann averaged charges (sum of occupancy-weighted charges),"
  over the confomers of each residue.""")

        new_pair = None, None  # conf, res
        new_found = False
        s2_lines = []
        logger.info("Processing each line in step2_out.pdb...")
        with open(self.s2_fp) as s2:
            for lx, line in enumerate(s2, start=1):
                line = line.strip()
                # fields for pqr: rec, seq, atm, res, resnum, x, y, z, crg, rad
                rec, seq, atm, alt, res, conf, x, y, z, rad, crg, _ = parse_mcce_line(line)
                resnum = str(int(conf[1:-4]))
                x, y, z, rad, crg = float(x), float(y), float(z), float(rad), float(crg)

                if conf.endswith("_000") or res[0] == "_":
                    keep = conf
                    s2_lines.append([rec, seq, conf, keep, res, resnum, atm,  x, y, z, rad, crg])
                else: 
                    # get occ weighted crg
                    pair = conf, res
                    if pair != new_pair:
                        new_pair = pair
                        # maybe get occ to keep until new pair
                        msk = (self.occ_df["confid"]==conf) & (self.occ_df["res"]==res)
                        ok_df = self.occ_df.loc[msk]
                        if ok_df.shape[0]:
                            occ = float(ok_df["occ"].values[0])
                            new_found = True
                        else:
                            new_found = False

                        if res in CONF01_RES:
                            search_keep = f"{res}01{conf[:5]}"
                        else:
                            search_keep = rf"{res}+1{conf[:5]}"
                        try:
                            keep = self.occ_df.loc[self.occ_df["conformer"].str.startswith(search_keep), "confid"].values[0]
                        except IndexError as e:
                            print(f"Could not get the conf to keep for: {res= }; {conf= }; {search_keep= }")
                            sys.exit()

                    if (pair == new_pair) and new_found:
                        crg = crg*occ
                        s2_lines.append([rec, seq, conf, keep, res, resnum, atm, x,y,x,rad,crg])

        logger.info(f"Number of s2 lines processed = {lx:,}")

        cols = "rec,seq,conf,keep,res,resnum,atm,x,y,z,rad,crg".split(",")
        s2_df = pd.DataFrame(s2_lines, columns=cols)
        s2_df["resid"] = s2_df["conf"].str[:5]

        logger.info("Summing the weighted charges...")
        grp1 = s2_df.groupby(["rec","resid","res","atm"]).agg({"crg":"sum"})
        df1 = grp1.stack(level=0,future_stack=True).reset_index([0,1,2,3])
        df1 = df1.reset_index(drop=True)
        df1.columns = ["rec","resid","res","atm","wq"]
        if self.debug:
            print("Intermediate result: df1, weighted crg:", df1, sep="\n")

        logger.info("Averaging the coordinates...")
        grp2 = s2_df.groupby(["rec","resid","res","atm","x","y","z"]).agg({"x":"mean", "y":"mean", "z":"mean"})
        df2 = grp2.stack(level=0,future_stack=True).reset_index([0,1,2,3, 4, 5, 6])
        df2 = df2.drop(0, axis=1)
        df2.drop_duplicates(inplace=True)
        df2 = df2.reset_index(drop=True)
        if self.debug:
            print("Intermediate result: df2, aver coords:", df2, sep="\n")

        logger.info("Merging the results...")
        wq_merg = s2_df.merge(df1, left_on=["rec","resid","res","atm"],
                              right_on=["rec","resid","res","atm"])
        wq_merg.drop(["x","y","z"], axis=1, inplace=True)
        
        all_merg = wq_merg.merge(df2, left_on=["rec","resid","res","atm"],
                                 right_on=["rec","resid","res","atm"])
        all_merg = all_merg.loc[all_merg["conf"]==all_merg["keep"]]
        if self.debug:
            print("Final result: all_merg df:", all_merg, sep="\n")

        logger.info(f"Writing {all_merg.shape[0]:,} pqr lines...")
        pqr2 = None
        if self.debug:
            self.pqr_with_resids_fp = self.pqr_fp.with_name(INTERM_FILNAME)
            exists, removed = check_file(self.pqr_with_resids_fp, self.replace, do_exit=False)
            if (exists, removed) != (True, False):
                pqr2 = open(self.pqr_with_resids_fp, "w") 
        
        with open(self.pqr_fp, "w") as pqr:
            for _, ro in all_merg.iterrows():
                pqr.write(pqr_frmt.format(ro.rec, ro.seq, ro.atm, ro.res, ro.resnum,
                                          f"{ro.x:.3f}", f"{ro.y:.3f}", f"{ro.z:.3f}",
                                          f"{ro.wq:.3f}", ro.rad)
                )
                if pqr2 is not None:
                    pqr2.write(pqr_frmt.format(ro.rec, ro.resid, ro.seq, ro.atm, ro.res,
                                               ro.resnum,f"{ro.x:.3f}", f"{ro.y:.3f}",
                                               f"{ro.z:.3f}", f"{ro.wq:.3f}", ro.rad)
                    )
        logger.info("Saved the occ-weighted charge pqr file: %s", str(self.pqr_fp))
        if pqr2 is not None:
            pqr2.close()
            logger.info(logmsg_save_interm, str(self.pqr_with_resids_fp))

        return

    def write_res_sumcrg_from_pqr(self):
        """List the residues sum charge from the pqr file."""
        if not self.debug:
            return
        hdr = "rec,resid,seq,atm,res,resnum,x,y,z,crg,rad".split(",")
        pqrdf = pd.read_csv(self.pqr_with_resids_fp, sep='\s+', header=None, names=hdr)
        lines = []
        for ur in pqrdf["resid"].unique().tolist():
            df = pqrdf.loc[pqrdf["resid"]==ur]
            lines.append(f"{ur} {df['res'].unique()}: {df['crg'].sum():.1f}\n")

        self.sumcrg_fp = self.pqr_fp.with_name(f"sumcrg_pqr_{self.titr_pt}.tsv")
        # this point is reached if debug, so quit if file exists and not replaced:
        _ = check_file(self.sumcrg_fp, self.replace)  # default: do_exit=True

        with open(self.sumcrg_fp, "w") as sumcrg:
            sumcrg.writelines(lines)
        logger.info("Saved the residue sum charge from the pqr file: %s", str(self.sumcrg_fp))

        return


def cli_parser():
    p = ArgumentParser(
        prog="get_aver_pqr",
        description="""Obtain a pqr file with Boltzmann averaged charges or one for the most occupied
        conformers in fort.38 together with step2_out.pdb data.
        """,
        add_help=True,
        usage=("\n"
        "  %(prog)s                             # minimal usage if pH=7.0 & -mcce_dir is cwd (default output is 'Boltzmann averaged' pqr file)\n"
        "  %(prog)s -mcce_dir dir -titr_pt 350  # case for a Eh titration in 'dir' (the titration point must be found in fort.38)\n"
        "  %(prog)s --most_occ_pqr             # to obtain the pqr of the most occupied conformers (with -titr_pt <value> if not default)\n"
        "  %(prog)s --replace                   # to overwrite any existing file\n"
        "  %(prog)s --debug                     # to obtain each residue sumcrg from the pqr file\n"
        "  %(prog)s --debug --replace           # save as above with overwriting of existing files\n"
        ), 
        epilog=CLI_EPILOG
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
    p.add_argument(
        "--most_occ_pqr",
        default=False,
        action="store_true",
        help="This option will create the pqr file of the most occupied conformers; default: %(default)s"
    )
    p.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Set logging level to DEBUG instead of INFO. This mode creates a sum crg file from the averaged pqr file; default: %(default)s"
    )
    p.add_argument(
        "--replace",
        default=False,
        action="store_true",
        help="Including this option will overwrite an existing pqr file; default: %(default)s"
    )
    return p


def get_aver_pqr_cli(argv=None):
    """Cli function for the `get_aver_pqr` tool.
    """
    p = cli_parser()
    args = p.parse_args(argv)

    pqr_fp = Path(args.mcce_dir).joinpath(f"average_{args.titr_pt}.pqr")
    _ = check_file(pqr_fp, args.replace)

    apqr = AverPQR(args.mcce_dir, args.titr_pt, args.replace, debug=args.debug)
    if args.most_occ_pqr:
        logger.info("Create the pqr file for the most occupied conformers...")
        apqr.most_occ_to_pqr()
    else:
        apqr.s2_to_aver_pqr()

    apqr.write_res_sumcrg_from_pqr()
    logger.info("Average pqr file creation over for '%s'\n", Path(args.mcce_dir).absolute().name)

    return


if __name__ == "__main__":
    get_aver_pqr_cli(sys.argv)
