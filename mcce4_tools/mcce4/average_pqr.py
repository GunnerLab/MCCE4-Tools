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
CONF01_RES = ACIDIC_RES + ["CYS", "CYD", "TYR", "CTR", "HOH"] + NEUTRAL_RES
DEFAULT_RUNPRM = "run.prm.record"


apqr_fname_frmt = "{kind}_{titr_pt}.pqr"
interim_fname_frmt = "{kind}_{titr_pt}_with_resids.pqr"
sumcr_fname_frmt = "{kind}_sumcrg_pqr_{titr_pt}.tsv"
occ_file_frmt = "{kind}_occ_df_{titr_pt}.tsv"


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
    def __init__(self, mcce_dir: str,
                 titr_pt: Union[str, int, float],
                 pqr_kind: str = "average",
                 replace: bool = False,
                 debug: bool = False):
        """
        Args:
         - mcce_dir (str): The mcce simulation folder.
         - titr_pt (str, int, float): The column name (titration point) from
           fort.38 that holds the occupancies; e.g.: '7.0' in a pH-titration,
           or '340.0' in a Eh-titration.
         - replace (bool): Whether to replace existing files.
         - pqr_kind (str): Which type of pqr file to create, default is Boltzmann average,
                           vs. most occupied conformers.
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

        self.sumcrg_out_fp = self.mcce_dir.joinpath("sum_crg.out")
        if not self.sumcrg_out_fp.exists():
            logger.warning("File 'sum_crg.out' was not found: No consistency check will be run.")
            self.do_sumcrg_check = False
        else:
            self.do_sumcrg_check = True
        
        self.pqr_kind = pqr_kind if pqr_kind=="average" else "mostocc"
        self.replace = replace
        self.debug = debug
        # file to verify res sum crg:
        self.pqr_with_resids_fp = None
        self.sumcrg_fp = None
        # set by AverPQR.get_occ_df if titr pt found in fort.38:
        self.pqr_fp = None
        self.uniq_res = None
        self.with_resid_frmt = "{:6s} {:>5} {:>5} {:^4} {:3} {:>5} {:>8} {:>8} {:>8} {:>6} {:>6}\n"
        # cast to str if num:
        self.titr_pt = str(titr_pt) if val_is_numeric(titr_pt) else titr_pt
        self.occ_df = self.get_occ_df()
        self.uniq_res = self.occ_df["res"].unique().tolist()

        return

    def get_occ_df(self) -> pd.DataFrame:
        """Get processed fort.38 in a pandas.DataFrame.
        Note:
        The last column 'keep' is a boolean flag indicating which confid & res entry to keep;
        If AverPQR.pqr_kind is 'average', the confids that are flagged with keep=True will be
        the first neutral conformers of acidic residues and the first '+1' conformers of basic
        residues.
        """
        def extract_res(ro: pd.Series):
            """Extract the res id from confid including 'DM' if found.
            Function for pd.DataFrame.apply.
            """
            if ro["conformer"][3:5] == "DM":
                return ro["conformer"][:5]
            return ro["conformer"][:3]

        df = pd.read_csv(self.f38_fp, sep=r"\s+")
        cols = df.columns.tolist()
        try:
            idx_titr = cols.index(self.titr_pt)
            self.pqr_fp = self.s2_fp.with_name(apqr_fname_frmt.format(kind=self.pqr_kind, titr_pt=self.titr_pt))
        except ValueError:
            pts = " ".join(cols[1:])
            logger.critical("Titration point %s is not in fort.38; Existing point(s): %s", self.titr_pt, pts)
            sys.exit(1)

        df = df.rename(columns={cols[0]:"conformer", cols[idx_titr]:"occ"})
        df["res"] = df.apply(extract_res, axis=1)
        df["confid"] = df["conformer"].str[5:]
        df["resid"] = df["conformer"].str[5:-4]

        uniq_resids = df["resid"].unique().tolist()
        df["keep"] = False
        df = df[["conformer","resid","confid","res","occ","keep"]]

        dummies = set()
        # update the 'keep' column:
        for ur in uniq_resids:
            if ur in dummies:
                continue

            df1 = df.loc[df["resid"]==ur]
            if len(df1) == 1:
                df.loc[df1.index, "keep"] = True
                continue

            # res with multiple confids:
            # could be free cofactors (HOH, HOHDM), TER groups, or res split with name.txt
            uniq_res = df1["res"].unique().tolist()

            DMfound = uniq_res[-1][-2:]=="DM"
            if DMfound:  # free cofactor:
                dm_res = uniq_res[-1]
                mx = df1["occ"].idxmax()
                if df.loc[mx, "res"] == dm_res:
                    # DM has max occ, eliminate all
                    df.loc[df1.index]["keep"] = False
                else:
                    df.loc[mx, "keep"] = True
                dummies.add(ur)
                continue

            for ures in uniq_res:
                res_df = df1.loc[df1["res"]==ures]
                
                if self.pqr_kind != "average":
                    resx = res_df["occ"].idxmax()
                else:
                    # flag the 1st listed neutral or charged conf
                    if ures in CONF01_RES:
                        search_keep = f"{ures}01{ur}_"
                    else:
                        search_keep = f"{ures}+1{ur}_"
                    try:
                        resx = df.loc[df["conformer"].str.startswith(search_keep), "confid"].idxmin()
                    except IndexError as e:
                        logger.error("Could not get the conformer to keep for: %s; %s; %s", ur, ures, search_keep)
                        sys.exit()

                df.loc[resx, "keep"] = True

        if self.debug:
            # save to file with 'keep' flag indicating which confs are selected for pqr as per 'pqr_kind':
            occ_fp = self.mcce_dir.joinpath(occ_file_frmt.format(kind=self.pqr_kind, titr_pt=self.titr_pt))
            exists, removed = check_file(occ_fp, self.replace, do_exit=False)
            if (exists, removed) != (True, False):
                df.to_csv(occ_fp, sep="\t")
                logger.info("Saved the occupancy dataframe as %s", str(occ_fp))

        return df

    def most_occ_to_pqr(self):
        """Create the pqr file of the most occ'd conformers.
        """
        df_keep = self.occ_df.loc[self.occ_df["keep"] == True]   # most occ'd confs only
        lines_out = []
        lines_conf = []

        new_pair = None, None  # conf, res
        new_found = False
        occ = 0
        with open(self.s2_fp) as s2:
            for line in s2:
                line = line.strip()
                rec, seq, atm, alt, res, conf, x, y, z, rad, crg, _ = parse_mcce_line(line)
                resnum = str(int(conf[1:-4]))

                if conf.endswith("_000") and (res[0] != "_"):
                    # write pqr line for BK as is
                    lines_out.append(pqr_frmt.format(rec, seq, atm, res, resnum, x, y, z, crg, rad))
                    if self.debug:
                        lines_conf.append(self.with_resid_frmt.format(rec,conf[:5],seq,atm,res,resnum,x,y,z,crg,rad))
                else:
                    pair = conf, res
                    if pair != new_pair:
                        new_pair = pair
                        # maybe get occ to keep until new pair
                        msk = (df_keep["confid"]==conf) & (df_keep["res"]==res)
                        ok_df = df_keep.loc[msk]
                        if ok_df.shape[0]:
                            occ = ok_df["occ"].values[0]
                            new_found = True
                        else:
                            new_found = False

                    if (pair == new_pair) and new_found:
                        q = f"{float(crg)*occ:.3f}"
                        lines_out.append(pqr_frmt.format(rec, seq, atm, res, resnum, x, y, z, q, rad))
                        if self.debug:
                            lines_conf.append(self.with_resid_frmt.format(rec,conf[:5],seq,atm,res,resnum,x,y,z,q,rad))

        logger.debug(f"Number of lines to write: {len(lines_out):,}")
        with open(self.pqr_fp, "w") as pqr:
            pqr.writelines(lines_out)
        logger.info("Saved the most occupied conformers pqr file: %s", str(self.pqr_fp))

        if self.debug:
            self.pqr_with_resids_fp = self.pqr_fp.with_name(interim_fname_frmt.format(kind=self.pqr_kind, titr_pt=self.titr_pt))
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
        new_pair_res = None, None  # resid, res
        new_found = False
        keep_conf = ""
        s2_lines = []

        logger.info("Processing each line in step2_out.pdb...")
        with open(self.s2_fp) as s2:
            for lx, line in enumerate(s2, start=1):
                line = line.strip()
                # fields for pqr: rec, seq, atm, res, resnum, x, y, z, crg, rad
                rec, seq, atm, alt, res, conf, x, y, z, rad, crg, _ = parse_mcce_line(line)
                resnum = str(int(conf[1:-4]))
                resid = conf[:5]
                x, y, z, rad, crg = float(x), float(y), float(z), float(rad), float(crg)

                if conf.endswith("_000") or res[0] == "_":
                    keep_conf = conf
                    s2_lines.append([rec, seq, conf, keep_conf, res, resnum, atm,  x, y, z, rad, crg])
                else:
                    pair_res = resid, res
                    if pair_res != new_pair_res:
                        new_pair_res = pair_res
                        msk0 = (self.occ_df["resid"]==resid) & (self.occ_df["res"]==res) & (self.occ_df["keep"]==True)
                        keep_df = self.occ_df.loc[msk0]
                        if keep_df.shape[0]:
                            keep_conf = keep_df["confid"].values[0]
                        else:
                            keep_conf = "?"

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

                    if (pair == new_pair) and new_found:
                        wq = crg*occ
                        s2_lines.append([rec, seq, conf, keep_conf, res, resnum, atm, x,y,x,rad,wq])

        logger.info(f"Number of s2 lines processed = {lx:,}")
    
        cols = "rec,seq,conf,keep,res,resnum,atm,x,y,z,rad,crg".split(",")
        s2_df = pd.DataFrame(s2_lines, columns=cols)
        s2_df["resid"] = s2_df["conf"].str[:5]

        logger.info("Summing the weighted charges...")
        grp1 = s2_df.groupby(["rec", "resid","res","atm"]).agg({"crg":"sum"})
        wq_df = grp1.reset_index(drop=False)
        wq_df.columns = ["rec","resid","res","atm","wq"]

        logger.info("Averaging the coordinates...")
        grp2 = s2_df.groupby(["rec","resid","res","atm"]).agg({"x":"mean", "y":"mean", "z":"mean"})
        xyz_df = grp2.reset_index(drop=False)

        logger.info("Merging the results...")
        wq_merg = s2_df.merge(wq_df, left_on=["rec","resid","res","atm"],
                            right_on=["rec","resid","res","atm"])
        wq_merg.drop(["x","y","z"], axis=1, inplace=True)
        wq_merg = wq_merg.loc[wq_merg["conf"]==wq_merg["keep"]]

        all_merg = wq_merg.merge(xyz_df, left_on=["rec","resid","res","atm"],
                                right_on=["rec","resid","res","atm"])
        all_merg = all_merg.loc[all_merg["conf"]==all_merg["keep"]]

        logger.info(f"Writing {all_merg.shape[0]:,} pqr lines...")
        pqr2 = None
        if self.debug:
            self.pqr_with_resids_fp = self.pqr_fp.with_name(interim_fname_frmt.format(kind=self.pqr_kind, titr_pt=self.titr_pt))
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
                    pqr2.write(self.with_resid_frmt.format(ro.rec, ro.resid, ro.seq, ro.atm, ro.res,
                                                           ro.resnum,f"{ro.x:.3f}", f"{ro.y:.3f}",
                                                           f"{ro.z:.3f}", f"{ro.wq:.3f}", ro.rad)
                    )
        logger.info("Saved the occ-weighted charge pqr file: %s", str(self.pqr_fp))
        if pqr2 is not None:
            pqr2.close()
            #logger.info(logmsg_save_interm, str(self.pqr_with_resids_fp))

        return

    def write_res_sumcrg_from_pqr(self):
        """List the residues sum charge from the pqr file for comparing with sum_crg.out."""
        if not self.debug:
            return
        
        self.sumcrg_fp = self.pqr_fp.with_name(sumcr_fname_frmt.format(kind=self.pqr_kind, titr_pt=self.titr_pt))
        # this point is reached if debug, so quit if file exists and not replaced:
        _ = check_file(self.sumcrg_fp, self.replace)  # default: do_exit=True

        hdr = "rec,resid,seq,atm,res,resnum,x,y,z,crg,rad".split(",")
        pqrdf = pd.read_csv(self.pqr_with_resids_fp, sep='\s+', header=None, names=hdr)
        lines = []
        new_pair = None, None
        neutral = NEUTRAL_RES + ["CYD"]
        for _, ro in pqrdf.iterrows():
            if ro["res"] in neutral:
                continue
            rid = ro["resid"]
            res = ro["res"]
            if (rid, res) == new_pair:
                continue
            lines.append(f'{rid} {res} {pqrdf.loc[(pqrdf["resid"]==rid) & (pqrdf["res"]==res)]["crg"].sum():6.2f}\n')
            new_pair = rid, res

        with open(self.sumcrg_fp, "w") as sumcrg:
            sumcrg.writelines(lines)
        logger.info("Saved the residue sum charge from the pqr file: %s", str(self.sumcrg_fp))

        return

    def check_sumcrg(self):
        if not self.debug:
            return
        if not self.do_sumcrg_check:
            return
        
        def add_resid_res_cols(row):
            return row["conf"][4:-1], row["conf"][:3]
        
        sumdf = pd.read_csv(self.sumcrg_fp, sep="\s+", header=None, names=["resid", "res", "wq"])
        sumdf.set_index(["resid", "res"], inplace=True)
     
        sumout = pd.read_csv(self.sumcrg_out_fp, sep="\s+", header=0, skipfooter=4, engine="python")
        cols = sumout.columns.tolist()
        prec = 1
        hdr_is_float = "." in cols[1]
        if hdr_is_float:
            prec = len(cols[1].split(".")[1])

        try:
            idx_titr = cols.index(self.titr_pt)
        except ValueError:
            if hdr_is_float:
                # maybe try a float with same prec:
                try:
                    f_titr = f"{float(self.titr_pt):.{prec}f}"
                    idx_titr = cols.index(f_titr)
                except ValueError:
                    logger.critical("Titration point %s is not in sum_crg.out. Exiting.", f_titr)
                    return
            else:
                # maybe try integer:
                try:
                    n_titr = int(float(self.titr_pt))
                    idx_titr = cols.index(str(n_titr))
                except ValueError:
                    logger.critical("Titration point %s is not in sum_crg.out. Exiting.", self.titr_pt)
                    return

        cols[0] = "conf"
        cols[idx_titr] = "crg"
        sumout.columns = cols
        sumout["crg"] = pd.to_numeric(sumout["crg"])
        sumout[["resid","res"]] = sumout.apply(add_resid_res_cols, axis=1, result_type='expand')
        sumout.drop("conf", axis=1, inplace=True)
        sumout.set_index(["resid","res"], inplace=True)

        df = sumout.merge(sumdf, left_on=sumout.index, right_on=sumdf.index)
        df["diff"] = abs(df["wq"]-df["crg"])
        dfnz = df.loc[df["diff"]>0.0101]
        if len(dfnz):
            logger.warning("Some sum_crg.out charges differ from the pqr file sum charge:\n%s\n", dfnz)
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

    if args.most_occ_pqr:
        pqr_kind = "most_occ"
    else:
        pqr_kind = "average"

    apqr = AverPQR(args.mcce_dir, args.titr_pt, pqr_kind = pqr_kind,
                   replace=args.replace,
                   debug=args.debug)
    
    logger.info("Creating the pqr file...")
    if args.most_occ_pqr:
        apqr.most_occ_to_pqr()
    else:
        apqr.s2_to_aver_pqr()

    apqr.write_res_sumcrg_from_pqr()
    if apqr.do_sumcrg_check:
        apqr.check_sumcrg()

    logger.info("Average pqr file creation over for '%s'\n", Path(args.mcce_dir).absolute().name)

    return


if __name__ == "__main__":
    get_aver_pqr_cli(sys.argv)
