#!/usr/bin/env python3

"""
Module: correlator.py

Codebase: mcce4/ms_hbnets/correlator.py

Handle the filtering and setup for generating heatmaps.
The states correlation heatmaps are obtain using parallel processing.
"""
from argparse import Namespace
from collections import defaultdict
import logging
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat
from re import split as re_split, findall, search
from tempfile import TemporaryDirectory
from time import time
from turtle import width
from typing import Tuple, Union

import numpy as np
import pandas as pd

from mcce4.constants import res3_to_res1
from mcce4.io_utils import show_elapsed_time
from mcce4.ms_hbnets.msout_hb import get_resid, group_confids_into_resids
from mcce4.ms_hbnets.msout_hb import MIN_OCC
from mcce4.ms_hbnets.msout_hb import MSout_hb
from mcce4.ms_hbnets.plotting import DEFAULT_FIGSIZE
from mcce4.ms_hbnets.plotting import heatmap_from_df


DTYPE = "float32"
HB_KINDS = ["pairs", "states"]  # matches ms_hbnets subcommands
VALID_AA1_CODES = [val for val in set(res3_to_res1.values()) if len(val) == 1]
RES_COFACTORS_SEP = "|"
VALID_COFACTOR_SEPS = " .,"
GROTTHUS_RES = "DESTCYHK"   # short list
GROTTHUS_RES3 = "NQR"       # long list


def get_file_header(file_fp: Union[str, Path]) -> str:
    """Return the first line of a file.
       In this context: a commented, informative line.
    """
    with open(file_fp) as fh:
        hdr = fh.readline().strip()
    if hdr.startswith("#"):
        return hdr
    else:
        return ""


def format_num(num):
    """Return condensed format for plots, up to millions.
    """
    M = 1_000_000
    K = 1_000
    if num > M:
        return f"{num/M:.4g}M"
    elif num > K:
        return f"{num/K:.4g}K"
    return num


def get_resnum(resid: str):
    """Get the res num from the resid in hb_pairs_res_pH*.csv  or hb_states_pairs_pH*.csv files.
    Examples:
      Aw1016 -> 1016; A_CL1 -> 1; _K90 -> 90; ADBK52 -> 52
    """
    match = search(r"\d+$", resid)
    if match:
        return int(match.group())
    else:
        logging.warning(f"Seq number of {resid = } not found: assigned 0.")
        return 0


def get_res_of_interest(res_of_interest: str, grotthus: bool = False) -> Tuple[str]:
    """Loads user's res of interest from input string.
    Outputs a tuple so that it can be used with str.startswith.
    Examples: 
      res_of_interest = "DEH+BK"       -> (D, DBK, E, EBK, H, HBK)
      res_of_interest = "DEH+|HOH PL9" -> (D, DBK, E, EBK, H, HBK, w, PL9)
      res_of_interest = "DEH|HOH.pl9"  -> (D, E, H, w, PL9)
      res_of_interest = "|HOH,PL9"     -> (w, PL9)
      res_of_interest = "|HOH,PO4"     -> (w, PO4)

    Residues and cofactors if grotthus is True:
    Short list: DESTCYHK|HOH
    Short list + any of the 3 add'l res in long list (NQR): DESTCYHKNQR|HOH
    + add'l non-water cofactors
    """
    if not res_of_interest and not grotthus:
        return ()

    out = []
    if not res_of_interest and grotthus:
        out.extend([f"{r}" for r in GROTTHUS_RES])
        out.append("w")
        return tuple(out)

    roi = res_of_interest.upper()
    add_bk = "+" in roi
    if add_bk and grotthus:
        roi = roi.replace("+","")
        add_bk = False

    if add_bk:
        ix = roi.index("+")
    else:
        ix = len(roi)

    add_hets = RES_COFACTORS_SEP in roi
    if add_hets:
        roi, het = roi.split(RES_COFACTORS_SEP)
        if "HOH" in het:
            if not grotthus:
                het = het.replace("HOH", "w")
            else:
                het = het.replace("HOH","")

    if not grotthus:
        # validate 1-letter codes:
        all_valid = all([res in VALID_AA1_CODES for res in roi[:ix]])
        if not all_valid:
            logging.error(f"Found invalid 1-letter residue code(s) in {roi[:ix]!r}. Not filtering.")
            return None
        else:
            # ok to populate list
            out.extend([f"{r}" for r in roi[:ix]])
            if add_bk:
                out.extend([f"{r}BK" for r in roi[:ix]])
            if add_hets:
                out.extend([f"{h}" for h in re_split(rf"[{VALID_COFACTOR_SEPS}]+", het)])
    else:
        out.extend([f"{r}" for r in GROTTHUS_RES])
        if roi[:ix]:
            all_valid = [res for res in roi[:ix] if res in GROTTHUS_RES3]
            if not all_valid:
                logging.warning(f"Ignoring additonal residue(s) in -res-of-interest (not N, Q or R).")
            else:
                # ok to populate user list
                out.extend([f"{r}" for r in all_valid])
        out.append("w")
        if add_hets and het:
            out.extend([f"{h}" for h in re_split(rf"[{VALID_COFACTOR_SEPS}]+", het)])

    return tuple(out)


def get_pairs_of_interest_df(pairs_of_interest_fp: Union[str, Path]) -> pd.DataFrame:
    """Loads user's pairs of interest from input file.
    Assumed format example per line: AR112,AN106
    Previous format was: R_A112,N_A106
    """
    poi_fp = Path(pairs_of_interest_fp)
    if not poi_fp.exists():
        logging.warning(f"File not found: {pairs_of_interest_fp!s}")
        return None

    poi_df = pd.read_csv(poi_fp, comment="#", names=["res_d", "res_a"])
    if not poi_df.shape[0]:
        logging.warning(f"Empty 'pairs of interest' file: {pairs_of_interest_fp!s}")
        return None

    return poi_df


def maybe_convert_condensed_ids(df: pd.DataFrame) -> bool:
    """Input dataframe df is loaded with either hb_pairs_res_*.csv or
    hb_states_pairs_*.csv data.
    New format: chain letter first, no underscore.
    """
    # check previous format: with underscore
    mskp = df["res_d"].str.contains("_") | df["res_a"].str.contains("_")

    return bool(mskp.any())


def num_prec(num: Union[float, int]) -> int:
    if isinstance(num, int):
        return 0
    num_str = str(num).lower()
    if 'e' in num_str:
        return int("".join(c for c in num_str.split('e')[1] if c.isnumeric()))
        
    if '.' in num_str:
        return len(num_str.split('.')[1])
    return 0


class HbCorrelator:
    def __init__(self, hbms: MSout_hb,
                 hb_kind: str = None,
                 res_of_interest: str = None,
                 grotthus: bool = False,
                 pairs_of_interest_fp: Union[str, Path] = None,
                 min_occ: float = MIN_OCC,
                 include_states_da_corr: bool = False,
                 split_BK: bool = False,
                 figs_args: Union[dict,Namespace] = None,
                ):
        """ 
        Args:
         - hb_kind (str, None): One of HB_KINDS=["pairs", "states"]; if None -> "pairs"
         - pairs_of_interest_fp ([str, Path], None): Obtain the correlation of a pair of 
            hb residues (comma-separated residue hb pairs).
         - grotthus (bool, False): Filter for Grotthuss-competent residues
         - min_occ (float, MIN_OCC): To change the default used by the msout data loader
         - include_states_da_corr (bool, False): If True, the states Donors and Acceptors correlation
           is calculated in addition to the pairs correlation.
         - split_BK (bool, False): If True, or no pairs of interest provided (no filtering), 
           separate correlation of the res-res and res-bk pairs will be computed.
         - figs_args([dict, argparse.Namespace]): To control the heatmap sizes, the lowest
           bound of the cbar and color of the data heatmaps, and pairs labels.
        """
        logging.info("HbCorrelator start...")
        self.ok = True
        if hb_kind not in HB_KINDS:
            logging.error(f"ValueError: hb_kind must be one of {HB_KINDS}")
            self.ok = False
            return
        self.hb_kind = hb_kind

        inputs = self.get_intput_files(hbms)
        if not self.ok:
            return
        self.input_fp1, self.input_fp2 = inputs
        self.file_hdr = get_file_header(self.input_fp1)

        self.mcce_dir = hbms.run_dir
        self.pheh_str = hbms.pheh_str
        self.min_occ = min_occ
        self.prec = num_prec(float(self.min_occ))
        if self.min_occ == MIN_OCC:
            self.min_occ_print = f"min occ > {self.min_occ:.{self.prec}f}"
        else:
            self.min_occ_print = f"min occ >= {self.min_occ:.{self.prec}f}"
        logging.info(f"Heatmaps min_occ: {self.min_occ_print}")

        self.grotthus = grotthus
        self.res_oi: tuple = None
        self.res_oi = get_res_of_interest(res_of_interest, self.grotthus)
        if self.res_oi is None:
            self.ok = False
            return

        self.poi_df: pd.DataFrame = None
        if pairs_of_interest_fp is not None:
            self.poi_df = get_pairs_of_interest_df(pairs_of_interest_fp)
            if self.poi_df is None:
                self.ok = False
                return

        # Populated by get_pairs_data :
        self.n_rows: int = 0
        self.n_cols: int = 0
        # defined for hb states:
        # dict to obtain the matrix index of the resid
        self.res2mat_ix: dict = None
        # dict to rename the corr matrix rows & columns:
        self.mat_ix2res: dict = None
        self.n_matrices: int = 0
        self.corr_matrix: np.ndarray = None

        self.corr_kind = "pairs" if self.hb_kind == "states" else "da"
        self.corr_wise: str = "col"
        self.split_kind: str = None
        self.include_states_da_corr = include_states_da_corr
        self.mmap_fps: dict = None
        self.dat_fps: dict = None

        self.pairs_df = None  # main df, ready for splitting if needed
        self.filtered = False
        # filtered with res_of_interest or pairs_of_interest_fp is given & min_occ:
        self.pairs_df = self.get_inputfile_pairs_data()
        if not self.ok:
            return

        self.split_bk = True
        if self.grotthus:
            self.split_bk = False
        else:
            if self.filtered:
                self.split_bk = split_BK

        if self.split_bk:
            logging.info("The hb pairs will be divided into res-res and res-BK pairs to reduce the heatmap size.")
  
        self.figure_captions = dict()
        if isinstance(figs_args, dict):
            self.figs_args = Namespace(**figs_args)
        else:
            self.figs_args = figs_args

        return

    def get_intput_files(self, hbms: MSout_hb) -> Union[tuple, None]:
        """
        Files for 'pairs' and 'states' processing
        """
        if self.hb_kind == "pairs":
            fp1 = hbms.pairs_csv
            fp2 = hbms.pairs_res_csv
        else:
            fp1 = hbms.states_csv
            fp2 = hbms.states_pairs_csv

        if fp1.exists() and fp2.exists():
            return fp1, fp2
        else:
            logging.error(f"Missing file(s): At least one of the csv was not found: {fp1.name}, {fp2.name}")
            self.ok = False

        return None

    def _redo_grouping_into_resid(self) -> pd.DataFrame:
        """Redo the grouping of confid-based pairs into resid-based
        pairs to update the new format.
        Called by _load_csv.
        """
        df = pd.read_csv(self.input_fp1, comment="#")
        if self.hb_kind == "states":
            # get the unique hb pairs in all the returned hb microstates:
            states_pairs_dict = defaultdict(lambda: [0, 0.])
            for _, ro in df.iterrows():
                tpls = set()
                # split state_id into pairs; exclude first '('
                for tpl in [val.split(",") for val in re_split(r",\(", ro["state_id"][1:])]:
                    tpl[1] = tpl[1][:-1]  # exclude trailing ')'
                    tpls.add((get_resid(tpl[0]), get_resid(tpl[1])))

                for tpl in tpls:
                    # get the effective count of each pair in the state key
                    states_pairs_dict[tpl][0] += ro["count"]
                    states_pairs_dict[tpl][1] += ro["occ"]

            # prep hb states pairs output:
            df = pd.DataFrame.from_dict(states_pairs_dict,
                                         orient="index",
                                         columns=["count", "occ"]).reset_index()
            df[["res_d","res_a"]] = df["index"].apply(lambda x: pd.Series([x[0], x[1]]))
        else:
            # hb pairs file update:
            df[["res_d","res_a"]] = df.apply(
                lambda x: pd.Series([get_resid(x["donor"]), get_resid(x["acceptor"])]),
                axis=1)

        # common processing, up to saving, res format: ADBK52
        df["with_bk"] = df.apply(
            lambda row: row["res_d"][2:4]=="BK" or row["res_a"][2:4]=="BK", axis=1)
        df_res = group_confids_into_resids(df)

        # if self.hb_kind == "states":
        #     with open(self.input_fp2, "w") as fo:
        #         fo.write(STATES_PAIRS_COMMENT)
        #         df_res.to_csv(fo, index=False)
        # else:
        df_res.to_csv(self.input_fp2, index=False)
        logging.info(f"Updated the format of {self.input_fp2.name}")

        return df_res
    
    def _load_csv(self, csv_fp: Path) -> pd.DataFrame:
        """Load state_pairs or pairs_res csv and maybe update the csv format.
        """
        # must be hb_states_pairs or hb_pairs_res:
        file_check = csv_fp.name == self.input_fp2.name
        if not file_check:
            logging.error("Data file to load must be either hb_states_pairs_*.csv or hb_pairs_res_*.csv")
            self.ok = False
            return

        redo = False
        df = pd.read_csv(csv_fp, comment="#")

        # previous format has underscore
        if maybe_convert_condensed_ids(df):
            df = self._redo_grouping_into_resid()  # & save updated file
            return df

        # older format:
        if "res_d" in df.columns:
            msk_hoh = (df["res_d"].str.startswith("HOH") | df["res_a"].str.startswith("HOH"))
            redo = msk_hoh.any()
        else:
            redo = True
        redo = redo or ("Mi" in df.columns) or ("index" in df.columns)
        if redo:
            df = self._redo_grouping_into_resid()  # & save updated file
            return df

        return df

    def _filter_res_of_interest(self, resdf: pd.DataFrame) -> pd.DataFrame:
        if not self.ok:
            return

        if self.res_oi:
            try:
                if self.grotthus:
                    # no res-bk
                    msk_d = (~resdf["res_d"].str.contains("BK") &
                             resdf["res_d"].str[1:].str.startswith(self.res_oi)) 
                    msk_a = (~resdf["res_a"].str.contains("BK") &
                             resdf["res_a"].str[1:].str.startswith(self.res_oi))
                    msk = msk_d & msk_a
                else:
                    msk = (resdf["res_d"].str[1:].str.startswith(self.res_oi) | 
                           resdf["res_a"].str[1:].str.startswith(self.res_oi))
                if msk.any():
                    resdf = resdf.loc[msk]
                    logging.info(f"Residue-filtered pairs count: {resdf.shape[0]}")
                    return resdf
                else:
                    return None
            except Exception as e:
                logging.error(f"Could not filter the dataframe for residues of interest. Error:\n{e}")
                self.ok = False
                return None
        else:
            return resdf

    def _filter_pairs_of_interest(self, resdf: pd.DataFrame) -> pd.DataFrame:
        if not self.ok:
            return

        if self.poi_df is not None:
            try:
                msk = (resdf[["res_d", "res_a"]].apply(tuple, axis=1)
                        .isin(self.poi_df[["res_d", "res_a"]].apply(tuple, axis=1))
                )
                if msk.any():
                    resdf = resdf.loc[msk]
                    logging.info(f"Filtered pairs count: {resdf.shape[0]}")
                    return resdf
                else:
                    logging.warning(f"Found no matches for the given pairs of interest: {self.poi_df!s}")
                    self.ok = False
                    return None
            except Exception as e:
                logging.error(f"Could not filter the dataframe for pairs of interest. Error:\n{e}")
                self.ok = False
                return None
        else:
            return resdf

    def get_inputfile_pairs_data(self) -> pd.DataFrame:
        """
        The states pairs file contains the unique pairs over the hb space returned.
        The pairs from the pairs_res file are the most occupied overall.
        These pairs define the size of the residue-based stored matrices.

        Prep of the main df: filtered for res oi, paits oi & occ.
        """
        if not self.ok:
            return
        logging.info("Getting pairs data...")

        resdf = self._load_csv(self.input_fp2)
        # maybe filter:
        if self.res_oi:
            df = self._filter_res_of_interest(resdf)
            if df is not None:
                resdf = df
                self.filtered = True
                if self.grotthus:
                    logging.info(f"Data filtered for Grotthuss res of interest: {self.res_oi}")
                else:
                    logging.info(f"Data filtered for res of interest: {self.res_oi}")
            else:
                self.ok = False
                return
        
        if self.poi_df is not None:
            df = self._filter_pairs_of_interest(resdf)
            if df is not None:
                resdf = df
                self.filtered = True
                logging.info("Data filtered for pairs of interest")
            else:
                self.ok = False
                return

        # apply min occ mask if occ not default:
        #resdf["occ"] = resdf["occ"].round(self.prec)
        if self.min_occ != MIN_OCC:
            occ_msk = resdf["occ"].ge(self.min_occ)
            if occ_msk.any():
                resdf = resdf.loc[occ_msk]
                self.filtered = True
            else:
                logging.warning(f"No hb pairs data @ {self.min_occ_print}")
                self.ok = False
                return

        # sort by resnum:
        resdf["di"] = resdf["res_d"].apply(get_resnum)
        resdf["ai"] = resdf["res_a"].apply(get_resnum)
        resdf = resdf.sort_values(by=["di","ai"])
        resdf = resdf.drop(columns=["di","ai"])
        resdf = resdf.rename(columns={"res_d":"Donor", "res_a":"Acceptor"})

        if self.filtered:
            resdf.to_csv(self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered.csv"),
                         index=False)
        return resdf

    def get_data_heatmaps(self, kind: str="pairs") -> Union[None, tuple]:
        """Create the data heatmap(s) from pairs in hb_states_pairs_ or
        hb_pairs_res_ csv files.

        Called by process_pairs, process_states.
        If split was done, outputs res_respairs_df, bk_respairs_df.
        """
        if not self.ok:
            return

        # these dfs maybe loaded if split was done:
        res_respairs_df = None
        bk_respairs_df = None
        if not self.split_bk:
            matrix_df = self.pairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
            # data heatmap:
            if self.filtered:
                if self.grotthus:
                    title = f"H-bond Donor - Acceptor pairs data (Grotthuss filtered)\n{self.input_fp2.name}"
                else:
                    title = f"H-bond Donor - Acceptor pairs data (filtered)\n{self.input_fp2.name}"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered.png")
            else:
                title = f"H-bond Donor - Acceptor pairs data\n{self.input_fp2.name}"
                png_fp = self.input_fp2.with_suffix(".png")
            self.figure_captions[str(png_fp)] = title
            heatmap_from_df(matrix_df,
                            fig_save_fp=png_fp,
                            title=title,
                            fig_size=self.figs_args.figsize_data,
                            min_bound=self.figs_args.data_cbar_min,
                            color=self.figs_args.data_map_color,
                            publication=self.figs_args.publication,
                            ) 
        else:
            # split non-BK, then BK:
            res_msk = self.pairs_df["with_bk"].eq(False)
            if res_msk.any():
                res_respairs_df = self.pairs_df.loc[res_msk, ["Donor", "Acceptor", "count", "occ"]]
                n_res_ud = len(res_respairs_df["Donor"].unique())
                n_res_ua = len(res_respairs_df["Acceptor"].unique())
                logging.info(f"{kind.capitalize()} res-res pairs, uniq donors: {n_res_ud}, uniq acceptors: {n_res_ua}")
        
                matrix_df = res_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                # data heatmap:
                if self.filtered:
                    title = f"H-bond Donor - Acceptor pairs data, res-res (filtered)\n{self.input_fp2.name}"
                else:
                    title = f"H-bond Donor - Acceptor pairs data, res-res\n({self.input_fp2.name})"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_res.png")
                self.figure_captions[str(png_fp)] = title
                heatmap_from_df(matrix_df,
                                fig_save_fp=png_fp,
                                title=title,
                                fig_size=self.figs_args.figsize_data,
                                min_bound=self.figs_args.data_cbar_min,
                                color=self.figs_args.data_map_color,
                                publication=self.figs_args.publication,
                                )
            else:
                logging.info(f"{kind.upper()}: No hb pairs of res-res kind in {self.input_fp2.name}.")

            # now BK:
            if (~res_msk).any():
                bk_respairs_df = self.pairs_df.loc[~res_msk, ["Donor", "Acceptor", "count", "occ"]]
                n_bk_ud = len(bk_respairs_df["Donor"].unique())
                n_bk_ua = len(bk_respairs_df["Acceptor"].unique())
                logging.info(f"{kind.capitalize()} res-bk pairs: uniq donors: {n_bk_ud} uniq acceptors: {n_bk_ua}")

                matrix_df = bk_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                # data heatmap:
                if self.filtered:
                    title = f"H-bond Donor - Acceptor pairs data, res-bk (filtered)\n{self.input_fp2.name}"
                else:
                    title = f"H-bond Donor - Acceptor pairs data, res-bk\n({self.input_fp2.name})"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_bk.png")
                self.figure_captions[str(png_fp)] = title
                heatmap_from_df(matrix_df,
                                fig_save_fp=png_fp,
                                title=title,
                                fig_size=self.figs_args.figsize_data_bk,
                                min_bound=self.figs_args.data_cbar_min,
                                color=self.figs_args.data_map_color,
                                publication=self.figs_args.publication,
                                )
            else:
                logging.info(f"{kind.upper()}: No pairs of res-bk kind in {self.input_fp2.name}.")
            return res_respairs_df, bk_respairs_df

        return

    def process_pairs(self):
        """Uses pandas for correlations.
        """
        logging.info(f"OK to process pairs? {self.ok}")
        if not self.ok:
            return

        process_start = time()
        # splitting done by data viz function:
        split_dfs = self.get_data_heatmaps()
        if split_dfs is not None:
            res_respairs_df, bk_respairs_df = split_dfs

        if not self.split_bk:
            # data heatmap:
            matrix_df = self.pairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)

            # Get donors and Acceptors correlation:
            # Transpose and compute Donor (row) correlation
            row_corr_mat = matrix_df.T.corr()
            if row_corr_mat.shape[0] < 2:
                logging.info("Not enough rows for correlation.")
            else:
                if self.filtered:
                    if self.grotthus:
                        title = "H-bond Donors correlation (Grotthuss filtered)"
                    else:
                        title = "H-bond Donors correlation (filtered)"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_donors.png")
                else:
                    title = "H-bond Donors correlation"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_donors.png")
                self.figure_captions[str(png_fp)] = title
                heatmap_from_df(row_corr_mat,
                                fig_save_fp=png_fp,
                                title=title,
                                map_kind="corr",
                                fig_size=self.figs_args.figsize_donor_corr,
                                min_bound=self.figs_args.data_cbar_min,
                                color=self.figs_args.data_map_color,
                                publication=self.figs_args.publication,
                                )
            
            # Compute Acceptor (col) correlation
            col_corr_mat = matrix_df.corr()
            if col_corr_mat.shape[0] < 2:
                logging.info("Not enough columns for correlation.")
            else:
                if self.filtered:
                    if self.grotthus:
                        title = "H-bond Acceptors correlation (Grotthuss filtered)"
                    else:
                        title = "H-bond Acceptors correlation (filtered)"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_acceptors.png")
                else:
                    title = "H-bond Acceptors correlation"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_acceptors.png")
                self.figure_captions[str(png_fp)] = title
                heatmap_from_df(col_corr_mat,
                                fig_save_fp=png_fp,
                                title=title,
                                map_kind="corr",
                                fig_size=self.figs_args.figsize_acceptor_corr,
                                min_bound=self.figs_args.data_cbar_min,
                                color=self.figs_args.data_map_color,
                                publication=self.figs_args.publication,
                                )
        else:
            # split non-BK, then BK:
            if res_respairs_df is not None:
                n_ud = len(res_respairs_df["Donor"].unique())
                n_ua = len(res_respairs_df["Acceptor"].unique())
                logging.info(f"RES-RES pairs, uniq donors: {n_ud}, uniq acceptors: {n_ua}")
                # data heatmap:
                matrix_df = res_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)

                # get the 2 kinds or corr:
                row_corr_mat = matrix_df.T.corr()
                if row_corr_mat.shape[0] < 2:
                    logging.info("Not enough rows for correlation.")
                else:
                    if self.filtered:
                        title = "H-bond Donors correlation, res-res (filtered)"
                        png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_donors_res.png")
                    else:
                        title = "H-bond Donors correlation, res-res"
                        png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_donors_res.png")
                    self.figure_captions[str(png_fp)] = title
                    heatmap_from_df(row_corr_mat,
                                    fig_save_fp=png_fp,
                                    title=title,
                                    map_kind="corr",
                                    fig_size=self.figs_args.figsize_donor_corr,
                                    min_bound=self.figs_args.data_cbar_min,
                                    color=self.figs_args.data_map_color,
                                    publication=self.figs_args.publication,
                                    )
                
                col_corr_mat = matrix_df.corr()
                if col_corr_mat.shape[0] < 2:
                    logging.info("Not enough columns for correlation.")
                else:
                    if self.filtered:
                        title = "H-bond Acceptors correlation, res-res (filtered)"
                        png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_acceptors_res.png")
                    else:
                        title = "H-bond Acceptors correlation, res-res"
                        png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_acceptors_res.png")
                    self.figure_captions[str(png_fp)] = title
                    heatmap_from_df(col_corr_mat,
                                    fig_save_fp=png_fp,
                                    title=title,
                                    map_kind="corr",
                                    fig_size=self.figs_args.figsize_acceptor_corr,
                                    min_bound=self.figs_args.data_cbar_min,
                                    color=self.figs_args.data_map_color,
                                    publication=self.figs_args.publication,
                                    )
            else:
                logging.info("No hb pairs of residue-residue kind.")

            # now BK:
            # if (~res_msk).any():
            #     bk_respairs_df = self.pairs_df.loc[~res_msk, ["Donor", "Acceptor", "count", "occ"]]
            if bk_respairs_df is not None:
                n_bk_ud = len(bk_respairs_df["Donor"].unique())
                n_bk_ua = len(bk_respairs_df["Acceptor"].unique())
                logging.info(f"RES-BK pairs, uniq donors: {n_bk_ud} uniq acceptors: {n_bk_ua}")
                # data heatmap:
                matrix_df = bk_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                # get the 2 kinds or corr:
                row_corr_mat = matrix_df.T.corr()
                if row_corr_mat.shape[0] < 2:
                    logging.info("Not enough rows for correlation.")
                else:
                    if self.filtered:
                        title = "H-bond Donors correlation, res-bk (filtered)"
                        png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_donors_bk.png")
                    else:
                        title = "H-bond Donors correlation, res-bk"
                        png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_donors_bk.png")
                    self.figure_captions[str(png_fp)] = title
                    heatmap_from_df(row_corr_mat,
                                    fig_save_fp=png_fp,
                                    title=title,
                                    map_kind="corr",
                                    fig_size=self.figs_args.figsize_donor_corr_bk,
                                    min_bound=self.figs_args.data_cbar_min,
                                    color=self.figs_args.data_map_color,
                                    publication=self.figs_args.publication,
                                    )
                
                col_corr_mat = matrix_df.corr()
                if col_corr_mat.shape[0] < 2:
                    logging.info("Not enough cols for correlation.")
                else:
                    if self.filtered:
                        title = "H-bond Acceptors correlation, res-bk (filtered)"
                        png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_acceptors_bk.png")
                    else:
                        title = "H-bond Acceptors correlation, res-bk"
                        png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_acceptors_bk.png")
                    self.figure_captions[str(png_fp)] = title
                    heatmap_from_df(col_corr_mat,
                                    fig_save_fp=png_fp,
                                    title=title,
                                    map_kind="corr",
                                    fig_size=self.figs_args.figsize_acceptor_corr_bk,
                                    min_bound=self.figs_args.data_cbar_min,
                                    color=self.figs_args.data_map_color,
                                    publication=self.figs_args.publication,
                                    )
            else:
                logging.info("No hb pairs of residue-backbone kind.")

        logging.info("Figure captions:\n" + pformat(self.figure_captions, sort_dicts=False, width=150) + "\n")
        logging.info("Processing of pairs correlation over.\n")
        logging.info(show_elapsed_time(process_start, info="H-bond pairs processing"))

        return

    # >>> Functions for states processing:
    def get_dims_dicts(self, df: pd.DataFrame):
        """
        State pairs processing.
        Get the df dimension and conversion dicts.
        Input df maybe a split df (res_df, bk_df).

        Sets these class attributes: n_rows, n_cols, res2mat_ix, mat_ix2res
        """
        if not self.ok:
            return 
        
        if self.hb_kind != "states":
            logging.info("ERROR: get_dims_dicts applies to 'states'")
            self.ok = False
            return
        
        if self.corr_kind == "da":
            uniq_res = list(set(df["Donor"]).union(df["Acceptor"]))
            n_uniq_res = len(uniq_res)
            # shape of matrix for each hb ms:
            self.n_rows, self.n_cols = n_uniq_res, n_uniq_res
            # dict to obtain the matrix index of the resid
            self.res2mat_ix = {res: rx
                               for rx, res in enumerate(sorted(uniq_res,
                                                               key=lambda x: int(x.rsplit("_",
                                                                                          maxsplit=1)[1][1:])))
                                }
            # dict to rename the corr matrix rows & columns:
            self.mat_ix2res = {v:k for k, v in self.res2mat_ix.items()}
            logging.info(f"States DA matrices shape: {self.n_rows}, {self.n_cols}")

        else:  # corr of pairs
            # Create a MultiIndex from the target columns
            df_index = pd.MultiIndex.from_frame(df[["Donor", "Acceptor"]])
            # whole matrix: (n_matrices, n_cols) (col-wise correl):
            self.n_cols = df_index.shape[0]
            self.n_rows = self.n_cols
            # dict to obtain the matrix index of each pair
            self.res2mat_ix = {pair: px for px, pair in enumerate(df_index.tolist())}
            # dict to rename the corr matrix rows & columns:
            # transform tuple to str: ('AH12', 'AV34') -> 'AH12|AV34'
            self.mat_ix2res = {v: f"{k[0]}|{k[1]}" for k, v in self.res2mat_ix.items()}
            logging.info(f"States PAIRS matrices columns: {self.n_cols}")
  
        return

    def _hb_ms2matrix(self, hbs, state_mat):
        """State pairs processing.
        Process a single microstate (row) of the hb_states file into a matrix.
        Used by save_hb_microstates_matrices.
        """
        # state res tuples to mat
        tpls = set()
        # split state_id into pairs
        for tpl in [val.split(",") for val in re_split(r",\(", hbs["state_id"][1:])]:
            tpl[1] = tpl[1][:-1]  # no trailing ")"
            tpls.add((get_resid(tpl[0]), get_resid(tpl[1])))
        for tpl in tpls:
            if self.corr_kind == "da":
                mi = self.res2mat_ix.get(tpl[0])
                if mi is None:
                    continue
                mj = self.res2mat_ix.get(tpl[1])
                if mj is None:
                    continue
                state_mat[mi, mj] = hbs["occ"]
            elif self.corr_kind == "pairs":
                pi = self.res2mat_ix.get(tpl)
                if pi is None:
                    continue
                state_mat[pi] = hbs["occ"]

        return state_mat    

    def save_hb_microstates_matrices(self):
        """State pairs processing.
        Save states microstates matrices to a memory-mapped binary file.
        """
        if not self.ok:
            return 
        # load ms_states file to df:
        dfs = pd.read_csv(self.input_fp1, comment="#")
        self.n_matrices = dfs.shape[0]
        if not self.n_matrices:
            logging.info("The hb states file is empty.")
            self.ok = False
            return

        occ_msk = dfs["occ"].ge(self.min_occ)
        dfs = dfs.loc[occ_msk].reindex()
        self.n_matrices = dfs.shape[0]
        if not self.n_matrices:
            logging.info(f"There are no occupied hb microstates @ {self.min_occ_print}.")
            self.ok = False
            return
        logging.info(f"Number of occupied states matrices: {self.n_matrices:,}.")

        # Initialize an empty binary file on disk to hold the stacked matrices
        if self.corr_kind == "da":
            logging.info("Creating stored matrix for Donors or Acceptors correlation")
            data_file = np.memmap(self.dat_fps["ms_matrices"], dtype=DTYPE, mode='w+',
                                  shape=(self.n_matrices, self.n_rows, self.n_cols))
            # Add each matrix sequentially into the memmap file:
            for i in range(self.n_matrices):
                data_file[i] = self._hb_ms2matrix(dfs.loc[i], np.zeros((self.n_rows, self.n_cols)))
        else:
            logging.info("Creating stored matrix for Pairs correlation")
            data_file = np.memmap(self.dat_fps["ms_matrices"], dtype=DTYPE, mode='w+',
                                    shape=(self.n_matrices, self.n_cols))
            # Add each matrix sequentially into the memmap file:
            for i in range(self.n_matrices):
                data_file[i] = self._hb_ms2matrix(dfs.loc[i], np.zeros((self.n_cols)))

        logging.info(f"Stored matrix shape: {data_file.shape}")
        # Flush changes to disk and clean up the write reference
        data_file.flush()
        del data_file

        return

    # --- Worker Functions ---
    def _process_row_chunk(self, r):
        """State pairs processing.
        Worker function to process a single row index.
        """
        # Re-open the memmap inside the worker process (safe for read-only)
        if self.corr_kind == "da":
            X_mmap = np.memmap(self.dat_fps["ms_matrices"], mode="r", dtype=DTYPE,
                                shape=(self.n_matrices, self.n_rows, self.n_cols))
            row_data = X_mmap[:, r, :].ravel()
            return r, row_data - np.mean(row_data)
        else:
            X_mmap = np.memmap(self.dat_fps["ms_matrices"], mode="r", dtype=DTYPE,
                               shape=(self.n_matrices, self.n_rows))
            # Extract the row across all matrices/columns and flatten to 1D
            row_data = X_mmap[r, :].ravel()
            # Return the centered row data
            return r, row_data - np.mean(row_data)

    def _process_col_chunk(self, c):
        """State pairs processing.
        Worker function to process a single column index.
        """
        # Re-open the memmap inside the worker process (safe for read-only)
        if self.corr_kind == "da":
            X_mmap = np.memmap(self.dat_fps["ms_matrices"], mode="r", dtype=DTYPE,
                               shape=(self.n_matrices, self.n_rows, self.n_cols))
            col_data = X_mmap[:, :, c].ravel()
            return c, col_data - np.mean(col_data)
        else:
            X_mmap = np.memmap(self.dat_fps["ms_matrices"], mode="r", dtype=DTYPE,
                               shape=(self.n_matrices, self.n_cols))
            col_data = X_mmap[:, c].ravel()
            return c, col_data - np.mean(col_data)

    def save_centered_matrices(self, centered_specs: dict):
        """State pairs processing.
        """
        if not self.ok:
            return 

        # Initialize a disk-backed memmap to collect the results from workers
        X_centered = np.memmap(centered_specs["centered_fp"],
                               shape=centered_specs["kind_shape"],
                               dtype=DTYPE, mode='w+')
        # Spin up a process pool using all available CPU cores
        with Pool() as pool:
            # imap_unordered is fast and memory-efficient as it streams results back
            for index, centered_vector in pool.imap_unordered(centered_specs["worker_fn"],
                                                              range(centered_specs["num_features"]),
                                                              chunksize=10):
                # Write the result directly into our output memmap
                X_centered[index] = centered_vector
        X_centered.flush()  # Ensure all writes are committed to disk
        del X_centered

        return

    def get_centered_specs(self) -> dict:
        """Depend on corr-kind, corr-wise
        """
        centered_specs: dict = None
        if self.corr_wise == "col":
            if self.corr_kind == "da":
                kind_shape = (self.n_cols, self.n_matrices * self.n_cols)
            else:
                kind_shape = (self.n_cols, self.n_matrices)

            centered_specs = {"num_features": self.n_cols,
                                "worker_fn": self._process_col_chunk,
                                "centered_fp": self.dat_fps["ms_matrices_cols"],
                                "kind_shape": kind_shape,
                                }
        else:  # self.corr_wise = "row"
            if self.corr_kind == "da":
                kind_shape = (self.n_rows, self.n_matrices * self.n_rows)
            else:
                kind_shape = (self.n_rows, self.n_matrices)

            centered_specs = {"num_features": self.n_rows,
                                "worker_fn": self._process_row_chunk,
                                "centered_fp": self.dat_fps["ms_matrices_rows"],
                                "kind_shape": kind_shape,
                                }
        return centered_specs

    def get_states_matrices_correlation(self, centered_fp: Path, kind_shape: tuple):
        if not self.ok:
            return        
        self.X_centered = np.memmap(centered_fp, dtype=DTYPE, mode='r', shape=kind_shape)
        # Perform the dot product (NumPy will auto-parallelize this using BLAS)
        covariance = np.dot(self.X_centered, self.X_centered.T)
        # Normalize covariance into Pearson correlation
        std_devs = np.sqrt(np.diag(covariance))

        # DIVIDE BY ZERO PROTECTION: indices where the standard deviation is exactly 0 (constant features)
        zero_std_mask = (std_devs == 0)
        if np.any(zero_std_mask):
            n_std0 = np.sum(zero_std_mask)
            if n_std0 == self.n_rows:
                logging.info("All matrix elements are constant (have 0 stdev). No correlation possible.")
                self.corr_matrix = None
                return
            logging.info(f"Warning: Found {n_std0} constant features with a standard deviation of 0.")
            # Replace 0 with sentinel temporarily to avoid a division by zero error.
            # No NaN propagation across the rest of the array.
            std_devs[zero_std_mask] = -999.0

        # Normalize covariance by outer product of standard deviations -> Pearson correlation 
        self.corr_matrix = covariance / np.outer(std_devs, std_devs)

        # POST-CORRECTION CLEANUP: If a feature had a standard deviation of 0, its correlation with everything 
        # (including itself) is mathematically undefined. We explicitly set those rows/columns to 1.
        if np.any(zero_std_mask):
            logging.info(" Setting undefined values for constant features in the final correlation matrix to 1")
            self.corr_matrix[zero_std_mask, :] = 1.
            self.corr_matrix[:, zero_std_mask] = 1.
        logging.info(f"States correlation matrix shape: {self.corr_matrix.shape}")

        return

    def _get_states_corr_mat_df(self):
        """
        Return the corr matrix as df for plotting.
        """
        if self.corr_matrix is None:
            return None

        mat_df = pd.DataFrame(self.corr_matrix)
        new_names = [self.mat_ix2res[c] for c in mat_df.columns.tolist()]
        # name the indices, to be automatically retrieved in plot
        mat_df.index = new_names
        mat_df.columns = new_names
        if self.corr_kind == "da":
            mat_df.index.name = "Donor"
            mat_df.columns.name = "Acceptor"
        else:
            mat_df.index.name = "State pairs"
            mat_df.columns.name = "State pairs"

        return mat_df

    def save_corr_heatmap(self,
                          fig_size=DEFAULT_FIGSIZE,
                          fig_save_fp=None,
                          title="H-bond Microstates Correlation",
                          split_x: bool = False,
                          split_y: bool = False,
                          publication: bool = False):
        """
        Wrapper for .plotting.heatmap_from_df
        """
        df = self._get_states_corr_mat_df()
        if df is None:
            logging.info("There is no correlation matrix to load.")
            return

        heatmap_from_df(df,
                        fig_save_fp=fig_save_fp,
                        title=title,
                        map_kind="corr",
                        fig_size=fig_size,
                        split_x=split_x,
                        split_y=split_y,
                        publication=publication,
                        )
        return

    def get_hdr_info(self) -> str:
        if not self.file_hdr:
            return ""
        info = re_split("represents ", self.file_hdr)[1]
        matches = findall(r"\(([\d,]+)\D+([\d,]+)\)", info)
        num1 = matches[0][0].replace(",","")
        num2 = matches[0][1].replace(",","")
        info = info.split("(")[0]
        info = info + f"(returned={format_num(float(num1))}/space={format_num(float(num2))})"

        return f"{self.n_matrices:,} occupied, {info}"

    def get_fig_specs(self) -> dict:
        """
        Get the figure parameters according to user's options.
        """
        if not self.ok:
            return

        if self.corr_kind == "pairs":  # corr_wise == "col"
            hdr = self.get_hdr_info()
            fig_size = self.figs_args.figsize_corr
            if not self.split_bk:
                if self.filtered:
                    if self.grotthus:
                        fig_title = f"H-bond States Pairs Correlation (Grotthuss filtered)\n{hdr}"
                    else:
                        fig_title = f"H-bond States Pairs Correlation (filtered)\n{hdr}"
                    fig_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr.png")
                else:
                    fig_title = f"H-bond States Pairs Correlation\n{hdr}"
                    fig_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr.png")
                self.figure_captions[str(fig_fp)] = fig_title
            else:
                if self.filtered:
                    fig_fp = self.input_fp2.with_name(
                                        f"{self.input_fp2.stem}_filtered_corr_{self.split_kind}.png"
                                        )
                else:
                    fig_fp = self.input_fp2.with_name(
                                        f"{self.input_fp2.stem}_corr_{self.split_kind}.png"
                                        )
                if self.split_kind == "bk":
                    fig_size = self.figs_args.figsize_corr_bk
                    if self.filtered:
                        fig_title = f"H-bond States Pairs Correlation, res-bk (filtered)\n{hdr}"
                    else:
                        fig_title = f"H-bond States Pairs Correlation, res-bk\n{hdr}"
                else:
                    if self.filtered:
                        fig_title = f"H-bond States Pairs Correlation, res-res (filtered)\n{hdr}"
                    else:
                        fig_title = f"H-bond States Pairs Correlation, res-res\n{hdr}"
                self.figure_captions[str(fig_fp)] = fig_title
        else:
            # da: Donors, Acceptors correl
            if self.corr_wise == "row":
                fig_size = self.figs_args.figsize_donor_corr
                if not self.split_kind:
                    if self.filtered:
                        if self.grotthus:
                            fig_title = "H-bond States Donors Correlation (Grotthuss filtered)"
                        else:
                            fig_title = "H-bond States Donors Correlation (filtered)"
                        fig_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_donors.png")
                    else:
                        fig_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_donors.png")
                        fig_title = "H-bond States Donors Correlation"
                else:
                    if self.filtered:
                        fig_fp = self.input_fp2.with_name(
                                                    f"{self.input_fp2.stem}_filtered_corr_donors_{self.split_kind}.png"
                                                    )
                    else:
                        fig_fp = self.input_fp2.with_name(
                            f"{self.input_fp2.stem}_corr_donors_{self.split_kind}.png"
                            )
                    if self.split_kind =="bk":
                        fig_size = self.figs_args.figsize_donor_corr_bk
                        if self.filtered:
                            fig_title = "H-bond States Donors Correlation, res-bk (filtered)"
                        else:
                            fig_title = "H-bond States Donors Correlation, res-bk"
                    else:
                        if self.filtered:
                            fig_title = "H-bond States Donors Correlation, res-res (filtered)"
                        else:
                            fig_title = "H-bond States Donors Correlation, res-res"
                self.figure_captions[str(fig_fp)] = fig_title
            else:
                fig_size = self.figs_args.figsize_acceptor_corr       
                if not self.split_kind:
                    if self.filtered:
                        fig_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_acceptors.png")
                        if self.grotthus:
                            fig_title = "H-bond States Acceptors Correlation (Grotthuss filtered)"
                        else:
                            fig_title = "H-bond States Acceptors Correlation (filtered)"
                    else:
                        fig_title = "H-bond States Acceptors Correlation"
                        fig_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_acceptors.png")
                else:
                    if self.filtered:
                        fig_fp = self.input_fp2.with_name(
                            f"{self.input_fp2.stem}_filtered_corr_acceptors_{self.split_kind}.png"
                            )
                    else:
                        fig_fp = self.input_fp2.with_name(
                            f"{self.input_fp2.stem}_corr_acceptors_{self.split_kind}.png"
                            )
                    if self.split_kind =="bk":
                        fig_size = self.figs_args.figsize_acceptor_corr_bk
                        if self.filtered:
                            fig_title = "H-bond States Acceptors Correlation, res-bk (filtered)"
                        else:
                            fig_title = "H-bond States Acceptors Correlation, res-bk"
                    else:
                        if self.filtered:
                            fig_title = "H-bond States Acceptors Correlation, res-res (filtered)"
                        else:
                            fig_title = "H-bond States Acceptors Correlation, res-res"
                self.figure_captions[str(fig_fp)] = fig_title

        return {"fig_fp": fig_fp,
                "fig_title": fig_title,
                "fig_size": fig_size,
                "split_x": self.figs_args.split_x,
                "split_y": self.figs_args.split_y,
                "publication": self.figs_args.publication,
                }

    def run_pipeline_pairs_corr(self, pairs_df: pd.DataFrame):
        """
        All mem maps are created in a temp folder.
        """
        if not self.ok:
            return

        self.get_dims_dicts(pairs_df)
        self.dat_fps = self.mmap_fps.copy()

        # ``delete`` was added to TemporaryDirectory in Python 3.12. The
        # supported MCCE mc4 environment uses Python 3.10; cleanup is already
        # explicit below, so the portable constructor is sufficient.
        tmp_dir = TemporaryDirectory()
        # set dat filenames to use tmp dir:
        for key in self.dat_fps:
            dp = Path(tmp_dir.name).joinpath(self.dat_fps[key])
            self.dat_fps[key] = dp

        # create main binary map using dims & self.corr_kind:
        self.save_hb_microstates_matrices()
        if not self.ok:
            tmp_dir.cleanup()
            return
        
        centered_specs = self.get_centered_specs()
        self.save_centered_matrices(centered_specs)
        self.get_states_matrices_correlation(centered_specs["centered_fp"],
                                             centered_specs["kind_shape"])
        runspecs = self.get_fig_specs()
        self.save_corr_heatmap(fig_size=runspecs["fig_size"],
                               fig_save_fp=runspecs["fig_fp"],
                               title=runspecs["fig_title"],
                               split_x=runspecs["split_x"],
                               split_y=runspecs["split_y"],
                               publication=runspecs["publication"],
                               )
        tmp_dir.cleanup()

        return

    def run_pipeline_da_corr(self, pairs_df: pd.DataFrame):
        """
        All mem maps are created in a temp folder.
        The Donors/Acceptors use the same hb_microstates matrices,
        but separate centered matrices.
        """
        if not self.ok:
            return

        self.get_dims_dicts(pairs_df)
        self.dat_fps = self.mmap_fps.copy()

        # Keep this compatible with Python 3.10 (no ``delete`` argument).
        tmp_dir = TemporaryDirectory()
        # set dat filenames to use tmp dir:
        for key in self.dat_fps:
            dp = Path(tmp_dir.name).joinpath(self.dat_fps[key])
            self.dat_fps[key] = dp

        # create main binary map using dims & self.corr_kind:
        self.save_hb_microstates_matrices()
        if not self.ok:
            tmp_dir.cleanup()
            return

        # donors:
        self.corr_wise = "row"
        centered_specs = self.get_centered_specs()
        self.save_centered_matrices(centered_specs)
        self.get_states_matrices_correlation(centered_specs["centered_fp"],
                                             centered_specs["kind_shape"])
        runspecs = self.get_fig_specs()
        self.save_corr_heatmap(fig_size=runspecs["fig_size"],
                               fig_save_fp=runspecs["fig_fp"],
                               title=runspecs["fig_title"],
                               publication=runspecs["publication"],
                               )
        # acceptors:
        self.corr_wise = "col"
        centered_specs = self.get_centered_specs()
        self.save_centered_matrices(centered_specs)
        self.get_states_matrices_correlation(centered_specs["centered_fp"],
                                             centered_specs["kind_shape"])
        runspecs = self.get_fig_specs()
        self.save_corr_heatmap(fig_size=runspecs["fig_size"],
                               fig_save_fp=runspecs["fig_fp"],
                               title=runspecs["fig_title"],
                               publication=runspecs["publication"],
                               )
        tmp_dir.cleanup()

        return

    def process_states(self):
        """
        Process hb states data using parallelism.
        """
        if self.hb_kind == "pairs":
            logging.info("ERROR: process_states should not be called when hb_kind is 'pairs'")
            self.ok = False

        logging.info(f"OK to process states? {self.ok}")
        if not self.ok:
            return

        process_start = time()
        # data viz: up to 2 heatmaps, also does splitting
        split_dfs = self.get_data_heatmaps()
        if split_dfs is not None:  # => self.split_bk == True
            res_pairs_df, bk_pairs_df = split_dfs

        # Name of ph-dependent memory maps:
        # The caller using this dict will construct the path using a new tempdir
        self.mmap_fps = {"ms_matrices": f"hb_matrices_{self.pheh_str}.dat",
                         "ms_matrices_rows": f"hb_matrices_row_centered_{self.pheh_str}.dat",
                         "ms_matrices_cols": f"hb_matrices_col_centered_{self.pheh_str}.dat",
                        }

        if not self.split_bk:
            self.corr_kind = "pairs"
            self.corr_wise = "col"
            self.split_kind = ""
            logging.info("Starting correlation pipeline for filtered hb states...")
            self.run_pipeline_pairs_corr(self.pairs_df)

            if self.include_states_da_corr:
                logging.info(" + Starting requested Donor & Acceptor correlation pipeline for filtered hb states...")
                # user requested Donor or Acceptor pairs correl:
                self.corr_kind = "da"
                self.run_pipeline_da_corr(self.pairs_df)
        else:
            if res_pairs_df is not None:
                logging.info("Starting correlation pipeline for res-res states pairs...")
                self.corr_kind = "pairs"
                self.corr_wise = "col"
                self.split_kind = "res"
                self.run_pipeline_pairs_corr(res_pairs_df)

                if self.include_states_da_corr:
                    logging.info(" + Starting requested Donor & Acceptor correlation pipeline for res-res states pairs...")
                    # user requested Donor or Acceptor pairs correl:
                    self.corr_kind = "da"
                    self.run_pipeline_da_corr(res_pairs_df)

            if bk_pairs_df is not None:
                logging.info("Starting correlation pipeline for res-bk states pairs...")
                self.corr_kind = "pairs"
                self.corr_wise = "col"
                self.split_kind = "bk"
                self.run_pipeline_pairs_corr(bk_pairs_df)

                if self.include_states_da_corr:
                    logging.info(" + Starting requested Donor & Acceptor correlation pipeline for res-bk states pairs...")
                    # user requested Donor or Acceptor pairs correl:
                    self.corr_kind = "da"
                    self.run_pipeline_da_corr(bk_pairs_df)

        logging.info("Figure captions:\n" + pformat(self.figure_captions, sort_dicts=False, width=150) + "\n")
        logging.info("Processing of states correlation over.")
        logging.info(show_elapsed_time(process_start, info="H-bond states processing"))

        return

