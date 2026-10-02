#!/usr/bin/env python3

"""
Module: correlator.py

Codebase: mcce4/ms_hbnets/correlator.py

Handle the filtering and setup for generating heatmaps.
The states correlation heatmaps are obtain using parallel processing.
"""
from argparse import Namespace
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat
from re import split as re_split, findall
from typing import Union

import numpy as np
import pandas as pd

from mcce4.ms_hbnets.msout_hb import get_resid
from mcce4.ms_hbnets.msout_hb import min_occ as MIN_OCC
from mcce4.ms_hbnets.msout_hb import MSout_hb
from mcce4.ms_hbnets.msout_hb import num_prec
from mcce4.ms_hbnets.plotting import DEFAULT_FIGSIZE
from mcce4.ms_hbnets.plotting import heatmap_from_df


DTYPE = "float32"
HB_KINDS = ["pairs", "states"]  # matches ms_hbnets subcommands


def get_resnum(resid: str):
    """Get the res num from the resid in hb_pairs_res_pH*.csv  or hb_states_pairs_pH*.csv files.
    Examples:
      HOH_W123 -> 123; _CL_1 -> 1; K__90 -> 90
    """
    if "__" in resid: # no chain
        return int(resid.rsplit("_")[-1])

    return int(resid.rsplit("_")[-1][1:])


def get_hb_states_header(hb_states_fp: Union[str, Path]) -> str:
    with open(hb_states_fp) as fh:
        hdr = fh.readline().strip()
    return hdr


def format_num(num):
    """Return condensed format up to millions.
    """
    M = 1_000_000
    K = 1_000
    if num > M:
        return f"{num / M:g}M"
    elif num > K:
        return f"{num / K:g}K"
    return num


def get_pairs_of_interest_df(pairs_of_interest_fp: Union[str, Path]) -> pd.DataFrame:
    poi_fp = Path(pairs_of_interest_fp)
    if not poi_fp.exists():
        print("File not found:", pairs_of_interest_fp)
        return None

    poi_df = pd.read_csv(poi_fp, comment="#", names=["res_d", "res_a"])
    if not poi_df.shape[0]:
        print("Empty 'pairs of interest' file:", pairs_of_interest_fp)
        return None

    return poi_df


class HbCorrelator:
    def __init__(self, hbms: MSout_hb,
                 hb_kind: str = None,
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
         - min_occ (float, MIN_OCC): To change the default used by the msout data loader
         - include_states_da_corr (bool, False): If True, the states Donors and Acceptors correlation
           is calculated in addition to the pairs correlation.
         - split_BK (bool, False): If True, or no pairs of interest provided (no filtering), 
           separate correlation of the res-res and res-bk pairs will be computed.
         - figs_args([dict, argparse.Namespace]): To control the heatmap sizes and the lowest
           bound of the cbar and color of the data heatmaps.
        """
        print("\nHbCorrelator start...")
        self.ok = True
        if hb_kind not in HB_KINDS:
            print(f" ValueError: hb_kind must be one of {HB_KINDS}")
            self.ok = False
            return
        self.hb_kind = hb_kind

        inputs = self.get_intput_files(hbms)
        if not self.ok:
            return
        self.input_fp1, self.input_fp2 = inputs

        self.mcce_dir = hbms.run_dir
        self.pheh_str = hbms.pheh_str
        self.min_occ = min_occ
        self.prec = num_prec(float(self.min_occ))
        self.min_occ_print = f"min occ >= {self.min_occ:.{self.prec}f}"
        print(f" Heatmaps min_occ: {self.min_occ_print}")

        self.poi_df: pd.DataFrame = None
        self.split_bk = True
        if pairs_of_interest_fp is not None:
            self.poi_df = get_pairs_of_interest_df(pairs_of_interest_fp)
            if self.poi_df is None:
                self.ok = False
                return
            self.split_bk = False or split_BK
        if self.split_bk:
            print(" The hb pairs will be divided into res-res and res-BK pairs to reduce the heatmap size.")
  
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
        # used in process_states:
        self.states_hdr: str = ""
        if self.hb_kind == "states":
             self.states_hdr = get_hb_states_header(self.input_fp1)
        
        self.pairs_df = None  # main df, ready for splitting if needed
        # filtered with pairs_of_interest_fp is given & min_occ:  
        self.pairs_df = self.get_inputfile_pairs_data()
        if not self.ok:
            return

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
            fp1 = hbms.pairs_csv   # not currently needed, integrity test
            fp2 = hbms.pairs_res_csv
        else:
            fp1 = hbms.states_csv
            fp2 = hbms.states_pairs_csv

        if fp1.exists() and fp2.exists():
            return fp1, fp2
        else:
            print(f"Missing file(s): At least once of the csv was not found: {fp1.name}, {fp2.name}")
            self.ok = False

        return None

    def _load_csv(self, csv_fp: Path) -> pd.DataFrame:
        """Load state_pairs or pairs_res csv and maybe update the csv format.
        """
        df = pd.read_csv(csv_fp, comment="#")
        save = False

        if "res_d" not in df.columns:
            save = True
            print(" Adding 'res_d' and 'res_a' columns to get residue-based names and indices...")
            # confids -> short resid:
            df[["res_d","res_a"]] = df.apply(lambda x: pd.Series([get_resid(x["donor"]),
                                                                  get_resid(x["acceptor"])]), axis=1)

        msk_hoh = (df["res_d"].str.startswith("HOH") | df["res_a"].str.startswith("HOH"))
        if msk_hoh.any():
            save = True
            df.loc[msk_hoh, "res_d"] = df.loc[msk_hoh, "res_d"].str.replace(r'^HOH', 'w', regex=True)
            df.loc[msk_hoh, "res_a"] = df.loc[msk_hoh, "res_a"].str.replace(r'^HOH', 'w', regex=True)

        if save:
            # output file predates codebase update
            if self.hb_kind == "states":  # preserve the commented header
                hdr = get_hb_states_header(csv_fp)
                with open(csv_fp, "w") as fo:
                    fo.write(hdr+"\n")
                    df.to_csv(fo, index=False)
            else:
                df.to_csv(csv_fp, index=False)

        if self.hb_kind == "states":  # file has both confids and resid
            resdf = (df.groupby(["res_d","res_a"], as_index=False)
                     .agg({"count": "max", "occ": "max",
                           "with_bk": "max", "Mi":"min", "Mj":"min"})
                     .sort_values(by=["Mi", "Mj"])
                     .reset_index(drop=True)
                     )
            return resdf

        return df

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
                    print(f" Filtered pairs df shape: {resdf.shape}")
                    return resdf
                else:
                    print(" Found no matches for the given pairs of interest:", self.poi_df, sep="\n")
                    self.ok = False
                    return None
            except Exception as e:
                print(f" Could not filter the dataframe for pairs of interest. Error:\n{e}")
                self.ok = False
                return None

    def get_inputfile_pairs_data(self) -> pd.DataFrame:
        """
        The states pairs file contains the unique pairs over the hb space returned.
        The pairs from the pairs_res file are the most occupied overall.
        These pairs define the size of the residue-based stored matrices.

        Prep of the main df : filtered for poi & occ.
        """
        if not self.ok:
            return

        resdf =  self._load_csv(self.input_fp2)
        # maybe filter:
        if self.poi_df is not None:
            df = self._filter_pairs_of_interest(resdf)
            if df is not None:
                resdf = df
            else:
                return

        # apply min occ mask:
        occ_msk = resdf["occ"].ge(self.min_occ)
        if occ_msk.any():
            resdf = resdf.loc[occ_msk]
        else:
            print(f" No hb pairs data at {self.min_occ_print}")
            self.ok = False
            return

        # sort by resnum:
        resdf["di"] = resdf["res_d"].apply(get_resnum)
        resdf["ai"] = resdf["res_a"].apply(get_resnum)
        resdf = resdf.sort_values(by=["di","ai"])
        resdf = resdf.drop(columns=["di","ai"])
        resdf = resdf.rename(columns={"res_d":"Donor", "res_a":"Acceptor"})

        return resdf

    def get_data_heatmaps(self, kind: str="pairs") -> Union[None, tuple]:
        """Create the data heatmap(s) from pairs in hb_states_pairs_ or
        hb_pairs_res_ csv files.

        Called by process_pairs, process_states.
        If split was done, outputs res_respairs_df, bk_respairs_df.
        """
        if not self.ok:
            return

        # will be output if split was done
        res_respairs_df = None
        bk_respairs_df = None
        if not self.split_bk:  # => self.pairs_df is filtered
            matrix_df = self.pairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
            # data heatmap:
            title = f"H-bond Donor - Acceptor pairs data\n({self.input_fp2.name})"
            png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered.png")
            heatmap_from_df(matrix_df,
                            fig_save_fp=png_fp,
                            title=title,
                            fig_size=self.figs_args.figsize_data,
                            min_bound=self.figs_args.data_cbar_min,
                            color=self.figs_args.data_map_color,
                            ) 
        else:
            # split non-BK, then BK:
            res_msk = self.pairs_df["with_bk"].eq(False)
            if res_msk.any():
                res_respairs_df = self.pairs_df.loc[res_msk, ["Donor", "Acceptor", "count", "occ"]]
                n_res_ud = len(res_respairs_df["Donor"].unique())
                n_res_ua = len(res_respairs_df["Acceptor"].unique())
                print(f" {kind.capitalize()} res-res pairs, uniq donors: {n_res_ud}, uniq acceptors: {n_res_ua}", sep="\n")
        
                matrix_df = res_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                # data heatmap:
                title = f"H-bond Donor - Acceptor pairs data, res-res\n({self.input_fp2.name})"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_res.png")
                heatmap_from_df(matrix_df,
                                fig_save_fp=png_fp,
                                title=title,
                                fig_size=self.figs_args.figsize_data,
                                min_bound=self.figs_args.data_cbar_min,
                                color=self.figs_args.data_map_color,
                                )
            else:
                print(f" {kind.upper()}: No hb pairs of res-res kind in {self.input_fp2.name}.")

            # now BK:
            if (~res_msk).any():
                bk_respairs_df = self.pairs_df.loc[~res_msk, ["Donor", "Acceptor", "count", "occ"]]
                n_bk_ud = len(bk_respairs_df["Donor"].unique())
                n_bk_ua = len(bk_respairs_df["Acceptor"].unique())
                print(f" {kind.capitalize()} res-bk pairs: uniq donors: {n_bk_ud} uniq acceptors: {n_bk_ua}", sep="\n")

                matrix_df = bk_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                # data heatmap:
                title = f"H-bond Donor - Acceptor pairs data, res-bk\n({self.input_fp2.name})"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_bk.png")
                heatmap_from_df(matrix_df,
                                fig_save_fp=png_fp,
                                title=title,
                                fig_size=self.figs_args.figsize_data_bk,
                                min_bound=self.figs_args.data_cbar_min,
                                color=self.figs_args.data_map_color,
                                )
            else:
                print(f" {kind.upper()}: No pairs of res-bk kind in {self.input_fp2.name}.")
            return res_respairs_df, bk_respairs_df

        return

    def process_pairs(self):
        """Uses pandas for correlations.
        """
        print(f"\n OK to process pairs? {self.ok}")
        if not self.ok:
            return

        self.pairs_df = self.get_inputfile_pairs_data()
        if not self.ok:
            return

        # data viz:
        split_dfs = self.get_data_heatmaps()
        if split_dfs is not None:
            res_respairs_df, bk_respairs_df = split_dfs

        if not self.split_bk:  # self.pairs_df is filtered, or split_BK was false
            # data heatmap:
            matrix_df = self.pairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
            # Get donors and Acceptors correlation:
            # Transpose and compute Donor (row) correlation
            row_corr_mat = matrix_df.T.corr()
            if row_corr_mat.shape[0] < 2:
                print("Not enough rows for correlation.")
            else:
                title = "H-bond Donors correlation"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_donors.png")
                heatmap_from_df(row_corr_mat,
                                fig_save_fp=png_fp,
                                title=title,
                                map_kind="corr",
                                fig_size=self.figs_args.figsize_donor_corr,
                                min_bound=self.figs_args.data_cbar_min,
                                color=self.figs_args.data_map_color,
                                )
            
            # Compute Acceptor (col) correlation
            col_corr_mat = matrix_df.corr()
            if col_corr_mat.shape[0] < 2:
                print(" Not enough columns for correlation.")
            else:
                title = "H-bond Acceptors correlation"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_corr_acceptors.png")
                heatmap_from_df(col_corr_mat,
                                fig_save_fp=png_fp,
                                title=title,
                                map_kind="corr",
                                fig_size=self.figs_args.figsize_acceptor_corr,
                                min_bound=self.figs_args.data_cbar_min,
                                color=self.figs_args.data_map_color,
                                )
        else:
            # split non-BK, then BK:
            # res_msk = self.pairs_df["with_bk"].eq(False)
            # if res_msk.any():
            #     res_respairs_df =  self.pairs_df.loc[res_msk, ["Donor", "Acceptor", "count", "occ"]]
            if res_respairs_df is not None:
                n_ud = len(res_respairs_df["Donor"].unique())
                n_ua = len(res_respairs_df["Acceptor"].unique())
                print(f" Non BK respairs, uniq donors: {n_ud}, uniq acceptors: {n_ua}", sep="\n")
                # data heatmap:
                matrix_df = res_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)

                # get the 2 kinds or corr:
                row_corr_mat = matrix_df.T.corr()
                if row_corr_mat.shape[0] < 2:
                    print(" Not enough rows for correlation.")
                else:
                    title = "H-bond Donors correlation, res-res"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_donors_res.png")
                    heatmap_from_df(row_corr_mat,
                                    fig_save_fp=png_fp,
                                    title=title,
                                    map_kind="corr",
                                    fig_size=self.figs_args.figsize_donor_corr,
                                    min_bound=self.figs_args.data_cbar_min,
                                    color=self.figs_args.data_map_color,
                                    )
                
                col_corr_mat = matrix_df.corr()
                if col_corr_mat.shape[0] < 2:
                    print("Not enough columns for correlation.")
                else:
                    title = "H-bond Acceptors correlation, res-res"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_acceptors_res.png")
                    heatmap_from_df(col_corr_mat,
                                    fig_save_fp=png_fp,
                                    title=title,
                                    map_kind="corr",
                                    fig_size=self.figs_args.figsize_acceptor_corr,
                                    min_bound=self.figs_args.data_cbar_min,
                                    color=self.figs_args.data_map_color,
                                    )
            else:
                print("No hb pairs of residue-residue kind.")

            # now BK:
            # if (~res_msk).any():
            #     bk_respairs_df = self.pairs_df.loc[~res_msk, ["Donor", "Acceptor", "count", "occ"]]
            if bk_respairs_df is not None:
                n_bk_ud = len(bk_respairs_df["Donor"].unique())
                n_bk_ua = len(bk_respairs_df["Acceptor"].unique())
                print(f" BK respairs, uniq donors: {n_bk_ud} uniq acceptors: {n_bk_ua}", sep="\n")
                # data heatmap:
                matrix_df = bk_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                # get the 2 kinds or corr:
                row_corr_mat = matrix_df.T.corr()
                if row_corr_mat.shape[0] < 2:
                    print(" Not enough rows for correlation.")
                else:
                    title = "H-bond Donors correlation, res-bk"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_donors_bk.png")
                    heatmap_from_df(row_corr_mat,
                                    fig_save_fp=png_fp,
                                    title=title,
                                    map_kind="corr",
                                    fig_size=self.figs_args.figsize_donor_corr_bk,
                                    min_bound=self.figs_args.data_cbar_min,
                                    color=self.figs_args.data_map_color,
                                    )
                
                col_corr_mat = matrix_df.corr()
                if col_corr_mat.shape[0] < 2:
                    print(" Not enough cols for correlation.")
                else:
                    title = "H-bond Acceptors correlation, res-bk"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_corr_acceptors_bk.png")
                    heatmap_from_df(col_corr_mat,
                                    fig_save_fp=png_fp,
                                    title=title,
                                    map_kind="corr",
                                    fig_size=self.figs_args.figsize_acceptor_corr_bk,
                                    min_bound=self.figs_args.data_cbar_min,
                                    color=self.figs_args.data_map_color,
                                    )
            else:
                print(" No hb pairs of residue-backbone kind.")

        print(" Processing of pairs correlation over.\n")

        return

    # >>> Functions for states processing:
    def get_dims_dicts(self, df: pd.DataFrame):
        """
        State pairs processing.
        Get the df dimension and conversion dicts.
        Input df maybe a split df (res_df, bk_df).

        Sets: n_rows, n_cols, res2mat_ix, mat_ix2res
        """
        if not self.ok:
            return 
        
        if self.hb_kind != "states":
            print("ERROR: get_dims_dicts applies to 'states'")
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
            print(f" States DA matrices shape: {self.n_rows}, {self.n_cols}")

        else:  # corr of pairs
            # Create a MultiIndex from the target columns
            df_index = pd.MultiIndex.from_frame(df[["Donor", "Acceptor"]])
            # whole matrix: (n_matrices, n_cols) (col-wise correl):
            self.n_cols = df_index.shape[0]
            self.n_rows = self.n_cols
            # dict to obtain the matrix index of each pair
            self.res2mat_ix = {pair: px for px, pair in enumerate(df_index.tolist())}
            # dict to rename the corr matrix rows & columns:
            # transform tuple to str: ('H_A12', 'V_A34') -> 'H_A12|V_A34'
            self.mat_ix2res = {v: f"{k[0]}|{k[1]}" for k, v in self.res2mat_ix.items()}
            print(f" States matrices columns: {self.n_cols}")
  
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
            else:  # states pairs
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
            print(" The hb states file is empty.")
            self.ok = False
            return

        occ_msk = dfs["occ"].ge(self.min_occ)
        dfs = dfs.loc[occ_msk].reindex()
        self.n_matrices = dfs.shape[0]
        if not self.n_matrices:
            print(f" There are no occupied hb microstates at {self.min_occ_print}.")
            self.ok = False
            return
        print(f" Number of occupied states matrices: {self.n_matrices:,}.")

        # Initialize an empty binary file on disk to hold the stacked matrices
        if self.corr_kind == "da":
            data_file = np.memmap(self.dat_fps["ms_matrices"], dtype=DTYPE, mode='w+',
                                  shape=(self.n_matrices, self.n_rows, self.n_cols))
            # Add each matrix sequentially into the memmap file:
            for i in range(self.n_matrices):
                data_file[i] = self._hb_ms2matrix(dfs.loc[i], np.zeros((self.n_rows, self.n_cols)))
        else:
            data_file = np.memmap(self.dat_fps["ms_matrices"], dtype=DTYPE, mode='w+',
                                  shape=(self.n_matrices, self.n_cols))
            # Add each matrix sequentially into the memmap file:
            for i in range(self.n_matrices):
                data_file[i] = self._hb_ms2matrix(dfs.loc[i], np.zeros((self.n_cols)))

        print(" Stored matrix shape:", data_file.shape)
        # Flush changes to disk and clean up the write reference
        data_file.flush()
        del data_file

        return

    # --- Worker Functions ---
    def _process_row_chunk(self, r):
        """State pairs processing.
        Worker function to process a single row index.
        """
        if self.corr_kind == "pairs":
            print("ERROR: _process_row_chunk called when self.corr_kind == 'pairs'")
            self.ok = False
            return r, 0
        
        # Re-open the memmap inside the worker process (safe for read-only)
        X_mmap = np.memmap(self.dat_fps["ms_matrices"], mode="r", dtype=DTYPE,
                           shape=(self.n_matrices, self.n_rows, self.n_cols))
        # Extract the row across all matrices/columns and flatten to 1D
        row_data = X_mmap[:, r, :].ravel()
        # Return the centered row data
        return r, row_data - np.mean(row_data)

    def _process_col_chunk(self, c):
        """State pairs processing.
        Worker function to process a single column index.
        """
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

        print(f" Starting parallel processing for {self.corr_wise}-wise correlation...")
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
                print(" All matrix elements are constant (have 0 std dev). No correlation possible.")
                self.corr_matrix = None
                return
            print(f" Warning: Found {n_std0} constant features with a standard deviation of 0.")
            # Replace 0 with 1 temporarily to avoid a division by zero error.
            # No NaN propagation across the rest of the array.
            std_devs[zero_std_mask] = -999.0

        # Normalize covariance by outer product of standard deviations -> Pearson correlation 
        self.corr_matrix = covariance / np.outer(std_devs, std_devs)

        # POST-CORRECTION CLEANUP: If a feature had a standard deviation of 0, its correlation with everything 
        # (including itself) is mathematically undefined. We explicitly set those rows/columns to 0.
        if np.any(zero_std_mask):
            print(" Setting undefined values for constant features in the final correlation matrix to 1")
            self.corr_matrix[zero_std_mask, :] = 1.
            self.corr_matrix[:, zero_std_mask] = 1.
        print(f" Obtained {self.corr_wise.capitalize()}-wise correlation matrix, shape:", self.corr_matrix.shape)

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
                          title="H-bond Microstates Correlation"):
        """
        Wrapper for .plotting.heatmap_from_df
        """
        df = self._get_states_corr_mat_df()
        if df is None:
            print(" There is no correlation matrix to load.")
            return

        heatmap_from_df(df,
                        fig_save_fp=fig_save_fp,
                        title=title,
                        map_kind="corr",
                        fig_size=fig_size,
                        )
        return

    def get_states_hdr_info(self) -> str:
        if not self.states_hdr:
            return ""
        info = re_split("represents ", self.states_hdr)[1]
        matches = findall(r"\(([\d,]+)\D+([\d,]+)\)", info)
        num1 = matches[0][0].replace(",","")
        num2 = matches[0][1].replace(",","")
        info = info.split("(")[0]
        info = info + f"(returned={format_num(float(num1))}/space={format_num(float(num2))})"

        return f"({self.n_matrices:,} occupied states, {info})"

    def get_run_specs(self) -> dict:
        """
        Get the figure parameters according to user's options.
        """
        if not self.ok:
            return

        corr_kind: str
        corr_wise: str
        fig_fp: Path
        fig_title: str
        fig_size: tuple
        corr_kind = self.corr_kind
        corr_wise = self.corr_wise

        if corr_kind == "pairs":
            hdr = self.get_states_hdr_info()
            fig_size = self.figs_args.figsize_corr
            if not self.split_bk:
                fig_fp = self.input_fp1.with_name(f"{self.input_fp1.stem}_corr.png")
                fig_title = f"H-bond States Pairs Correlation\n{hdr}"
            else:
                fig_fp = self.input_fp1.with_name(
                    f"{self.input_fp1.stem}_corr_{self.split_kind}.png"
                    )
                if self.split_kind == "bk":
                    fig_size = self.figs_args.figsize_corr_bk
                    fig_title = f"H-bond States Pairs Correlation, res-bk\n{hdr}"
                else:
                    fig_title = f"H-bond States Pairs Correlation, res-res\n{hdr}"
        else:
            # da: Donors, Acceptors correl
            if corr_wise == "row":
                fig_size = self.figs_args.figsize_donor_corr
                if not self.split_kind:
                    fig_fp = self.input_fp1.with_name(f"{self.input_fp1.stem}_corr_donors.png")
                    fig_title = "H-bond States Donors Correlation"
                else:
                    fig_fp = self.input_fp1.with_name(
                        f"{self.input_fp1.stem}_corr_donors_{self.df_split_kind}.png"
                        )
                    if self.split_kind =="bk":
                        fig_size = self.figs_args.figsize_donor_corr_bk
                        fig_title = "H-bond States Donors Correlation, res-bk"
                    else:
                        fig_title = "H-bond States Donors Correlation, res-res"
            else:
                fig_size = self.figs_args.figsize_acceptor_corr       
                if not self.split_kind:
                    fig_fp = self.input_fp1.with_name(f"{self.input_fp1.stem}_corr_acceptors.png")
                    fig_title = "H-bond States Acceptors Correlation"
                else:
                    fig_fp = self.input_fp1.with_name(
                        f"{self.input_fp1.stem}_corr_acceptors_{self.split_kind}.png"
                        )
                    if self.split_kind =="bk":
                        fig_size = self.figs_args.figsize_acceptor_corr_bk
                        fig_title = "H-bond States Acceptors Correlation, res-bk"
                    else:
                        fig_title = "H-bond States Acceptors Correlation, res-res"

        return {
                 "corr_kind": corr_kind,
                 "corr_wise": corr_wise,
                 "fig_fp": fig_fp,
                 "fig_title": fig_title,
                 "fig_size": fig_size,
                }

    def run_pipeline(self, df: pd.DataFrame, centered_specs: dict, runspecs: dict):
        if not self.ok:
            return

        self.corr_kind = runspecs["corr_kind"]
        self.corr_wise = runspecs["corr_wise"]
        self.save_centered_matrices(centered_specs)
        self.get_states_matrices_correlation(centered_specs["centered_fp"],
                                             centered_specs["kind_shape"])
        self.save_corr_heatmap(fig_size=runspecs["fig_size"],
                               fig_save_fp=runspecs["fig_fp"],
                               title=runspecs["fig_title"])
        return
    
    def process_states(self):
        """
        Process hb states data using parallelism.
        """
        if self.hb_kind == "pairs":
            print("ERROR: process_states should not be called when hb_kind is 'pairs'")
            self.ok = False

        print(f"\n OK to process states? {self.ok}")
        if not self.ok:
            return

        self.pairs_df = self.get_inputfile_pairs_data()
        if not self.ok:
            return

        # data viz: up to 2 heatmaps, also does splitting
        split_dfs = self.get_data_heatmaps()
        if split_dfs is not None:  # => self.split_bk == True
            res_pairs_df, bk_pairs_df = split_dfs

        # dat files always re-created as they will depends on the list of 'pairs of interest'
        # save all matrices in memory mapped binary file:
        dat_dir = self.mcce_dir.joinpath(".dats")
        dat_dir.mkdir(exist_ok=True)
        self.dat_fps = {"ms_matrices": dat_dir.joinpath(f"hb_matrices_{self.pheh_str}.dat"),
                        "ms_matrices_rows": dat_dir.joinpath(f"hb_matrices_row_centered_{self.pheh_str}.dat"),
                        "ms_matrices_cols": dat_dir.joinpath(f"hb_matrices_col_centered_{self.pheh_str}.dat"),
                      }

        centered_specs: dict = None
        if not self.split_bk:
            self.get_dims_dicts(self.pairs_df)
            self.corr_kind = "pairs"
            # create main binary mat using dims & self.corr_kind:
            self.save_hb_microstates_matrices()
            if not self.ok:
                return            
            centered_specs = {
                    "num_features": self.n_cols,
                    "worker_fn": self._process_col_chunk,
                    "centered_fp": self.dat_fps["ms_matrices_cols"],
                    "kind_shape": (self.n_cols, self.n_matrices),
                    }
            self.split_kind = ""
            runspecs = self.get_run_specs()
            self.run_pipeline(self.pairs_df, centered_specs, runspecs)

            if self.include_states_da_corr:
                self.corr_kind = "da"
                self.get_dims_dicts(self.pairs_df)
                self.save_hb_microstates_matrices()
                if not self.ok:
                    return                
                # donors:
                self.corr_wise = "row"
                centered_specs = {
                    "num_features": self.n_rows,
                    "worker_fn": self._process_row_chunk,
                    "centered_fp": self.dat_fps["ms_matrices_rows"],
                    "kind_shape": (self.n_rows, self.n_matrices * self.n_rows),
                    }
                runspecs = self.get_run_specs()
                self.run_pipeline(self.pairs_df, centered_specs, runspecs)

                # acceptors:
                self.corr_wise = "col"
                centered_specs = {
                    "num_features": self.n_cols,
                    "worker_fn": self._process_col_chunk,
                    "centered_fp": self.dat_fps["ms_matrices_cols"],
                    "kind_shape": (self.n_cols, self.n_matrices * self.n_cols),
                    }
                runspecs = self.get_run_specs()
                self.run_pipeline(self.pairs_df, centered_specs, runspecs)
        else:
            if res_pairs_df is not None:
                self.get_dims_dicts(res_pairs_df)
                self.corr_kind = "pairs"
                self.save_hb_microstates_matrices()
                if not self.ok:
                    return
                centered_specs = {
                    "num_features": self.n_cols,
                    "worker_fn": self._process_col_chunk,
                    "centered_fp": self.dat_fps["ms_matrices_cols"],
                    "kind_shape": (self.n_cols, self.n_matrices),
                    }
                self.split_kind = "res"
                runspecs = self.get_run_specs()
                self.run_pipeline(res_pairs_df, centered_specs, runspecs)

                if self.include_states_da_corr:
                    self.corr_kind = "da"
                    self.get_dims_dicts(res_pairs_df)
                    self.save_hb_microstates_matrices()
                    if not self.ok:
                        return                    
                    # donors:
                    self.corr_wise = "row"
                    centered_specs = {
                        "num_features": self.n_rows,
                        "worker_fn": self._process_row_chunk,
                        "centered_fp": self.dat_fps["ms_matrices_rows"],
                        "kind_shape": (self.n_rows, self.n_matrices * self.n_rows),
                        }
                    runspecs = self.get_run_specs()
                    self.run_pipeline(res_pairs_df, centered_specs, runspecs)
                    # acceptors:
                    self.corr_wise = "col"
                    centered_specs = {
                        "num_features": self.n_cols,
                        "worker_fn": self._process_col_chunk,
                        "centered_fp": self.dat_fps["ms_matrices_cols"],
                        "kind_shape": (self.n_cols, self.n_matrices * self.n_cols),
                        }
                    runspecs = self.get_run_specs()
                    self.run_pipeline(res_pairs_df, centered_specs, runspecs)

            if bk_pairs_df is not None:
                self.split_kind = "bk"
                self.get_dims_dicts(bk_pairs_df)
                self.corr_kind = "pairs"
                self.save_hb_microstates_matrices()
                if not self.ok:
                    return
                centered_specs = {
                    "num_features": self.n_cols,
                    "worker_fn": self._process_col_chunk,
                    "centered_fp": self.dat_fps["ms_matrices_cols"],
                    "kind_shape": (self.n_cols, self.n_matrices),
                    }
                runspecs = self.get_run_specs()
                self.run_pipeline(bk_pairs_df, centered_specs, runspecs)

                if self.include_states_da_corr:
                    self.corr_kind = "da"
                    self.get_dims_dicts(res_pairs_df)
                    self.save_hb_microstates_matrices()
                    if not self.ok:
                        return
                    # donors:
                    self.corr_wise = "row"
                    centered_specs = {
                        "num_features": self.n_rows,
                        "worker_fn": self._process_row_chunk,
                        "centered_fp": self.dat_fps["ms_matrices_rows"],
                        "kind_shape": (self.n_rows, self.n_matrices * self.n_rows),
                        }
                    runspecs = self.get_run_specs()
                    self.run_pipeline(res_pairs_df, centered_specs, runspecs)
                    # acceptors:
                    self.corr_wise = "col"
                    centered_specs = {
                        "num_features": self.n_cols,
                        "worker_fn": self._process_col_chunk,
                        "centered_fp": self.dat_fps["ms_matrices_cols"],
                        "kind_shape": (self.n_cols, self.n_matrices * self.n_cols),
                        }
                    runspecs = self.get_run_specs()
                    self.run_pipeline(res_pairs_df, centered_specs, runspecs)

        print(" Processing of states correlation over.")
        return

