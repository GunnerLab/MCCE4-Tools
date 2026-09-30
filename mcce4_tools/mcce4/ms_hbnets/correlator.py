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
from re import split as re_split
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


HB_KINDS = ["pairs", "states"]

class HbCorrelator:
    def __init__(self, hbms: MSout_hb,
                 hb_kind: str = None,
                 pairs_of_interest_fp: Union[str, Path] = None,
                 min_occ: float = MIN_OCC,
                 figs_args: Union[dict,Namespace] = None,
                ):
        """ 
        Args:
         - pairs_of_interest_fp ([str, Path], None): Obtain the correlation of a pair of 
            hb residues (comma-separated residue hb pairs).
         - hb_kind (str, None): One of HB_KINDS=["pairs", "states"]; if None -> "pairs"
         - min_occ (float, MIN_OCC): To change the default used by the msout data loader
        """
        print("\nHbCorrelator start...")
        self.ok = True
        if hb_kind not in HB_KINDS:
            print(f" ValueError: hb_kind must be one of {HB_KINDS}")
            self.ok = False
            return
        self.hb_kind = hb_kind

        inputs = self.get_intput_files(hbms)
        if inputs is None:
            self.ok = False
            return
        self.input_fp1, self.input_fp2 = inputs
        
        self.mcce_dir = hbms.run_dir
        self.pheh_str = hbms.pheh_str
        self.min_occ = min_occ
        self.prec = num_prec(float(self.min_occ))
        self.min_occ_print = f"min occ >= {self.min_occ:.{self.prec}f}"
        print(f" Heatmaps min_occ: {self.min_occ_print}")

        self.poi_df: pd.DataFrame = None
        #self.poi_fp: Path = None
        self.split_bk = True
        if pairs_of_interest_fp is not None:
            self.poi_df = get_pairs_of_interest_df(pairs_of_interest_fp)
            if self.poi_df is None:
                self.ok = False
                return
            #self.poi_fp = Path(pairs_of_interest_fp)
            self.split_bk = False

        if self.ok and self.split_bk:
            print(" WARNING: No file with pairs of interest provided for filtering:\n",
                  " the hb pairs will be divided into res-res and res-BK pairs to reduce the heatmap size.")

        # Populated by get_pairs_data :
        self.n_rows: int = 0
        self.n_cols: int = 0

        # defined for hb states:
        # dict to obtain the matrix index of the resid
        self.res2mat_ix: dict = None
        # dict to rename the corr matrix rows & columns:
        self.mat_ix2res: dict = None

        # filtered if pairs_of_interest_fp is given:
        self.pairs_df = self.get_pairs_data()

        if isinstance(figs_args, dict):
            self.figs_args = Namespace(**figs_args)
        else:
            self.figs_args = figs_args

        # set process_states pipeline:
        self.n_matrices: int = 0
        self.corr_matrix: np.ndarray = None
        
        return

    def get_intput_files(self, hbms: MSout_hb) -> Union[tuple, None]:
        """
        self.states_csv, self.states_pairs_csv
        """
        if self.hb_kind == "pairs":
            fp1 = hbms.pairs_csv
            fp2 = hbms.pairs_res_csv
        else:
            fp1 = hbms.states_csv
            fp2 = hbms.states_pairs_csv

        if fp1.exists() and fp2.exists():
            return fp1, fp2

        return None

    def get_pairs_data(self) -> pd.DataFrame:
        # Return filtered df if pairs_of_interest_fp is given
        #print("Populating pairs_df with pairs data...")
        if self.hb_kind == "states":
            return self._get_states_pairs_data()
        else:
            return self._get_pairs_res_data()

    def load_csv(self, csv_fp: Path) -> pd.DataFrame:
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
            if self.hb_kind == "states":
                hdr = get_hb_states_header(csv_fp)
                with open(csv_fp, "w") as fo:
                    fo.write(hdr+"\n")
                    df.to_csv(fo, index=False)
            else:
                df.to_csv(csv_fp, index=False)
    
        if self.hb_kind == "states":
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

    def _get_pairs_res_data(self):
        """
        The pairs res file contains the hb pairs over residues.
        Populate self.n_rows, self.n_cols and self.res2mat_ix
        """
        if not self.ok:
            return

        if self.hb_kind != "pairs":
            print("Function '_get_pairs_res_data' should only be called when hb_kind='pairs'.")
            self.ok = False
            return None

        resdf = self.load_csv(self.input_fp2)
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
            print(f"No hb pairs data at {self.min_occ_print}")
            self.ok = False
            return
        
        # sort by resnum:
        resdf["di"] = resdf["res_d"].apply(get_resnum)
        resdf["ai"] = resdf["res_a"].apply(get_resnum)
        resdf = resdf.sort_values(by=["di","ai"])
        resdf = resdf.drop(columns=["di","ai"])
        resdf = resdf.rename(columns={"res_d":"Donor", "res_a":"Acceptor"})

        uniq_d = resdf["Donor"].unique().tolist()
        uniq_a = resdf["Acceptor"].unique().tolist()
        self.n_rows = len(uniq_d)
        self.n_cols = len(uniq_a)
        # # dict to obtain the matrix index of the resid
        # self.res2mat_ix = {res: rx for rx, res in enumerate(sorted(uniq_res, key=lambda x: int(x.rsplit("_", maxsplit=1)[1][1:])))}
        # # dict to rename the corr matrix rows & columns:
        # self.mat_ix2res = {v:k for k, v in self.res2mat_ix.items()}
        print(f" Pairs matrix shape: ({self.n_rows}, {self.n_cols})")

        return resdf

    def _get_states_pairs_data(self) -> pd.DataFrame:
        """
        The states pairs file contains the unique pairs over the hb ms returned.
        These pairs define the size of the residue-based stored matrices.

        Populate self.n_rows, self.n_cols and self.res2mat_ix
        """
        if not self.ok:
            return

        if self.hb_kind != "states":
            print(" Function '_get_states_pairs_data' should only be called when hb_kind='states'.")
            self.ok = False
            return None
        
        resdf =  self.load_csv(self.input_fp2)
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

        uniq_res = list(set(resdf["Donor"]).union(resdf["Acceptor"]))
        n_uniq_res = len(uniq_res)
        # shape of matrix for each hb ms:
        self.n_rows, self.n_cols = n_uniq_res, n_uniq_res
        # dict to obtain the matrix index of the resid
        self.res2mat_ix = {res: rx for rx, res in enumerate(sorted(uniq_res, key=lambda x: int(x.rsplit("_", maxsplit=1)[1][1:])))}
        # dict to rename the corr matrix rows & columns:
        self.mat_ix2res = {v:k for k, v in self.res2mat_ix.items()}
        print(f" States matrices shape: ({self.n_rows}, {self.n_cols})")

        return resdf

    def _hb_ms2matrix(self, hbs, state_mat):
        """Process a single microstate (row) of the hb_states file into a matrix.
        Used by save_hb_microstates_matrices.
        """
        # state res tuples to mat
        tpls = set()
        for tpl in [val.split(",") for val in re_split(r",\(", hbs["state_id"][1:])]:
            tpl[1] = tpl[1][:-1]  # no trailing ")"
            tpls.add((get_resid(tpl[0]), get_resid(tpl[1])))
        for tpl in tpls:
            mi = self.res2mat_ix.get(tpl[0])
            if mi is None:
                continue
            mj = self.res2mat_ix.get(tpl[1])
            if mj is None:
                continue
            state_mat[mi, mj] = hbs["occ"]

        return state_mat

    def save_hb_microstates_matrices(self):
        """Save states microstates matrices to a memory-mapped binary file.
        """
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
        data_file = np.memmap(self.dat_fps["ms_matrices"], dtype=DTYPE, mode='w+',
                              shape=(self.n_matrices, self.n_rows, self.n_cols))
        # Add each matrix sequentially into the memmap file:
        for i in range(self.n_matrices):
            data_file[i] = self._hb_ms2matrix(dfs.loc[i], np.zeros((self.n_rows, self.n_cols)))

        # Flush changes to disk and clean up the write reference
        data_file.flush()
        del data_file

        return

    # --- Worker Functions ---
    def process_row_chunk(self, r):
        """Worker function to process a single row index."""
        # Re-open the memmap inside the worker process (safe for read-only)
        X_mmap = np.memmap(self.dat_fps["ms_matrices"], mode="r", dtype=DTYPE,
                           shape=(self.n_matrices, self.n_rows, self.n_cols))
        # Extract the row across all matrices/columns and flatten to 1D
        row_data = X_mmap[:, r, :].ravel()
        # Return the centered row data
        return r, row_data - np.mean(row_data)

    def process_col_chunk(self, c):
        """Worker function to process a single column index."""
        X_mmap = np.memmap(self.dat_fps["ms_matrices"], mode="r", dtype=DTYPE,
                           shape=(self.n_matrices, self.n_rows, self.n_cols))
        col_data = X_mmap[:, :, c].ravel()
        return c, col_data - np.mean(col_data)

    def save_centered_matrices(self):
        """Correlation 'kind' must be one of ["row", "col"] for 
        row-, column-wise centering & correlation
        """
        # always re-create as it will depends on the list of 'pairs of interest'
        centered_fp = self.centered_specs[self.kind]["centered_fp"]
        num_features = self.centered_specs[self.kind]["num_features"]
        worker_fn = self.centered_specs[self.kind]["worker_fn"]
        kind_shape = self.centered_specs[self.kind]["kind_shape"]
        #print(f"Specs: {self.centered_specs[self.kind]}\n")
        
        print(f" Starting parallel processing for {self.kind}-wise correlation...")
        # Initialize a disk-backed memmap to collect the results from workers
        X_centered = np.memmap(centered_fp, dtype=DTYPE, mode='w+', shape=kind_shape)
        # Spin up a process pool using all available CPU cores
        with Pool() as pool:
            # imap_unordered is fast and memory-efficient as it streams results back
            for index, centered_vector in pool.imap_unordered(worker_fn, range(num_features), chunksize=10):
                # Write the result directly into our output memmap
                X_centered[index] = centered_vector
        X_centered.flush()  # Ensure all writes are committed to disk
        del X_centered

        return

    def get_states_matrices_correlation(self):
        centered_fp = self.centered_specs[self.kind]["centered_fp"]
        kind_shape = self.centered_specs[self.kind]["kind_shape"]
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
                print(" All elements in the matrices are constant (have 0 std dev). No correlation possible.")
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
            print(" Setting undefined values for constant features in the final correlation matrix to 0")
            self.corr_matrix[zero_std_mask, :] = 0.0
            self.corr_matrix[:, zero_std_mask] = 0.0

        print(f" Obtained {self.kind.capitalize()}-wise correlation matrix, shape:", self.corr_matrix.shape)

        return

    def _get_states_corr_mat_df(self):
        if self.corr_matrix is None:
            return None

        mat_df = pd.DataFrame(self.corr_matrix)
        new_names = [self.mat_ix2res[c] for c in mat_df.columns.tolist()]
        # drop 0 rows, 0 cols:
        #msk_rows0 = mat_df.sum(axis=1) == 0
        #if msk_rows0.any():
        #    mat_df = mat_df.loc[~msk_rows0, :]
        #msk_cols0 = mat_df.sum(axis=0) == 0
        #if msk_cols0.any():
        #    mat_df = mat_df.loc[:, ~msk_cols0]
        #print(f"(Reduced) correlation matrix df shape: {mat_df.shape}")
        
        # name the indices, to be automatically retrieved in plot
        mat_df.index = new_names
        mat_df.columns = new_names
        mat_df.index.name = "Donor"
        mat_df.columns.name = "Acceptor"
    
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
            print("There is no correlation matrix to load.")
            return

        heatmap_from_df(df,
                        fig_size=fig_size,
                        fig_save_fp=fig_save_fp,
                        title=title)
        return

    def process_pairs(self):
        """Uses pandas for correlations.
        """
        print(f"\n OK to process pairs? {self.ok}")
        if not self.ok:
            return
        
        if not self.split_bk:
            # self.pairs_df is filtered
            # data heatmap:
            matrix_df = self.pairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
            title = f"H-bond Donor - Acceptor pairs\n({self.input_fp2.stem}, filtered)"
            png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered.png")
            heatmap_from_df(matrix_df,
                            fig_size=self.figs_args.figsize_data,
                            fig_save_fp=png_fp, title=title)
            
            # Get donors and Acceptors correlation:
            # Transpose and compute Donor (row) correlation
            row_corr_mat = matrix_df.T.corr()
            if row_corr_mat.shape[0] < 2:
                print("Not enough rows for correlation.")
            else:
                title = "H-bond Donors correlation (filtered)"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_donors_corr.png")
                heatmap_from_df(row_corr_mat,
                                fig_size=self.figs_args.figsize_donor_corr,
                                fig_save_fp=png_fp, title=title)
            
            # Compute Acceptor (col) correlation
            col_corr_mat = matrix_df.corr()
            if col_corr_mat.shape[0] < 2:
                print(" Not enough columns for correlation.")
            else:
                title = "H-bond Acceptors correlation (filtered)"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered_acceptors_corr.png")
                heatmap_from_df(col_corr_mat,
                                fig_size=self.figs_args.figsize_acceptor_corr,
                                fig_save_fp=png_fp, title=title)
        else:
            # split non-BK, then BK:
            res_msk = self.pairs_df["with_bk"].eq(False)
            if res_msk.any():
                res_respairs_df =  self.pairs_df.loc[res_msk, ["Donor", "Acceptor", "count", "occ"]]
                n_ud = len(res_respairs_df["Donor"].unique())
                n_ua = len(res_respairs_df["Acceptor"].unique())
                print(f" Non BK respairs, uniq donors: {n_ud}, uniq acceptors: {n_ua}", sep="\n")
                # data heatmap:
                matrix_df = res_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                title = f"H-bond Donor - Acceptor pairs, no BK\n({self.input_fp2.stem})"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}.png")
                heatmap_from_df(matrix_df,
                                fig_size=self.figs_args.figsize_data,
                                fig_save_fp=png_fp,
                                title=title)
                
                # get the 2 kinds or corr:
                row_corr_mat = matrix_df.T.corr()
                if row_corr_mat.shape[0] < 2:
                    print(" Not enough rows for correlation.")
                else:
                    title = f"H-bond Donors correlation, no BK\n({self.input_fp2.stem})"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_donors_corr.png")
                    heatmap_from_df(row_corr_mat,
                                    fig_size=self.figs_args.figsize_donor_corr,
                                    fig_save_fp=png_fp,
                                    title=title)
                
                col_corr_mat = matrix_df.corr()
                if col_corr_mat.shape[0] < 2:
                    print("Not enough columns for correlation.")
                else:
                    title = f"H-bond Acceptors correlation, no BK\n({self.input_fp2.stem})"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_acceptors_corr.png")
                    heatmap_from_df(col_corr_mat,
                                    fig_size=self.figs_args.figsize_acceptor_corr,
                                    fig_save_fp=png_fp,
                                    title=title)
            else:
                print("No hb pairs of residue-residue kind.")

            # now BK:
            if (~res_msk).any():
                bk_respairs_df = self.pairs_df.loc[~res_msk, ["Donor", "Acceptor", "count", "occ"]]
                n_bk_ud = len(bk_respairs_df["Donor"].unique())
                n_bk_ua = len(bk_respairs_df["Acceptor"].unique())
                print(f" BK respairs, uniq donors: {n_bk_ud} uniq acceptors: {n_bk_ua}", sep="\n")
                # data heatmap:
                matrix_df = bk_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                title = f"H-bond Donor - Acceptor pairs, BK\n({self.input_fp2.stem})"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_bk.png")
                heatmap_from_df(matrix_df,
                                fig_size=self.figs_args.figsize_data_bk,
                                fig_save_fp=png_fp,
                                title=title) 
                
                # get the 2 kinds or corr:
                row_corr_mat = matrix_df.T.corr()
                if row_corr_mat.shape[0] < 2:
                    print(" Not enough rows for correlation.")
                else:
                    title = f"H-bond Donors correlation, BK\n({self.input_fp2.stem})"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_donors_corr_bk.png")
                    heatmap_from_df(row_corr_mat,
                                    fig_size=self.figs_args.figsize_donor_corr_bk ,
                                    fig_save_fp=png_fp,
                                    title=title)
                
                col_corr_mat = matrix_df.corr()
                if col_corr_mat.shape[0] < 2:
                    print(" Not enough cols for correlation.")
                else:
                    title = f"H-bond Acceptors correlation, BK\n({self.input_fp2.stem})"
                    png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_acceptors_corr_bk.png")
                    heatmap_from_df(col_corr_mat,
                                    fig_size=self.figs_args.figsize_acceptor_corr_bk,
                                    fig_save_fp=png_fp,
                                    title=title)
            else:
                print(" No hb pairs of residue-backbone kind.")

        print(" Processing of pairs correlation over.\n")

        return


    def get_states_row_correl(self, figsize: tuple = DEFAULT_FIGSIZE):
        if not self.ok:
            return

        self.kind = "row"  # donors
        self.save_centered_matrices()
        self.get_states_matrices_correlation()
        title = "H-bond States Donors Correlation"
        save_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_donors_corr.png")
        self.save_corr_heatmap(fig_size=figsize,
                                fig_save_fp=save_fp,
                                title=title
                                )
        return

    def get_states_col_correl(self, figsize: tuple = DEFAULT_FIGSIZE):
        if not self.ok:
            return

        self.kind = "col"  # acceptors
        self.save_centered_matrices()
        self.get_states_matrices_correlation()
        title = "H-bond States Acceptors Correlation"
        save_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_acceptors_corr.png")
        self.save_corr_heatmap(fig_size=figsize,
                                fig_save_fp=save_fp,
                                title=title
                                )
        return

    def _get_states_pairs_data_heatmaps(self):
        """Create the data heatmap(s) from pairs in hb_states_pairs_ or
        hb_pairs_res_ csv files.
        """
        if not self.ok:
            return

        if not self.split_bk:  # => self.pairs_df is filtered
            matrix_df = self.pairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
            # data heatmap:
            title = "States H-bond Donor - Acceptor pairs data"
            png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_filtered.png")
            heatmap_from_df(matrix_df,
                            fig_size=self.figs_args.figsize_data,
                            fig_save_fp=png_fp, title=title)
        else:
            # split non-BK, then BK:
            res_msk = self.pairs_df["with_bk"].eq(False)
            if res_msk.any():
                res_respairs_df = self.pairs_df.loc[res_msk, ["Donor", "Acceptor", "count", "occ"]]
                n_res_ud = len(res_respairs_df["Donor"].unique())
                n_res_ua = len(res_respairs_df["Acceptor"].unique())
                print(f" States Non BK respairs, uniq donors: {n_res_ud}, uniq acceptors: {n_res_ua}", sep="\n")
        
                matrix_df = res_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                # data heatmap:
                title = "States H-bond Donor - Acceptor pairs data, no BK"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}.png")
                heatmap_from_df(matrix_df,
                                fig_size=self.figs_args.figsize_data,
                                fig_save_fp=png_fp,
                                title=title)
            else:
                print(f" No states hb pairs of res-res kind in {self.input_fp2.name}.")

            # now BK:
            if (~res_msk).any():
                bk_respairs_df = self.pairs_df.loc[~res_msk, ["Donor", "Acceptor", "count", "occ"]]
                n_bk_ud = len(bk_respairs_df["Donor"].unique())
                n_bk_ua = len(bk_respairs_df["Acceptor"].unique())
                print(f" BK respairs, uniq donors: {n_bk_ud} uniq acceptors: {n_bk_ua}", sep="\n")

                matrix_df = bk_respairs_df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
                # data heatmap:
                title = "States H-bond Donor - Acceptor pairs data, BK)"
                png_fp = self.input_fp2.with_name(f"{self.input_fp2.stem}_bk.png")
                heatmap_from_df(matrix_df,
                                fig_size=self.figs_args.figsize_data_bk,
                                fig_save_fp=png_fp,
                                title=title)
            else:
                print(f" No states hb pairs of res-bk kind in {self.input_fp2.name}.")

        return

    def process_states(self):
        """
        Process hb states data using parallelism.
        """
        print(f"\n OK to process states? {self.ok}")
        if not self.ok:
            return

        # data viz:
        self._get_states_pairs_data_heatmaps()

        dat_dir = self.mcce_dir.joinpath("dats")
        dat_dir.mkdir(exist_ok=True)
        self.dat_fps = {"ms_matrices": dat_dir.joinpath(f"hb_matrices_{self.pheh_str}.dat"),
                        "ms_matrices_rows": dat_dir.joinpath(f"hb_matrices_row_centered_{self.pheh_str}.dat"),
                        "ms_matrices_cols": dat_dir.joinpath(f"hb_matrices_col_centered_{self.pheh_str}.dat"),
                      }

        # dat files always re-created as they will depends on the list of 'pairs of interest'
        # save all matrices in memory mapped binary file:
        self.n_matrices: int = 0
        self.save_hb_microstates_matrices()  # populates self.n_matrices
        if not self.ok:
            return

        # specs for saving centered matrices:
        self.centered_specs = {
            "row": {
                "num_features": self.n_rows,
                "worker_fn": self.process_row_chunk,
                "centered_fp": self.dat_fps["ms_matrices_rows"],
                "kind_shape": (self.n_rows, self.n_matrices * self.n_rows),
            },
            "col": {
                "num_features": self.n_cols,
                "worker_fn": self.process_col_chunk,
                "centered_fp": self.dat_fps["ms_matrices_cols"],
                "kind_shape": (self.n_cols, self.n_matrices * self.n_cols),
            },
        }
    
        self.kind = "row"  # donors
        self.get_states_row_correl(figsize=self.figs_args.figsize_donor_corr)

        self.kind = "col"  # acceptors
        self.get_states_col_correl(figsize=self.figs_args.figsize_acceptor_corr)

        print(" Processing of states correlation over.")
        return

