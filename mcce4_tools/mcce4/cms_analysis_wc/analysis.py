#!/usr/bin/env python3

"""
Module: analysis.py

  Protonation microstate analysis with weighted correlation.
  Uses  MSout_np as the msout file loader class.
"""
from collections import defaultdict
import logging
from pathlib import Path
import sys
from typing import Dict, List, Union
import warnings

logger = logging.getLogger(__name__)
try:
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    from scipy.stats import rankdata
except ImportError as e:
    logger.critical("Oops! Forgot to activate an appropriate environment?\n", exc_info=e)
    sys.exit(1)

from mcce4.constants import res3_to_res1
from mcce4.cms_analysis_wc import IONIZABLES, ACIDIC_RES, BASIC_RES, POLAR_RES
from mcce4.cms_analysis_wc import parameters as prm
from mcce4.cms_analysis_wc.plots import energy_distribution, crgms_energy_histogram, corr_heatmap
from mcce4.io_utils import get_mcce_filepaths
from mcce4.msout_np import MSout_np, MAX_INT


CORR_METHODS = ["pearson", "spearman"]


class WeightedCorr:
    def __init__(
        self,
        xyw: pd.DataFrame = None,
        x: pd.Series = None,
        y: pd.Series = None,
        w: pd.Series = None,
        df: pd.DataFrame = None,
        wcol: str = None,
        cutoff: float = 0.02,
    ):
        """Weighted Correlation class.
        To instantiate WeightedCorr, either supply:
          1. xyw as pd.DataFrame,
          2. 3 pd.Series: (x, y, w),
          3. a pd.DataFrame and the name of the weight column.
        Args:
          xyw: pd.DataFrame with shape(n,3) containing x, y, and w columns;
          x: pd.Series (n,) containing values for x;
          y: pd.Series (n,) containing values for y;
          w: pd.Series (n,) containing weights;
          df: pd.Dataframe (n,m+1) containing m phenotypes and a weight column;
          wcol: str column of the weight in the df argument.
          cutoff: if given, return values whose absolute values are greater.
        Usage:
          ```
          # input as DataFrame and weight column name:
          wcorr = WeightedCorr(df=input_df, wcol="Count", cutoff=0.01)(method='pearson')

          # input as a DataFrame subset:
          wcorr = WeightedCorr(xyw=df[["xcol", "ycol", "wcol"]])(method='pearson')
          ```
        """
        self.cutoff = cutoff

        if (df is None) and (wcol is None):
            if np.all([i is None for i in [xyw, x, y, w]]):
                raise ValueError("No data supplied")

            if not (
                (isinstance(xyw, pd.DataFrame)) != (np.all([isinstance(i, pd.Series) for i in [x, y, w]]))
            ):
                raise TypeError("xyw should be a pd.DataFrame, or x, y, w should be pd.Series")

            xyw = pd.concat([x, y, w], axis=1).dropna() if xyw is None else xyw.dropna()
            self.x, self.y, self.w = (pd.to_numeric(xyw[i], errors="coerce").values for i in xyw.columns)
            self.df = None

        elif (wcol is not None) and (df is not None):
            if (not isinstance(df, pd.DataFrame)) or (not isinstance(wcol, str)):
                raise ValueError("df should be a pd.DataFrame and wcol should be a string")

            if wcol not in df.columns:
                raise KeyError("wcol not found in column names of df")

            if not df.shape[0] > 1:
                sys.exit("Too few rows for correlation.")

            cols = df.columns.to_list()
            _ = cols.pop(cols.index(wcol))
            self.df = df.loc[:, cols]
            self.w = pd.to_numeric(df.loc[:, wcol], errors="coerce")

        else:
            raise ValueError("Incorrect arguments specified, please specify xyw, or (x, y, w) or (df, wcol)")

    def _wcov(self, x, y, ms):
        return np.sum(self.w * (x - ms[0]) * (y - ms[1]))

    def _pearson(self, x=None, y=None) -> float:
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        mx, my = (np.sum(i * self.w) / np.sum(self.w) for i in [x, y])

        # needed for unchanging values (fixed res):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                wcov = self._wcov(x, y, [mx, my]) / np.sqrt(
                    self._wcov(x, x, [mx, mx]) * self._wcov(y, y, [my, my])
                )
            except RuntimeWarning:
                wcov = 0

        if abs(wcov) > self.cutoff:
            return wcov
        else:
            return 0

    def _wrank(self, x):
        (unique, arr_inv, counts) = np.unique(rankdata(x), return_counts=True, return_inverse=True)
        a = np.bincount(arr_inv, self.w)
        return (np.cumsum(a) - a)[arr_inv] + ((counts + 1) / 2 * (a / counts))[arr_inv]

    def _spearman(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        return self._pearson(self._wrank(x), self._wrank(y))

    def __call__(self, method: str = "pearson") -> Union[float, pd.DataFrame]:
        """
        WeightedCorr call method.
        Args:
          method (str, "pearson"): Correlation method to be used:
                                   'pearson' for pearson r, 'spearman' for spearman rank-order correlation.
        Return:
          - The correlation value as float if xyw, or (x, y, w) were passed to __init__;
          - A m x m pandas.DataFrame holding the correlaton matrix if (df, wcol) were passed to __init__.
        """
        method = method.lower()
        if method not in CORR_METHODS:
            raise ValueError(f"`method` should be one of {CORR_METHODS}.")

        # define which of the defined methods to use:
        cor = self._pearson
        if method == "spearman":
            cor = self._spearman

        if self.df is None:  # run the method over series
            return cor()
        else:
            # run the method over matrix
            df_out = pd.DataFrame(np.nan, index=self.df.columns, columns=self.df.columns)
            for i, x in enumerate(self.df.columns):
                for j, y in enumerate(self.df.columns):
                    if i >= j:
                        df_out.loc[x, y] = cor(
                            x=pd.to_numeric(self.df[x], errors="coerce"),
                            y=pd.to_numeric(self.df[y], errors="coerce"),
                        )
                        df_out.loc[y, x] = df_out.loc[x, y]

            if self.cutoff is not None:
                # values will be all 0 when cutoff is applied in cor():
                msk = df_out == 0
                df_out = df_out.loc[~msk.all(axis=1), ~msk.all(axis=0)]

            return df_out


def choose_res_data(top_df: pd.DataFrame, correl_resids: list) -> pd.DataFrame:
    """Prereq: correl_resids is either empty or is a subset of the residues in top_df.
    """
    df = top_df.copy()
    if correl_resids is None or not correl_resids:
        correl_resids = df.columns[:-3].tolist()

    out_cols = correl_resids + df.columns[-3:-1].tolist()  # no SumCharge col
    df = df[out_cols]
    # final reduction: get unique microstates viz those specific residues
    df = df.groupby(correl_resids).agg({"Count": "sum", "Occupancy": "sum"}).reset_index()
    df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)
    df.index = df.index + 1

    return df


def add_fixed_resoi_crg_to_topdf(top_df: pd.DataFrame, fixed_resoi_crg_df: pd.DataFrame,
                                 cms_wc_format: bool = False) -> pd.DataFrame:

    if fixed_resoi_crg_df is None or not fixed_resoi_crg_df.shape[0]:
        return top_df

    tots_col_idx = 3 if cms_wc_format else 4

    top_res_cols = top_df.columns[:-tots_col_idx].tolist()
    top_tots_cols = top_df.columns[-tots_col_idx:].tolist()
    n_rows = top_df.shape[0]

    # Create a dictionary of new columns for fixed residues
    new_cols_dict = {}
    for i, res in enumerate(fixed_resoi_crg_df.Residue.tolist()):
        # Broadcast the fixed charge value to all rows
        new_cols_dict[res] = np.repeat(fixed_resoi_crg_df.iloc[i, 1], n_rows)

    fixed_cols_df = pd.DataFrame(new_cols_dict)

    # Concatenate the original DataFrame parts with the new fixed residue columns
    df = pd.concat([top_df[top_res_cols], fixed_cols_df, top_df[top_tots_cols]], axis=1)

    return df


def rename_reorder_df_cols(choose_res_data_df: pd.DataFrame) -> pd.DataFrame:
    """Output a new df with resids in res_cols with this format:
    chain + 1-letter res code + seq num,  and with this group order:
    acid, polar, base, ub_q, non_res kinds.
    """
    res_cols = choose_res_data_df.columns[0:-2].tolist()
    # termini
    ter_cols = []
    for i, col in enumerate(choose_res_data_df.columns[0:-2].tolist()):
        if col[:3].startswith(("NTR", "CTR")):
            ter_cols.append(res_cols.pop(i))

    res_cols = res_cols + ter_cols
    choose_res_data_df = choose_res_data_df[res_cols + ["Count"]]
    res_cols = choose_res_data_df.columns[0:-1].tolist()

    col_order = defaultdict(list)
    mapping = {}
    for res in res_cols:
        r3 = res[:3]
        rout = f"{res[3]}{res3_to_res1.get(r3, r3)}{int(res[4:-1])}"
        mapping[res] = rout
        if r3 in ACIDIC_RES:
            col_order[1].append((rout, ACIDIC_RES.index(r3)))
        elif r3 in POLAR_RES:
            col_order[2].append((rout, POLAR_RES.index(r3)))
        elif r3 in BASIC_RES:
            col_order[3].append((rout, BASIC_RES.index(r3)))
        elif r3 == "PL9":
            col_order[4].append((rout, 8888))
        else:
            col_order[5].append((rout, 9999))
    new_order = [v[0] for k in col_order for v in sorted(col_order[k], key=lambda x: x[1])]

    return choose_res_data_df.rename(columns=mapping)[new_order + ["Count"]]


def combine_all_free_fixed_residues(free_res_crg_df: pd.DataFrame,
                                    fixed_res_crg_df: pd.DataFrame) -> pd.DataFrame:
    if free_res_crg_df is not None and free_res_crg_df.shape[0]:
        free_res_crg_df = free_res_crg_df.copy()
        free_res_crg_df["status"] = "free"

    if fixed_res_crg_df is not None and fixed_res_crg_df.shape[0]:
        fixed_res_crg_df = fixed_res_crg_df.copy()
        fixed_res_crg_df["status"] = "fixed"
    
    try:
        df = pd.concat([free_res_crg_df, fixed_res_crg_df])
        df.set_index("Residue", inplace=True)
    except Exception as e:
        logger.error("Could not combine free and fixed residues.", exc_info=e)
        return None

    return df


def cluster_corr_matrix(corr_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Return the clustered correlation matrix.
    Args:
      - corr_df (pd.DataFrame): correlation dataframe, i.e. df.corr();
      - n_clusters (int, 5): Number of candidate clusters, minimum 3;
    """
    # Convert correlation matrix to distance matrix
    dist_matrix = pdist(1 - np.abs(corr_df))

    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method="complete")

    if n_clusters < 3:
        n_clusters = 3

    clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    # Get the order of columns based on clustering
    ordered_cols = [corr_df.columns[i] for i in np.argsort(clusters)]

    # Return the reordered correlation matrix:
    return corr_df.loc[ordered_cols, ordered_cols]


class CMSWC_Pipeline:
    """
    Performs protonation microstate analysis with weighted correlation.
    """
    def __init__(self, main_params: Dict, histo_params: Dict):
        """
        Initializes the pipeline with parameters.

        Args:
            main_params (Dict): Dictionary of general parameters.
            histo_params (Dict): Dictionary of charge histogram parameters.
        """
        self.main_prms = main_params
        self.histo_prms = histo_params
        self.main_defaults: Dict = None

        self.mcce_dir: Path = None
        self.output_dir: Path = None
        self.h3_fp: Path = None
        self.msout_fp: Path = None
        self.ph: str = None
        self.eh: str = None

        self.residue_kinds: List = None
        self.correl_resids: List = None
        self.correl_all: bool = False
        self.show_fig: bool = False
        
        self.mc: MSout_np = None
        self.top_cms: List = None
        self.top_df: pd.DataFrame = None
        self.fixed_resoi_crg_df: pd.DataFrame = None

        logger.info("Initializing CMSWC_Pipeline...")
        self._setup_paths_and_params()

        return

    def _setup_paths_and_params(self):
        """Sets up file paths and validates parameters.
        """
        self.mcce_dir = Path(self.main_prms.get("mcce_dir", ".")).resolve()
        if not self.mcce_dir.exists():
            logger.error("mcce_dir not found.")
            sys.exit(1)
        
        # Extract pH and Eh from msout_file parameter
        msout_filename = self.main_prms.get("msout_file")
        if msout_filename is None:
            msout_filename = "pH7eH0ms.txt"
            logger.info("Using default msout file: %s", msout_filename)

        try:
            p, e = msout_filename[:-4].lower().split("eh")
            self.ph = p.removeprefix("ph")
            self.eh = e.removesuffix("ms")
        except ValueError:
            logger.error(f"Could not parse pH/Eh from msout_file name: {msout_filename}")
            sys.exit(1)
        self.main_defaults = prm.params_main(ph=self.ph, eh=self.eh)

        # Setup output directory
        self.output_dir = self.mcce_dir.joinpath(self.main_prms.get("output_dir",
                                                 self.main_defaults["output_dir"])
        )
        if not self.output_dir.exists():
            self.output_dir.mkdir()
        logger.info(f"Using output directory: {self.output_dir}")

        # Get MCCE input files (step2_out not used here)
        self.h3_fp, _, self.msout_fp = get_mcce_filepaths(self.mcce_dir, self.ph, self.eh)
        logger.info(f"Using head3.lst: {self.h3_fp}\nUsing msout file: {self.msout_fp}")

        # Process residue kinds
        self.residue_kinds = self.main_prms.get("residue_kinds", IONIZABLES)
        if (set(self.residue_kinds).symmetric_difference(IONIZABLES)
            and len(self.residue_kinds) > 1):
            self.residue_kinds = prm.sort_resoi_list(self.residue_kinds)
        logger.info(f"Residue kinds for analysis: {self.residue_kinds}")

        self.correl_resids = self.main_prms.get("correl_resids")
        self.show_fig = eval(self.main_prms.get("fig_show", "False"))

        return

    def load_data(self):
        """Loads data using MSout_np.
        """
        logger.info("Loading microstate data...")
        # Instantiate the 'fast loader' class; with_tautomers=False (default)
        self.mc = MSout_np(self.h3_fp, self.msout_fp, mc_load="crg",
                           res_kinds=self.residue_kinds)
        logger.info(self.mc)
        self.mc.get_uniq_ms()

        # Validate correlation residues after loading data
        if self.correl_resids:
            logger.info("Validating correlation residues...")
            self.correl_resids = prm.check_res_list(
                self.correl_resids,
                res_lst = self.residue_kinds,
                conf_info = self.mc.conf_info
            )
            if not self.correl_resids:
                logger.warning("Empty 'correl_resids' post-validation. Correlation will be skipped.")
                self.correl_resids = None
            elif len(self.correl_resids) < 2:
                logger.warning("Not enough 'correl_resids' (< 2) left post-validation. Correlation will be skipped.")
                self.correl_resids = None
            else:
                 logger.info(f"Residues for correlation: {self.correl_resids}")
        else:
            # TODO : implement
            logger.info("No specific residues provided for correlation analysis => do correlation on all.")
            self.correl_all = True

        return

    def process_residue_charges(self):
        """Calculates and saves average/fixed residue charges.
        """
        logger.info("Processing residue charges...")
        # Fixed res info:
        all_fixed_res_crg_df = self.mc.get_fixed_residues_df()
        # Free res average crg
        free_res_aver_crg_df = self.mc.get_free_res_aver_crg_df()

        # Combine free aver crg & fixed res with crg and save to csv:
        all_res_crg_df = combine_all_free_fixed_residues(free_res_aver_crg_df, all_fixed_res_crg_df)
        if all_res_crg_df is not None:
            csv_fp = self.output_dir.joinpath(
                self.main_prms.get("all_res_crg_csv",
                                self.main_defaults["all_res_crg_csv"])
            )
            msg = (
                f"Saving combined residue charge status to {csv_fp!s}.\n"
                "\tNote: For residues with 'free' status, the charge is the average charge."
            )
            logger.info(msg)
            all_res_crg_df.to_csv(csv_fp)

        # Fixed res of interest info:
        self.fixed_resoi_crg_df = self.mc.get_fixed_res_of_interest_df()
        n_fixed_resoi = self.fixed_resoi_crg_df.shape[0]
        if n_fixed_resoi:
            csv_fp = self.output_dir.joinpath(
                self.main_prms.get("fixed_res_of_interest_csv",
                                   self.main_defaults["fixed_res_of_interest_csv"])
            )
            msg = (
                f"Fixed residues in 'residue_kinds': {n_fixed_resoi}\n"
                f"\tSaving fixed_resoi_crg_df to {csv_fp!s}."
            )
            logger.info(msg)
            self.fixed_resoi_crg_df.to_csv(csv_fp, index=False)
        else:
            self.fixed_resoi_crg_df = None # Ensure it's None if empty
            logger.info("No fixed residues found within the specified 'residue_kinds'.")

        return

    def analyze_top_states(self):
        """Analyzes the top N microstates and prepares data for correlation.
        """
        logger.info("\nAnalyzing top microstates...")
        # default is ""
        n_top_str = self.main_prms.get("n_top", self.main_defaults["n_top"])
        if n_top_str == "":
            n_top = MAX_INT
        else:
            n_top = int(n_top_str)

        min_occ = float(self.main_prms.get("min_occ", self.main_defaults["min_occ"]))

        self.top_cms, _ = self.mc.get_topN_data(N=n_top, min_occ=min_occ)
        if not self.top_cms:
             logger.warning(f"No microstates found with occupancy >= {min_occ:.2%}. Correlation analysis will be skipped.")
             self.top_df = None # Ensure top_df is None if no states
             return

        # Create DataFrame in the format needed for WeightedCorr (cms_wc_format=True)
        self.top_df = self.mc.top_cms_df(self.top_cms, cms_wc_format=True)

        # Add fixed residues of interest if they exist
        if self.fixed_resoi_crg_df is not None:
            all_res_crg_df = add_fixed_resoi_crg_to_topdf(self.top_df,
                                                          self.fixed_resoi_crg_df,
                                                          cms_wc_format=True)
            all_res_crg_df.to_csv(self.output_dir.joinpath(self.main_prms.get("all_crg_count_resoi_csv",
                                                           self.main_defaults["all_crg_count_resoi_csv"])))
            logger.info("Saved ranked cms data including fixed residues of interest to all_crg_count_resoi_csv")

        return

    def perform_correlation(self):
        """Performs weighted correlation analysis and generates heatmap.
        """
        # if self.correl_resids is None or self.top_df is None:
        #     logger.info("Skipping correlation analysis (no residues specified or no top states found).")
        #     return
        if self.top_df is None:
            logger.info("Skipping correlation analysis (no top states found).")
            return
        logger.info("\nPerforming weighted correlation analysis...")
        # Select data for chosen residues
        choose_res_data_df = choose_res_data(self.top_df, self.correl_resids)

        # Save the selected data
        csv_path = self.output_dir.joinpath(self.main_prms.get("res_of_interest_data_csv",
                                            self.main_defaults["res_of_interest_data_csv"]))
        choose_res_data_df.to_csv(csv_path)
        logger.info(f"Saved data for correlation to {csv_path}")

        # Relabel residues for better plot labels
        df_chosen_res_renamed = rename_reorder_df_cols(choose_res_data_df)
        if df_chosen_res_renamed.shape[0] < 2:
            logger.warning("Too few residues (< 2) for correlation after filtering/selection.")
            return

        # Perform weighted correlation
        corr_method = self.main_prms.get("corr_method", self.main_defaults["corr_method"])
        corr_cutoff = float(self.main_prms.get("corr_cutoff", self.main_defaults["corr_cutoff"])) # Use 0.02 as default cutoff

        logger.info(f"Calculating weighted correlation (method={corr_method}, cutoff={corr_cutoff})...")
        df_correlation = WeightedCorr(df=df_chosen_res_renamed, wcol="Count", cutoff=corr_cutoff)(
            method=corr_method
        )
        if df_correlation is None or df_correlation.empty:
            logger.warning("Correlation matrix is empty after calculation/cutoff. Skipping heatmap.")
            return

        # Maybe generate heatmap
        savename = self.main_prms.get("corr_heatmap.save_name",
                                      self.main_defaults["corr_heatmap.save_name"])
        figsize = eval(self.main_prms.get("corr_heatmap.fig_size",
                                          self.main_defaults["corr_heatmap.fig_size"]))

        if df_correlation.shape[0] >= 2: # Need at least 2 residues for heatmap & clustering
            n_res_needed = int(self.main_prms.get("cluster_min_res_count",
                                                  self.main_defaults["cluster_min_res_count"]))
           
            if df_correlation.shape[0] >= n_res_needed:
                n_clusters = int(self.main_prms.get("n_clusters", self.main_defaults["n_clusters"]))
                logger.info(f"Clustering the correlation matrix (max {n_clusters} clusters).")
                try:
                    clustered_corr = cluster_corr_matrix(df_correlation, n_clusters=n_clusters)
                    corr_heatmap(
                        clustered_corr, out_dir=self.output_dir, save_name=savename,
                        show=self.show_fig, fig_size=figsize
                    )
                except Exception as e:
                    logger.error(f"Error during clustering or heatmap generation: {e}")
                    logger.info("Attempting heatmap without clustering...")
                    corr_heatmap(
                        df_correlation, out_dir=self.output_dir, save_name=savename,
                        show=self.show_fig, fig_size=figsize
                    )
            else: # Plot without clustering if only 2 residues
                 logger.info("Plotting heatmap (no clustering for less than 6 residues).")
                 corr_heatmap(
                    df_correlation, out_dir=self.output_dir, save_name=savename,
                    show=self.show_fig, fig_size=figsize
                 )
        else:
            logger.warning("Correlation matrix has less than 2 residues after processing. Skipping heatmap.")

        return

    def generate_energy_plots(self):
        """Generates energy distribution and histogram plots.
        """
        logger.info("\nGenerating energy distribution plots...")
        # Energies distribution plot:
        save_name = self.output_dir.joinpath(
            self.main_prms.get("energy_histogram.save_name",
                               self.main_defaults["energy_histogram.save_name"])
        )
        fig_size = eval(self.main_prms.get("energy_histogram.fig_size",
                                           self.main_defaults["energy_histogram.fig_size"]))
        energy_distribution(self.mc.all_cms, self.output_dir, kind="cms",
                            save_name=save_name, 
                            show=self.show_fig, fig_size=fig_size)

        # Charge Microstate Energy Histograms based on bounds
        cms_E_stats = self.mc.get_cms_energy_stats()
        logger.info(f"Charge microstate energy stats (min, avg, max): {cms_E_stats}")

        for hist_key, hist_d in self.histo_prms.items():
            title = hist_d.get("title", "Protonation Microstates Energy")
            bounds_str = hist_d.get("bounds")
            save_name = hist_d.get("save_name", f"{hist_key}_hist.png")
            ebounds = (None, None) # Default: no bounds

            if bounds_str:
                bounds_str = bounds_str.strip()
                if bounds_str == "(None, None)" or "None" in bounds_str:
                    pass # Keep default ebounds
                elif "Emin" in bounds_str:
                    try:
                        offset = float(bounds_str[:-1].split("+")[1].strip())
                        ebounds = (cms_E_stats[0], cms_E_stats[0] + offset)
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse Emin bounds: {bounds_str}. Using no bounds.")
                elif "Eaver" in bounds_str:
                     try:
                        offset = float(bounds_str[:-1].split("+")[1].strip())
                        ebounds = (cms_E_stats[1] - offset, cms_E_stats[1] + offset)
                     except (IndexError, ValueError):
                        logger.warning(f"Could not parse Eaver bounds: {bounds_str}. Using no bounds.")
                elif "Emax" in bounds_str:
                    try:
                        offset = float(bounds_str[1:].split("-")[1].split(",")[0].strip())
                        ebounds = (cms_E_stats[2] - offset, cms_E_stats[2])
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse Emax bounds: {bounds_str}. Using no bounds.")
                else: # Assume free bounds like "(val1, val2)"
                    try:
                        b, e = bounds_str[1:-1].strip().split(",")
                        ebounds = (float(b.strip()), float(e.strip()))
                    except (IndexError, ValueError):
                         logger.warning(f"Could not parse free bounds: {bounds_str}. Using no bounds.")

            # Filter and plot
            logger.info(f"Filtering charge microstates data for energy bounds: {bounds_str} -> {ebounds}")
            filtered_cms = self.mc.filter_cms_E_within_bounds(self.mc.all_cms, ebounds)
            if filtered_cms is not None and len(filtered_cms) > 0:
                logger.info(f"Plotting histogram for {len(filtered_cms)} states: {title}")
                crgms_energy_histogram(
                    filtered_cms, self.mc.CI.background_crg, title, self.output_dir,
                    save_name=save_name, show=self.show_fig
                )
            else:
                logger.warning(f"No charge microstates found within bounds {ebounds} for '{title}'. Skipping histogram.")
            
        # Per Issue #10: plot crgms_logcount for residues in self.correl_resids
        resoi_cms = self.mc.get_resoi_cms(self.correl_resids)
        if resoi_cms is not None and len(resoi_cms):
            logger.info(f"Plotting histogram for residues of interest")
            title = "Protonation MS Counts for Residues of Interest"
            save_name = "crgms_logcount_resoi.png"
            crgms_energy_histogram(resoi_cms, self.mc.CI.background_crg, title,
                                   self.output_dir,
                                   save_name=save_name, show=self.show_fig)

        return

    def run(self):
        """Executes the full analysis pipeline.
        """
        logger.info("Starting analysis pipeline...")
        self.load_data()
        self.process_residue_charges()
        self.generate_energy_plots()
        self.analyze_top_states()
        self.perform_correlation()
        logger.info("CMS Analysis pipeline end.")

        return
