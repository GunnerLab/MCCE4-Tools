#!/usr/bin/env python3

"""
Module: msout_hb.py

Module for MSout_hb class, the numpy-based ms_out/ file loader for outputing
H-bonds information from microstates using the 'hah' file', the output of the
detect_hbonds tool.
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict
from pathlib import Path
from pprint import pformat
import re
import sys
import time
from typing import Dict, List, Tuple, Union

try:
    from numba import njit
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
except ImportError as e:
    print(f"Oops! Forgot to activate an appropriate environment?\n{e}")
    sys.exit(1)

from mcce4.constants import res3_to_res1
from mcce4.io_utils import MsoutHeaderData
from mcce4.io_utils import N_HDR, N_STATES
from mcce4.io_utils import get_mcce_filepaths, get_msout_size_info
from mcce4.io_utils import reader_gen
from mcce4.io_utils import show_elapsed_time
from mcce4.io_utils import table_to_df


NO_MONTE_MSG = """
The MC method for H-bonding states must be MONTE.
If you can rerun step4.py, change the NSTATE_MAX key to a low
value, e.g.: 100, instead of 1M.
If you want the processing of the analytical solution to be reinstated 
please, open a feature request at https://github.com/GunnerLab/MCCE4-Tools/issues
"""


HAH_FNAME_INIT = "step2_out_hah.txt"
# output filename as fstring formats to receive MSout_hb.pheh_str,
# or msout_fp.stem[:-2], because they are ph/eh dependent.
# reduced hah.txt file, no hb pairs involving conformer that
# are fixed & always off:
fHAH_FNAME = "hah_{}.txt"
fHAH_EXPANDED = "expanded_hah_{}.csv"


# mapping of iconf (donor, acceptor) pairs for classification purposes:
pair_classes = {(-1, -1): -2,  # bk, bk: not in matrix
                (-1, 0): -1,   # bk, fixed: not in matrix
                (0, -1): -1,   # fixed, bk: not in matrix
                (0, 0): 0,     # fixed, fixed: not in matrix
                # in matrix:
                (1, 1): 1,     # free, free: initial matrix elements
                (1, 0): 2,     # free, fixed: extended matrix
                (1,-1): 2,     # free, bk: extended matrix
                (0, 1): 3,     # fixed, free: extended matrix
                (-1,1): 3,     # bk, free: extended matrix
                }


def is_int(val:str) -> bool:
    try:
        return str(int(val)) == val
    except ValueError:
        return False


def get_titr_vec(titr_fp: Path, titr_point: str, non_zeros: bool = None) -> np.ndarray:
    """"
    Return the vector of values at the given titr_point.
     - non_zeros (bool, None) can be used as a filter for the output values:
       If None (default): return the entire vector;
       If True: output the non-zero values;
       If False: output the zero values.
    """
    if is_int(titr_point):
        titr_point = titr_point + ".0"
    df = table_to_df(titr_fp)
    vec = None
    vec = df.filter(items=[df.columns[0], titr_point], axis=1)
    if vec.shape[1] < 2:
        print(f"Titration point not found in {titr_fp!s}: {titr_point}")
        return None

    if non_zeros is not None:
        vec = vec[vec[titr_point]!=0] if non_zeros else vec[vec[titr_point]==0]
        if not vec.shape[0]:
            return None

    return vec.to_numpy()


def get_hb_paths(mcce_dir: Path, ph: str = "7", eh: str = "0") -> Tuple[Path]:
    """Return the paths to head3, step2 pdb, msout file, the hah file, 
    and 'microstate ready', expanded hah file path.
    If no _hah.txt file is found (hah_fp is None), detect_hbonds will be run.
    """
    h3_fp, step2_fp, msout_fp = get_mcce_filepaths(mcce_dir, ph=ph, eh=eh)
    # reset to match precision in msout file name:
    # case where reduced function was already run:
    pheh = msout_fp.stem[:-2]
    # use the reduced file if found:
    hah_fp = mcce_dir.joinpath(fHAH_FNAME.format(pheh))
    if not hah_fp.exists():
        # use the output file from detect_hbonds:
        hah_fp = mcce_dir.joinpath(HAH_FNAME_INIT)
        if not hah_fp.exists():
            sys.exit("Run detect_hbonds first (step2_out_hah.txt not found)")
        else:
            with open(hah_fp) as fh:
                has_xyz = fh.readline().split()[-1] == "xyz"
            if not has_xyz:
                sys.exit("Rerun detect_hbonds for new format")

    return (h3_fp, step2_fp, msout_fp, hah_fp,
            mcce_dir.joinpath(fHAH_EXPANDED.format(pheh)),
            h3_fp.with_name(f"hb_pairs_{pheh}.csv"),
            h3_fp.with_name(f"hb_states_{pheh}.csv"))


def get_da_pairs(hah_fp: Path) -> np.ndarray:
    """Return array with 4 fields:
        confid_donor","confid_acceptor","d_occ","a_occ"
    """
    if hah_fp.suffix != ".txt":
        print(("FileTypeError:\n "
               "Expected a '.txt' file (output of detect_hbounds or its reduced version), "
               f"got: {hah_fp.name}"))
        return None

    if not hah_fp.exists():
        print(f"FileNotFoundError: {hah_fp!r}")
        return None

    return pd.read_csv(hah_fp,
                       usecols=["confid_donor","confid_acceptor","d_occ","a_occ"],
                       sep=r"\s+").drop_duplicates().to_numpy()


def get_states_keys(states_csv: Path) -> list:
    return pd.read_csv(states_csv,
                       usecols=["state_id"]).to_string(header=False,
                                                       index=False).splitlines()

    
def check_state_pairs(state_pairs: str, hah_da_pairs: np.ndarray) -> Union[str, None]:
    """Check that the state id (str(tuple of hb pairs) pairs are found
    in the hah file, return a string of invalid pairs if found, else None.
    """
    out = ""
    state_key_pattern = r"\((\w+),(\w+)\)"
    da_tuples_lst = re.findall(state_key_pattern, state_pairs)
    for tx, (d, a) in enumerate(da_tuples_lst, start=1):
        if (d, a) not in hah_da_pairs[:,[0,1]]:
            out += f"state #{tx}: ({d}, {a}) not found\n"
    if not out:
        return None
    return out


def check_states(hah_da_pairs: np.ndarray, states_csv: Path,
                 output_fname: str = "states_pairs_errors.txt"):
    ok = True
    states_keys = get_states_keys(states_csv)
    if not states_keys:
        return

    print(" Checking da_pairs in states against da_pairs in hah file...")
    out_fp = states_csv.parent.joinpath(output_fname)
    with open(out_fp, "a") as fha:
        for skey in states_keys:
            result = check_state_pairs(skey, hah_da_pairs)
            if result is not None:
                ok = False
                fha.write(result)
    if ok:
        print(" Passed.")
    else:
        print(f" Spurious states pairs saved in {out_fp!r}")

    return

def do_checks(mcce_dir: str, ph: str = "7", eh: str = "0"):
    """Wrapper to perform check on ms_hbnets main output files.
    """
    run_dir = Path(mcce_dir)
    print(f"Run dir: {run_dir!s}")
    (_,_,_, hah_fp, _, _, states_csv) = get_hb_paths(run_dir, ph=ph, eh=eh)

    hah_da_pairs = get_da_pairs(hah_fp)
    if hah_da_pairs is None:
        return

    if not states_csv.exists():
        print(f"Not found: {states_csv.name}, skipping check on states file.")
    else:
        check_states(hah_da_pairs[:,[0,1]], states_csv)

    print((" Checks on the hb pairs (not yet integrated), are obtained with this command:\n"
           f"  ms_sanity.py {hah_fp.relative_to(mcce_dir)}"))

    return


class ConfInfo:
    """This class handles the loading of head3 data into a numpy array
    and provides these accessor functions:
     - get_iconf(confid)
     - get_confid(iconf)
     - get_ires(iconf)
     - is_free_conf(confid)
     - is_fixed_off(confid)
    The 'conf_info' attribute (np.ndarray) is a lookup 'table' for
    these fields: confid:0, crg:1, iconf:2, is_fixed:3, ires:4, is_free:5

    Note: Only is_free is a true boolean field; fields preset with -1 need
          interpretation, e.g. after fixed iconfs are set to 1 if they are
          'fixed on', the remaining -1 values mean 'fixed off'.
    """
    def __init__(self, h3_fp: Path, verbose: bool = False):
        self.h3_fp = h3_fp
        self.verbose = verbose
        self.conf_info: np.ndarray = None
        self.n_confs: int = None
        self.max_iconf: int = None
        self.max_ires: int = None
        self.background_crg: int = None

    def load(self, iconf2ires: Dict, fixed_iconfs: List[int]):
        """Popuate the 'conf_info' attribute (np.ndarray): a lookup 'table' for:
        confid:0, crg:1, iconf:2, is_fixed:3, ires:4, is_free:5
        """
        print("\nPopulating the lookup array with head3.lst and msout file header data")
        conf_info = []
        with open(self.h3_fp) as h3:
            lines = h3.readlines()[1:]

        for line in lines:
            # ignored columns: FL & fields past confid
            iConf, confid, _, _, Crg, *_ = line.split()
            cx = int(iConf) - 1  # as python index
            crg = int(float(Crg))
            # extend ires indices; to use in matrix
            # confid:0, crg:1, cx:2, is_fixed:3, rx:4, is_free:5
            conf_info.append([confid, crg, cx, -1, -1, 0])

        # temp list structure is now sized & has h3 info; cast to np.ndarray:
        conf_info = np.array(conf_info, dtype=object)

        # is_free field
        free_ics = list(iconf2ires.keys())
        conf_info[free_ics, -1] = 1
        # fixed on
        conf_info[fixed_iconfs, 3] = 1

        # populate the rx of free res, if possible
        for i, (_, _, cx, *_) in enumerate(conf_info): 
            conf_info[i][-2] = iconf2ires.get(cx, -1)

        self.n_confs = conf_info.shape[0]
        self.max_iconf, self.max_ires = np.max(conf_info[:,[2,4]], axis=0)
        # sumcrg for not is_free & is_fixed on:
        self.background_crg = conf_info[np.where((conf_info[:,-1]==0) 
                                                  & (conf_info[:,3]==1)), 1].sum()
        print(f" Background crg: {self.background_crg}",
              f" n_confs: {self.n_confs}", sep="\n")
        if self.verbose:
            self.fieldnames()
        self.conf_info = conf_info

        return
    
    def fieldnames(self):
        print(" CI.conf_info fields: 0:confid, 1:crg, 2:iconf, 3:is_fixed, 4:ires, 5:is_free")
    
    def get_iconf(self, confid: str) -> int:
        """Get the conf index of a confid;
        confid:0, crg:1, cx:2, is_fixed:3, rx:4, is_free:5
        Note: -1 is return for confid of BK (not in head3.lst)
        """
        if "BK" in confid:  # not in conf_info
            return -1
        try:
            return self.conf_info[np.where(self.conf_info[:,0]==confid)][0,2]
        except IndexError:
            return -1

    def get_confid(self, iconf: int) -> str:
        """Get the confid of a conf index;
        confid:0, crg:1, cx:2, is_fixed:3, rx:4, is_free:5
        """
        try:
            return self.conf_info[np.where(self.conf_info[:,2]==iconf)][0,0]
        except IndexError:
            return "?"

    def get_ires(self, iconf: int) -> int:
        """Get the res index given a conformer index;
        confid:0, crg:1, cx:2, is_fixed:3, rx:4, is_free:5
        """
        try:
            return self.conf_info[np.where((self.conf_info[:,-1]==1) 
                                          & (self.conf_info[:,2]==iconf))][0,-2]
        except IndexError:
            return -1

    def is_free_conf(self, confid: str) -> int:
        """confid:0, crg:1, cx:2, off:3, rx:4, is_free:5
        Note: -1 is return for confid of BK
        """
        if "BK" in confid:
            return -1
        try:
            return self.conf_info[np.where(self.conf_info[:,0]==confid)][0,-1]
        except IndexError:
            return -1

    def is_fixed_off(self, confid: str) -> int:
        """confid:0, crg:1, cx:2, is_fixed:3, rx:4, is_free:5
        Notes:
          -1 is return for confid of BK, so it's not eliminated
           is_fixed values are either 1 or -1; -1 means not in ms fixed on => off
        """
        if "BK" in confid:
            return -1
        try:
            val = self.conf_info[np.where((self.conf_info[:,0]==confid)
                                          & (self.conf_info[:,-1]==0))][0,3]
            if val == -1:
                val = 1
            else:
                val = 0
        except IndexError:
            val = 0

        return val


@njit
def _process_hbpairs_numba(microstate: np.ndarray,
                           count: int,
                           hb_adj_indices: np.ndarray,
                           hb_adj_indptr: np.ndarray,
                           ms_mask: np.ndarray,
                           hb_pairs: np.ndarray,
                           #effective_count = True
                           ):
    """Update hb_pairs matrix with microstate data.

    hb_pairs is the hb pairs matrix, MSout_hb.P;
    ms_mask is a boolean mask preset to True with the given ms iconfs, i.e.:
    ```
      ms_mask = np.zeros(self.n_hb_confs, dtype=np.uint8)
      ms_mask[microstate] = 1
    ```
    """
    for d in microstate:
        for p in range(hb_adj_indptr[d], hb_adj_indptr[d + 1]):
            a = hb_adj_indices[p]
            if ms_mask[a]:
                hb_pairs[d, a] += count
            # else:
            # MG's algo does not make sense; pair count is incremented only
            # if found in microstate; if not found, why would the count be
            # decremented by the current count, which was never applied in the first place?
            # + yields incorrect occupancies
            #     if effective_count:
            #         prev = hb_pairs[d, a]
            #         hb_pairs[d, a] = max(0, prev - count)  # 0 if decreased value is < 0

    return


class MSout_hb:
    """Class to process the'msout file' for H-bonds data.

    Arguments:
        - head3_file, msout_file (str): Paths to head3.lst & msout files.
    """
    def __init__(self, mcce_dir: str, ph: str = "7", eh: str = "0",
                 n_target_states: int = N_STATES,
                 load_states: bool = False,
                 #pairs_sum_count: bool = False,
                 verbose: bool = False):
        self.verbose = verbose
        #self.effective_count = not pairs_sum_count
        self.load_states = load_states
        self.run_dir = Path(mcce_dir)

        start_tot = time.time()
        (self.h3_fp, self.step2_fp, self.msout_fp,
         self.hah_fp, self.hah_ms_fp,
         self.pairs_csv, self.states_csv) = get_hb_paths(self.run_dir, ph=ph, eh=eh)

        # ph, eh as string to match msoutfile:
        self.pheh_str = self.msout_fp.stem[:-2]

        self.mc_lines, self.n_skip, self.n_MC = get_msout_size_info(self.msout_fp,
                                                n_target_states=n_target_states)
        print(f"Approximate number of lines in msout file: {self.mc_lines:,}")

        # data from the msout file 'header':
        self.HDR = MsoutHeaderData(self.msout_fp)
        if not self.HDR.is_monte:
            sys.exit(NO_MONTE_MSG)

        self.CI = ConfInfo(self.h3_fp, verbose=self.verbose)
        # load the self.CI.conf_info lookup array:
        # fields: confid:0, crg:1, iconf:2, is_fixed:3, ires:4, is_free:5
        self.CI.load(self.HDR.iconf2ires, self.HDR.fixed_iconfs)
        # + CI.n_confs, CI.max_iconf, CI.max_ires

        # attributes populated by get_extended_iconfs:
        self.n_fx: int = 0
        self.fx_iconfs: List[int] = []
        self.n_bk: int = 0
        self.dm_iconfs: List[int] = []
        self.bkid2dmic = {}
        self.n_hb_confs: int = 0
        self.extended_iconfs: List[int] = []

        # free and mixed hb pairs + needed indices
        print(f"Creating the expanded hah file {self.hah_ms_fp!s}...")
        start_t = time.time()
        self.df = self.expand_hah_data()
        show_elapsed_time(start_t, info="Expanding the hah file into a dataframe")

        if not self.expd_df_checks():
            sys.exit("Some conformer indices appear to be missing.")

        # self.iconf2confid is used to convert indices to confids in the 
        # pairs file; the 1st dict is extra
        self.confid2iconf, self.iconf2confid = self.get_confs_mappings()
        # states space size, incremented by either load_ functions
        self.n_space: int = 0

        if self.load_states:
            self.hb_states: dict = None
            self.I: csr_matrix = None

            print("Running the pipeline for H-bonding states...")
            start_t = time.time()
            self.I = self.get_sparse_matrix()
            self.load_hb_states()
            show_elapsed_time(start_t, info="H-bonding states pipeline")
        else:
            self.hb_pairs: dict = None
            self.hb_adj = {}
            self.hb_adj_indices: np.ndarray
            self.hb_adj_indptr: np.ndarray
            self.P: np.ndarray   # dense pairs matrix

            print("Running the pipeline for H-bonding pairs...")
            start_t = time.time()
            self.hb_adj = self.get_adjacency_dict()
            self.hb_adj_indices, self.hb_adj_indptr = self.get_adj_idx_idxptr()
            self.P = np.zeros((self.n_hb_confs, self.n_hb_confs), dtype=np.int32)
            self.load_hb_pairs()
            show_elapsed_time(start_t, info="H-bonding pairs pipeline")

        print("Converting Mij to confid, updating the expanded file with pairs data, if any",
              "& saving dicts to csv files...", sep="\n")
        start_t = time.time()
        self.dicts2csv()
        show_elapsed_time(start_t, info="Converting, updating & saving dicts to csv")

        show_elapsed_time(start_tot, info="Start to end")

        return
            
    def load_hah_file(self) -> Union[pd.DataFrame, None]:
        """Wrapper for hah file preparation.
         - If path not set, or old format: run detect_hbonds
         - If file has default output name 'step2_out_hah.txt',
           create a reduced file.
         """
        df = table_to_df(self.hah_fp)
        print(f"H-bonding pairs in {self.hah_fp.name!s}: {df.shape[0]}")
        if self.hah_fp.name==HAH_FNAME_INIT:
            df = self.reduce_hah_file(df)
        if self.verbose:
            print(f"Loaded {self.hah_fp.name!s} into a dataframe with {df.shape[0]} rows")

        return df

    def reduce_hah_file(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce the inital hah.txt file by removing entries that have:
          - H-bonding pairs with fixed conformers that are always OFF
          - H-bonding pairs between 2 BK conformers
          - 0 occupancy in fort.38
        The the reduced file is save as hah_{pheh}.txt, as it is ph dependent and
        used in extend_hah_data function.
        """
        assert self.hah_fp.name == HAH_FNAME_INIT

        # remove confs with 0 occupancy
        occ_fp = self.step2_fp.with_name("fort.38")
        vec = get_titr_vec(occ_fp, str(self.HDR.pH))
        if vec is not None:
            # assign occ to each conf; keep occ cols
            def get_occ(cid):
                try:
                    return vec[np.where(vec[:,0]==cid)][0,1]
                except IndexError:
                    if "BK" in cid:
                        return 1
                    else:
                        return -1

            df["d_occ"] = df["confid_donor"].apply(get_occ)
            df["a_occ"] = df["confid_acceptor"].apply(get_occ)
            msk_occ = (df["d_occ"]==0) | (df["a_occ"]==0)
            if msk_occ.any():
                df = df.drop(index=df.loc[msk_occ].index, axis=0)

        # remove BK-BK:
        msk_bk = (df["confid_donor"].str.contains("BK")) & (df["confid_acceptor"].str.contains("BK"))
        if msk_bk.any():
            df = df.drop(index=df.loc[msk_bk].index, axis=0)

        df["d_off"] = df["confid_donor"].apply(self.CI.is_fixed_off)
        df["a_off"] = df["confid_acceptor"].apply(self.CI.is_fixed_off)
        msk_off = (df["d_off"]==1) | (df["a_off"]==1)
        if msk_off.any():
            df = df.drop(index=df.loc[msk_off].index, axis=0)
        df = df.drop(columns=["d_off", "a_off"])

        reduced_fp = self.hah_fp.with_name(fHAH_FNAME.format(self.pheh_str))
        # reset hah path:
        self.hah_fp = reduced_fp
        self.hah_fp.write_text(df.to_string(index=False)+"\n")
        print(f"H-bonding pairs in reduced file {self.hah_fp.name!s}: {df.shape[0]}")

        return df

    def get_fixed_or_bk_confids(self, df: pd.DataFrame) -> Tuple[List[str],List[str],List[str],List[str]]:
        """Return lists for fixed and bk confids for each donor/acceptor:
           fixd, fixa, bkd, bka (in that order).
        """
        # Operations on pair_class > 1, includes all mixed pairs
        # class 2::donor is free; class 3::acceptor is free
        # lists hold confids, donor/acceptor split is needed for:
        # - getting unique values for each kind
        # - filtering the master df to assign final Mij using dummy ics
        fixd, fixa, bkd, bka = [],[],[],[]
        params = {2:{"not_free": "free_a",
                     "col": "confid_acceptor",
                     0: {"lst":fixa}, -1: {"lst":bka},
                    },
                  3:{"not_free": "free_d",
                     "col": "confid_donor",
                      0: {"lst":fixd}, -1: {"lst":bkd},
                     }}

        K = ["fixed", "BK"]
        for cx in [2, 3]:  # mixed pairs
            class_msk = df["free"]==cx
            if class_msk.any():
                not_free, kcol = params[cx]["not_free"], params[cx]["col"]
                for i, k in enumerate([0, -1]):
                    k_msk = df[not_free]==k
                    msk = class_msk & k_msk
                    if msk.any():
                        kdf = df.loc[msk]
                        gp = kdf.groupby(kcol)
                        if len(gp.groups):
                            keys = list(gp.indices.keys())  # unique & sorted
                            params[cx][k]["lst"].extend(keys)
                            if self.verbose:
                                print(f" Class {cx}: Extended slots for {K[i]} {kcol!r}: {len(keys)}")
                        else:
                            if self.verbose:
                                print(f" Class {cx}: No extended slots for {K[i]} {kcol!r}")
                    else:
                        if self.verbose:
                            print(f" Class {cx}: No extended slots for {K[i]} {kcol!r}")
            else:
                if self.verbose:
                    print(f" No extended slots for empty pair class {cx}")

        return fixd, fixa, bkd, bka

    def get_extended_iconfs(self, fixd, fixa, bkd, bka):
        """Obtain the list of conformer indices to add to the current state of msout file.
        """
        extend_fixed = []
        extend_bk = []
        if fixd and fixa:
            extend_fixed = list(set(fixd).union(fixa))
        elif fixd:
            extend_fixed = fixd
        elif fixa:
            extend_fixed = fixa
        if bkd and bka:
            extend_bk = sorted(set(bkd).union(bka), key=lambda x: x[-9:-4])
        elif bkd:
            if len(bkd) > 1:
                extend_bk = sorted(bkd, key=lambda x: x[-9:-4])
            else:
                extend_bk = bkd
        elif bka:
            if len(bka) > 1:
                extend_bk = sorted(bka, key=lambda x: x[-9:-4])
            else:
                extend_bk = bka
        
        print(" Number of extension slots from mixed pairs with fixed:",len(extend_fixed))
        print(" Number of extension slots from mixed pairs with BK:",len(extend_bk))
        self.n_fx = len(extend_fixed)
        self.n_bk = len(extend_bk)
        if self.n_fx:
            # get iconfs values for fixed
            self.fx_iconfs = [self.CI.get_iconf(xid) for xid in extend_fixed]
            self.extended_iconfs.extend(self.fx_iconfs)

        if self.n_bk:
            # to avoid collisions, dummy ics for bk confs start past last iconf:
            dmic_start = self.CI.n_confs
            # create dm iconfs values for bk:
            self.dm_iconfs = [dmic_start + i for i in range(self.n_bk)]
            self.extended_iconfs.extend(self.dm_iconfs)
            self.bkid2dmic = dict(p for p in zip(extend_bk, self.dm_iconfs))

        # set the total count of confs:
        if self.n_bk:
            # tot from head3 needed as dummy bk indices start past the last h3 iconf
            # to avoid collision or confusion:
            self.n_hb_confs = self.CI.n_confs + self.n_bk
        else:
            if self.n_fx:
                # h3 confs include 'fixed on' confs; reduce_hah_file removes 'fixed off'
                self.n_hb_confs = self.CI.n_confs
            else:
                # smallest collection:
                self.n_hb_confs = self.HDR.n_free_ics
        print(f" n_hb_confs: {self.n_hb_confs:,}")

        return

    def expand_hah_data(self) -> pd.DataFrame:
        """Load the hah file into a dataframe & add:
         -conformer indices from h3
         -free conformer flags
         -the conformer indices used to build the hb pairs matrix P or adj list
        """
        def is_free_pair(ro) -> int:
            """Assign classes to pair kinds."""
            return pair_classes[(ro["free_d"], ro["free_a"])]
    
        df = self.load_hah_file()

        # these may help creating an adjacency list.
        df["dina"] = df["confid_donor"].isin(df["confid_acceptor"]).astype("int32")
        df["aind"] = df["confid_acceptor"].isin(df["confid_donor"]).astype("int32")

        # conf indices
        df["iconf1"] = df["confid_donor"].apply(self.CI.get_iconf)
        df["iconf2"] = df["confid_acceptor"].apply(self.CI.get_iconf)
        # dummy iconfs cols for pairs with BK -> iconf == -1:
        df["iconf1dm"] = -1
        df["iconf2dm"] = -1

        # free flags for each d/a: for later use:
        df["free_d"] = df["confid_donor"].apply(self.CI.is_free_conf)
        df["free_a"] = df["confid_acceptor"].apply(self.CI.is_free_conf)
        df["free"] = df.apply(is_free_pair, axis=1)

        # Add cols for final (extended) matrix i,j; init with dummy value
        df["Mi"] = -1
        df["Mj"] = -1

        fixd, fixa, bkd, bka = self.get_fixed_or_bk_confids(df)
        self.get_extended_iconfs(fixd, fixa, bkd, bka)
           
        def get_bk_ic(ro: pd.Series) -> int:
            """Return dummy iconf for BK confs. Function passed to df.apply()
            Bad values: -2 :: dict is None; -3: confid not in dict.
            """
            if self.bkid2dmic is None:
                return -2
            return self.bkid2dmic.get(ro, -3)

        # get final Mij for building adj list
        # classes 1,2 & 3:
        valid = df["free"].gt(0)

        # free or fixed confs:
        d_msk = valid & (df["free_d"] != -1)
        if d_msk.any():
            df.loc[d_msk, "Mi"] = df.loc[d_msk, "iconf1"]
        a_msk = valid & (df["free_a"] != -1)
        if a_msk.any():
            df.loc[a_msk, "Mj"] = df.loc[a_msk, "iconf2"]

        # bk confs:
        bk_msk1 = valid & (df["free_d"] == -1)
        if bk_msk1.any():
            df.loc[bk_msk1, "iconf1dm"] = df.loc[bk_msk1, "confid_donor"].apply(get_bk_ic)
            df.loc[bk_msk1, "Mi"] = df.loc[bk_msk1, "iconf1dm"]

        bk_msk2 = valid & (df["free_a"] == -1)
        if bk_msk2.any():
            df.loc[bk_msk2, "iconf2dm"] = df.loc[bk_msk2, "confid_acceptor"].apply(get_bk_ic)
            df.loc[bk_msk2, "Mj"] = df.loc[bk_msk2, "iconf2dm"] 

        df["d_occ"] = df["d_occ"].astype("float32")
        df["a_occ"] = df["a_occ"].astype("float32")
        int_cols = df.select_dtypes(include=["int64"]).columns
        df[int_cols] = df[int_cols].astype("int32")

        # save
        df.to_csv(self.hah_ms_fp, index=False)
        print(f"Accepted H-bonding pairs in df: {df.loc[df['free'].gt(0)].shape[0]}")
    
        return df

    def expd_df_checks(self) -> bool:
        passed = True
        valid = self.df["free"].gt(0)
        # final iconfs columns:
        for col in ["Mi", "Mj"]:
            if not all(self.df[col].isna() == False):
                print(f"Failed check: column {col!r} should not have NA values")
                passed = False
            neg = self.df.loc[valid & (self.df[col]<0)]
            if neg.shape[0] != 0:
                # save to investigate:
                neg_fp = self.hah_ms_fp.with_suffix(f"_{col}.unmapped.csv")
                neg.to_csv(neg_fp)
                print(f"Failed check: column {col!r} should not have negative values",
                      f"Unmapped iconfs saved to {neg_fp!s}", sep="\n")
                passed = False
        return passed

    def get_confs_mappings(self) -> Tuple[dict, dict]:
        """Return confid2iconf and iconf2confid dicts.
        """
        confid2iconf = {}
        if not (self.n_fx + self.n_bk):
            # only free ics:
            free = self.CI.conf_info[:,-1] == 1
            # confid2iconf, iconf2confid
            return dict(self.CI.conf_info[free][:,[0,2]]), dict(self.CI.conf_info[free][:,[2,0]])

        if self.n_bk:
            confid2iconf = dict(self.CI.conf_info[:,[0,2]])
            confid2iconf.update(self.bkid2dmic)
        else:
            if self.n_fx:
                return dict(self.CI.conf_info[:,[0,2]]), dict(self.CI.conf_info[:,[2,0]])

        # reverse to get iconf2confid:
        iconf2confid = dict((v, k) for k, v in confid2iconf.items())

        return confid2iconf, iconf2confid

    def get_sparse_matrix(self) -> csr_matrix:
        """Create a sparse matrix set with 1 for each hb pairs (Mi,Mj)
        from the dataframe.
        """
        # Drop duplicates Mij indices:
        # - dups occur when a donor can donate multiple H-atoms
        # - needed for scipy sparse matrix: the 1 flag for h-bond pairs
        #   would be summed for dups anyways.
        # Dropping dups or the reducing of indices by scipy may be a problem
        # when converting the 'free ires iconf' indices back to h3 iconfs
        # mask with the valid pair classes:
        msk = self.df["free"].gt(0)
        #df = self.df.loc[msk]
        df = self.df.loc[msk].drop_duplicates(subset=["Mi", "Mj"])
        n_rows = df.shape[0]
        print(f"Unique H-bonding pairs for matrix: {n_rows}")
        return csr_matrix(([1]*n_rows, (df["Mi"].values, df["Mj"].values)),
                           dtype=np.uint8,
                           shape=(self.n_hb_confs, self.n_hb_confs)
                           )

    def get_adjacency_dict(self) -> dict:
        hb_adj = defaultdict(set)
        for dx, ax in sorted(self.df[["Mi","Mj"]].values.tolist()):
            hb_adj[dx].add(ax)

        return dict(hb_adj)

    def get_adj_idx_idxptr(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the column indices and index pointer arrays for data in adj_lst as per
        CSR format.
        """
        # CSR format
        # data         : An array containing all the non-zero values of the matrix,
        #                stored row by row.
        # indices      : An array containing the column index for each value in the
        #                data array
        # index pointer: An array that points to the start and end of the data for each
        #                row within the data and indices arrays.
        # The column indices for row i are stored in indices[indptr[i]:indptr[i+1]], and
        # their corresponding values are in data[indptr[i]:indptr[i+1]].

        indptr = np.zeros(self.n_hb_confs + 1, dtype=np.int32)
        #indptr = np.zeros(self.CI.n_confs + 1, dtype=np.int32)
        indices = []
        #for d in range(self.CI.n_confs):
        for d in range(self.n_hb_confs):
            neighbors = self.hb_adj.get(d, ())
            indices.extend(sorted(neighbors))
            indptr[d + 1] = indptr[d] + len(neighbors)

        return np.array(indices, dtype=np.int32), indptr

    def process_ms_pairs(self, microstate: np.ndarray, count: int):
        """
        Processes a single microstate and updates H-bond pairs matrix.
        """
        # Boolean membership mask
        ms_mask = np.zeros(self.n_hb_confs, dtype=np.uint8)
        ms_mask[microstate] = 1
        _process_hbpairs_numba(microstate, count,
                               self.hb_adj_indices, self.hb_adj_indptr, ms_mask,
                               self.P
                               #, effective_count=self.effective_count
                               )
        return
    
    def load_hb_states(self):
        """Process the 'msout file' for H-bond microstates.
        """
        found_mc = False
        newmc = False
        hb_states =  defaultdict(lambda: [0, 0.])
        mc_lines = 0

        msout_data = reader_gen(self.msout_fp)
        for lx, line in enumerate(msout_data, start=1):
            if lx <= N_HDR:
                continue
            line = line.strip()
            if not line or line[0] == "#":
                continue
            else:
                # find the next MC record
                if line.startswith("MC:"):
                    found_mc = True
                    newmc = True
                    continue

                if newmc:
                    # line with candidate state for MC sampling, e.g.:
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free residues in state
                    current_state = [int(i) for i in line.split(":")[1].split()]
                    if self.extended_iconfs:
                        xs = np.array(current_state + self.extended_iconfs, dtype=np.int32)
                    else:
                        xs = np.array(current_state, dtype=np.int32)
                    newmc = False
                    continue

                if found_mc:
                    fields = line.split(",")
                    if len(fields) < 3:
                        continue

                    mc_lines += 1
                    count = int(fields[1])
                    self.n_space += count
                    # flipped: 
                    for ic in [int(c) for c in fields[2].split()]:
                        xs[self.HDR.iconf2ires[ic]] = ic

                    if mc_lines % self.n_skip == 0:
                        si, sj = self.I[np.ix_(xs, xs)].nonzero()
                        sij = tuple(zip(xs[si], xs[sj]))
                        hb_states[sij][0] += count

                    if mc_lines % (self.n_skip*5000) == 0:
                        print(f" Trace: processed mc lines {mc_lines:,}...")

        # update states dict with occ:
        for s in hb_states:
            hb_states[s][1] = hb_states[s][0]/self.n_space

        self.hb_states = dict(hb_states)
        print(f"State space: {self.n_space:,}")
        print(f"H-bonding states: {len(self.hb_states):,}")
        if self.verbose:
            sum_cnt = np.array(list(self.hb_states.values()))[:,0].sum()
            sum_occ = np.array(list(self.hb_states.values()))[:,1].sum()
            print("Check on sum totals:")
            print(f" states count: {sum_cnt:,.0f}")
            print(f"   states occ: {sum_occ:.2%}")

        if mc_lines != self.mc_lines:
            # accepted ms with flipped iconfs
            print(f"Processed mc lines: {mc_lines:,}")
        else:
            print(f"Accepted states lines: ~ {self.mc_lines:,}\n")

        return
    
    def load_hb_pairs(self):
        """Process the 'msout file' for H-bonding pairs.
        """
        found_mc = False
        newmc = False
        mc_lines = 0

        msout_data = reader_gen(self.msout_fp)
        for lx, line in enumerate(msout_data, start=1):
            if lx <= N_HDR:
                continue
            line = line.strip()
            if not line or line[0] == "#":
                continue
            else:
                # find the next MC record
                if line.startswith("MC:"):
                    found_mc = True
                    newmc = True
                    continue

                if newmc:
                    # line with candidate state for MC sampling, e.g.:
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free residues in state
                    current_state = [int(i) for i in line.split(":")[1].split()]
                    if self.extended_iconfs:
                        xs = np.array(current_state + self.extended_iconfs, dtype=np.int32)
                    else:
                        xs = np.array(current_state, dtype=np.int32)

                    newmc = False
                    continue

                if found_mc:
                    fields = line.split(",")
                    if len(fields) < 3:
                        continue

                    mc_lines += 1
                    count = int(fields[1])
                    self.n_space += count

                    # flipped: 
                    for ic in [int(c) for c in fields[2].split()]:
                        xs[self.HDR.iconf2ires[ic]] = ic

                    self.process_ms_pairs(xs, count)

                    if mc_lines % (self.n_skip*5000) == 0:
                        print(f" Trace: processed mc lines {mc_lines:,}...")

        # create pairs dict:
        pi, pj = self.P.nonzero()
        hb_pairs =  defaultdict(lambda: [0, 0.])
        for p in zip(pi, pj):
            cnt = self.P[p[0], p[1]]
            hb_pairs[p][0] += cnt
            hb_pairs[p][1] = hb_pairs[p][0]/self.n_space

        self.hb_pairs = dict(hb_pairs)
        print(f"State space: {self.n_space:,}")
        print(f"H-bonding pairs: {len(self.hb_pairs):,}")

        if mc_lines != self.mc_lines:
            # accepted ms with flipped iconfs
            print(f"Processed mc lines: {mc_lines:,}")
        else:
            print(f"Accepted states lines: ~ {self.mc_lines:,}\n")

        return

    def dicts2csv(self):
        if not self.load_states:
            if self.hb_pairs:
                def get_resid(confid:str) -> str:
                    id1 = res3_to_res1.get(confid[:3], confid[:3])
                    return f"{id1}_" + confid[5] + str(int(confid[6:-4]))

                # temporarily, save the unconverted dicts to text files
                # so that they can be loaded without re-running ms_hbnets
                # in order to implement the grouping
                pdict_fp = self.pairs_csv.with_suffix(".dict")
                pdict_fp.write_text(pformat(self.hb_pairs)+"\n")

                dfp = pd.DataFrame.from_dict(self.hb_pairs, orient="index",
                                            columns=["ms_count","ms_occ"]).reset_index()
                dfp[["Mi","Mj"]] = dfp["index"].apply(lambda x: pd.Series([int(x[0]),int(x[1])]))
                dfp[["donor","acceptor"]] = dfp["index"].apply(
                    lambda x: pd.Series([self.iconf2confid[x[0]],
                                         self.iconf2confid[x[1]]])
                                         )
                dfp[["res_d","res_a"]] = dfp["index"].apply(
                    lambda x: pd.Series([get_resid(self.iconf2confid[x[0]]),
                                         get_resid(self.iconf2confid[x[1]])])
                                         )
                
                pairs_out = dfp[["Mi","Mj","donor","acceptor","res_d","res_a","ms_count","ms_occ"]]
                pairs_out = pairs_out.sort_values(by="ms_count", ascending=False)
                pairs_out.to_csv(self.pairs_csv, index=False)

                dfp = dfp.drop(columns=["index","donor","acceptor","res_d","res_a"])
                # update expanded hah file with hb_pairs count, occ:
                hah_df = pd.read_csv(self.hah_ms_fp)
                hah_df = hah_df.merge(dfp, left_on=["Mi","Mj"], right_on=["Mi","Mj"])
                hah_df.to_csv(self.hah_ms_fp, index=False)

            return
        
        if self.hb_states:
            sdict_fp = self.states_csv.with_suffix(".dict")
            sdict_fp.write_text(pformat(self.hb_states)+"\n")
        
            dfs = pd.DataFrame.from_dict(self.hb_states, orient='index',
                                        columns = ["ms_count","ms_occ"]).reset_index()
            dfs["state_id"] = None
            for rx, ro in dfs.iterrows():
                dfs.loc[rx,"state_id"] = ",".join(f"({self.iconf2confid[tp[0]]},{self.iconf2confid[tp[1]]})"
                                                    for tp in ro["index"])
            dfs = dfs[["state_id", "ms_count","ms_occ"]]
            dfs = dfs.sort_values(by="ms_count", ascending=False)
            dfs.to_csv(self.states_csv, index=False)

        return

    def __str__(self):
        return (f"Conformers: {self.CI.n_confs:,}\n"
                f"Free residues: {self.HDR.n_free_res:,}\n"
                f"Fixed residues: {self.HDR.n_fixed_ics:,}\n"
                f"Background charge: {self.CI.background_crg:.0f}\n"
                f"State space: {self.n_space:,}\n"
                )


def cli_parser():
    p = ArgumentParser(prog="ms_hbnets",
        description="""Gather the H-bonding conformer pairs and states occupancies 
        from the microstates file given a mcce dir, pH & Eh.""",
        usage="""ms_hbnets
       ms_hbnets --load_states    # to get hb states instead of pairs
       ms_hbnets -ph 5
       ms_hbnets -n_states 30000
       ms_hbnets --run_checks     # + -mcce_dir, -ph, -eh if needed; all other options: ignored
       """,
        formatter_class=RawDescriptionHelpFormatter,
    )
    p.add_argument("-mcce_dir",
                    default=".",
                    type=str,
                    help="MCCE run directory; Default: %(default)s",
                    )
    # ph, eh: as strings to easily determine the precision
    p.add_argument("-ph",
                    default="7",
                    type=str,
                    help="Titration pH; Default: %(default)s"
                    )
    p.add_argument("-eh",
                    default="0",
                    type=str,
                    help="Titration Eh; Default: %(default)s"
                    )
    p.add_argument("--load_states",
                    action="store_true",
                    default=False,
                    help="""
Include to load the H-bonding states instead of the H-bonding pairs (default)"""
                    )
    p.add_argument("-n_states",
                    default=N_STATES,
                    type=int,
                    help="""
Number of H-bonding states to return, possibly (no effect without --load_states); Default: %(default)s"""
                    )
    p.add_argument("--run_checks",
                    action="store_true",
                    default=False,
                    help="Perform checks on main outputs; Default: %(default)s"
                    )
    p.add_argument("-v", "--verbose",
                    action="store_true",
                    default=False,
                    help="To ouput more details; Default: %(default)s"
                    )
    return p


def cli(argv=None):
    p = cli_parser()
    args = p.parse_args(argv)

    if args.run_checks:
        do_checks(args.mcce_dir, args.ph, args.eh)
        print("Microstates H_bonds checks over.")
    else:
        mshb = MSout_hb(args.mcce_dir, args.ph, args.eh,
                        n_target_states=args.n_states,
                        #pairs_sum_count=args.pairs_sum_count,
                        load_states=args.load_states,
                        verbose=args.verbose)
        print("Microstates H_bonds collection over.")

    return
