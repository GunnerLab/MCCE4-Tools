#!/usr/bin/env python3

"""
Module: msout_hb.py

Module for MSout_hb class, the numpy-based ms_out/ file loader for outputing
H-bonds information from microstates using the 'hah' file', the output of the
detect_hbonds tool.

Notes:

CHANGELOG:
"""
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from shutil import copyfile
import sys
import time
from typing import Dict, List, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
except ImportError as e:
    print(f"Oops! Forgot to activate an appropriate environment?\n{e}")
    sys.exit(1)

from mcce4.constants import ROOMT
from mcce4.detect_hbonds import detect_hbonds
from mcce4.io_utils import CalledProcessError, subprocess_run
from mcce4.io_utils import get_mcce_filepaths
from mcce4.io_utils import reader_gen
from mcce4.io_utils import show_elapsed_time
from mcce4.io_utils import table_to_df
from mcce4.ms_split_msout import split_mc_file


N_HDR = 6      # min header lines in msout file (non mc data)
MC_METHODS = ["MONTERUNS", "ENUMERATE"]
MIN_OCC = 0.0  # occ threshold
N_STATES = 25000  # target number of hb states to return

HAH_FNAME_INIT = "step2_out_hah.txt"
# output filename as fstring formats to receive MSout_hb.pheh_str,
# or msout_fp.stem[:-2], because they are ph/eh dependent.
# reduced hah.txt file, no hb pairs involving conformer that
# are fixed & always off:
fHAH_FNAME = "hah_{}.txt"
fHAH_EXPANDED = "expanded_hah_{}.tsv"

# mapping of iconf (donor, acceptor) pairs for classification purposes:
pair_classes = {(-1, -1): -2,  # bk, bk: not in matrix
                (-1, 0): -1,   # bk, fixed: not in matrix
                (0, -1): -1,   # fixed, bk: not in matrix
                (0, 0): 0,     # fixed, fixed: not in matrix
                # in matrix:
                (1, 1): 1,     # free, free: initial matrix elements
                (1, 0): 2,     # free, fixed: extended matrix
                (1, -1): 2,    # free, bk: extended matrix
                (0, 1): 3,     # fixed, free: extended matrix
                (-1, 1): 3,    # bk, free: extended matrix
                }


def load_hah_ms(hah_ms_fp: Path) -> pd.DataFrame:
    """Load the main output file into a dataframe."""
    return pd.read_csv(hah_ms_fp, sep="\t")


def get_hb_paths(mcce_dir: Path, ph: str = "7", eh: str = "0") -> Tuple[Path]:
    """Return the paths to head3, step2 pdb, msout file, the hah file, 
    and 'microstate ready', expanded hah file path.
    If no _hah.txt file is found (hah_fp is None), detect_hbonds will be run.
    """
    h3_fp, step2_fp, msout_fp = get_mcce_filepaths(mcce_dir, ph=ph, eh=eh)
    # reset to match precision in msout file name:
    # case where reduced function was already run:
    pheh = msout_fp.stem[:-2]
    hah_fp = mcce_dir.joinpath(fHAH_FNAME.format(pheh))
    if not hah_fp.exists():
        hah_fp = mcce_dir.joinpath(HAH_FNAME_INIT)
        if not hah_fp.exists():
            hah_fp = None

    return (h3_fp, step2_fp, msout_fp, hah_fp,
            mcce_dir.joinpath(fHAH_EXPANDED.format(pheh)),
            h3_fp.with_name("hb_pairs.csv"),
            h3_fp.with_name("hb_states.csv"))


def get_msout_size_info(msout_fp: Path,
                        n_target_states: int = N_STATES) -> Tuple[int, int, int]:
    """Return n_lines, n_skip_lines, n_mc_runs
    """
    mso = str(msout_fp)
    cmd = f"egrep '^MC' {mso}; wc -l {mso};"
    out = subprocess_run(cmd, shell=True, check=True)
    if isinstance(out, CalledProcessError):
        print(out.stderr)
        sys.exit(1)
    out = out.stdout.splitlines()
    n_mc_runs = len(out[:-1])
    # to implement skipping accepted states every n lines:
    # lines count is approximate
    n_lines = int(out[-1].split()[0]) - N_HDR - 4  # - 2 if method not monte
    n_skip_lines = int(np.floor(n_lines / n_target_states))
    print(f"Microstates to be saved every {n_skip_lines:,} lines",
          f"({n_lines=:,} / {n_target_states=:,})")
    if n_mc_runs > 1:
        print(f"The msout file {msout_fp!s} has {n_mc_runs} MC runs")

    return n_lines, n_skip_lines, n_mc_runs


class MsoutHeaderData:
    """This class handles the loading of the data in the header of a 'msout file'.
    """
    def __init__(self, msout_fp: Path):
        self.msout_fp = msout_fp
        self.T: float = ROOMT
        self.pH: float = 7.0
        self.Eh: float = 0.0
        self.method: str = ""
        self.is_monte: bool = False

        self.fixed_iconfs: List[int] = []
        self.n_fixed_ics: int = 0

        self.free_residues: List[List[int]] = []
        self.n_free_res: int = 0
        self.free_iconfs: List[int] = []
        self.n_free_ics: int = 0
        self.iconf2ires: Dict = {}

        self.load()

        return
    
    def load(self):
        """Process an unadulterated 'msout file' header rows to populate
        the class attributes.
        """
        with open(self.msout_fp) as fh:
            head = list(islice(fh, 6))
        for i, line in enumerate(head, start=1):
            if i == 1:
                fields = line.split(",")
                for field in fields:
                    key, value = field.split(":")
                    key = key.strip().upper()
                    value = float(value)
                    if key == "T":
                        self.T = value
                    elif key == "PH":
                        self.pH = value
                    elif key == "EH":
                        self.Eh = value
            if i == 2:
                key, value = line.split(":")
                self.method = value.strip().upper() 
                if key.strip() != "METHOD" or self.method not in MC_METHODS:
                    msg = (f"File {self.msout_fp!s} is not a valid microstate file; "
                           "method: {self.method}")
                    sys.exit(msg)

                self.is_monte = self.method == "MONTERUNS"

            if i == 4:
                fix, iconfs = line.split(":")
                self.fixed_iconfs = [int(i) for i in iconfs.split()]
                self.n_fixed_ics = int(fix)
            if i == 6:  # free residues
                free, residues_str = line.split(":")
                self.n_free_res = int(free)
                residues = residues_str.split(";")
                for f in residues:
                    if f.strip():
                        self.free_residues.append([int(i) for i in f.split()])
                for idx, lst in enumerate(self.free_residues):
                    for iconf in lst:
                        self.iconf2ires[iconf] = idx

        self.free_iconfs = list(self.iconf2ires.keys())
        self.n_free_ics = len(self.free_iconfs)

        return


class ConfInfo:
    """This class handles the loading of head3 data into a numpy array
    and provides accessor functions.
    """
    def __init__(self, h3_fp: Path, verbose: bool = False):
        self.h3_fp = h3_fp
        self.conf_info: np.ndarray = None
        self.n_confs: int = None
        self.max_ic, self.max_ir = None, None
        self.background_crg: int = None
        self.verbose = verbose

        return
    

    def load(self, iconf2ires: Dict, fixed_iconfs: List[int]):
        """Popuate the 'conf_info' attribute (np.ndarray): a lookup 'table' for:
        confids, crg, iconfs, ires, and is_free flag.
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
            # confid:0, crg:1, cx:2, off:3, rx:4, is_free:5
            conf_info.append([confid, crg, cx, 0, -1, 0])

        # temp list structure is now sized & has h3 info; cast to np.ndarray:
        conf_info = np.array(conf_info, dtype=object)

        # populate the ir of free res, if possible
        for i, (_, _, cx, *_) in enumerate(conf_info):
            conf_info[i][-2] = iconf2ires.get(cx, -1)

        # populate the 'is_free' field using valid ires
        conf_info[np.where(conf_info[:,-2]>=0), -1] = 1

        # populate the fixed 'off' field
        conf_info[np.where((conf_info[:,-1]==0) 
                            & np.logical_not(np.isin(conf_info[:,2], fixed_iconfs))
                          ), 3] = 1
        if self.verbose:
            print(" Head3 lookup array 'conf_info' fields:", 
                "confid:0, crg:1, cx:2, off:3, rx:4, is_free:5")
        self.n_confs = conf_info.shape[0]
        self.max_iconf, self.max_ires = np.max(conf_info[:,[2,4]], axis=0)
        # sumcrg for not is_free:
        self.background_crg = conf_info[np.where(conf_info[:,-1]==0), 1].sum()
        print(f" Background crg: {self.background_crg}",
              f" n_confs: {self.n_confs}", sep="\n")
        self.conf_info = conf_info

        return
    
    def get_iconf(self, confid: str) -> int:
        """Get the conf index of a confid;
        confid:0, crg:1, cx:2, off:3, rx:4, is_free:5
        """
        if "BK" in confid:
            return -1
        return self.conf_info[np.where(self.conf_info[:,0]==confid)][0,2]

    def get_confid(self, iconf: int) -> str:
        """Get the confid of a conf index;
        confid:0, crg:1, cx:2, off:3, rx:4, is_free:5
        """
        try:
            val = self.conf_info[np.where(self.conf_info[:,2]==iconf)][0,0]
        except IndexError:
            val = None

        return val

    def get_ires(self, iconf: int) -> int:
        """Get the res index given a conformer index;
        confid:0, crg:1, cx:2, off:3, rx:4, is_free:5
        """
        try:
            val = self.conf_info[np.where((self.conf_info[:,-1]==1) 
                                          & (self.conf_info[:,2]==iconf))][0,-2]
        except IndexError:
            val = None

        return val

    def is_free_conf(self, confid: str) -> int:
        """confid:0, crg:1, cx:2, off:3, rx:4, is_free:5
        Note: -1 is return for confid of BK
        """
        if "BK" in confid:
            return -1
        try:
            val = self.conf_info[np.where(self.conf_info[:,0]==confid)][0,-1]
        except IndexError:
            val = None

        return val

    def is_fixed_off(self, confid: str) -> int:
        """confid:0, crg:1, cx:2, off:3, rx:4, is_free:5
        Note: -1 is return for confid of BK, so it's not eliminated
        """
        if "BK" in confid:
            return -1
        try:
            val = self.conf_info[np.where(self.conf_info[:,0]==confid)][0,3]
            if val == -1: val = 0
        except IndexError:
            val = None

        return val


class MSout_hb:
    """Class to process the'msout file' for H-bonds data.

    Arguments:
        - head3_file, msout_file (str): Paths to head3.lst & msout files.
    """
    def __init__(self, mcce_dir: str, ph: str = "7", eh: str = "0",
                 n_target_states: int = 25_000,
                 interactive: bool = False,
                 verbose: bool = False):
        self.verbose = verbose
        self.run_dir = Path(mcce_dir)
        print(f"Run dir: {self.run_dir!s}")

        start_tot = time.time()
        (self.h3_fp, self.step2_fp, self.msout_fp,
         self.hah_fp, self.hah_ms_fp,
         self.pairs_csv, self.states_csv) = get_hb_paths(self.run_dir, ph=ph, eh=eh)

        # ph, eh as string for keeping same precision as in msoutfile:
        self.pheh_str = self.msout_fp.stem[:-2]

        self.mc_lines, self.n_skip, self.n_MC = get_msout_size_info(self.msout_fp,
                                                n_target_states=n_target_states)
        print(f"Approximate number of lines in msout file: {self.mc_lines:,}")
        
        if interactive:
            print("not implemented yet")
            # TODO
            #check n_mc_runs > 1
            # ask reduce to 1 or smaller range?
            #   reset_master = n_mc_runs != 6 & all_msout file exists
            #   y: split_mc_file(msout_fp, mc_range, reset_master)
            # ask n_target_states ok? Enter or give new number = new_target
            #re-assign counts:
            # n_lines, n_skip_lines, n_mc_runs = get_msout_size_info(self.msout_fp,
            #                                                   n_target_states=new_target)

        # data from the msout file 'header':
        self.HDR = MsoutHeaderData(self.msout_fp)

        # conformer info -> conf_info head3 & HDR lookup array
        # fields: confid:0, crg:1, ic:2, off:3, ir:4, is_free:5
        start_t = time.time()
        self.CI = ConfInfo(self.h3_fp, verbose=self.verbose)
        # load the self.CI.conf_info lookup array:
        self.CI.load(self.HDR.iconf2ires, self.HDR.fixed_iconfs)
        # + CI.n_confs, CI.max_ic, CI.max_ir
        show_elapsed_time(start_t, info="Loading conf_info array")

        # attributes populated by get_extended_iconfs:
        self.extend_fixed: List[str] = []
        self.n_fx: int = None
        self.fx_iconfs: List[int] = None
        self.extend_bk: List[str] = []
        self.n_bk: int = None
        self.dm_iconfs: List[int] = None

        # attributes populated by set_extended_accessors
        self.mat_res: List[List[int]] = None
        self.n_mat_res: int = None
        self.mat_iconfs: List[int] = None
        self.n_mat_ics: int = None
        self.mat_iconf2ires: dict = None
        self.mat_ires2confid: dict = None
        self.dmic2bkid: dict = None
        self.bkid2dmic: dict = None

        # free and mixed hb pairs + needed indices
        print(f"Creating the expanded hah file {self.hah_ms_fp!s}...")
        start_t = time.time()
        self.df = self.expand_hah_data()
        show_elapsed_time(start_t, info="Expanding the hah file into a dataframe")

        start_t = time.time()
        self.M = self.get_hah_matrix()
        if self.M is None:
            sys.exit("Could not create the hb pairs matrix.")
        if self.verbose:
            mat_pair_i, mat_pair_j = self.M.nonzero()
            print("Mij set to 1 in matrix:", list(zip(mat_pair_i, mat_pair_j)), sep="\n")
        show_elapsed_time(start_t, info="Creating the hb pairs matrix M")

        # Attributes populated by the 'load_ms_hb' function:
        self.n_space: int = None     # size of state space
        # dicts to receive each H-bond pair (str key: space-less d/a tuple of iconfs) 
        # or the H-bond state (str key: semi-colon separated, space-less d/a tuples).
        # Values for both dicts items: [count, occ]
        self.hb_pairs: dict = None
        self.hb_states: dict = None

        print("Loading the H-bond data from microstates...")
        start_t = time.time()
        self.load_ms_hb()
        show_elapsed_time(start_t, info="Loading msout for H-bond data")

        print("Converting Mij to confid, updating the expanded file with pairs data, if any",
              "& saving dicts to csv files...", sep="\n")
        start_t = time.time()
        self.dicts2csv()
        show_elapsed_time(start_t, info="Converting, updating & saving dicts to csv")

        show_elapsed_time(start_tot, info="Start to end")

        return

    def run_detect_hbonds(self):
        # run detect_hbonds (with bk atoms by default):
        status = detect_hbonds(str(self.step2_fp),
                               no_empty_files=True)
        if not status[0]:
            raise ValueError("No H-bonding pairs in step2_out.pdb.")
        # reset
        self.hah_fp = self.run_dir.joinpath(HAH_FNAME_INIT)

        return
            
    def load_hah_file(self) -> Union[pd.DataFrame, None]:
        """Wrapper for hah file preparation.
         - If path not set, or old format: run detect_hbonds
         - If original file saved as step2_out_hah0.txt, does not exist,
           create a reduced file.
         """
        if self.hah_fp is None:
            self.run_detect_hbonds()
        
        df = table_to_df(self.hah_fp)
        if df.columns[-1] != "xyz":
            # old format, re-run
            self.run_detect_hbonds()
            df = table_to_df(self.hah_fp)

        print(f"H-bonding pairs in {self.hah_fp.name!s}: {df.shape[0]}")
        if self.hah_fp.name==HAH_FNAME_INIT:
            df = self.reduce_hah_file(df)

        return df

    def reduce_hah_file(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce the inital hah.txt file by removing entries that have
        H-bonding pairs with fixed conformers that are always OFF.
        The original file is saved as hah0.txt, the reduced one as hah.txt,
        indicating that the presence of the hah0.txt file means that the hah.txt
        file is reduced.
        The fixed conformers that are always OFF are identified by these criteria:
         - iconf is not free (not in the free_iconfs list)
         - iconf is not in the fixed_iconfs list
        """
        assert self.hah_fp.name == HAH_FNAME_INIT
        df["d_off"] = df["confid_donor"].apply(self.CI.is_fixed_off)
        df["a_off"] = df["confid_acceptor"].apply(self.CI.is_fixed_off)
        msk_off = (df["d_off"]==1) | (df["a_off"]==1)
        df.drop(index=df.loc[msk_off].index, axis=0, inplace=True)
        # drop the temp columns before saving:
        df.drop(columns=["d_off", "a_off"], inplace=True)
        reduced_fp = self.hah_fp.with_name(fHAH_FNAME.format(self.pheh_str))
        # reset hah path:
        self.hah_fp = reduced_fp
        self.hah_fp.write_text(df.to_string(index=False)+"\n")
        print(f"H-bonding pairs in reduced file {self.hah_fp.name!s}: {df.shape[0]}")

        return df

    def get_fixed_or_bk_confids(self, df: pd.DataFrame) -> List:
        """Return lists for fixed and bk confids for each donor/acceptor:
           fid, fixa, bkd, bka (in that order).
        """
        # Operations on pair_class > 1, includes all mixed pairs
        # class 2::donor is free; class 3::acceptor is free
        # lists hold confids, donor/acceptor split is needed for:
        # - getting unique values for each kind
        # - filtering the master df to assign final Mij using dummy ics
        fixd, fixa, bkd, bka = [],[],[],[]
        params = {2:{"not_free": "free_a", "col": "confid_acceptor", "dmic": "iconf2dm",
                     0: {"lst":fixa}, -1: {"lst":bka},
                    },
                  3:{"not_free": "free_d", "col": "confid_donor", "dmic": "iconf1dm",
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

    def set_extended_accessors(self) -> Tuple:
        if not (self.n_fx + self.n_bk):
            # assign current accessors to outputs:
            self.mat_res = self.HDR.free_residues
            self.n_mat_res = self.HDR.n_free_res
            self.mat_iconf2ires = self.HDR.iconf2ires
            self.mat_iconfs = self.HDR.free_iconfs
            self.n_mat_ics = self.HDR.n_free_ics
            self.mat_ires2confid = [self.CI.get_confid(iconf)
                                 for iconf in self.mat_iconfs]
            return

        self.mat_res = self.HDR.free_residues.copy()
        n_free_res = len(self.mat_res)
        ix = n_free_res - 1  # last index of free res

        if self.n_fx:
            # extend for fixed, same format as free_residues:
            self.mat_res.extend([fic] for fic in self.fx_iconfs)
            assert len(self.mat_res) == n_free_res + self.n_fx

            if self.n_bk:
                self.mat_res.extend([[dic] for dic in self.dm_iconfs])
                assert len(self.mat_res) == n_free_res + self.n_fx + self.n_bk
                ix = ix + self.n_fx + 1 # 1st index of bk slots
        else:
            if self.n_bk:
                self.mat_res.extend([[dic] for dic in self.dm_iconfs])
                assert len(self.mat_res) == n_free_res + self.n_bk
                ix = ix + self.n_bk + 1
        
        if self.n_bk:
            self.bkid2dmic = {}
            # mapping dm confid-> new slot:
            for bx, cid in enumerate(self.extend_bk):
                self.bkid2dmic.update({cid: self.mat_res[(ix + bx)][0]})
            self.dmic2bkid = {v: k for k, v in self.bkid2dmic.items()}

        self.n_mat_res = len(self.mat_res)
        self.mat_iconf2ires = {}  # iconf to extended
        self.mat_ires2confid = {}
        ix = self.HDR.n_free_res -1 + self.n_fx + 1
        for ires, lst in enumerate(self.mat_res):
            for iconf in lst:
                self.mat_iconf2ires[iconf] = ires
                if iconf <= self.CI.max_iconf:
                    self.mat_ires2confid[ires] = self.CI.get_confid(iconf)
                else:
                    self.mat_ires2confid[ires] = self.dmic2bkid.get(iconf)

        self.mat_iconfs = list(self.mat_iconf2ires.keys())
        self.n_mat_ics = len(self.mat_iconfs)
        N = n_free_res + self.n_fx + self.n_bk
        if self.verbose:
            print("Check:",
                f"n_free_res {n_free_res} + n_fixed {self.n_fx} +",
                f"n_bk {self.n_bk} ?= n_mat_res {self.n_mat_res} ::",
                f"{N==self.n_mat_res}")
        
        return

    def get_extended_iconfs(self, fixd, fixa, bkd, bka):
      
        if fixd and fixa:
            self.extend_fixed = sorted(set(fixd).union(fixa))
        elif fixd:
            self.extend_fixed = fixd
        elif fixa:
            self.extend_fixed = fixa
        if bkd and bka:
            self.extend_bk = sorted(set(bkd).union(bka))
        elif bkd:
            self.extend_bk = bkd
        elif bka:
            self.extend_bk = bka
        
        print(" Number of extension slots from mixed pairs with fixed:",len(self.extend_fixed))
        print(" Number of extension slots from mixed pairs with BK:",len(self.extend_bk))
        self.n_fx = len(self.extend_fixed)
        self.n_bk = len(self.extend_bk)
        if self.n_fx:
            # get iconfs values fo fixed
            self.fx_iconfs = [self.CI.get_iconf(xid) for xid in self.extend_fixed]
        if self.n_bk:
            # to avoid collisions, dummy ics for bk confs start past last iconf:
            dmic_start = self.CI.n_confs
            # create dm iconfs values for bk:
            self.dm_iconfs = [dmic_start + i for i in range(self.n_bk)]

        return
    
    def expand_hah_data(self) -> pd.DataFrame:
        """Load the hah file into a dataframe & add:
         -conformer indices from h3
         -free conformer flag
         -free iconf to free residue index used to build the hb pairs matrix P
        """
        # df.apply functions:
        def is_free_pair(ro) -> int:
            """Assign classes to pair kinds. df.apply fn; axis=1.
            """
            return pair_classes[(ro["free_d"], ro["free_a"])]
        
        def get_Mx(ro: pd.Series) -> int:
            return self.HDR.iconf2ires.get(ro)
        
        def get_dm_ic(ro: pd.Series) -> int:
            return self.bkid2dmic.get(ro)

        def get_dm_Mx(ro: pd.Series) -> int:
            return self.mat_iconf2ires.get(ro)

        df = self.load_hah_file()

        # these may help creating an adjacency list.
        df["dina"] = df["confid_donor"].isin(df["confid_acceptor"])
        df["aind"] = df["confid_acceptor"].isin(df["confid_donor"])

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

        # Add cols for final (extended) matrix i,j; mixed pairs to be set later
        df["Mi"] = -1
        df["Mj"] = -1

        # assign ires to the free iconf(s) in pairs destined for matrix;
        # classes 1,2 & 3):
        free_msk = df["free"].gt(0)
        # fixed & bk iconfs will have NaN:
        df.loc[free_msk, "Mi"] = df.loc[free_msk,"iconf1"].apply(get_Mx)
        df.loc[free_msk, "Mj"] = df.loc[free_msk,"iconf2"].apply(get_Mx)

        fixd, fixa, bkd, bka = self.get_fixed_or_bk_confids(df)
        self.get_extended_iconfs(fixd, fixa, bkd, bka)
        self.set_extended_accessors()

        params = {2:{"not_free": "free_a", "col": "confid_acceptor",
                        "dmic": "iconf2dm",
                        "iconf": "iconf2",
                        "Mx": "Mj"},
                    3:{"not_free": "free_d", "col": "confid_donor",
                        "dmic": "iconf1dm",
                        "iconf": "iconf1",
                        "Mx": "Mi"},}

        # update dummy iconfs & fixed Mx:
        for cx in [2, 3]:
            mixed = df["free"]==cx
            for k in [0, -1]:  # fixed, BK
                # update dummy iconfs cols & Mx of fixed
                cidf = df[params[cx]["not_free"]]==k
                msk = mixed & cidf
                if msk.any():
                    Mx = params[cx]["Mx"]
                    if k == 0:
                        iconf = params[cx]["iconf"]
                        # update NaN Mij of fixed iconfs
                        df.loc[msk, Mx] = df.loc[msk, iconf].apply(get_dm_Mx)
                    else:
                        dmicol, idcol = params[cx]["dmic"], params[cx]["col"]
                        df.loc[msk, dmicol] = df.loc[msk, idcol].apply(get_dm_ic)
                        # update NaN Mij of dm iconfs
                        df.loc[msk, Mx] = df.loc[msk, dmicol].apply(get_dm_Mx)

        df["Mi"] = df["Mi"].astype("int8")
        df["Mj"] = df["Mj"].astype("int8")
        # save
        df.to_csv(self.hah_ms_fp, index=False, sep="\t")
        print(f"Accepted H-bonding pairs in df: {df.loc[df['free'].gt(0)].shape[0]}")
    
        return df

    def get_hah_matrix(self) -> csr_matrix:
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
        df = self.df.loc[msk]
        df = df.drop_duplicates(subset=["Mi", "Mj"])
        print(f"Unique H-bonding pairs for matrix: {df.shape[0]}")
        N = self.HDR.n_free_ics + self.n_fx + self.n_bk

        return csr_matrix(([1]*df.shape[0],
                           (df["Mi"].values, df["Mj"].values)
                           ),
                          shape=(N, N))

    def load_ms_hb(self):
        """Process the 'msout file' for H-bond data using the hb pairs matrix
        """
        found_mc = False
        newmc = False
        hb_pairs =  defaultdict(lambda: [0, 0.])
        hb_states =  defaultdict(lambda: [0, 0.])
        mc_lines = 0
        states = 0
        # precompute the unchanging indices:
        extended = []
        if not (self.n_fx + self.n_bk):
            if self.n_fx:
                extended.extend([self.mat_iconfs.index(ic) for ic in self.fx_iconfs])
            if self.n_bk:
                extended.extend([self.mat_iconfs.index(ic) for ic in self.dm_iconfs])

        msout_data = reader_gen(self.msout_fp)
        for lx, line in enumerate(msout_data, start=1):
            if lx <= N_HDR:
                continue
            line = line.strip()
            if not line or line[0] == "#":
                continue
            else:
                if self.HDR.is_monte:
                    # find the next MC record
                    if line.startswith("MC:"):
                        found_mc = True
                        newmc = True
                        continue
                else:
                    if line.startswith(tuple("0123456789")) and ":" in line:
                        found_mc = True
                        current_state = [int(i) for i in line.split(":")[1].split()]
                        newmc = False
                        continue

                if newmc:
                    # newmc is still True for MONTERUNS
                    # line with candidate state for MC sampling, e.g.:
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                    current_state = [int(i) for i in line.split(":")[1].split()]
                    newmc = False
                    continue

                if found_mc:
                    fields = line.split(",")
                    if len(fields) < 3:
                        continue
                    mc_lines += 1

                    # state_e = float(fields[0])
                    if self.HDR.is_monte:
                        count = int(fields[1])
                        states += count
                    else:  # occ
                        count = float(fields[1])

                    flipped = [int(c) for c in fields[2].split()]
                    for ic in flipped:
                        ir = self.HDR.iconf2ires[ic]
                        current_state[ir] = ic

                    # get the index vector of the current, extended state for querying the matrix:
                    xs = np.array([self.mat_iconfs.index(ic) for ic in current_state] + extended)
                    ni, nj = self.M[xs].nonzero()
                    ni = np.array([xs[i] for i in ni])
                    #print(f"corrected ni, {ni = }\n{nj = }")
                    nij = list(zip(ni, nj))
                    # Note: tuples are (Mi,Mj)
                    for p in nij:
                        hb_pairs[p][0] += count

                    if mc_lines % self.n_skip == 0:
                        hb_states[tuple(nij)][0] += count

        self.hb_pairs = dict(hb_pairs)
        self.hb_states = dict(hb_states)
        # update occ:
        if self.HDR.is_monte:
            self.n_space = states            
            for p in self.hb_pairs:
                cnt = self.hb_pairs[p][0]
                self.hb_pairs[p][1] = cnt/states

            for s in self.hb_states:
                cnt = self.hb_states[s][0]
                self.hb_states[s][1] = cnt/states
        else:
            tot_occ = sum(self.hb_states[s][1] for s in self.hb_states)
            if tot_occ > 0:
                self.n_space = (self.HDR.n_free_ics * 2000 * self.n_MC) // tot_occ # ??

        print(f"State space: {self.n_space:,}")
        print(f"H-bonding states: {len(self.hb_states):,}")
        print(f"H-bonding pairs: {len(self.hb_pairs):,}")
        if mc_lines != self.mc_lines:
            # accepted ms with flipped iconfs
            print(f"Processed mc lines: {mc_lines:,}")
        else:
            print(f"Accepted states lines: ~ {self.mc_lines:,}\n")

        return

    def dicts2csv(self):
        if self.hb_pairs:
            dfp = pd.DataFrame.from_dict(self.hb_pairs, orient='index',
                                         columns = ["ms_count","ms_occ"]).reset_index()
            dfp = dfp.rename({"index":"Mij"}, axis=1)
            dfp[["Mi","Mj"]] = dfp["Mij"].apply(lambda x: pd.Series([x[0],x[1]]))
            dfp[["donor","acceptor"]] = dfp["Mij"].apply(lambda x: pd.Series([self.mat_ires2confid[x[0]],
                                                                            self.mat_ires2confid[x[1]]]))
            dfp = dfp.sort_values(by="ms_count", ascending=False)
            dfp[["Mij","donor","acceptor","ms_count","ms_occ"]].to_csv(self.pairs_csv, index=False)

            # update expanded hah file:
            dfp = dfp.drop(columns=["Mij","donor","acceptor"])
            hah_df = pd.read_csv(self.hah_ms_fp, sep="\t")
            hah_df = hah_df.merge(dfp, left_on=["Mi","Mj"], right_on=["Mi","Mj"])
            hah_df.to_csv(self.hah_ms_fp, index=False)

        if self.hb_states:
            dfs = pd.DataFrame.from_dict(self.hb_states, orient='index',
                                        columns = ["ms_count","ms_occ"]).reset_index()
            dfs["state_id"] = None
            for rx, ro in dfs.iterrows():
                dfs.loc[rx,"state_id"] = ", ".join(f"({self.mat_ires2confid[tp[0]]},{self.mat_ires2confid[tp[1]]})"
                                                    for tp in ro["index"])
            dfs = dfs[["state_id", "ms_count","ms_occ"]]
            dfs = dfs.sort_values(by="ms_count", ascending=False)
            dfs.to_csv(self.states_csv, index=False)

        return


    def __str__(self):
        out = (f"Conformers: {self.CI.n_confs:,}\n"
                f"Free residues: {self.HDR.n_free_res:,}\n"
                f"Free conformers: {self.CI.n_confs:,}\n"
                f"Fixed residues: {self.HDR.n_fixed_ics:,}\n"
                f"Background charge: {self.CI.background_crg:.0f}\n"
                )
        if self.n_space is not None:
            out = out + f"State space: {self.n_space:,}\n"

        return out


def cli_parser():
    p = ArgumentParser(prog="ms_hbnets",
        description="""Gather the H-bonding conformer pairs and states occupancies 
        from the microstates file given a mcce dir, pH & Eh.""",
        usage="""ms_hbnets
       ms_hbnets -ph 5
       ms_hbnets -n_states 30000""",
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
    p.add_argument("-n_states",
                    default=N_STATES,
                    type=int,
                    help="Number of hb states to return, possibly; Default: %(default)s"
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

    mshb = MSout_hb(args.mcce_dir, args.ph, args.eh, args.n_states, args.verbose)
    print("Microstates H_bonds collection over.")

    return
