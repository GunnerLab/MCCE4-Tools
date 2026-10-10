#!/usr/bin/env python3

"""
Module: msout_hb.py

Module for MSout_hb class, the numpy-based ms_out/ file loader for outputing
H-bonds information from microstates using the 'hah' file', the output of the
detect_hbonds tool.

Notes:
 - Loading of H-bond data is either for 'pairs' or for 'states', reflected in the
   ms_hbnets tool subcommands.
   This split follows the msout loading routines that differ for computational efficiency.

Warning:
 The main output of the 'pairs' subcommand is the hb_pairs_pH*.csv file. 
 Unless a higher min_occ is given, ~99.+% of the microstates space is returned. Thus, this
 output lists the uniques pairs over most of the space.
 (The difference is due to rounding of occ to the millionth.)
 The states pairs output by the 'states' subcommand, hb_states_pairs_pH*.csv, lists the unique
 pairs in the returned sample of hb microstates.
 Comparing the pairs' occ in these two files is only meaning full if their counts amount to
 to a (very) similar percentage of the entire state space.
 This information is found in the commented first line in hb_states_pH*.csv and hb_pairs_pH*.csv,
 obtained with this command:
   head -n1 hb_states_pH*.csv
"""
from collections import defaultdict
import logging
from operator import itemgetter
from pathlib import Path
import re
import sys
from time import time
from typing import Dict, List, Tuple, Union

try:
    from numba import njit
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
except ImportError as e:
    logging.critical(f"Oops! Forgot to activate an appropriate environment?\n{e}")
    sys.exit(1)

from mcce4.constants import res3_to_res1
from mcce4.io_utils import MsoutHeaderData, N_HDR, N_STATES
from mcce4.io_utils import get_mcce_filepaths, get_msout_size_info
from mcce4.io_utils import reader_gen
from mcce4.io_utils import show_elapsed_time
from mcce4.io_utils import table_to_df


NO_MONTE_MSG = """
The MC method for H-bonding states must be MONTE.
If you can rerun step4.py, change the 1M default NSTATE_MAX key to 1: > step4.py ... -u NSTATE_MAX=1
If you want the processing of the analytical solution to be reinstated 
please, open a feature request at https://github.com/GunnerLab/MCCE4-Tools/issues
"""

# Input file, output of `detect_hbonds`:
HAH_FNAME_INIT = "step2_out_hah.txt"
# Output filenames: f-strings to receive MSout_hb.pheh_str,
#                   (msout_fp.stem[:-2]), because they are ph/eh dependent.
# Reduced hah_*.txt file: has no hb pairs involving conformers that are:
#                       fixed & always off, or two BK conformers, or conformers with 0 occupancy
fHAH_FNAME = "hah_{}.txt"
fHAH_EXPANDED = "expanded_hah_{}.csv"
fPAIR_RES = "hb_pairs_res_{}.csv"


# the largest spaces ever seen are in the order of millions;
# the occ returned but the loading functions are rounded to this precision:
PREC = 6
MIN_OCC = 10**(-PREC)
printed_occ = "0." + "".join('0' for i in range(PREC - 1)) + "1"


# mapping of iconf (donor, acceptor) pairs for filtering purposes:
pair_classes = {
                # not in matrix:
                (-1, -1): -2,  # bk, bk
                (-1, 0): -1,   # bk, fixed
                (0, -1): -1,   # fixed, bk
                (0, 0): 0,     # fixed, fixed
                # in matrix:
                (1, 1): 1,     # free, free: initial matrix elements
                (1, 0): 2,     # free, fixed: extended matrix
                (1,-1): 2,     # free, bk: extended matrix
                (0, 1): 3,     # fixed, free: extended matrix
                (-1,1): 3,     # bk, free: extended matrix
                }


def num_prec(num: Union[float, int]) -> int:
    if isinstance(num, int):
        return 0
    num_str = str(num).lower()
    if 'e' in num_str:
        return int("".join(c for c in num_str.split('e')[1] if c.isnumeric()))
        
    if '.' in num_str:
        return len(num_str.split('.')[1])
    return 0


def get_resid(confid:str) -> str:
    """Return the one-letter resid in this format:
    1-letter code + _ + Chain + resnum, ex: H_A59
    """
    id1 = res3_to_res1.get(confid[:3], confid[:3])
    if id1 == "HOH":
        id1 = "w"
    return f"{id1}_" + confid[5] + str(int(confid[6:-4]))


def is_int(val:str) -> bool:
    try:
        return str(int(val)) == val
    except ValueError:
        return False


def get_titr_vec(titr_fp: Path, titr_point: str, non_zeros: bool = None) -> Union[np.ndarray, None]:
    """"
    Return the vector of values at the given titr_point.
     - non_zeros (bool, None) can be used as a filter for the output values:
       If None (default): return the entire vector;
       If True: output the non-zero values;
       If False: output the zero values.

    Used by MSout_hb.reduce_hah_file.
    """
    if is_int(titr_point):
        titr_point = titr_point + ".0"
    df = table_to_df(titr_fp)
    if df is None:
        return None
    vec = None
    vec = df.filter(items=[df.columns[0], titr_point], axis=1)
    if vec.shape[1] < 2:
        logging.critical(f"Titration point not found in {titr_fp!s}: {titr_point}")
        return None

    if non_zeros is not None:
        vec = vec[vec[titr_point].ne(0)] if non_zeros else vec[vec[titr_point].eq(0)]
        if not len(vec):
            return None

    return vec.to_numpy()


def get_hb_paths(mcce_dir: Path, ph: str = "7", eh: str = "0") -> Tuple[Path, None]:
    """Return the paths to all the required input files, and the output
    files in this order:
      - Inputs: head3.lst, step2_out.pdb, the msout file, fort.38 the hah file (reduced if found),
      then
      - Outputs: the expanded hah file, and the final csv files:
        - hb_pairs,
        - hb_states_pairs (hb_pairs count, occ & state membership)
        - hb_states
    """
    if not isinstance(ph, str):
        ph = str(ph)
    if not isinstance(eh, str):
        eh = str(eh)

    ms_files_fps = get_mcce_filepaths(mcce_dir, ph=ph, eh=eh)
    if ms_files_fps is not None:
        h3_fp, step2_fp, msout_fp = ms_files_fps
    else:
        return None

    pheh = msout_fp.stem[:-2]
    # hah with msout signature: reduced for that microstate;
    # use the reduced file if found:
    hah_fp = mcce_dir.joinpath(fHAH_FNAME.format(pheh))
    if not hah_fp.exists():
        # use the output file from detect_hbonds:
        hah_fp = mcce_dir.joinpath(HAH_FNAME_INIT)
        if not hah_fp.exists():
            logging.critical("MISSING: step2_out_hah.txt. Run detect_hbonds first.")
            return None
        else:
            with open(hah_fp) as fh:
                has_xyz = fh.readline().split()[-1] == "xyz"
            if not has_xyz:
                logging.critical("OUTDATED file format: Rerun detect_hbonds for new file format (includes xyz).")
                return None

    # lastly, check fort.38:
    fort38_fp = h3_fp.with_name("fort.38")
    if not fort38_fp.exists():
        logging.critical("MISSING: fort.38")
        return None
    
    return (h3_fp,
            step2_fp,
            msout_fp,
            fort38_fp,
            hah_fp,
            mcce_dir.joinpath(fHAH_EXPANDED.format(pheh)),
            mcce_dir.joinpath(f"hb_pairs_{pheh}.csv"),
            mcce_dir.joinpath(fPAIR_RES.format(pheh)),
            mcce_dir.joinpath(f"hb_states_{pheh}.csv"),
            mcce_dir.joinpath(f"hb_states_pairs_{pheh}.csv")
            )


def get_da_pairs(hah_fp: Path) -> Union[np.ndarray, None]:
    """Return array of donor/acceptor pairs info with 4 slots:
    confid_donor, confid_acceptor, d_occ, a_occ
    """
    if hah_fp.suffix != ".txt":
        logging.error(("FileTypeError:\n "
               "Expected a '.txt' file (output of detect_hbounds or its reduced version), "
               f"got: {hah_fp.name}"))
        return None

    if not hah_fp.exists():
        logging.critical(f"FileNotFoundError: {hah_fp!r}")
        return None

    try:
        return pd.read_csv(hah_fp,
                           usecols=["confid_donor","confid_acceptor","d_occ","a_occ"],
                           sep=r"\s+").drop_duplicates().to_numpy()
    except Exception as err:
        logging.critical(f"Could not load {hah_fp.name} data into array.\n{err}")
        return None


def get_ms_pairs(pairs_csv: Path) -> Union[np.ndarray, None]:
    """Return array with 3 slots: donor, acceptor, occ
    """
    try:
        return pd.read_csv(pairs_csv, usecols=["donor","acceptor","occ"],
                           comment="#").to_numpy()
    except Exception as err:
        logging.critical(f"Could not load {pairs_csv.name} data into array.\n{err}")
        return None


def get_states_keys(states_csv: Path) -> list:
    try:
        return pd.read_csv(states_csv,
                           usecols=["state_id"],
                           comment="#").to_string(header=False,
                                                  index=False).splitlines()
    except FileNotFoundError:
        logging.info((f"Did not find the specified HB states file ({states_csv.name})\n"
                "Hint: the hb states file is created when calling `ms_hbnets states`")
                )
        return []


def check_state_pairs(state_pairs: str, hah_da_pairs: np.ndarray) -> Union[str, None]:
    """
    Output Consistency function. Checks that the pairs from a hb_state are found
    in the hah file, return a string of invalid pairs if found, else None (ok).
    """
    out = ""
    # the hb_states state id is a string :: str(tuple(d, a),...) 
    state_key_pattern = r"\(([\w+-]+),\s*([\w+-]+)\)"
    da_tuples_lst = re.findall(state_key_pattern, state_pairs)
    for tx, (d, a) in enumerate(da_tuples_lst, start=1):
        result = hah_da_pairs[(hah_da_pairs[:,0]==d) & (hah_da_pairs[:,1] == a), [0,1]]
        if not result.any():
            out += f"state #{tx}: ({d}, {a}) not found\n"
        # else:
        #     if result.size > 2:
        #         logging.info("Multiple matching pairs found")
           
    if not out:
        return None
    return out


def check_states(hah_da_pairs: np.ndarray,
                 states_csv: Path,
                 output_fname: str = "states_pairs_errors.txt") -> bool:
    """
    Output Consistency function. Checks that the pairs in the hah file are found
    in the hb states file, return a string of invalid pairs if found, else None (ok).
    """
    states_keys = get_states_keys(states_csv)
    if not states_keys:
        return False

    logging.info(" Checking da_pairs in states against da_pairs in hah file...")
    out_fp = states_csv.parent.joinpath(output_fname)
    ok = True
    with open(out_fp, "a") as fha:
        for skey in states_keys:
            result = check_state_pairs(skey, hah_da_pairs)
            if result is not None:
                ok = False
                fha.write(result)
    if not ok:
        logging.warning(f" Spurious states pairs saved in {out_fp!r}")
    else:
        logging.info(" Check on states passed.")

    return ok


def check_pairs(hah_da_pairs, pairs_csv) -> bool:
    """
    Output consistency checks.
    """
    ms_pairs = get_ms_pairs(pairs_csv)
    if ms_pairs is None:
        return False

    correct_pairs = True
    correct_occ = True

    logging.info(" Checking da_pairs in pairs file against da_pairs in hah file & fort.38...")
    for d,a,occ in ms_pairs:
        result = hah_da_pairs[(hah_da_pairs[:,0]==d) & (hah_da_pairs[:,1]==a)]
        correct_pairs = result.any()
        if not correct_pairs:
            logging.warning(f"Pair not found in hah_file: ({d}, {a})")
            continue

        occ = round(occ, 3)
        d_occ, a_occ = result[:,[-2,-1]][0]
        occ_max = min(d_occ, a_occ)
        occ_min = max(0.0, d_occ + a_occ - 1.0)
        if occ > round(occ_max + 1e-3, 3):
            logging.warning((f" occ {occ:.3f} of pair ({d}, {a}) exceeds max bound {occ_max:.3f} "
                   f"obtained from {d_occ:.3f} (donor) and {a_occ:.3f} (acceptor)"))
            correct_occ = correct_occ or False
        if occ < round(occ_min - 1e-3, 3):
            logging.warning((f" occ {occ:.3f} of pair ({d}, {a}) below min bound {occ_min:.3f} "
                   f"obtained from {d_occ:.3f} (donor) and {a_occ:.3f} (acceptor)"))
            correct_occ = correct_occ or False

    if correct_pairs and correct_occ:
        logging.info(" Check on pairs passed.")
        return True

    msg = ""
    if not correct_pairs:
        msg = msg + " Failed due to spurious pairs in the hb_pairs file.\n"
    if not correct_occ:
        msg = msg + " Failed due to out-of-bound occupancies in the hb_pairs file."
    logging.info(msg)

    return False


# Temporarily excluded: under review
def do_checks(mcce_dir: str, ph: str = "7", eh: str = "0") -> bool:
    """Wrapper to perform check on ms_hbnets main output files.
    """
    logging.info("The consistency checks are under review and TEMPORARILY BYPASSED.")
    return True

    run_dir = Path(mcce_dir)
    logging.info(f"Run dir: {run_dir!s}")
    needed_fps = get_hb_paths(run_dir, ph=ph, eh=eh)
    if needed_fps is None:
        return False

    _,_,_,_, hah_fp, _, pairs_csv,_, states_csv, _ = needed_fps

    hah_da_pairs = get_da_pairs(hah_fp)
    if hah_da_pairs is None:
        return False

    if not pairs_csv.exists():
        logging.info(f" Not found: {pairs_csv.name}, skipped check on pairs file.")
    else:
        status = check_pairs(hah_da_pairs, pairs_csv)
        if not status:
            return False

    if not states_csv.exists():
        logging.info(f" Not found: {states_csv.name}, skipped check on states file.")
    else:
        status = check_states(hah_da_pairs[:,[0,1]], states_csv)
        if not status:
            return False

    return True


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
        conf_info = []
        with open(self.h3_fp) as h3:
            lines = h3.readlines()[1:]

        for line in lines:
            # ignored columns: FL & fields past Crg
            iConf, confid, _, _, Crg, *_ = line.split()
            cx = int(iConf) - 1  # as python 0-based index
            crg = int(float(Crg))
            # extend ires indices; to use in matrix
            # confid:0, crg:1, cx:2, is_fixed:3, rx:4, is_free:5
            conf_info.append([confid, crg, cx, -1, -1, 0])

        # temp list structure is now sized & has h3 info; cast to np.ndarray:
        conf_info = np.array(conf_info, dtype=object)

        try:
            # is_free field
            free_ics = list(iconf2ires.keys())
            conf_info[free_ics, -1] = 1
            # fixed on
            conf_info[fixed_iconfs, 3] = 1
        except IndexError:
            logging.critical("Mismatch between conformer data in head3.lst and",
                    "given inputs (typically extracted from the msout file header).")
            return
        
        # populate the rx of free res, if possible
        for i, (_, _, cx, *_) in enumerate(conf_info): 
            conf_info[i][-2] = iconf2ires.get(cx, -1)

        self.n_confs = conf_info.shape[0]
        self.max_iconf, self.max_ires = np.max(conf_info[:,[2,4]], axis=0)
        # sumcrg for not is_free & is_fixed on:
        self.background_crg = conf_info[np.where((conf_info[:,-1]==0) 
                                                  & (conf_info[:,3]==1)), 1].sum()
        if self.verbose:
            logging.info((f"Background crg: {self.background_crg}; n_confs: {self.n_confs}"))
    
        self.conf_info = conf_info

        return

    def fieldnames(self):
        logging.info(" CI.conf_info fields: 0:confid, 1:crg, 2:iconf, 3:is_fixed, 4:ires, 5:is_free")

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
                           ):
    """Update hb_pairs matrix with microstate data.

    hb_pairs is the hb pairs dense matrix, MSout_hb.P;
    ms_mask is a boolean mask preset to True with the given state iconfs, i.e.:
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

    return


class MSout_hb:
    """Class to process the 'msout file' for H-bonds data.
    """
    def __init__(self, mcce_dir: str, ph: str = "7", eh: str = "0",
                 n_target_states: int = N_STATES,
                 load_states: bool = False,
                 min_occ: float = MIN_OCC,
                 reload: bool = False,
                 verbose: bool = False):
        
        """
        MSout_hb class constructor.

        Arguments:
         - mcce_dir (str): Path to a mcce simulation directory.
         - ph (str, '7'): Titration pH.
         - eh (str, '0'): Titration Eh.
         - n_target_states (int, 25,000): Number of hb states to return.
         - load_states (bool, False): Must be True to load hb states instead of hb pairs.
           When True, the output file 'hb_states_pairs_{pheh}.csv' has the hb pair state
           membership in the column 'state_id', which is the index found in the 'ix' column
           of the 'hb_states_{pheh}.csv' file.
         - min_occ (bool, 0.00001): Minimal ms occ to return.
         - reload (bool, False): Reload msout file data (overwrite existing output files)
         - verbose (bool, False): Print more details if True.
        """
        self.proceed = True
        self.verbose = verbose
        self.load_states = load_states
        self.reload = reload
        self.hb_kind = "states" if self.load_states else "pairs"
        self.n_target_states = n_target_states
        self.run_dir = Path(mcce_dir)
        self.msinfo_dict: dict = None

        start_setup = time()
        needed_fps = get_hb_paths(self.run_dir, ph=ph, eh=eh)
        if needed_fps is None:
            self.proceed = False
            return

        (self.h3_fp, self.step2_fp, self.msout_fp, self.fort38_fp,
         self.hah_fp,
         # outputs:
         self.hah_ms_fp,
         self.pairs_csv, self.pairs_res_csv,
         self.states_csv, self.states_pairs_csv) = needed_fps
        self.reductions_fp = self.hah_ms_fp.with_name(f"{self.hah_ms_fp.stem}_reductions.tsv")

        # ph, eh as string to match msout file:
        self.pheh_str = self.msout_fp.stem[:-2]

        # data from the msout file 'header':
        self.HDR = MsoutHeaderData(self.msout_fp)
        if self.HDR.method == "Invalid":
            self.proceed = False
            return

        if not self.HDR.is_monte:
            logging.info(NO_MONTE_MSG)
            self.proceed = False
            return

        self.CI = ConfInfo(self.h3_fp, verbose=self.verbose)
        # load the self.CI.conf_info lookup array:
        # fields: confid:0, crg:1, iconf:2, is_fixed:3, ires:4, is_free:5
        self.CI.load(self.HDR.iconf2ires, self.HDR.fixed_iconfs)
        # + CI.n_confs, CI.max_iconf, CI.max_ires
        if self.CI.conf_info is None:
            logging.info("[DATA MISMATCH]: Conformer info could not be loaded.")
            self.proceed = False
            return

        self.min_occ = float(min_occ)
        self.prec = num_prec(self.min_occ)
        self.min_occ_print = f"min occ >= {self.min_occ:.{self.prec}f}"

        # attributes populated by get_extended_iconfs:
        self.n_fx: int = 0
        self.fx_iconfs: List[int] = []
        self.n_bk: int = 0
        self.dm_iconfs: List[int] = []
        self.bkid2dmic = {}
        self.n_hb_confs: int = 0
        self.extended_iconfs: List[int] = []

        self.df = self.expand_hah_data()
        if self.df is None:
            self.proceed = False
            return

        if not self.expd_df_checks():
            logging.info("Some conformer indices appear to be missing.")
            self.proceed = False
            return

        # self.iconf2confid is used to convert indices to confids in the 
        # pairs file; the 1st dict, confid2iconf, is unused:
        _, self.iconf2confid = self.get_confs_mappings()
        self.setup_time = show_elapsed_time(start_setup, info="MS setup", return_time=True)

        if self.reload:  # no check
            self.missing_outputs = True
        else:
            self.missing_outputs = self.do_load_ms()

        # states space size, incremented by either load_ functions
        self.n_space: int = 0

        self.I: csr_matrix = None
        self.hb_states: dict = None
        self.n_hb_states: int = 0
        self.n_hb_space: int = 0
        self.hb_pairs: list = []
        self.n_hb_pairs: int = 0
        self.hb_adj = {}
        self.hb_adj_indices: np.ndarray
        self.hb_adj_indptr: np.ndarray
        self.P: np.ndarray   # dense pairs matrix

        return

    def reduce_hah_file(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        """Reduce the inital hah.txt file by removing entries that have:
          - H-bonding pairs with fixed conformers that are always OFF
          - H-bonding pairs between 2 BK conformers
          - 0 occupancy in fort.38
        The the reduced file is save as hah_{pheh}.txt, as it is ph/eh dependent,
        and is used in extend_hah_data function.

        Called by load_hah_file.
        """
        # remove confs with 0 occupancy
        vec = get_titr_vec(self.fort38_fp, str(self.HDR.pH))
        if vec is None:
            return None
        
        # assign occ to each conf; keep occ cols
        def get_occ(cid):
            try:
                return vec[np.where(vec[:,0]==cid)][0,1]
            except IndexError:
                if "BK" in cid:
                    return 1
                else:
                    return -1

        # write whatever was removed to self.reductions_fp:
        # occ columns are kept
        df["d_occ"] = df["confid_donor"].apply(get_occ)
        df["a_occ"] = df["confid_acceptor"].apply(get_occ)
        msk_negocc = (df["d_occ"]==-1) | (df["a_occ"]==-1)
        if msk_negocc.any():
            # problem!
            logging.error(f"Conformers with negative occ?!")
            return None

        msk_occ = (df["d_occ"]==0) | (df["a_occ"]==0)
        if msk_occ.any():
            with open(self.reductions_fp, "a") as fr:
                fr.write(".......................\n\n# Dropped: zero occ confs:\n")
                df.loc[msk_occ].to_csv(fr, sep="\t")
            df = df.drop(index=df.loc[msk_occ].index, axis=0)

        # remove BK-BK:
        msk_bk = (df["confid_donor"].str.contains("BK")) & (df["confid_acceptor"].str.contains("BK"))
        if msk_bk.any():
            with open(self.reductions_fp, "a") as fr:
                fr.write(".......................\n\n# Dropped: BK-BK pairs\n")
                df.loc[msk_bk].to_csv(fr, sep="\t")
            df = df.drop(index=df.loc[msk_bk].index, axis=0)

        df["d_off"] = df["confid_donor"].apply(self.CI.is_fixed_off)
        df["a_off"] = df["confid_acceptor"].apply(self.CI.is_fixed_off)
        msk_off = (df["d_off"]==1) | (df["a_off"]==1)
        if msk_off.any():
            with open(self.reductions_fp, "a") as fr:
                fr.write(".......................\n\n# Dropped: pairs with fixed-off member(s)\n")
                df.loc[msk_off].to_csv(fr, sep="\t")
            df = df.drop(index=df.loc[msk_off].index, axis=0)

        df = df.drop(columns=["d_off", "a_off"])

        reduced_fp = self.hah_fp.with_name(fHAH_FNAME.format(self.pheh_str))
        # reset hah path:
        self.hah_fp = reduced_fp
        self.hah_fp.write_text(df.to_string(index=False)+"\n")
        logging.info(f"H-bonding pairs in reduced file {self.hah_fp.name!s}: {df.shape[0]}")

        return df

    def load_hah_file(self) -> Union[pd.DataFrame, None]:
        """Used by expand_hah_data.
        Produces the ph-eh dependent 'hah' file (aka the reduced hah file) if not found.
         - If path not set, or old format: run detect_hbonds
         - If hah file has default output name 'step2_out_hah.txt',
           create the reduced file.
        """
        df = table_to_df(self.hah_fp)
        if df is None:
            self.proceed = False
            return

        if self.hah_fp.name==HAH_FNAME_INIT:
            df = self.reduce_hah_file(df)
            if df is None:
                logging.info("Could not reduce the input hah file.")
                self.proceed = False
                return
        logging.info(f"H-bonding pairs in {self.hah_fp.name!s}: {df.shape[0]}")

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
                                logging.info(f" Class {cx}: Extended slots for {K[i]} {kcol!r}: {len(keys)}")
                        else:
                            if self.verbose:
                                logging.info(f" Class {cx}: No extended slots for {K[i]} {kcol!r}")
                    else:
                        if self.verbose:
                            logging.info(f" Class {cx}: No extended slots for {K[i]} {kcol!r}")
            else:
                if self.verbose:
                    logging.info(f" No extended slots for empty pair class {cx}")

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

        logging.info(f"Number of extension slots from mixed pairs with fixed: {len(extend_fixed)}")
        logging.info(f"Number of extension slots from mixed pairs with BK: {len(extend_bk)}")
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

        logging.info(f"H-bonding conformers: {self.n_hb_confs:,}")

        return

    def expand_hah_data(self) -> pd.DataFrame:
        """Load the hah file into a dataframe & add:
         -conformer indices from h3
         -free conformer flags
         -the conformer indices used to build the hb pairs matrix P or adj list
        """
        if not self.proceed:
            return None

        if self.hah_ms_fp.exists():
            if self.reload:
                self.hah_ms_fp.unlink()
            else:
                exp_df = pd.read_csv(self.hah_ms_fp)
                # populate the needed attributes:
                fixd, fixa, bkd, bka = self.get_fixed_or_bk_confids(exp_df)
                self.get_extended_iconfs(fixd, fixa, bkd, bka)
                return exp_df

        # free and mixed hb pairs + needed indices
        logging.info(f"Creating the expanded hah file {self.hah_ms_fp!s}...")

        df = self.load_hah_file()
        if df is None:
            self.proceed = False
            return None

        def is_free_pair(ro) -> int:
            """Assign classes to pair kinds."""
            return pair_classes[(ro["free_d"], ro["free_a"])]

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
        # classes 1, 2 & 3:
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
        logging.info(f"Accepted H-bonding pairs in {self.hah_ms_fp.name}: {df.loc[df['free'].gt(0)].shape[0]}")

        return df

    def expd_df_checks(self) -> bool:
        passed = True
        valid = self.df["free"].gt(0)
        # final iconfs columns:
        for col in ["Mi", "Mj"]:
            if not all(self.df[col].isna() == False):
                logging.info(f"Failed check: column {col!r} should not have NA values")
                passed = False
            neg = self.df.loc[valid & (self.df[col]<0)]
            if neg.shape[0] != 0:
                # save to investigate:
                neg_fp = self.hah_ms_fp.with_suffix(f"_{col}.unmapped.csv")
                neg.to_csv(neg_fp)
                logging.error((f"Failed check: column {col!r} should not have negative values. "
                              f"Unmapped iconfs saved to {neg_fp!s}"))
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
        df = self.df.loc[msk].drop_duplicates(subset=["Mi", "Mj"])
        n_rows = df.shape[0]
        logging.info(f"Unique H-bonding pairs for matrix: {n_rows}")
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

        The CSR format:
            data         : An array containing all the non-zero values of the matrix,
                           stored row by row.
            indices      : An array containing the column index for each value in the
                           data array
            index pointer: An array that points to the start and end of the data for each
                           row within the data and indices arrays.

        The values for row i are in data[indptr[i]:indptr[i+1]] and the corresponding
        column indices are in indices[indptr[i]:indptr[i+1]].
        """
        indptr = np.zeros(self.n_hb_confs + 1, dtype=np.int32)
        indices = []
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
                               )
        return

    def load_hb_states(self):
        """Process the 'msout file' for H-bond microstates.
        """
        found_mc = False
        newmc = False
        # value list: E, count, occ
        hb_states =  defaultdict(lambda: [0., 0, 0.])
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
                        hb_states[sij][0] += float(fields[0])*count   # state energy
                        hb_states[sij][1] += count

                    if mc_lines % (self.n_skip*5000) == 0:
                        logging.info(f" Trace - mc lines processed: {mc_lines:,}...")

        # accepted ms with flipped iconfs
        logging.info(f"Processed mc lines: {mc_lines:,}")

        # update final states dict with aver E & occ if occ > 0.00001%:
        self.hb_states = defaultdict(lambda: [0., 0, 0.])   # [E, count, occ]
        for s in hb_states:
            occ = round(hb_states[s][1]/self.n_space, PREC)
            if occ >= self.min_occ:
                # size of the returned hb states space:
                self.n_hb_space += hb_states[s][1]
                # average E:
                self.hb_states[s][0] = hb_states[s][0]/self.n_space
                self.hb_states[s][1] = hb_states[s][1]
                self.hb_states[s][2] = occ
        del hb_states
        self.hb_states = dict(self.hb_states)
        self.n_hb_states = len(self.hb_states)
        if not self.n_hb_states:
            logging.warning(f"No hb states found @ {self.min_occ_print}")

        return

    def load_hb_pairs(self):
        """Process the 'msout file' for H-bonding pairs.
        Populate hb_pairs dict with the pair as index and [count, occ] as value.
        Notes:
         - The occ of the pair is that obtained over the entire state space, meaning
           it is consistent with the occupancies of the pair members from fort.38.
         - The 'effective' occ of a hb_pair is hb_state-dependent: it is the occ of the
           hb_state where it occurs; It can be obtained when the `ms_hbnets` tool is called
           with the --load_states option.
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

                    self.process_ms_pairs(xs, count)  # update pair matrix, self.P

                    if mc_lines % (self.n_skip*5000) == 0:
                        logging.info(f" Trace: processed mc lines {mc_lines:,}...")

        # accepted ms with flipped iconfs
        logging.info(f"Processed mc lines: {mc_lines:,}")

        # Save to list where each item is [(pi, pj), count, occ]
        self.hb_pairs = []
        # Added: Missing filtering for given min occ:
        indices = np.argwhere(self.P/self.n_space >= self.min_occ)
        self.n_hb_pairs = indices.shape[0]
        if not self.n_hb_pairs: 
            logging.warning(f"No hb pairs found @ {self.min_occ_print}")
        else:
            if self.n_hb_pairs > 1:
                for p in indices:
                    self.hb_pairs.append([(int(p[0]),int(p[1])),
                                          int(self.P[p[0], p[1]].sum(axis=0)),
                                          round(self.P[p[0], p[1]].sum(axis=0)/self.n_space, PREC)
                                          ])
            else:   # no for-loop
                p = indices[0]
                self.hb_pairs.append([(int(p[0]),int(p[1])),
                                       int(self.P[p[0], p[1]].sum(axis=0)),
                                       round(self.P[p[0], p[1]].sum(axis=0)/self.n_space, PREC)
                                       ])
        self.n_hb_space = max(self.hb_pairs, key=itemgetter(1))[1]

        return

    def hb_pairs_data2csv(self):
        if self.hb_pairs:
            df = pd.DataFrame(self.hb_pairs, columns=["index","count","occ"])
            df["occ"] = df["occ"].round(PREC)
            df["count"] = df["count"].astype("int32")
            df[["Mi","Mj"]] = df["index"].apply(lambda x: pd.Series([int(x[0]),int(x[1])]))
            df[["donor","acceptor"]] = df["index"].apply(
                lambda x: pd.Series([self.iconf2confid[x[0]],self.iconf2confid[x[1]]]))
            pairs_out = df[["Mi","Mj","donor","acceptor","count","occ"]]
            pairs_out = pairs_out.sort_values(by=["count", "Mi"], ascending=[False, True])

            # build comment about % returned pairs
            note = (f"# Data for the {self.n_hb_pairs:,} "
                    f"occupied hb_pairs (@ {self.min_occ_print}) whose max count represents "
                    f"{self.n_hb_space/self.n_space:.2%} of the "
                    f"entire space ({self.n_hb_space:,}/{self.n_space:,})\n"
            )
            with open(self.pairs_csv, "w") as fo:
                fo.write(note)
                pairs_out.to_csv(fo, index=False)

            # rename conf to 1-letter res codes for loading in cytoscape
            df[["res_d","res_a"]] = df.apply(
                lambda x: pd.Series([get_resid(x["donor"]), get_resid(x["acceptor"])]),
                axis=1)
            # add with_bk flag:
            df["with_bk"] = df.apply(
                lambda row: row["donor"][3:5]=="BK" or row["acceptor"][3:5]=="BK",
                axis=1)
            
            # collapse confs to resid with res occ = max conf occ:
            df_res = df.groupby(["res_d","res_a"], as_index=False).agg(
                 {"count":"max", "occ": "max", "with_bk": "max"})
            df_res["occ"] = df_res["occ"].round(PREC)
            df_res["count"] = df_res["count"].round(0).astype("int32")
            df_res = df_res.sort_values(by="count", ascending=False)
            df_res.to_csv(self.pairs_res_csv, index=False)

            # save new cols to expanded files
            df = df.drop(columns=["index","donor","acceptor","res_d","res_a"])
            # update expanded hah file with hb_pairs count, occ:
            hah_df = pd.read_csv(self.hah_ms_fp, comment="#")
            hah_df = hah_df.merge(df, left_on=["Mi","Mj"], right_on=["Mi","Mj"])
            hah_df.to_csv(self.hah_ms_fp, index=False)

            logging.info(("\nMain output files:\n"
                          f"  {self.pairs_csv!s} ({pairs_out.shape[0]:,} rows): hb pairs data\n"
                          f"  {self.pairs_res_csv!s} ({df_res.shape[0]:,} rows): "
                          "residues hb pairs (for loading in Cytoscape)\n"
                          ))

        return

    def hb_states_data2csv(self):
        """
        Outputs hb states data into two csv files:
         - hb_states_pH*.csv: the hb microstates file with the state id defined by hb pairs in that hb ms;
         - hb_states_pairs_pH*.csv: the unique hb pairs in all the returned hb microstates
        """
        if self.hb_states:
            dfs = pd.DataFrame.from_dict(self.hb_states,
                                         orient="index",
                                         columns=["averE", "count", "occ"]
                                         ).reset_index()
            dfs["averE"] = dfs["averE"].round(2)
            dfs["occ"] = dfs["occ"].round(PREC)
            dfs["state_id"] = None

            # get the unique hb pairs in all the returned hb microstates:
            states_pairs_dict = defaultdict(lambda: [0, 0.])
            # convert the index as tuple of integers to tuple of confids:
            for rx, ro in dfs.iterrows():
                state_id = ""
                for tpl in ro["index"]:
                    # get the effective count of each pair in the state key
                    states_pairs_dict[tpl][0] += ro["count"]
                    states_pairs_dict[tpl][1] += ro["occ"]
                    # convert conf indices to confids:
                    state_id = state_id + f"({self.iconf2confid[tpl[0]]},{self.iconf2confid[tpl[1]]}),"
                # update converted state id, minus trailing comma:
                dfs.loc[rx,"state_id"] = state_id[:-1]
     
            # prep hb states pairs output:
            dfsp = pd.DataFrame.from_dict(states_pairs_dict,
                                          orient="index",
                                          columns=["count", "occ"]).reset_index()
            dfsp[["donor","acceptor"]] = dfsp["index"].apply(
                lambda x: pd.Series([self.iconf2confid[x[0]],
                                     self.iconf2confid[x[1]]]))
            dfsp["occ"] = (dfsp["count"]/self.n_space).round(PREC)

            # add short residue-based confid as in pairs_res file
            dfsp[["res_d","res_a"]] = dfsp.apply(
                lambda x: pd.Series([get_resid(x["donor"]), get_resid(x["acceptor"])]), axis=1)
            # boolean indicator: one pair member involves BK conf; used for splitting heatmaps
            dfsp["with_bk"] = dfsp.apply(
                lambda row: row["donor"][3:5]=="BK" or row["acceptor"][3:5]=="BK", axis=1)

            # output file has same format as hb_pairs_res file
            # collapse confs to resid with res occ = max conf occ:
            dfsp_res = dfsp.groupby(["res_d","res_a"], as_index=False).agg(
                                    {"count":"max", "occ": "max", "with_bk": "max"})
            dfsp_res["occ"] = dfsp_res["occ"].round(PREC)
            dfsp_res["count"] = dfsp_res["count"].round(0).astype("int32")
            #dfsp_res = dfsp_res[["res_d", "res_a", "count", "occ", "with_bk"]]
            dfsp_res = dfsp_res.sort_values(by="count", ascending=False)

            with open(self.states_pairs_csv, "w") as fo:
                fo.write("# States hb pairs; state count, occ: per occupied returned states\n")
                dfsp.to_csv(fo, index=False)

            # exclude indices, sort & save
            dfs = dfs[["state_id","averE","count","occ"]].sort_values(by="count",
                                                                      ascending=False)
            # build comment about % returned states
            note = (f"# Data for the {self.n_hb_states:,} returned occupied "
                    f"hb_states (@ {self.min_occ_print}) whose sum count represents "
                    f"{self.n_hb_space/self.n_space:.2%} of the "
                    f"entire space ({self.n_hb_space:,}/{self.n_space:,})\n"
            )
            with open(self.states_csv, "w") as fo:
                fo.write(note)
                dfs.to_csv(fo, index=False)

            logging.info((f"\nMain output files:\n"
                          f"  {self.states_csv!s} ({dfs.shape[0]:,} rows): hb states data\n"
                          f"  {self.states_pairs_csv!s} ({dfsp.shape[0]:,} rows): hb pairs with state space occ\n"
                          ))
        return

    def __str__(self):
        if self.n_space:
            return (f"\nConformers: {self.CI.n_confs:,}\n"
                    f"Fixed residues: {self.HDR.n_fixed_ics:,}\n"
                    f"Free residues: {self.HDR.n_free_res:,}\n"
                    f"Free conformers: {self.HDR.n_free_ics:,}\n"
                    f"Background charge: {self.CI.background_crg:.0f}\n"
                    f"State space: {self.n_space:,}\n"
                    )
        return (f"\nConformers: {self.CI.n_confs:,}\n"
                f"Fixed residues: {self.HDR.n_fixed_ics:,}\n"
                f"Free residues: {self.HDR.n_free_res:,}\n"
                f"Free conformers: {self.HDR.n_free_ics:,}\n"
                f"Background charge: {self.CI.background_crg:.0f}\n"
                )

    def get_processed_ms_info_txt(self) -> str:
        """Text to output similar info for 'pairs' or 'states' processing.
        """
        txt = f"\nMicrostates space: {self.n_space:,}\n"
        txt = txt + f"Occupied H-bonding space @ {self.min_occ_print}: {self.n_hb_space:,} "
        txt = txt + f"({self.n_hb_space/self.n_space:.2%} of state space)\n"
        if self.load_states:       
            txt = txt + f"Saved occupied H-bonding states: {self.n_hb_states:,} (target: {self.n_target_states:,})\n"
        else:
            txt = txt + f"Occupied H-bonding pairs: {self.n_hb_pairs:,}\n"

        return txt

    def do_load_ms(self) -> bool:
        """Check presence of output according to self.load_states and self.reload.
        Returns:
         - go (boolean): True: load microstates, False: don't.
        """
        go = True
        if self.load_states:
            if self.states_csv.exists() and self.states_pairs_csv.exists():
                logging.info("Output files for hb states already exist.")
                if not self.reload:
                    go = False
                else:  # delete all
                    self.states_csv.unlink()
                    self.states_pairs_csv.unlink()
        else:
            if self.hah_ms_fp.exists() and self.pairs_csv.exists() and self.pairs_res_csv.exists():
                logging.info("Output files for hb pairs already exist.")
                if not self.reload:
                    go = False
                else:  # delete all
                    if self.reductions_fp.exists():
                        self.reductions_fp.unlink()
                    self.hah_ms_fp.unlink()
                    self.pairs_csv.unlink()
                    self.pairs_res_csv.unlink()
        
        return go

    def run_ms_pipeline(self, load_states: bool = False):
        """
        Takes load_states as parameter to enable ouputing hb_pairs and hb_states
        programmatically via calls to run_ms_pipeline with two load_states
        values, i.e. True, then False.
        """
        if not self.proceed:
            return

        self.load_states = load_states
        self.hb_kind = "states" if self.load_states else "pairs"

        if not self.missing_outputs:
            return
        
        start_pipeline = time()
        self.mc_lines, self.n_skip, self.n_MC = get_msout_size_info(self.msout_fp,
                                                n_target_states=self.n_target_states)
        logging.info(f"Approximate number of lines in msout file: {self.mc_lines:,}")
        if self.n_skip == 1:
            logging.info("Warning: Returning all hb_states and printing every 5000th accepted state.")

        if self.load_states:
            start_t = time()
            self.I = self.get_sparse_matrix()
            self.load_hb_states()
            logging.info(self.__str__())
            logging.info(show_elapsed_time(start_t, info="Loading H-bonding states"))
            if self.n_hb_states:
                logging.info(self.get_processed_ms_info_txt())
                # save loaded hb_states dict to csv:
                #start_t = time()
                self.hb_states_data2csv()
                #logging.info(show_elapsed_time(start_t, info="Processing final outputs for hb states"))
        else:
            start_t = time()
            self.hb_adj = self.get_adjacency_dict()
            self.hb_adj_indices, self.hb_adj_indptr = self.get_adj_idx_idxptr()
            self.P = np.zeros((self.n_hb_confs, self.n_hb_confs), dtype=np.int32)
            self.load_hb_pairs()
            logging.info(self.__str__())
            logging.info(show_elapsed_time(start_t, info="Loading H-bonding pairs"))
            if self.n_hb_pairs:
                logging.info(self.get_processed_ms_info_txt())
                # save loaded hb_pairs list to csv:
                #start_t = time()
                self.hb_pairs_data2csv()
                #logging.info(show_elapsed_time(start_t, info="Processing final outputs for hb pairs"))
            else:
                # remove expanded file
                self.hah_ms_fp.unlink()

        elapsed = time() - start_pipeline
        logging.info(show_elapsed_time(start_pipeline, info="MS H-bond pipeline"))
        tot_time = self.setup_time + elapsed
        logging.info(f"Elapsed time - Start to end: {tot_time:,.2f} s ({tot_time/60:,.2f} min)")

        return
