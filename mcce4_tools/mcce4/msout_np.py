#!/usr/bin/env python3

"""
Module: msout_np.py

Module for MSout_np class, the faster ms_out/ file loader, that loads
different microstates depending on the value of 'mc_load', the 'loading
mode' argument, which can be:
 - 'conf' :: only conformer microstates in MSout_np.all_ms
 - 'crg' :: only protonation microstates in MSout_np.all_cms
 - 'all :: both protonation and conformer microstates in both arrays.
When 'mc_load' is "all", boolean argument 'reduced_ms_row' set to True will
output in MSout_np.all_ms as many as charge microstates saved in MSout_np.all_cms.
With 'reduced_ms_row' set to False, all conformer microstates are saved.

Note:
* Naming convention:
- 'charge ms', 'crg ms' and 'cms' are shortcuts for protonation microstate.
- 'ms' is a shortcut for conformational microstate.
- 'msout file' refers to a .txt file that starts with 'pH<ph>' in the ms_out
  subfolder of an mcce run.

CHANGELOG:
2025-08-08: Added functionality to allow for the format of the ENUMERATE method,
            which is the analyticall method triggered when the number of accepted
            states is <= 1_000_000 (per NSTATE_MAX key default).
2025-07-30: Changed default value for argument N to None in MSout.get_topN_data.
            For backward compatibility, will default to N_top=5 if None.
            A calling cli tool would be responsible to define a default.
"""
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Union

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Oops! Forgot to activate an appropriate environment?\n{e}")
    sys.exit(1)

from mcce4.constants import IONIZABLE_RES as IONIZABLES, ROOMT, res3_to_res1
from mcce4.io_utils import MsoutHeaderData, MC_METHODS, N_HDR
from mcce4.io_utils import reader_gen, show_elapsed_time


MIN_OCC = 0.0  # occ threshold
N_TOP = 5
MAX_INT = np.iinfo(np.int32).max
HIS0_tautomers = {0: "NE2", 1: "ND1", 2: 1}


def topN_loadtime_estimate(n_freeres: int) -> str:
    # # TODO: Redo estimate
    #     """Returns the time estimate given the number of free residues
    #     for reading the mc lines to getting the topN ms in a formatted
    #     string showing seconds and minutes.
    #     """
    #     # fit of 5 runs:
    #     # -2 offset: improvements since fit
    #     return round(-14.9897855 -2 + 0.451883977*n_freeres + 6.25518650e-04*n_freeres**2)
    pass


class ConfInfo:
    """This class handles the loading of head3 data into a numpy array
    and provides one accessor function is_fixed_off(iconf).
    The 'conf_info' attribute (np.ndarray) is a lookup 'table' for
    these fields: conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
    """
    def __init__(self, h3_fp: Path, verbose: bool = False):
        self.h3_fp = h3_fp
        self.verbose = verbose

        self.conf_info: np.ndarray = None
        self.conf_ids: List = []
        self.n_confs: int = 0
        self.cms_resids: List = []
        self.n_resids: int = 0
        self.background_crg: int = None

    def load(self, iconf2ires: Dict, fixed_iconfs: List[int], with_tautomers: bool,
             residue_kinds: List[str] = None):
        """Popuate the 'conf_info' attribute (np.ndarray): a lookup 'table' for:
        conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        """
        print("\nPopulating the lookup array with head3.lst and msout file header data")
        conf_info = []
        conf_vec = []
        with open(self.h3_fp) as h3:
            lines = h3.readlines()[1:]

        for line in lines:
            # ignored columns: FL & fields past confid
            iConf, confid, _, _, Crg, *_ = line.split()
            iconf = int(iConf) - 1  # as python index
            kind = confid[:3]
            resid = kind + confid[5:11]
            crg = int(float(Crg))
            if with_tautomers:
                if kind == "HIS":
                    # reset crg to pseudo crg:
                    # 0 :: HIS01->" NE2"; 1 :: HIS02->" ND1"; 2 :: HIS+1
                    crg = int(confid[4]) - 1 if confid[3] == "0" else 2

            is_ioniz = int(resid[:3] in IONIZABLES)
            in_kinds = 1  # preset to accept all if next condition is False
            if residue_kinds is not None and len(residue_kinds):
                in_kinds = int(kind in residue_kinds)
            # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
            #              0      1        2       3         4         5        6     7
            #                                               -4        -3       -2    -1
            conf_info.append([iconf, resid, in_kinds, is_ioniz, -1, 0, -1, crg])
            conf_vec.append([iconf, confid])

        # temp list structure is now sized & has h3 info; cast to np.ndarray:
        conf_info = np.array(conf_info, dtype=object)
        self.conf_ids = np.array(conf_vec, dtype=object)

        # is_free field
        # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        free_ics = list(iconf2ires.keys())
        conf_info[free_ics, -3] = 1

        # populate the ires of free res, if possible
        for i, (iconf, *_) in enumerate(conf_info): 
            conf_info[i][-2] = iconf2ires.get(iconf, -1)

        # populate the 'is_fixed' as not free & not fixed 'off'
        conf_info[np.where((conf_info[:,-3]==0) & np.isin(conf_info[:,0], fixed_iconfs)), -4] = 1
        
        # get cms unique resids list via filtering conf_info for valid confs for
        # protonation state vec: is_ioniz & is_free & in user list if given.

        # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        sum_conditions = conf_info[:, 3] + conf_info[:, -3]   # ionizable & free
        sum_tot = 2
        if residue_kinds:
            # in_kinds & is_ioniz & is_free
            sum_conditions = conf_info[:, 2] + sum_conditions
            sum_tot = 3
        # Note: dict in use instead of a set (or np.unique) to preserve the order:
        d = defaultdict(int)
        for r in conf_info[np.where(sum_conditions == sum_tot)][:, 1]:
            d[r] += 1
        # uniq resids from dict keys:
        self.cms_resids = list(d.keys())
        self.n_resids = len(self.cms_resids)
        # create mapping from confs space to protonation resids space:
        # reset conf_info resix field to the index from cms_resids list or -1:

        # Getting resix w/o checking again for is_free was not sufficient,
        # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        for i, (_, resid, _, _, _, is_free, *_) in enumerate(conf_info):
            try:
                resix = self.cms_resids.index(resid)
                if not is_free:
                    resix = -1
            except ValueError:
                # put sentinel flag for unmatched res:
                resix = -1
            conf_info[i][-2] = resix
        if self.verbose:
            print("\nHead3 lookup array 'conf_info' fields ::",
                "iconf:0, resid:1, in_kinds:2, is_ioniz:3,",
                "is_fixed:4, is_free:5, resix:6, crg:7\n")

        self.n_confs = conf_info.shape[0]
        # sumcrg for not is_free & is_fixed on:
        # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        self.background_crg = conf_info[np.where((conf_info[:,-3]==0) 
                                                  & (conf_info[:,-4]==1)), -1].sum()
        if self.verbose:
            print(f" Background crg: {self.background_crg}",
                  f" n_confs: {self.n_confs}", sep="\n")
        self.conf_info = conf_info

        return

    def is_fixed_off(self, iconf: int) -> int:
        """conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        """
        try:
            val = self.conf_info[np.where((self.conf_info[:,0]==iconf)
                                          & (self.conf_info[:,-4]!=1))][0,-4]
            if val == -1: val = 1
        except IndexError:
            val = 0

        return val


# TODO: test res_kinds with mc_load="conf"
class MSout_np:
    """Class to load 'msout file' to obtain MCCE microstates data in numpy.arrays.
    * Naming convention:
        - 'charge ms', 'crg ms' and 'cms' are shortcuts for protonation microstate.
        - 'ms' is a shortcut for conformational microstate.
        - 'msout file' refers to the .txt file in the ms_out subfolder of an mcce run
           and starts with 'pH<ph>'.

    Arguments:
        - head3_file, msout_file (str): Paths to head3.lst & msout files.
        - mc_load (str, "all"): Specifies what to load from the msout file:
           - 'all': crg and conf ms.
           - 'conf': conformation ms only (as in ms_analysis.MSout class);
           - 'crg': protonation ms only;
        - res_kinds (list, None): List of 3-letter residue/ligand names, e.g. ['GLU', 'HEM'];
          Defaults to IONIZABLES with mc_load='all' or 'crg'.
        - with_tautomers (bool, False): Whether to return the tautomer string instead of the charge
        - reduced_ms_rows (bool, False): With mc_load="all", indicates whether to save all
          conformers into MSout_np.all_ms array or only as many as the charge microstates.

    Details:
        1. Reads head3.lst & the msout file header and creates a conformer data
           'lookup table' into np.array attribute 'conf_info'.
        2. Loads accepted states information from msout file MC lines into a list,
           which is recast to np.array at end of processing, yielding MSout_np.all_ms
           or MSout_np.all_cms.

    Example for obtaining topN data (crg ms and related conf ms):
        ```
        from msout_np import MSout_np

        h3_fp = "path/to/head3.lst"
        msout_fp = "path/to/ms_out/msout_file/filename"

        # using defaults: mc_load="all", res_kinds=IONIZABLES, loadtime_estimate=False
        mso = MSout_np(h3_fp, msout_fp)
        print(mso)

        # run get_uniq_ms() method, which will populate mso.uniq_cms (as per mc_load)
        mso.get_uniq_ms()
 
        # Get topN data using defaults: N = 5; min_occ = 0.0
        top_cms, top_ms = mso.get_topN_data()
        ```
    """
    def __init__(self, head3_file: str, msout_file: str,
                 mc_load: str = "all",
                 res_kinds: list = None,
                 with_tautomers: bool = False,
                 #loadtime_estimate: bool = False,
                 reduced_ms_rows: bool = False,
                 verbose: bool = False,
                 ):
        self.verbose = verbose
        self.h3_fp = Path(head3_file)
        self.msout_fp = Path(msout_file)
        self.reduced_ms_rows = reduced_ms_rows
        self.mc_load: str = None

        # populate these attributes:
        self.with_tautomers: bool = None
        self.res_kinds: list = None
        self.validate_kwargs(mc_load, res_kinds, with_tautomers)

        self.HDR = MsoutHeaderData(self.msout_fp)

        self.CI = ConfInfo(self.h3_fp, verbose=self.verbose)
        # load the self.CI.conf_info lookup array:
        # fields: iconf:0, resid:1, in_kinds:2, is_ioniz:3, is_fixed:4, is_free:5, resix:6, crg:7
        self.CI.load(self.HDR.iconf2ires, self.HDR.fixed_iconfs, self.with_tautomers,
                     residue_kinds=self.res_kinds)
        # copy from CI.conf_info
        self.conf_info = self.CI.conf_info
        self.cms_resids = self.CI.cms_resids  # :: list of resids defining a cms
        self.n_resids = self.CI.n_resids

        # attributes populated by the 'load' functions:
        self.N_space: int = None     # size of state space
        self.N_mc_lines: int = None  # total number of mc lines (accepted states data)
        self.N_cms: int = None       # total number of crg (protonation) ms
        # np.arrays to receive the lists of conf/crg ms: 
        self.all_ms: np.ndarray = None
        self.all_cms: np.ndarray = None

        # attributes populated by the 'get_uniq_' functions:
        self.uniq_ms: np.ndarray = None
        self.uniq_cms: np.ndarray = None
        self.N_ms_uniq: int = None  # unique number of conf ms
        self.N_cms_uniq: int = None  # unique number of crg ms

        # load accepted states:
        if self.mc_load == "conf":
            start_t = time.time()
            self.load_conf()
            show_elapsed_time(start_t, info="Loading msout for conf ms")

        elif self.mc_load == "crg":
            start_t = time.time()
            self.load_crg()
            show_elapsed_time(start_t, info="Loading msout for cms")

        elif self.mc_load == "all":
            # if loadtime_estimate:
            #     yt = topN_loadtime_estimate(len(self.HDR.free_residues))  #free_iconfs?
            #     print(f"\nESTIMATED TIME to topN: {yt:,.2f} s ({yt/60:,.2f} min).\n")
            start_t = time.time()
            self.load_all()
            show_elapsed_time(start_t, info="Loading msout for ms & cms")
        else:
            print("No processing function associated with:", self.mc_load)

        return

    def validate_kwargs(self, mc_load: str, res_kinds: list, with_tautomers: bool):
        # valid loading modes:
        loading_modes = ["conf", "crg", "all"]
        self.mc_load = mc_load.lower()
        if self.mc_load not in loading_modes:
            msg = ("Argument mc_load must be one of "
                   f"{loading_modes} "
                   "to load either conformer or charge microstates, or both.")
            sys.exit(msg)

        if with_tautomers and self.mc_load == "conf":
            # not applicable (no cms returned), mc_load has precedence, reset:
            self.with_tautomers = False
        else:
            self.with_tautomers = with_tautomers

        if res_kinds is None:
            if self.mc_load != "conf":
                self.res_kinds = IONIZABLES
            else:
                self.res_kinds = None
        else:
            if self.mc_load == "conf":
                # CodeReview: it is `self.load_conf()` that is not setup for filtering
                #             per res_kinds; `ConfInfo.load` can handle res_kinds if not None.
                print("WARNING: Residue selection when loading conformer microstates",
                      "is not implemented: res_kinds reset to None.")
                self.res_kinds = None
            else:
                self.res_kinds = res_kinds

        return

    def get_ter_dict(self) -> dict:
        """Return a dict for res with multiple entries, such as
        terminal residues.
        Sample output, dict: {'A0001': ['NTR', 'LYS'],
                              'A0129': ['LEU', 'CTR']}
        """
        ter_dict = defaultdict(list)
        for confid in self.CI.conf_ids[:,1]:
            res = confid[:3]
            res_id = confid[5:].split("_")[0]
            # order needed, can't use set():
            if res not in ter_dict[res_id]:
                ter_dict[res_id].append(res)

        return dict((k, v) for k, v in ter_dict.items() if len(v) > 1)

    def load_conf(self):
        """Process the 'msout file' mc lines to populate a list of
        [state, state.E, count] items if method is MONTERUNS or [state, state.E, occ] if
        method is ENUMERATE, where state is a list of conformal microstates.
        This list is then assigned to MSout.all_ms as a numpy.array.
        """
        # print("Loading function: load_conf")
        found_mc = False
        newmc = False
        ms_vec = []  # list to hold conf ms info

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

                    state_e = float(fields[0])
                    if self.HDR.is_monte:
                        count = int(fields[1])
                    else:  # occ
                        count = float(fields[1])
                    flipped = [int(c) for c in fields[2].split()]
                    for ic in flipped:
                        ir = self.HDR.iconf2ires[ic]
                        current_state[ir] = ic

                    ms_vec.append([list(current_state), state_e, count])

        if self.HDR.is_monte:
            self.N_mc_lines = lx - N_HDR - 4
        else:
            self.N_mc_lines = lx - N_HDR - 2
        print(f"Accepted states lines: ~ {self.N_mc_lines:,}\n")

        if ms_vec:
            self.all_ms = np.array(ms_vec, dtype=object)
            if self.HDR.is_monte:
                self.N_space = self.all_ms[:, -1].sum()
                print(f"State space: {self.N_space:,}")
            self.N_ms = len(self.all_ms)
            print(f"Conformer microstates loaded: {self.N_ms:,}\n")
        else:
            return ValueError("Something went wrong in loading msout file: 'ms_vec' is empty!")

        return

    def load_crg(self):
        """Process the accepted microstates lines to populate a list of
        [state, totE, averE, count] items, where state is a list of protonation
        microstates for the free & ionizable residues in the simulation.
        This list is then assignedd to MCout.all_cms as a numpy.array.
        """
        found_mc = False
        newmc = False
        ro = -1
        # list to hold crg ms info:
        cms_vec = []

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
                        ro += 1  # will be 0 at 1st mc line (state)
                        # line with candidate state for MC sampling, e.g.:
                        # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                        current_state = [int(i) for i in line.split(":")[1].split()]

                        # cms_vec :: [state, totE, averE, occ]
                        cms_vec.append([[0] * self.n_resids, 0, 0, 0])
                        # update cms_vec state:
                        curr_info = self.conf_info[current_state]

                        # acceptable conformer: ionizable, free, and in res_kinds if provided, meaning
                        # field 'resix' has a valid index (positive int) => resix != -1
                        # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
                        upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                        for u in upd:
                            cms_vec[ro][0][u[0]] = u[1]
                        newmc = False
                        continue

                if newmc:
                    ro += 1  # will be 0 at "MC:0" + 1 line
                    # line with candidate state for MC sampling, e.g.:
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                    current_state = [int(i) for i in line.split(":")[1].split()]
                    # cms_vec :: [state, totE, averE, count]
                    cms_vec.append([[0] * self.n_resids, 0, 0, 0])
                    # update cms_vec state:
                    curr_info = self.conf_info[current_state]

                    # acceptable conformer: ionizable, free, and in res_kinds if provided, meaning
                    # field 'resix' has a valid index (positive int) => resix != -1
                    # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
                    upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                    for u in upd:
                        cms_vec[ro][0][u[0]] = u[1]
                    newmc = False
                    continue

                if found_mc:
                    fields = line.split(",")
                    if len(fields) < 3:
                        continue

                    state_e = float(fields[0])
                    if self.HDR.is_monte:
                        count = int(fields[1])
                    else:
                        count = float(fields[1])
                    flipped = [int(c) for c in fields[2].split()]
                    for ic in flipped:
                        ir = self.HDR.iconf2ires[ic]
                        current_state[ir] = ic

                    # flipped iconfs from non-ionizable or fixed res
                    # => same protonation state: increment totE & count;
                    # Note: -1 is a sentinel index for this situation.
                    update_cms = np.all(self.conf_info[flipped, -2] == -1)
                    if update_cms:
                        if self.HDR.is_monte:
                            # cms_vec ::  [state, totE, averE, count]
                            cms_vec[ro][1] += state_e * count
                            cms_vec[ro][3] += count
                            cms_vec[ro][2] = cms_vec[ro][1] / cms_vec[ro][3]
                        else:
                            # cms_vec ::  [state, totE, np.nan, occ]
                            cms_vec[ro][1] += state_e
                            cms_vec[ro][2] = np.nan
                            cms_vec[ro][3] += count

                    else:  # new crg ms
                        ro += 1
                        if self.HDR.is_monte:
                            cms_vec.append([[0] * self.n_resids, state_e * count, state_e, count])
                        else:
                            cms_vec.append([[0] * self.n_resids, state_e, np.nan, count])
                        # update cms_vec state:
                        curr_info = self.conf_info[current_state]
                        upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                        for u in upd:
                            cms_vec[ro][0][u[0]] = u[1]

        if self.HDR.is_monte:
            self.N_mc_lines = lx - N_HDR - 4
        else:
            self.N_mc_lines = lx - N_HDR - 2
        print(f"Accepted states lines: ~ {self.N_mc_lines:,}\n")

        if cms_vec:
            self.all_cms = np.array(cms_vec, dtype=object)
            # remove 0 count; hack to remove first initialized vec
            self.all_cms = self.all_cms[np.where(self.all_cms[:,-1] != 0)]
            if self.HDR.is_monte:
                self.N_space = self.all_cms[:, -1].sum()
                print(f"State space: {self.N_space:,}")
            self.N_cms = len(self.all_cms)
            print(f"Protonation microstates: {self.N_cms:,}\n")
        else:
            return ValueError("Something went wrong in loading msout file: 'cms_vec' is empty!")

        return

    def load_all(self):
        """Process the 'msout file' mc lines to output both conformal
        and protonation microstates to numpy.arrays MSout_np.all_ms 
        and MSout_np.all_cms.
        """
        print("Loading ms and cms data into arrays.")

        found_mc = False
        newmc = False
        ro = -1  # list item accessor
        current_state = []
        # lists to hold conf and crg ms info; they can be related by their common index;
        cms_vec = []
        ms_vec = []

        msout_data = reader_gen(self.msout_fp)
        # start MUST be 1
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
                        # newmc = True
                        # current_state = [int(i) for i in line.split(":")[1].split()]
                        # continue

                        ro += 1  # will be 0 at 1st mc line (state)
                        # line with candidate state for MC sampling, e.g.:
                        # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                        current_state = [int(i) for i in line.split(":")[1].split()]

                        # initialize the vectors:
                        ms_vec.append([ro, list(current_state), 0, 0])
                        #  cms_vec ::  [idx, state, totE, averE, count]
                        cms_vec.append([ro, [0] * self.n_resids, 0, 0, 0])

                        # update cms_vec state:
                        curr_info = self.conf_info[current_state]
                        # acceptable conformer: ionizable, free, and in res_kinds if provided, meaning
                        # field 'resix' has a valid index (positive int) => resix != -1
                        # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
                        upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                        for u in upd:
                            cms_vec[ro][1][u[0]] = u[1]
                        newmc = False
                        continue
                
                if newmc:
                    ro += 1  # will be 0 at "MC:0" + 1 line
                    #if not current_state:
                    current_state = [int(i) for i in line.split(":")[1].split()]

                    # initialize the vectors:
                    ms_vec.append([ro, list(current_state), 0, 0])

                    # cms_vec ::  [idx, state, totE, averE, count]
                    cms_vec.append([ro, [0] * self.n_resids, 0, 0, 0])
                    # update cms_vec state:
                    curr_info = self.conf_info[current_state]
                    upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                    for u in upd:
                        cms_vec[ro][1][u[0]] = u[1]
                    newmc = False
                    continue

                if found_mc:
                    fields = line.split(",")
                    if len(fields) < 3:
                        continue
                    state_e = float(fields[0])
                    if self.HDR.is_monte:
                        count = int(fields[1])
                    else:
                        count = float(fields[1])
                    flipped = [int(c) for c in fields[2].split()]
                    for ic in flipped:
                        ir = self.HDR.iconf2ires[ic]
                        current_state[ir] = ic

                    if ro == 0:
                        #ms_vec.append([ro, list(current_state), state_e, count])
                        # update the 1st ms:
                        ms_vec[0][2] = state_e
                        ms_vec[0][3] = count
                    #else:
                    if not self.reduced_ms_rows:
                        # save all conformer ms:
                        ms_vec.append([ro, list(current_state), state_e, count])

                    # if the flipped iconfs are from non-ionizable or fixed res,
                    # the protonation state is the same: increment count & E;
                    # Note: -1 is a sentinel index for this situation.
                    update_cms = np.all(self.conf_info[flipped, -2] == -1)
                    if update_cms:
                        if self.HDR.is_monte:
                            # cms_vec ::  [idx, state, totE, averE, count]
                            cms_vec[ro][2] += state_e * count
                            cms_vec[ro][4] += count
                            cms_vec[ro][3] = cms_vec[ro][2] / cms_vec[ro][4]
                        else:
                            # cms_vec ::  [idx, state, totE, np.nan, occ]
                            cms_vec[ro][2] += state_e
                            cms_vec[ro][3] = np.nan
                            cms_vec[ro][4] += count
                    else:
                        ro += 1  # new cms
                        if self.reduced_ms_rows:
                            # save the 'associated' conformer ms:
                            ms_vec.append([ro, list(current_state), state_e, count])
                        # save new cms, create new list item & update with data from
                        # lookup array for the current state
                        if self.HDR.is_monte:
                            cms_vec.append([ro,
                                            [0] * self.n_resids,
                                            state_e * count, state_e, count])
                        else:
                            cms_vec.append([ro,
                                            [0] * self.n_resids,
                                            state_e, np.nan, count])
                        # update cms_vec state:
                        curr_info = self.conf_info[current_state]
                        upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                        for u in upd:
                            cms_vec[ro][1][u[0]] = u[1]

        if self.HDR.is_monte:
            self.N_mc_lines = lx - N_HDR - 4
        else:
            self.N_mc_lines = lx - N_HDR - 2
        print(f"Accepted states lines: ~ {self.N_mc_lines:,}\n")

        if ms_vec:
            self.all_ms = np.array(ms_vec, dtype=object)
            # remove 0 count; hack to remove first initialized vec
            self.all_ms = self.all_ms[np.where(self.all_ms[:,-1] != 0)]
            self.N_ms = len(self.all_ms)
            print(f"Conformer microstates loaded: {self.N_ms:,}\n")
        else:
            return ValueError("Something went wrong in loading msout file: 'ms_vec' is empty!")
        
        if cms_vec:
            self.all_cms = np.array(cms_vec, dtype=object)
            # remove 0 count; hack to remove first initialized vec
            self.all_cms = self.all_cms[np.where(self.all_cms[:,-1] != 0)]
            self.N_cms = len(self.all_cms)
            print(f"Protonation microstates found: {self.N_cms:,}\n")
            if self.HDR.is_monte:
                self.N_space = self.all_cms[:, -1].sum()
                print(f"State space: {self.N_space:,}")
        else:
            return ValueError("Something went wrong in loading msout file: 'cms_vec' is empty!")

        return

    def get_uniq_ms(self):
        """Semaphore function to call the 'get unique' function corresponding
        to .mc_load loading mode.
        Note:
          If the loading mode is "all", it's assume that crg ms are the main
          focus thus, there is no need to get the unique conf ms as well.
          The equivalent function for conformers, `MSout_np.get_uniq_all_ms()`
          is available if needed.
        """
        if self.mc_load == "conf":
            start_t = time.time()
            self._get_uniq_conf()
            show_elapsed_time(start_t, info="Populating MSout_np.uniq_ms array")
            print(f"Unique conformer microstates: {self.N_ms_uniq:,}\n")

        elif self.mc_load == "crg":
            start_t = time.time()
            self._get_uniq_cms()
            show_elapsed_time(start_t, info="Populating MSout_np.uniq_cms array")
            print(f"Unique protonation microstates: {self.N_cms_uniq:,}\n")

        elif self.mc_load == "all":
            start_t = time.time()
            self._get_uniq_all_cms()
            show_elapsed_time(start_t, info="Populating MSout_np.uniq_cms array, 'all' mode")
            print(f"Unique protonation microstates: {self.N_cms_uniq:,}\n")

        else:
            print(f"WARNING: No processing function associated with: {self.mc_load}")

        return

    def _get_uniq_cms(self):
        """Assign unique crg ms info to self.uniq_cms and assign count of unique ms to self.N_cms_uniq.
        The values of MSout.uniq_cms, the populated array, depend on the msout method:
          - [state, totE, averE, occ, count] if MONTERUNS
          - [state, totE, np.nan, occ, np.nan] if ENUMERATE
        """
        subtot_d = {}
        if self.HDR.is_monte:
            # crg_ms in :: [state, totE, averE, count]
            for _, itm in enumerate(self.all_cms):
                key = tuple(itm[0])
                if key in subtot_d:
                    subtot_d[key][1] += itm[1]
                    subtot_d[key][3] += itm[3]
                    subtot_d[key][2] = subtot_d[key][1] / subtot_d[key][3]
                else:
                    subtot_d[key] = itm.copy()

            self.N_cms_uniq = len(subtot_d)
            # add occ, sort by count & assign to self.uniq_cms as np.array:
            # crg_ms out :: [state, totE, averE, occ, count]
            self.uniq_cms = np.array(
                sorted(
                [
                    [list(k), subtot_d[k][1], subtot_d[k][2], subtot_d[k][3] / self.N_space, subtot_d[k][3]]
                    for k in subtot_d
                ],
                key=lambda x: x[-1],
                reverse=True,
            ), dtype=object)
        else:
            # cms in :: [state, totE, np.nan, occ]
            for _, itm in enumerate(self.all_cms):
                key = tuple(itm[0])
                if key in subtot_d:
                    subtot_d[key][1] += itm[1]
                    subtot_d[key][2] = np.nan
                    subtot_d[key][3] += itm[3]
                else:
                    subtot_d[key] = itm.copy()

            self.N_cms_uniq = len(subtot_d)

            # cms out :: [state, totE, np.nan, occ, np.nan]  to keep same shape
            # sort by occ:
            self.uniq_cms = np.array(
                sorted(
                    [
                    [list(k), subtot_d[k][1], subtot_d[k][2], subtot_d[k][3], np.nan]
                    for k in subtot_d
                    ],
                    key=lambda x: x[-2], reverse=True), dtype=object)

        return

    def _get_uniq_conf(self):
        """Assign unique conf ms info to self.uniq_ms and assign count of unique ms to self.N_ms_uniq.
        The values of MSout.uniq_ms, the populated array, depend on the msout method:
          - [state, state.e, occ, count] if MONTERUNS
          - [state, state.e, occ, np.nan] if ENUMERATE
        """
        if self.mc_load != "conf":
            sys.exit("CRITICAL: Wrong call to '_get_uniq_conf': 'mc_load' must be 'conf'.")

        if self.HDR.is_monte:
            # ms in ::  [state, state.e, count]
            # use dict to get unique states
            subtot_d = {}
            for _, itm in enumerate(self.all_ms):
                key = tuple(itm[0])
                if key in subtot_d:
                    subtot_d[key][2] += itm[2]
                else:
                    subtot_d[key] = itm.copy()

            self.N_ms_uniq = len(subtot_d)
        
            # add occ, sort by count & assign to self.uniq_ms as np.array:
            # ms out ::  [state, state.e, occ, count]
            mslist = [
                [list(k), subtot_d[k][1], subtot_d[k][-1] / self.N_space, subtot_d[k][-1]]
                 for k in subtot_d
                 ]
            # sort by count:
            self.uniq_ms = np.array(sorted(mslist, key=lambda x: x[-1], reverse=True), dtype=object)
        else:
            # ENUMERATE :: ANALYTICAL SOLUTION => all unique despite occ precision problem, see
            #              https://github.com/GunnerLab/MCCE4-Tools/issues/22
            self.N_ms_uniq = len(self.all_ms)
            # ms in  :: [state, state.e, occ]
            # ms out :: [state, state.e, occ, np.nan]  to keep same shape
            # sort by occ:
            self.uniq_ms = np.array(sorted([[ms[0], ms[1], ms[2], np.nan] for ms in self.all_ms],
                                           key=lambda x: x[-2], reverse=True), dtype=object)

        return

    def _get_uniq_all_cms(self):
        """Get the unique charge ms array when the `all_cms` array
        was produced together with the `all_ms` array, i.e. mc_load='all'.
        In this case, each of their items starts with an index,
        which can be used to match conf ms to each unique cms.
        """
        print("Getting unique cms array.")
        subtot_d = {}
        if self.HDR.is_monte:
            # vec :: [idx, state, totE, averE, count]
            for _, itm in enumerate(self.all_cms):
                key = tuple(itm[1])
                if key in subtot_d:
                    subtot_d[key][2] += itm[2]
                    subtot_d[key][4] += itm[4]
                    subtot_d[key][3] = subtot_d[key][2] / subtot_d[key][4]
                else:
                    subtot_d[key] = itm.copy()

            self.N_cms_uniq = len(subtot_d)
            # add occ, sort by count & assign to self.uniq_cms as np.array:
            # crg ms ::  [idx, state, totE, averE, occ, count]
            # sort by count
            self.uniq_cms = np.array(
                sorted(
                    [
                        [
                            subtot_d[k][0],
                            list(k),
                            subtot_d[k][2],
                            subtot_d[k][3],
                            subtot_d[k][4] / self.N_space,
                            subtot_d[k][4],
                        ]
                        for k in subtot_d
                    ],
                    key=lambda x: x[-1],
                    reverse=True,
                ),
                dtype=object,
            )
        else:
            # cms in  :: [idx, state, totE, np.nan, occ]
            for _, itm in enumerate(self.all_cms):
                key = tuple(itm[1])
                if key in subtot_d:
                    subtot_d[key][2] += itm[2]
                    subtot_d[key][3] = np.nan
                    subtot_d[key][4] += itm[4]
                else:
                    subtot_d[key] = itm.copy()
            self.N_cms_uniq = len(subtot_d)

            # cms out :: [id, state, totE, np.nan, occ, np.nan]  to keep same shape
            # sort by occ:
            self.uniq_cms = np.array(
                sorted(
                    [
                    [subtot_d[k][0], list(k), subtot_d[k][1], subtot_d[k][2], subtot_d[k][3], np.nan]
                    for k in subtot_d
                    ],
                    key=lambda x: x[-2], reverse=True), dtype=object)
        return

    def _get_uniq_all_ms(self):
        print("Getting unique ms array.")
        if self.method == "ENUMERATE":
            #print("The conformer microstates returned by the analytical method are unique.")
            self.N_ms_uniq = len(self.all_ms)
            # ms out :: [id, state, state.e, occ, np.nan]
            # sort by occ:
            self.uniq_ms = np.array(
                sorted([[ms[0], ms[1], ms[2], np.nan] for ms in self.all_ms],
                       key=lambda x: x[-2], reverse=True
                       ),
                    dtype=object
                    )
            return
            
        # ms in ::  [idx, state, state.e, count]
        subtot_d = {}
        for _, itm in enumerate(self.all_ms):
            key = tuple(itm[1])
            if key in subtot_d:
                subtot_d[key][3] += itm[3]
            else:
                subtot_d[key] = itm.copy()

        self.N_ms_uniq = len(subtot_d)
        # add occ, sort by count & assign to self.uniq_ms as np.array:
        # ms out ::  [idx, state, state.e, occ, count]
        mslist = [
            [subtot_d[k][0], list(k), subtot_d[k][2], subtot_d[k][-1] / self.N_space, subtot_d[k][-1]]
            for k in subtot_d
        ]
        self.uniq_ms = np.array(sorted(mslist, key=lambda x: x[-1], reverse=True), dtype=object)

        return
    
    def get_free_residues_df(self) -> pd.DataFrame:
        """Extract resid for is_free from lookup array into a pandas.DataFrame."""
        free_residues_df = pd.DataFrame(self.conf_info[np.where(self.conf_info[:,-3]==1), 1][0],
                                        columns=["Residue"])
        free_residues_df.drop_duplicates(inplace=True)
        return free_residues_df

    def get_fixed_residues_arr(self) -> np.ndarray:
        """Extract resid, crg for is_ioniz & is_fixed from lookup array."""
        # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        return self.conf_info[np.where(np.logical_and(self.conf_info[:,3]==1,
                                                      self.conf_info[:,4]==1))][:, [1,-1]]

    def get_fixed_residues_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_fixed_residues_arr(), columns=["Residue", "crg"])

    def get_fixed_res_of_interest_arr(self) -> np.ndarray:
        """Extract resid, crg for in_kinds & is_fixed from lookup array."""
        # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        return self.conf_info[np.where(np.logical_and(self.conf_info[:,2]==1,
                                                      self.conf_info[:,4]==1))][:, [1, -1]]

    def get_fixed_res_of_interest_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_fixed_res_of_interest_arr(),
                            columns=["Residue", "crg"])

    def get_cms_energy_stats(self) -> Tuple[float, float, float]:
        """Return the minimum, average, and maximum energies of the cms in .all_cms."""
        cms_e = self.all_cms[:, -2]
        # Charge microstate energy stats (min, avg, max): (-706.26, np.float64(-691.7), 0)
        return round(cms_e.min(), 2), round(cms_e.mean(), 2), round(cms_e.max(), 2)

    def get_ms_energy_stats(self) -> Tuple[float, float, float]:
        """Return the minimum, average, and maximum energies of the conf ms in .all_ms."""
        ms_e = self.all_ms[:, -2]
        return round(np.min(ms_e), 2), round(np.mean(ms_e), 2), round(np.max(ms_e), 2)

    def get_resoi_cms(self, resoi: Union[list, np.ndarray]) -> Union[np.ndarray, None]:
        """Obtain a filtered .all_cms array for residues of interest if any."
        """
        # changed to filter for all free ionizable residues if no user list
        if not self.cms_resids:
            # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
            resoi_info_idx = np.array(self.conf_info[(self.conf_info[:,-3]==1)
                                                     & (self.conf_info[:,3]==1)][:,0],
                                      dtype="int")
        else:
            # get iconfs of res of interest into array: will be used to filter all_cms
            resoi_info_idx = np.array(self.conf_info[np.isin(self.conf_info[:,1], resoi)][:,0],
                                    dtype="int")

        return self.all_cms[resoi_info_idx]

    @staticmethod
    def filter_cms_E_within_bounds(top_cms: Union[list, np.ndarray],
                                   E_bounds: Tuple[float, float]) -> list:
        """
        Filter top_cms for cms with energies within E_bounds.
        """
        if E_bounds == (None, None):
            return top_cms

        # index of energy item:
        # 3 :: array from mc_load=="all"; 2 :: array from mc_load=="crg"
        E = 3 if len(top_cms[0]) == 6 else 2   
        filtered = []
        for i, ro in enumerate(top_cms[:, 0]):
            # ignore top cms out of bounds:
            if top_cms[i, E] < E_bounds[0] or top_cms[i, E] > E_bounds[1]:
                continue
            filtered.append(top_cms[i])

        return filtered

    def n_ms_to_ooc_pct(self, occ_pct: list=[50, 90]) -> str:
        """
        Return a string giving the number of (c)ms needed to reach an occ of 50%, 90%, etc.
        """
        if self.HDR.is_monte:
            p1 = occ_pct[0]/100
            p2 = occ_pct[1]/100
            # determine which topn data to return as per mc_load:
            which_top = {"conf":1, "crg":2, "all":3}
            process_top = which_top[self.mc_load]
            if process_top in (2, 3):
                to_pct1 = self.all_cms[np.less_equal(np.cumsum(self.all_cms[:, -1]/self.N_space), p1)]
                to_pct2 = self.all_cms[np.less_equal(np.cumsum(self.all_cms[:, -1]/self.N_space), p2)]
            else:
                to_pct1 = self.all_ms[np.less_equal(np.cumsum(self.all_ms[:, -1]/self.N_space), p1)]
                to_pct2 = self.all_ms[np.less_equal(np.cumsum(self.all_ms[:, -1]/self.N_space), p2)]

        return ("Number of microstates to occpuancy of:\n"
                f" {p1:.1%}: {to_pct1.shape[0]:,}\n"
                f" {p2:.1%}: {to_pct2.shape[0]:,}\n")

    def get_topN_data(self, N: int=None, min_occ: float = MIN_OCC,
                      all_ms_out: bool = False) -> Tuple[list, Union[dict, None]]:
        """
        Return a 2-tuple:
         - top_c|ms (list): Containing the first N most numerous c|ms array with occ >= min_occ;
         - top_ms (dict|None): Dict with key as the index shared by the .all_cms and .all_ms arrays
           (integer at position 0) when mc_load='all', else None;
           The dict values depend on argument 'all_ms_out': if False (default), they are the most
           numerous conformer ms for the given index, else they are all the associated ms for that index.

        Notes:
         - HIS have pseudo charges if MSout_np was instantiated using 'with_tautomers'=True,
           see MSout_np.get_conf_info for details.
         - The output will be ([], None) if all unique c|ms's occupancies are below the threshold.

        Call examples:
          1. Using defaults: dict top_ms values are a single related conformer ms:
            > top_cms, top_ms_dict = msout_np.get_topN_data()
          2. With 'all_ms_out' set to True: dict values are all related conf ms:
            > top_cms, top_ms_dict = msout_np.get_topN_data(all_ms_out=True)
        """
        print("Getting top N data.")
        # determine which topn data to return as per mc_load:
        which_top = {"conf":1, "crg":2, "all":3}
        process_top = which_top[self.mc_load]
        min_occ = float(min_occ)
        if N is None:  # for backward compatibility:
            N = N_TOP
        elif N == MAX_INT:  # output all
            N = self.N_ms_uniq if process_top == 1 else self.N_cms_uniq
        else:
            # recast input as they can be strings via the cli:
            N = int(N)

        if process_top in (2, 3):
            if N > self.N_cms_uniq:
                print(f"Requested topN ({N:,}) greater than available ({self.N_cms_uniq:,}):",
                       "processing all.")
                N = self.N_cms_uniq
        else:
            if N > self.N_ms_uniq:
                print(f"Requested topN ({N:,}) greater than available ({self.N_cms_uniq:,}):",
                       "processing all.")
                N = self.N_ms_uniq

        if process_top == 3:
            print(f"Processing unique cms & ms for requested top {N:,} at {min_occ = :.1%}")
            topN_cms_occ = self.uniq_cms[np.where(self.uniq_cms[:, -2] > min_occ)][:N]
            if len(topN_cms_occ):
                top_cms = topN_cms_occ.tolist()
                print(f"Number of top cms returned: {len(top_cms):,}")
                top_ms = {}
                for ro in topN_cms_occ[:,0]:
                    matched_ms = self.all_ms[np.where(self.all_ms[:, 0] == ro)]
                    n_matched = len(matched_ms)
                    if not n_matched:
                        sys.exit("ERROR - 'get_topN_data': No associated ms found!")

                    if all_ms_out:
                        top_ms[ro] = matched_ms
                    else:
                        if self.reduced_ms_rows:
                            top_ms[ro] = matched_ms[0]
                        elif n_matched > 1:
                            # get the most numerous:
                            top_ms[ro] = sorted(matched_ms,
                                                key=lambda x: x[-1], reverse=True)[0]
                        else:
                            top_ms[ro] = matched_ms[0]
                return top_cms, top_ms
            else:
                return [], None

        if process_top == 2:  # cms only
            print(f"Processing unique cms for requested top {N} at {min_occ = :.1%}")
            topN_cms_occ = self.uniq_cms[np.where(self.uniq_cms[:, -2] > min_occ)][:N]
            top_cms = []
            if len(topN_cms_occ):
                top_cms = topN_cms_occ.tolist()
                print(f"Number of top cms returned: {len(top_cms):,}")
            return top_cms, None

        if process_top == 1:  # ms only
            print(f"Processing unique ms for requested top {N} at {min_occ = :.1%}")
            topN_ms_occ = self.uniq_ms[np.where(self.uniq_ms[:, -2] > min_occ)][:N]
            top_ms = []
            if len(topN_ms_occ):
                top_ms = topN_ms_occ.tolist()
                print(f"Number of top ms returned: {len(top_ms):,}")
            return top_ms, None

    def top_cms_df(self, top_cms: list,
                   cms_wc_format: bool = False,
                   cms_wc_keep_E: bool = False) -> pd.DataFrame:
        """
        Arguments:
          - top_cms: List of top cms data
          - cms_wc_format: Set to True to get df formatted for crg ms analysis with
                           weighted correlation.
          - cms_wc_keep_E: Set to True to keep the energy column in cms_wc_format mode.
        """
        fixed_free_res = None
        n_ffres = 0
        if not cms_wc_format:
            fixed_free_res = self.get_fixed_res_of_interest_arr()
            n_ffres = len(fixed_free_res) 
            if not n_ffres:
                fixed_free_res = None

        data = []
        ix_state = 1 if self.mc_load == "all" else 0
        for it, itm in enumerate(top_cms):
            if ix_state == 1:
                # [ idx, list(state), totE, averE, occ, count ]
                fields = [itm[0]]   # the shared index
            else:
                fields = [it]  # current index

            state = itm[ix_state].copy()
            for i, s in enumerate(state):
                if self.cms_resids[i][:3] == "HIS":
                    if self.with_tautomers:
                        s = HIS0_tautomers[s]

                fields.extend([s])
            if fixed_free_res is not None:
                fields.extend(fixed_free_res[:,1])

            # sci notation for occ, e.g.: 1.23e+06
            if ix_state == 1:
                fields.extend([round(itm[3], 2), sum(state) + self.CI.background_crg, itm[5],
                               round(itm[4], 6)])
            else:
                fields.extend([round(itm[2], 2), sum(state) + self.CI.background_crg, itm[4],
                               round(itm[3], 6)])
            data.append(fields)

        if not cms_wc_format:
            res_cols = self.cms_resids

            if fixed_free_res is not None:
                # add fixed ionizable res
                res_cols = res_cols + fixed_free_res[:,0].tolist() 
                info_dat = ["tmp"] + ["free"] * self.n_resids + ["fixed"] * n_ffres + ["totals"] * 4
            else:
                # order as in data
                info_dat = ["tmp"] + ["free"] * self.n_resids + ["totals"] * 4

            # always remove trailing underscore
            res_cols = [c.rstrip("_") for c in res_cols]
            cols = ["idx"] + res_cols +  ["E", "sum_crg", "size", "occ"]
       
            df = pd.DataFrame(data, columns=cols)
            df["res"] = df.index + 1
            df.set_index("res", inplace=True)
            df = df.T
            df.columns.name = ""
            df.reset_index(inplace=True)
            df.rename(columns={"index":"residues"}, inplace=True)
            df["info"] = info_dat
            
            return df
        
        # Format as per original specs in Raihan's microstate_analysis_code:
        cols = ["Order"] + self.cms_resids + ["E", "SumCharge", "Count", "Occupancy"]
        df = pd.DataFrame(data, columns=cols)
        df["Order"] = df.index + 1
        if not cms_wc_keep_E:
            df = df.drop("E", axis=1)
            # move SumCharge to end:
            new_cols = df.columns[:-3].tolist() + ["Count", "Occupancy", "SumCharge"]
        else:
            new_cols = df.columns[:-4].tolist() + ["E", "Count", "Occupancy", "SumCharge"]

        return df[new_cols].set_index("Order")

    def get_sampled_ms(self, size: int, ms_kind: str, seed: int = None) -> Union[List, None]:
        """
        Obtain a random sample of microstates using the ms count probability. 
        The random number generator is fixed with 'seed' if not None.

        Args:
            size (int): Sample size
            ms_kind (str): One of 'ms' for conformer microstates or 'cms' for charge microstates.
            seed (int, None): Seed for fixing the random number generator.

        Returns:
            A random sample of conformer or charge microstates in a list ([[selected index, selected ms]..]),
            or None if MSout_np was intantiated with an incompatible 'mc_load' mode.
        """
        ms_kind = ms_kind.lower()
        if ms_kind not in ("ms", "cms"):
            print("Argument 'kind' must be one of 'ms' or 'cms'. Returning None.")
            return None

        if ms_kind == "ms":
            if self.mc_load == "crg":
                print("Not applicable: Request is for sampling conformer microstates, but", 
                      "MSout_np was instantiated with mc_load='crg'; 'mc_load' must be 'conf' or 'all'.")
                return None
        else:
            if self.mc_load == "conf":
                print("Not applicable: Request is for sampling charge microstates, but",
                      "MSout_np was instantiated with mc_load='conf'; 'mc_load' must be 'crg' or 'all'.")
                return None
 
        if size <= 0:
            print("ERROR: Requested sample size must be > 0: Returning None.")
            return None

        if size > self.N_space:
            print(f"Requested sample size ({size :,}) > available data: reset to {self.N_space:,}")
            size = self.N_space

        rng = np.random.default_rng(seed=seed)
        if ms_kind == "ms":
            # Occupancies as probability of selection (sum to 1)
            probs = np.array(self.all_ms[:,-1] / self.N_space, dtype=float)
            indices = rng.choice(len(self.all_ms), size=size, p=probs)
            return [[idx, self.all_ms[idx]] for idx in indices]

        probs = np.array(self.all_cms[:,-1] / self.N_space, dtype=float)
        indices = rng.choice(len(self.all_cms), size=size, p=probs)
        return [[idx, self.all_cms[idx]] for idx in indices]
   
    def _sum_crg_state_vec(self) -> np.ndarray:
        """Sums a numpy array of cms state lists at each index.
        Returns:
            A numpy array containing the sums at each index.
        """
        if self.mc_load == "conf":
            raise TypeError(("Not applicable: Method `sum_crg_state_vec` operates on 'all_cms' "
                              "attribute, which is not loaded when 'mc_load' is set to 'conf'."))

        ix_state = 1 if self.mc_load == "all" else 0
        result = np.zeros(len(self.all_cms[0,ix_state]))
        for cms in self.all_cms:
            cms_cnt = cms[-1]
            for i, val in enumerate(cms[ix_state]):
                result[i] += val * cms_cnt

        return result

    def _free_res_aver_crg_df_from_all_ms(self) -> pd.DataFrame:
        """Return a pd.DataFrame with average charge of all free residues calculated
        from 'all_ms' attribute (conformer microstates).
        """
        charges_total = defaultdict(float)
        ix_state = 1 if self.mc_load == "all" else 0

        for _, ms in enumerate(self.all_ms):
            for _, (resid, crg) in enumerate(self.conf_info[ms[ix_state]][:, [1, -1]]):
                charges_total[resid] += crg * ms[-1]  # crg * count

        return pd.DataFrame([(k, round(charges_total[k]/self.N_space))
                                for k in charges_total],
                                columns=["Residue", "crg"])

    def _free_res_aver_crg_df_from_all_cms(self) -> pd.DataFrame:
        """Return a pd.DataFrame with average charge of all free residues calculated
        from 'all_cms' attribute (protonation microstates).
        """
        crg_aver = np.round(self._sum_crg_state_vec()/self.N_space, 0)
        return pd.DataFrame(list(zip(self.cms_resids, crg_aver)),
                            columns=["Residue", "crg"])

    def get_free_res_aver_crg_df(self) -> pd.DataFrame:
        """Convert the conformer ms state in MSout_np.all_ms array to crg
        and return the average charge of all free residue into a pandas.DataFrame.
        """
        # use the smaller input array if possible:
        if self.mc_load == "conf":
            # use the largest and only available array
            return self._free_res_aver_crg_df_from_all_ms()

        if self.mc_load == "crg" and not self.with_tautomers:
            # use all_cms
            return self._free_res_aver_crg_df_from_all_cms()

        if self.with_tautomers:
            # the charge ms may have 'pseudo charges' on HIS: use the
            # conformer ms data in attribute 'all_ms' if possible
            if self.mc_load == "all":
                return self._free_res_aver_crg_df_from_all_ms()
            else:
                raise TypeError(("Not applicable: Cannot calculate average sum charge if 'with_tautomer' is True "
                                 "and mc_load='crg'.\nEither instantiate MSout_np with mc_load='all' "
                                 "or mc_load='crg' and with_tautomers=False."))
        else:
            return self._free_res_aver_crg_df_from_all_cms()

    def __str__(self):
        out = (f"\nConformers: {self.CI.n_confs:,}\n"
                f"Free residues: {len(self.HDR.free_residues):,}\n"
                f"Fixed residues: {len(self.HDR.fixed_iconfs):,}\n"
                f"Background charge: {self.CI.background_crg:.0f}\n"
                )
        if self.HDR.is_monte:
            out = out + f"State space: {self.N_space:,}\n"
            out = out + self.n_ms_to_ooc_pct()

        return out
