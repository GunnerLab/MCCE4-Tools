#!/usr/bin/env python

"""
Module: msout_np.py

Module for MSout_np class, the faster ms_out/ file loader, that loads
different microstates depending on the value of 'mc_load', the 'loading
mode' argument, which can be:
 - 'conf' :: only conformer microstates in MSout_np.all_ms
 - 'crg' :: only protonation microstates in MSout_np.all_cms
 - 'all :: both protonation and conformer microstates in both arrays.

Note:
* Naming convention:
- 'charge ms', 'crg ms' and 'cms' are shortcuts for protonation microstate.
- 'ms' is a shortcut for conformational microstate.
- 'msout file' refers to a .txt file that starts with 'pH<ph>' in the ms_out
  subfolder of an mcce run.
"""

from collections import defaultdict
from itertools import islice
import sys
import time
from typing import List, Tuple, Union

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Oops! Forgot to activate an appropriate environment?\n{e}")
    sys.exit(1)

from mcce4_tools.constants import IONIZABLE_RES as IONIZABLES, ROOMT
from mcce4_tools.io_utils import reader_gen, show_elapsed_time


MIN_OCC = 0.0  # occ threshold
N_top = 5      # top N default
HIS0_tautomers = {0: "NE2", 1: "ND1", 2: 1}


# TODO: Redo estimate
def topN_loadtime_estimate(n_freeres: int) -> str:
    """Returns the time estimate given the number of free residues
    for reading the mc lines to getting the topN ms in a formattted
    string showing seconds and minutes.
    """
    # fit of 5 runs:
    # -2 offset: improvements since fit
    return round(-14.9897855 -2 + 0.451883977*n_freeres + 6.25518650e-04*n_freeres**2)


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
                 loadtime_estimate: bool = False,
                 ):

        self.validate_kwargs(mc_load, res_kinds, with_tautomers)

        # attributes populated by self.load_header:
        self.T: float = ROOMT
        self.pH: float = 7.0
        self.Eh: float = 0.0
        self.fixed_iconfs: list = []
        self.free_residues: list = []  # needed in get_conf_info
        self.iconf2ires: dict = {}

        # attributes populated by self.get_conf_info:
        self.N_confs: int = None
        self.N_resid: int = None
        # conformer lookup table using head3 data:
        self.conf_info: np.ndarray = None
        self.conf_ids: np.ndarray = None
        # list of resids defining a cms:
        self.cms_resids: list = None
        # sum crg of fixed res:
        self.background_crg: int = None

        # attributes populated by the 'load' functions:
        self.N_space: int = None     # size of conformal state space
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

        # load msout file header data:
        self.msout_file = msout_file
        self.load_header()

        # create the head3 lookup array:
        self.conf_info, self.cms_resids, self.conf_ids = self.get_conf_info(head3_file)
        self.N_confs = len(self.conf_ids)
        self.N_resid = len(self.cms_resids)
        # fields :: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        # sumcrg for is_fixed:
        self.background_crg = self.conf_info[np.where(self.conf_info[:, -4]), -1].sum()

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
            if loadtime_estimate:
                yt = topN_loadtime_estimate(len(self.free_residues))
                print(f"\nESTIMATED TIME to topN: {yt:,.2f} s ({yt/60:,.2f} min).\n")
            start_t = time.time()
            self.load_all()
            show_elapsed_time(start_t, info="Loading msout for ms & cms")
        else:
            print("No processing function associated with:", self.mc_load)

        return

    def validate_kwargs(self, mc_load: str, res_kinds: list, with_tautomers: bool):
        # valid loading modes:
        loading_modes = ["conf", "crg", "all"]
        if mc_load.lower() not in loading_modes:
            msg = ("Argument mc_load must be one of "
                   f"{loading_modes} "
                   "to load either conformer or charge microstates, or both.")
            sys.exit(msg)

        self.mc_load = mc_load.lower()
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
                # likely not applicable: not implemented
                print("WARNING: Residue selection when loading conformer microstates",
                      "is not implemented: res_kinds reset to None.")
                self.res_kinds = None
            else:
                self.res_kinds = res_kinds

        return
 
    def load_header(self):
        """Process an unadulterated 'msout file' header rows to populate
        these attributes: T, pH, Eh, fixed_iconfs, free_residues, iconf2ires.
        """
        with open(self.msout_file) as fh:
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
                if key.strip() != "METHOD" or value.strip() != "MONTERUNS":
                    print("This file %s is not a valid microstate file" % self.msout_file)
                    sys.exit(-1)
            if i == 4:
                _, iconfs = line.split(":")
                self.fixed_iconfs = [int(i) for i in iconfs.split()]
            if i == 6:  # free residues
                _, residues_str = line.split(":")
                residues = residues_str.split(";")
                for f in residues:
                    if f.strip():
                        self.free_residues.append([int(i) for i in f.split()])
                for idx, lst in enumerate(self.free_residues):
                    for iconf in lst:
                        self.iconf2ires[iconf] = idx
        return

    def get_conf_info(self, h3_fp: str) -> Tuple[np.ndarray, list, np.ndarray]:
        """Output these variables:
         - conf_info (np.ndarray): a lookup 'table' for iconfs, resids, and charges
           initially from head3.lst;
         - cms_resids (list): list of unique, free & ionizable resids in a MCCE simulation;
         - conf_ids (np.ndarray): array of iconfs, confids;

        Note:
        Final extended format of conf_info:
          [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg];
        Field 'is_fixed' is needed for proper accounting of net charge.
        """
        with open(h3_fp) as h3:
            lines = h3.readlines()[1:]

        conf_info = []
        conf_vec = []
        for line in lines:
            # ignored columns: FL, Occ & fields beyond crg
            iConf, confid, _, _, Crg, *_ = line.split()
            iconf = int(iConf) - 1  # as python index
            kind = confid[:3]
            resid = kind + confid[5:11]
            crg = int(float(Crg))
            if self.with_tautomers:
                if kind == "HIS":
                    # reset crg to pseudo crg:
                    # 0 :: HIS01->" NE2"; 1 :: HIS02->" ND1"; 2 :: HIS+1
                    crg = int(confid[4]) - 1 if confid[3] == "0" else 2

            is_ioniz = int(resid[:3] in IONIZABLES)
            in_kinds = 1
            if self.res_kinds is not None and len(self.res_kinds):
                in_kinds = int(kind in self.res_kinds)
            is_fixed = int(iconf in self.fixed_iconfs)
            # conf_info last 3 :: [..., is_free, resix, crg]
            conf_info.append([iconf, resid, in_kinds, is_ioniz, is_fixed, 0, 0, crg])
            conf_vec.append([iconf, confid])
        # temp list structure is now set & has h3 info; cast to np.ndarray:
        conf_info = np.array(conf_info, dtype=object)
        conf_ids = np.array(conf_vec, dtype=object)

        # update conf_info: use free iconfs from free_residues to
        # populate the 'is_free' field.
        free_iconfs = [ic for free in self.free_residues for ic in free]
        conf_info[free_iconfs, -3] = 1

        # get cms unique resids list via filtering conf_info for valid confs for
        # protonation state vec: ionizable & free & in user list if given.
        # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        sum_conditions = conf_info[:, 3] + conf_info[:, -3]   # ionizable & free
        sum_tot = 2
        if self.res_kinds:
            # in_kinds & is_ioniz & is_free
            sum_conditions = conf_info[:, 2] + sum_conditions
            sum_tot = 3
        # Note: dict in use instead of a set (or np.unique) to preserve the order:
        d = defaultdict(int)
        for r in conf_info[np.where(sum_conditions == sum_tot)][:, 1]:
            d[r] += 1
        # uniq resids to list:
        cms_resids = list(d.keys())

        # create mapping from confs space to protonation resids space:
        # update conf_info resix field with the index from cms_resids list:
        # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]

        # Getting resix w/o checking again for is_free was not sufficient,
        # e.g. GLUA0007; iconfs 12, 13, 14 needed resix = -1, since not free:
        # [12 'GLUA0007_' 1 0 0 2 0]
        # [13 'GLUA0007_' 1 0 0 2 0]
        # [14 'GLUA0007_' 1 0 0 2 0]
        # [15 'GLUA0007_' 1 0 1 2 0]
        # [16 'GLUA0007_' 1 0 1 2 -1]
        for i, (_, resid, _, _, _, is_free, *_) in enumerate(conf_info):
            try:
                resix = cms_resids.index(resid)
                if not is_free:
                    resix = -1
            except ValueError:
                # put sentinel flag for unmatched res:
                resix = -1
            conf_info[i][-2] = resix

        print("\nHead3 lookup array 'conf_info'\n\tfields ::",
              "iconf:0, resid:1, in_kinds:2, is_ioniz:3,",
              "is_fixed:4, is_free:5, resix:6, crg:7\n")
        return conf_info, cms_resids, conf_ids

    def get_ter_dict(self) -> dict:
        """Return a dict for res with multiple entries, such as
        terminal residues.
        Sample output, dict: {'A0001': ['NTR', 'LYS'],
                              'A0129': ['LEU', 'CTR']}
        """
        ter_dict = defaultdict(list)
        for confid in self.conf_ids[:,1]:
            res = confid[:3]
            res_id = confid[5:].split("_")[0]
            # order needed, can't use set():
            if res not in ter_dict[res_id]:
                ter_dict[res_id].append(res)

        return dict((k, v) for k, v in ter_dict.items() if len(v) > 1)

    def load_conf(self):
        """Process the 'msout file' mc lines to populate a list of
        [state, state.E, count] items, where state is a list of conformal
        microstates.
        This list is then assignedd to MCout.all_ms as a numpy.array.
        """
        # print("Loading function: load_conf")
        found_mc = False
        newmc = False
        ms_vec = []  # list to hold conf ms info

        msout_data = reader_gen(self.msout_file)
        for lx, line in enumerate(msout_data):
            if lx < 9:
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
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                    state_e = 0.0
                    current_state = [int(i) for i in line.split(":")[1].split()]
                    newmc = False
                    continue

                if found_mc:
                    fields = line.split(",")
                    if len(fields) >= 3:
                        state_e = float(fields[0])
                        count = int(fields[1])
                        flipped = [int(c) for c in fields[2].split()]
                        for ic in flipped:
                            ir = self.iconf2ires[ic]
                            current_state[ir] = ic

                        ms_vec.append([list(current_state), state_e, count])
                    else:
                        continue

        if ms_vec:
            self.all_ms = np.array(ms_vec, dtype=object)
            self.N_mc_lines = len(self.all_ms)
            print(f"Accepted states lines: {self.N_mc_lines:,}\n")
            self.N_space = self.all_ms[:, -1].sum()
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

        msout_data = reader_gen(self.msout_file)
        for lx, line in enumerate(msout_data):
            if lx < 7:
                continue
            line = line.strip()
            if not line or line[0] == "#":
                continue
            else:
                # find the next MC record, e.g. MC:4
                if line.startswith("MC:"):
                    found_mc = True
                    newmc = True
                    continue

                if newmc:
                    ro += 1  # will be 0 at "MC:0" + 1 line
                    # line with candidate state for MC sampling, e.g.:
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                    state_e = 0.0
                    current_state = [int(i) for i in line.split(":")[1].split()]

                    # cms_vec :: [state, totE, averE, count]
                    cms_vec.append([[0] * len(self.cms_resids), 0, 0, 0])
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
                    if len(fields) >= 3:
                        state_e = float(fields[0])
                        count = int(fields[1])
                        flipped = [int(c) for c in fields[2].split()]
                        for ic in flipped:
                            ir = self.iconf2ires[ic]
                            current_state[ir] = ic

                        # flipped iconfs from non-ionizable or fixed res
                        # => same protonation state: increment totE & count;
                        # Note: -1 is a sentinel index for this situation.
                        update_cms = np.all(self.conf_info[flipped, -2] == -1)
                        if update_cms:
                            # cms_vec ::  [state, totE, averE, count]
                            cms_vec[ro][1] += state_e * count
                            cms_vec[ro][3] += count
                            cms_vec[ro][2] = cms_vec[ro][1] / cms_vec[ro][3]
                        else:
                            ro += 1
                            cms_vec.append([[0] * len(self.cms_resids), state_e * count, state_e, count])
                            curr_info = self.conf_info[current_state]
                            upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                            for u in upd:
                                cms_vec[ro][0][u[0]] = u[1]
                    else:
                        continue
        if cms_vec:
            self.all_cms = np.array(cms_vec, dtype=object)
            self.N_space = self.all_cms[:, -1].sum()
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
        found_mc = False
        newmc = False
        ro = -1  # list item accessor
        # lists to hold conf and crg ms info; they can be related by their common index;
        cms_vec = []
        ms_vec = []

        msout_data = reader_gen(self.msout_file)
        for lx, line in enumerate(msout_data):
            if lx < 9:
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
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                    state_e = 0.0
                    ro += 1  # will be 0 at "MC:0" + 1 line
                    current_state = [int(i) for i in line.split(":")[1].split()]

                    # cms_vec ::  [idx, state, totE, averE, count]
                    cms_vec.append([ro, [0] * len(self.cms_resids), 0, 0, 0])
                    # update cms_vec state:
                    curr_info = self.conf_info[current_state]
                    upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                    for u in upd:
                        cms_vec[ro][1][u[0]] = u[1]
                    newmc = False
                    continue

                if found_mc:
                    fields = line.split(",")
                    if len(fields) >= 3:
                        state_e = float(fields[0])
                        count = int(fields[1])
                        flipped = [int(c) for c in fields[2].split()]
                        for ic in flipped:
                            ir = self.iconf2ires[ic]
                            current_state[ir] = ic

                        ms_vec.append([ro, list(current_state), state_e, count])

                        # if the flipped iconfs are from non-ionizable or fixed res,
                        # the protonation state is the same: increment count & E;
                        # Note: -1 is a sentinel index for this situation.
                        update_cms = np.all(self.conf_info[flipped, -2] == -1)
                        if update_cms:
                            # cms_vec ::  [idx, state, totE, averE, count]
                            cms_vec[ro][2] += state_e * count
                            cms_vec[ro][4] += count
                            cms_vec[ro][3] = cms_vec[ro][2] / cms_vec[ro][4]
                        else:
                            ro += 1  # new cms
                            cms_vec.append([ro, [0] * len(self.cms_resids), state_e * count, state_e, count])

                            curr_info = self.conf_info[current_state]
                            upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                            for u in upd:
                                cms_vec[ro][1][u[0]] = u[1]

        if ms_vec:
            self.all_ms = np.array(ms_vec, dtype=object)
            self.N_mc_lines = len(self.all_ms)
            print(f"Accepted states lines: {self.N_mc_lines:,}\n")
            self.N_space = self.all_ms[:, -1].sum()
        else:
            return ValueError("Something went wrong in loading msout file: 'ms_vec' is empty!")
        
        if cms_vec:
            self.all_cms = np.array(cms_vec, dtype=object)
            self.N_cms = len(self.all_cms)
            print(f"Protonation microstates: {self.N_cms:,}\n")
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

    def _get_uniq_cms(self):
        """Assign unique crg ms info (state, totE, averE, count) to self.uniq_cms;
        Assign count of unique cms to self.N_cms_uniq.
        """
        subtot_d = {}
        # crg_e ::  [state, totE, averE, count]
        for ix, itm in enumerate(self.all_cms):
            key = tuple(itm[0])
            if key in subtot_d:
                subtot_d[key][1] += itm[1]
                subtot_d[key][3] += itm[3]
                subtot_d[key][2] = subtot_d[key][1] / subtot_d[key][3]
            else:
                subtot_d[key] = itm.copy()

        self.N_cms_uniq = len(subtot_d)
        # add occ, sort by count & assign to self.uniq_cms as np.array:
        # crg_e ::  [state, totE, averE, occ, count]
        self.uniq_cms = np.array(
            sorted(
                [
                    [list(k), subtot_d[k][1], subtot_d[k][2], subtot_d[k][3] / self.N_space, subtot_d[k][3]]
                    for k in subtot_d
                ],
                key=lambda x: x[-1],
                reverse=True,
            ),
            dtype=object,
        )

        return

    def _get_uniq_conf(self):
        """Assign unique conf ms info (state, stateE, occ, count) to self.uniq_ms;
        Assign count of unique ms to self.N_ms_uniq.
        """
        if self.mc_load != "conf":
            print("WARNING: Redirecting to 'get_uniq_all_ms' function as per 'mc_load'.")
            self.get_uniq_all_ms()
            return
        # ms in ::  [state, state.e, count]
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
            [list(k), subtot_d[k][1], subtot_d[k][-1] / self.N_space, subtot_d[k][-1]] for k in subtot_d
        ]
        self.uniq_ms = np.array(sorted(mslist, key=lambda x: x[-1], reverse=True), dtype=object)
        return

    def _get_uniq_all_cms(self):
        """Get the unique charge ms array when the `all_cms` array
        was produced together with the `all_ms` array, i.e. mc_load='all'.
        In this case, each of their items starts with an index,
        which can be used to match conf ms to each unique cms.
        """
        subtot_d = {}
        # vec :: [idx, state, totE, averE, count]
        for ix, itm in enumerate(self.all_cms):
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

        return

    def _get_uniq_all_ms(self):
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
        free_residues_df = pd.DataFrame(self.conf_info[np.where(self.conf_info[:, -3]), 1][0],
                                        columns=["Residue"])
        free_residues_df.drop_duplicates(inplace=True)
        return free_residues_df

    def get_fixed_residues_arr(self) -> np.ndarray:
        """Extract resid, crg for is_ioniz & is_fixed from lookup array."""
        # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        is_ioniz_fixed = self.conf_info[np.where(np.logical_and(self.conf_info[:, 3],
                                                                self.conf_info[:, 4]))]
        return is_ioniz_fixed[:, [1, -1]]

    def get_fixed_residues_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_fixed_residues_arr(), columns=["Residue", "crg"])

    def get_fixed_res_of_interest_arr(self) -> np.ndarray:
        """Extract resid, crg for in_kinds & is_fixed from lookup array."""
        # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        return self.conf_info[np.where(np.logical_and(self.conf_info[:, 2],
                                                      self.conf_info[:, 4]))][:, [1, -1]]

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

    @staticmethod
    def filter_cms_E_within_bounds(top_cms: Union[list, np.ndarray],
                                   E_bounds: Tuple[float, float]) -> list:
        """
        Filter top_cms for cms with energies within E_bounds.
        """
        if E_bounds == (None, None):
            return top_cms

        # index of energy data:
        E = 3 if len(top_cms[0]) == 6 else 2   # 2 :: array from mc_load=="crg"
        filtered = []
        for i, ro in enumerate(top_cms[:, 0]):
            # ignore top cms out of bounds:
            if top_cms[i, E] < E_bounds[0] or top_cms[i, E] > E_bounds[1]:
                continue
            filtered.append(top_cms[i])

        return filtered

    def get_topN_data(self, N: int = 5, min_occ: float = MIN_OCC,
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
        # determine which topn data to return as per mc_load:
        which_top = {"conf":1, "crg":2, "all":3}
        process_top = which_top[self.mc_load]
        
        # recast input as they can be strings via the cli:
        N = int(N)
        min_occ = float(min_occ)

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
                    if all_ms_out:
                        top_ms[ro] = matched_ms
                    else:
                        if len(matched_ms) > 1:
                            # get the most numerous:
                            top_ms[ro] = sorted(matched_ms, key=lambda x: x[-1], reverse=True)[0]
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
                print(f"Number of top ms returned: {len(top_cms):,}")
            return top_ms, None

    def top_cms_df(self, top_cms: list,
                   cms_wc_format: bool = False) -> pd.DataFrame:
        """
        Arguments:
          - output_tauto: Set to False to keep the charge instead of the string tautomer.
          - cms_wc_format: Set to True to get df formatted for crg ms analysis with
                           weighted correlation.
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
            if ix_state == 1:
                fields.extend([round(itm[3], 2), sum(state) + self.background_crg, itm[5], round(itm[4], 4)])
            else:
                fields.extend([round(itm[2], 2), sum(state) + self.background_crg, itm[4], round(itm[3], 4)])
            data.append(fields)

        if not cms_wc_format:
            res_cols = self.cms_resids

            if fixed_free_res is not None:
                # add fixed ionizable res
                res_cols = res_cols + fixed_free_res[:,0].tolist() 
                info_dat = ["tmp"] + ["free"] * len(self.cms_resids) + ["fixed"] * n_ffres + ["totals"] * 4
            else:
                # order as in data
                info_dat = ["tmp"] + ["free"] * len(self.cms_resids) + ["totals"] * 4

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
        df.drop("E", axis=1, inplace=True)
        df["Order"] = df.index + 1

        # move SumCharge to end:
        new_cols = df.columns[:-3].tolist() + ["Count", "Occupancy", "SumCharge"]

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
        return (f"Conformers: {self.N_confs:,}\n"
                f"Conformational state space: {self.N_space:,}\n"
                f"Free residues: {len(self.free_residues):,}\n"
                f"Fixed residues: {len(self.fixed_iconfs):,}\n"
                f"Background charge: {self.background_crg:.0f}\n"
                )
