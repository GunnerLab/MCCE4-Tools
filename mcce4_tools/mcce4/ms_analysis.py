#!/usr/bin/env python3

"""
Module: ms_analysis.py

Provides class and functions to load the 'msout file' and work with
conformer and charge microstates.

Classes:
  - Conformer
  - Microstate
  - Charge_Microstate
  - Charge_Microstates
  - MSout

"""
from collections import defaultdict
from dataclasses import dataclass
import logging
import operator
from pathlib import Path
import sys
from typing import ByteString, Dict, List, Tuple, Union
import zlib
import numpy as np
import pandas as pd
from mcce4.constants import Kcal2kT


# TODO: Fix functions so that they work via import;


logger = logging.getLogger()
logger.setLevel(logging.INFO)


MIN_OCC = 0.0


@dataclass(init=True, repr=False, order=True, slots=True)
class Conformer:
    """Minimal Conformer class for use in microstate analysis.
    Attributes: iconf, confid, ires, resid, crg.
    See also:
      - mcce4.pdbio.Conformer class: full class for writing head3.lst.
    """
    iconf: int
    confid: str
    ires: int
    resid: str
    crg: float
    occ: float

    def __init__(self):
        self.iconf = 0
        self.confid = ""
        self.ires = 0
        self.resid = ""
        self.crg = 0.0
        self.occ = 0.0

    def load_from_head3lst(self, line):
        fields = line.split()
        self.iconf = int(fields[0]) - 1
        self.confid = fields[1]
        self.resid = self.confid[:3] + self.confid[5:11]
        self.crg = float(fields[4])
        self.occ = float(fields[3])

    def __str__(self):
        return f"{self.confid} ({self.iconf}), {self.resid} ({self.ires}), {self.crg:.2f}, {self.occ:.2f}"


def read_conformers(head3_path: str = "head3.lst") -> list:
    """Load conformerrs from given head3.lst path;
    Uses ./head3.lst by default; returns empty list if file not found.
    """
    if not Path(head3_path).exists():
        return []

    with open(head3_path) as h3:
        lines = h3.readlines()[1:]

    conformers = []
    for line in lines:
        conf = Conformer()
        conf.load_from_head3lst(line)
        conformers.append(conf)

    return conformers


class Conformers:
    """Collection of Conformer objects.
    Has methods and attributes pertaining to that collection.
    """

    def __init__(self, head3_path: str):
        """Given the path to 'head3.lst', process it to populate these attributes:
        - conformers (list): List of Conformer objects
        - N (int): size of conformers list
        """
        self.h3_fp = Path(head3_path)
        self.conformers = None
        self.N = None
        self.load_data()

    def load_data(self) -> list:
        """Populate the class attributes.
        """
        if not self.h3_fp.exists():
            # print(f"File not found: {self.h3_fp}")
            return
        with open(self.h3_fp) as fp:
            lines = fp.readlines()
        lines.pop(0)
        self.conformers = []
        for line in lines:
            conf = Conformer()
            conf.load_from_head3lst(line)
            self.conformers.append(conf)
        self.N = len(self.conformers)
        return

    def get_fixed_resid2crg_dict(self, fixed_iconfs: list) -> dict:
        dd = defaultdict(float)
        for conf in self.conformers:
            if conf.iconf in fixed_iconfs:
                dd[conf.resid] = conf.crg
        return dd


try:
    # conformers will be an empty list if module is imported
    # or called outside of a MCCE output folder
    # or if head3.lst is missing or corrupted.
    Confs = Conformers("head3.lst")
    conformers = Confs.conformers
except FileNotFoundError:
    conformers = []


@dataclass(init=True, repr=True, order=True, slots=True)
class Microstate:
    """Sortable class for mcce conformer microstates.
    """
    state: list
    E: float
    count: int

    def __str__(self):
        return f"Microstate(\n\tcount={self.count:,},\n\tE={self.E:,},\n\tstate={self.state}\n)"


@dataclass(init=True, repr=False, order=True, slots=True)
class Charge_Microstate:
    """
    Sortable class for charge microstates.
    Notes:
      For usage symmetry with Microstate class, the energy accessor E exists,
      but here: Charge_Microstate.E == Charge_Microstate.average_E. Thus, accessing
      E via Charge_Microstate.E, returns the average energy of a charge microstate.
      The Charge_Microstate.crg_stateid is implemented as compressed bytes as in the
      ms_analysis.py Microstate class in Junjun Mao's demo, but the 'key' that
      is encoded is a 2-tuple: (resid, crg) to facilitate further processing.
    """
    crg_stateid: ByteString
    count: int
    average_E: float
    total_E: float
    E: float  # alias for average E

    def __init__(self, crg_state: list, total_E: float, count: int):
        # crg_state is a list of (resid, crg) tuples:
        self.crg_stateid = zlib.compress(" ".join(x for x in crg_state).encode())
        self.count = count
        self.average_E = self.E = 0
        self.total_E = total_E

    def state(self):
        # recover (resid, crg) tuple:
        return [i for i in zlib.decompress(self.crg_stateid).decode().split()]

    def split_key(self) -> list:
        res_states = [tuple(itm.split("|")) for itm in self.state() ]
        return res_states
   
    def crg(self):
        """Return the sum charge of charges in the state key."""
        return sum(int(split_key[1]) for split_key in self.split_key())
    
    def __str__(self):
        return (f"\nCharge_Microstate(\n\tcount = {self.count:,}"
                f"\tE = {self.average_E:,.2f}\tcrg = {self.crg}\n\tstate = {self.state()}\n"
        )


def reader_gen(fpath: Path):
    """
    Generator function yielding a file line.
    """
    with open(fpath) as fh:
        for line in fh:
            yield line


def ms_counts(microstates: Union[dict, list]) -> int:
    """Sum the microstates count attribute."""
    if isinstance(microstates, dict):
        return sum(ms.count for ms in microstates.values())
    else:
        return sum(ms.count for ms in microstates)


@dataclass(init=False, repr=False, slots=True)
class MSout:
    # Define attribute types (initial values are set in __init__)
    T: float
    pH: float
    Eh: float
    fixed_iconfs: List
    # list of free residues per their list of conformer indices, iconf:
    free_residues: List
    # mapping from conformer index to free residue index:
    iconf2ires: Dict[int, int]
    microstates: Dict[str, Microstate]
    # attibutes that depend on the microstates collection:
    N_ms: int
    N_uniq: int
    lowest_E: float
    average_E: float
    highest_E: float

    def __init__(self, fname):
        self.T = 298.15
        self.pH = 7.0
        self.Eh = 0.0
        self.N_ms = 0
        self.N_uniq = 0
        self.lowest_E = 0.0
        self.average_E = 0.0
        self.highest_E = 0.0
        self.fixed_iconfs = []
        self.free_residues = []
        self.iconf2ires = {}
        self.microstates = {}

        self.load_msout(fname)

        self.N_uniq = len(self.microstates)
        # find N_ms, lowest, highest, averge E
        E_sum = 0.0
        msvals = self.microstates.values()
        # initialize with actual, first values:
        ms = next(iter(msvals))
        lowest_E = ms.E
        highest_E = ms.E
        for ms in msvals:
            self.N_ms += ms.count
            E_sum += ms.E * ms.count
            if ms.E < lowest_E:
                lowest_E = ms.E
            elif ms.E > highest_E:
                highest_E = ms.E
        # set attributes:
        self.lowest_E = lowest_E
        self.average_E = E_sum / self.N_ms
        self.highest_E = highest_E

        return

    def load_msout(self, fname):
        """Process the 'msout file' to populate the class attributes.
        """
        found_mc = False
        newmc = False

        msout_data = reader_gen(fname)
        for i, line in enumerate(msout_data, start=1):
            line = line.strip()
            if not line or line[0] == "#":
                continue

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
                    print("This file %s is not a valid microstate file" % fname)
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
            else:
                # find the next MC record
                if line.startswith("MC:"):
                    found_mc = True
                    newmc = True
                    continue

                if newmc:
                    f1, f2 = line.split(":")
                    current_state = [int(c) for c in f2.split()]
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

                        ms = Microstate(list(current_state), state_e, count)
                        key = ",".join(str(i) for i in ms.state)
                        if key in self.microstates:
                            self.microstates[key].count += ms.count
                        else:
                            self.microstates[key] = ms
        return

    def get_fixed_res_crg(self, conformers: list) -> float:
        """Return the sum charge contributed by fixed conformers."""
        return sum([conf.crg for conf in conformers if conf.iconf in self.fixed_iconfs])

    def get_sampled_ms0(
        self,
        size: int,
        kind: str = "random",
        seed: Union[None, int] = None,
    ) -> list:
        """
        Implement a sampling of MSout.microstates depending on `kind`.
        Args:
            size (int): sample size
            kind (str, 'random'): Sampling kind: one of ['deterministic', 'random'].
                 If 'deterministic', the microstates in ms_list are sampled at regular intervals
                 otherwise, the sampling is random. Case insensitive.
            seed (int, None): For testing purposes, fixes random sampling.
        Returns:
            A list of lists: [[selection index, selected microstate], ...]
        """

        if not len(self.microstates):
            print("The microstates dict is empty.")
            return []

        kind = kind.lower()
        if kind not in ["deterministic", "random"]:
            raise ValueError(f"Values for `kind` are 'deterministic' or 'random'; Given: {kind}")

        ms_sampled = []
        ms_list = list(self.microstates.values())
        sampled_cumsum = np.cumsum([mc.count for mc in ms_list])

        if kind == "deterministic":
            sampled_ms_indices = np.arange(size, self.N_ms - size, self.N_ms / size, dtype=int)
        else:
            rng = np.random.default_rng(seed=seed)
            sampled_ms_indices = rng.integers(low=0, high=self.N_ms, size=size, endpoint=True)

        for i, c in enumerate(sampled_ms_indices):
            ms_sel_index = np.where((sampled_cumsum - c) > 0)[0][0]
            ms_sampled.append([ms_sel_index, ms_list[ms_sel_index]])

        return ms_sampled

    def get_sampled_ms(
        self,
        size: int,
        kind: str = "random",
        seed: Union[None, int] = None,
    ) -> list:
        """
        Implement a sampling of MSout.microstates using the ms count probability.
        Args:
            size (int): sample size
            kind (str, 'random'): Sampling kind: one of ['deterministic', 'random'].
                 If 'deterministic', the seed is set to 42. Case insensitive.
            seed (int, None): For testing purposes, fixes random sampling.
        Returns:
            A list of lists: [[selection index, selected microstate], ...]
        """
        print("MSout.get_sample_ms :: Sampling of MSout.microstates using the ms count probability\n",
              "The original, slow sampling scheme (using cumsum) is MSout.get_sampled_ms0.")

        if not len(self.microstates):
            print("The microstates dict is empty.")
            return []

        kind = kind.lower()
        if kind not in ["deterministic", "random"]:
            raise ValueError(f"Values for `kind` are 'deterministic' or 'random'; Given: {kind}")

        ms_list = list(self.microstates.values()) 
        if size > self.N_ms:  # ms space size
            print(f"Requested sample size ({size :,}) > available data: reset to {self.N_ms:,}")
            size = self.N_ms

        if  kind == "deterministic":
            seed = 42
        rng = np.random.default_rng(seed=seed)

        # Occupancies as probability of selection (sum to 1)
        probs = np.array(np.array([ms.count for ms in ms_list]) / self.N_ms, dtype=float)
        indices = rng.choice(len(self.microstates), size=size, p=probs)

        return [[idx, ms_list[idx]] for idx in indices]

    def sort_microstates(self, sort_by: str = "E", sort_reverse: bool = False) -> Union[list, None]:
        """Return the list of microstates sorted by one of these attributes: ["count", "E"],
        and in reverse order (descending) if sort_reverse is True.
        Args:
          sort_by (str, "E"): Attribute as sort key;
          sort_reverse (bool, False): Sort order: ascending if False (default), else descending.
        Return None if 'sort_by' is not recognized.
        """
        if sort_by not in ["count", "E"]:
            print(f"{sort_by = } is not a valid microstate attribute; choices: ['count', 'E']")
            return None

        return sorted(
            list(self.microstates.values()),
            key=operator.attrgetter(sort_by),
            reverse=sort_reverse,
        )

    def __str__(self):
        return (
            f"\nConformer microstates: {self.N_ms:,}\n"
            f"Accepted mc lines: {self.N_uniq:,}\n"
            f"Energies: lowest_E: {self.lowest_E:,.2f}; average_E: "
            f"{self.average_E:,.2f}; highest_E: {self.highest_E:,.2f}\n"
            f"Free residues: {len(self.free_residues):,}\n"
            f"Fixed residues: {len(self.fixed_iconfs):,}\n"
        )


# FIX divisor in occ calc should be the entire conf ms space: MSout.N_ms
class Charge_Microstates:
    """
    Holds a collection of charge microstates. Has methods over the collection.
    Attributes:
      - charge_microstates (list): List of ms_analysis.Charge_Microstate objects
                                   processed from ms_analysis.MSOut.microstates.
      - orig_ms_by_crg_stateid (dict): Grouping of original conformer microstates
                                   for each charge state in charge_microstates.
      - crg_group_stats (dict): Values: (low_state, lowE), (hi_state, hi_E), average.
    """

    __slots__ = ["N_uniq", "confms_by_crg_stateid", "charge_microstates"]

    def __init__(self, microstates: dict, conformers: list, residue_kinds: list = None):
        """
        Args:
          - microstates (dict): MSout.microstates
          - conformers (list): conformers, e.g. Confs.conformers
        """
        self.N_ms = sum(ms.count for ms in microstates.values())
        # for 'bucketing' all ms with the same crg state:
        self.confms_by_crg_stateid = defaultdict(list)
        # cms data per cms stateid, counterpart to MSout.microstates
        self.charge_microstates = {}

        # populate the empty init attributes:
        self.ms_to_charge_ms(microstates, conformers)

    def __str__(self):
        return (f"Unique charge microstates: {len(self.confms_by_crg_stateid.keys()):,}\n"
                )
    
    def ms_to_charge_ms(self, microstates: dict, conformers: list, residue_kinds: list = None):
        """Process the conformer microstates collection to
        obtain a list of charge microstate objects.
        Populate:
           self.confms_by_crg_stateid dict;
           self.charge_microstates;
        """
        for ms in microstates.values():
            current_crg_state = [f"{conformers[ic].resid}|{round(conformers[ic].crg)}" for ic in ms.state]

            crg_ms = Charge_Microstate(current_crg_state, ms.E * ms.count, ms.count)
            crg_id = crg_ms.crg_stateid  # compressed bytes for key
            # append conf ms in this crg ms key:
            self.confms_by_crg_stateid[crg_id].append(ms)

            if crg_id in self.charge_microstates:
                # increment totals viz the crg ms collection
                self.charge_microstates[crg_id].count += crg_ms.count
                self.charge_microstates[crg_id].total_E += crg_ms.total_E
            else:
                # create item
                self.charge_microstates[crg_id] = crg_ms

        # finalize each charge_microstate with average_E (=E) attribute:
        # & update self.charge_microstates dict:
        for k in self.charge_microstates:
            crg_ms = self.charge_microstates[k]
            crg_ms.average_E = crg_ms.E = crg_ms.total_E / crg_ms.count
            self.charge_microstates[k] = crg_ms

        return

    def get_sorted_cms_data(self, return_top_ms: bool = True, min_occ: float = MIN_OCC) -> Union[dict, None]:
        """
        Combine self.confms_by_crg_stateid and data from self.charge_microstates
        filtered for states with occ >= min_occ (0.01 default) into a new dict.
        Args:
            - return_top_ms (bool, True): If True, return the ms with highest count in the bin
              else, return the sorted bin by count.
            - min_occ (float, 0.01): selected cms have at least this minimum occupancy.
        Returns: A dict
            Output dict format:
                key = crg state
                value (list) = [ size, occ, averE, sum_crg, sorted_ms_lst  ]  # most frequent cms data
                                OR
                               [ size, occ, averE, sum_crg, ms0.count, ms0.E, ms0.state ] # most frequent ms data
        Note:
          Output is None when return_top_ms is True and no state within the occupancy
          threshold was found.
        """
        # sort the ms lists to get most numerous ms first:
        ms_by_crg_stateid = dict(sorted(
                                        self.confms_by_crg_stateid.items(),
                                        key=lambda x: len(x[1]),
                                        reverse=True,
                                        )
                                 )
        cms_data = {}
        for k in ms_by_crg_stateid:
            size = len(ms_by_crg_stateid[k])
            sum_crg = self.charge_microstates[k].crg()

            occ = size / self.N_ms
            if round(occ, 2) < min_occ:
                continue
            E = self.charge_microstates[k].average_E
            if return_top_ms:
                # sort conf_ms by count, descendingly:
                sorted_lst = sorted(ms_by_crg_stateid[k], key=lambda x: x.count, reverse=True)
                ms0 = sorted_lst[0]  # most frequent conf ms
                cms_data[k] = [size, occ, E, sum_crg, ms0.count, ms0.E, ms0.state]
            else:
                cms_data[k] = [size, occ, E, sum_crg, ms_by_crg_stateid[k]]

        return cms_data or None

    def get_topN_cms(self, n_top: int = 5) -> list:
        """
        Return list of [[state, averE], size, occ].
        """
        # get sorted cms data with the most numerous ms in each crg state:
        sorted_cms_data = self.get_sorted_cms_data()  # return_top_ms==True (default)
        N = len(sorted_cms_data)
        if n_top > N:
            print(f"{n_top = } > {N = :,}: outputing all charge microstates.")
            n_top = N

        lst = []
        for i, cms in enumerate(sorted_cms_data):
            if i == n_top:
                break
            # cms data :: [size, occ, E, sum_crg, ms0.count, ms0.E, ms0.state]
            size, occ, averE, sum_crg, count, E, state = sorted_cms_data[cms]
            lst.append([[state, averE], size, occ, sum_crg])

        return lst


def free_residues_df(free_res: list, conformers: list, colname: str = "FreeRes") -> pd.DataFrame:
    """Return the free residues' ids in a pandas DataFrame."""
    free_residues = [conformers[res[0]].resid for res in free_res]

    return pd.DataFrame(free_residues, columns=[colname])


def fixed_res_crg(
    conformers: list,
    fixed_iconfs: list,
    res_of_interest: list = None,
    return_df: bool = False,
    no_trailing_underscore: bool = False,
) -> Tuple[float, Union[dict, pd.DataFrame]]:
    """
    Args:
      fixed_iconfs (list): List of fixed conformers.
      conformers (list): List of Conformer instances.
      res_of_interest (list, None): List of resid for filtering.
      return_df (bool, False): If True, the second item of the output tuple
                               will be a pandas.DataFrame, else a dict.
      no_trailing_underscore (bool, False): If True, output resid w/o trailing "_".

    Returns:
      A 2-tuple:
      The net charge contributed by the fixed residues in `fixed_iconfs`;
      A dictionary: key=conf.resid, value=conf.iconf, int(conf.crg).
    """
    fixed_net_charge = 0.0
    dd = defaultdict(float)
    for conf in conformers:
        if conf.iconf in fixed_iconfs:
            fixed_net_charge += conf.crg
            resid = conf.resid[:-1] if no_trailing_underscore else conf.resid
            dd[resid] = conf.iconf, int(conf.crg)
    if res_of_interest:
        dd = {k: dd[k] for k in dd if k[:3] in res_of_interest}
    if return_df:
        fixed_res_crg_df = pd.DataFrame(
            [[k, v[0], v[1]] for k, v in dd.items()],
            columns=["residues", "iconf", "crg"],
        )
        return fixed_net_charge, fixed_res_crg_df

    return fixed_net_charge, dict(dd)


def ms_charge(ms: Microstate):
    """Compute microstate charge"""
    crg = 0.0
    for ic in ms.state:
        crg += conformers[ic].crg
    return crg


def get_ms_crg(ms: Microstate, conformers: list):
    """Compute microstate charge.
    Alternate version of `ms_charge` for use when
    conformers is not a global variable (case when
    module is imported).
    """
    crg = 0.0
    for ic in ms.state:
        crg += conformers[ic].crg
    return crg


def groupms_byenergy(microstates: list, ticks: List[float]) -> list:
    """
    Group the microstates' energies into bands provided in `ticks`.
    Args:
      microstates (list): List of microstates
      ticks (list(float)): List of energies.
    """
    N = len(ticks)
    ticks.sort()
    ticks.append(1.0e100)  # add a big number as the last boundary
    resulted_bands = [[] for i in range(N)]

    for ms in microstates:
        it = -1
        for itick in range(N):
            if ticks[itick] <= ms.E < ticks[itick + 1]:
                it = itick
                break
        if it >= 0:
            resulted_bands[it].append(ms)

    return resulted_bands


def groupms_byiconf(microstates: list, iconfs: list) -> tuple:
    """
    Divide the microstates by the conformers indices provided in `iconfs`
    into 2 groups: the first contains one of the given conformers, the
    second one contains none of the listed conformers.
    Args:
      microstates (list): List of microstates
      iconfs (list): List of conformer indices.
    Return:
      A 2-tuple: (Microstates with any of `iconfs`,
                  microstates with none)
    """
    ingroup = []
    outgroup = []
    for ms in microstates:
        contain = False
        for ic in iconfs:
            if ic in ms.state:
                ingroup.append(ms)
                contain = True
                break
        if not contain:
            outgroup.append(ms)

    return ingroup, outgroup


def groupms_byconfid(microstates: list, confids: list) -> tuple:
    """
    Divide the microstates by the conformers ids provided in `confids`
    into 2 groups: the first contains ALL of the given conformers, the
    second one does not; contains only some but not all.
    Note: An ID is a match if it is a substring of the conformer name.
    Args:
      microstates (list): List of microstates
      confids (list): List of conformer ids.
    Return:
      A 2-tuple: Microstates with all of `confids`, microstates with some or none.
    """
    ingroup = []
    outgroup = []
    for ms in microstates:
        contain = True
        names = [conformers[ic].confid for ic in ms.state]
        for confid in confids:
            innames = False
            for name in names:
                if confid in name:
                    innames = True
                    break
            contain = contain and innames
        if contain:
            ingroup.append(ms)
        else:
            outgroup.append(ms)

    return ingroup, outgroup


def ms_energy_stat(microstates: list) -> tuple:
    """
    Return the lowest, average, and highest energies of the listed
    microstates.
    """
    ms = next(iter(microstates))
    lowest_E = highest_E = ms.E
    N_ms = 0
    total_E = 0.0
    for ms in microstates:
        if lowest_E > ms.E:
            lowest_E = ms.E
        elif highest_E < ms.E:
            highest_E = ms.E
        N_ms += ms.count
        total_E += ms.E * ms.count

    average_E = total_E / N_ms

    return lowest_E, average_E, highest_E


def ms_convert2occ(microstates: list) -> dict:
    """
    Given a list of microstates, convert to conformer occupancy
    for conformers that appear at least once in the microstates.
    Return:
      A dict: {ms.state: occ}
    """
    occurrence = defaultdict(int)
    occ = {}
    N_ms = 0
    for ms in microstates:
        N_ms += ms.count
        for ic in ms.state:
            occurrence[ic] += ms.count

    for k in occurrence:
        occ[k] = occurrence[k] / N_ms

    return occ


def ms_convert2sumcrg(microstates: list, free_res: list) -> list:
    """
    Given a list of microstates, convert to net charge of each free residue.
    """
    # FIX: dependence on global conformers variable

    iconf2ires = {}
    for i_res in range(len(free_res)):
        for iconf in free_res[i_res]:
            iconf2ires[iconf] = i_res

    charges_total = [0.0 for i in range(len(free_res))]
    N_ms = 0
    for ms in microstates:
        N_ms += ms.count
        for ic in ms.state:
            ir = iconf2ires[ic]
            charges_total[ir] += conformers[ic].crg * ms.count

    charges = [x / N_ms for x in charges_total]

    return charges


def e2occ(energies: list) -> float:
    """Given a list of energy values in unit Kacl/mol,
    calculate the occupancy by Boltzmann Distribution.
    """
    e = np.array(energies)
    e = e - min(e)
    Pi_raw = np.exp(-Kcal2kT * e)
    Pi_sum = sum(Pi_raw)
    Pi_norm = Pi_raw / Pi_sum

    return Pi_norm


def bhata_distance(prob1: list, prob2: list) -> float:
    """Bhattacharyya distance between 2 probability distributions."""

    d_max = 10000.0  # Max possible value
    p1 = np.array(prob1) / sum(prob1)
    p2 = np.array(prob2) / sum(prob2)
    if len(p1) != len(p2):
        d = d_max
    else:
        bc = sum(np.sqrt(p1 * p2))
        if bc <= np.exp(-d_max):
            d = d_max
        else:
            d = -np.log(bc)

    return d


def whatchanged_conf(msgroup1: list, msgroup2: list) -> dict:
    "Given two group of microstates, calculate what changed at conformer level."

    occ1 = ms_convert2occ(msgroup1)
    occ2 = ms_convert2occ(msgroup2)

    all_keys = list(set(occ1.keys()) | set(occ2.key()))
    all_keys.sort()
    diff_occ = {}
    for key in all_keys:
        if key in occ1:
            p1 = occ1[key]
        else:
            p1 = 0.0
        if key in occ2:
            p2 = occ2[key]
        else:
            p2 = 0.0
        diff_occ[key] = p2 - p1

    return diff_occ


def whatchanged_res(msgroup1: list, msgroup2: list, free_res: list) -> list:
    "Return a list of Bhattacharyya distance of free residues."

    occ1 = ms_convert2occ(msgroup1)
    occ2 = ms_convert2occ(msgroup2)

    bhd = []
    for res in free_res:
        p1 = []
        p2 = []
        for ic in res:
            if ic in occ1:
                p1.append(occ1[ic])
            else:
                p1.append(0.0)
            if ic in occ2:
                p2.append(occ2[ic])
            else:
                p2.append(0.0)
        bhd.append(bhata_distance(p1, p2))

    return bhd


def example(msout_file: str):
    msout = MSout(msout_file)

    n_bands = 20
    e_step = (msout.highest_E - msout.lowest_E) / n_bands
    ticks = [msout.lowest_E + e_step * (i) for i in range(n_bands)]
    ms_in_bands = groupms_byenergy(msout.microstates.values(), ticks)
    print([len(band) for band in ms_in_bands])
    netural, charged = groupms_byiconf(msout.microstates.values(), [12, 13, 14, 15])
    lo_E, av_E, hi_E = ms_energy_stat(msout.microstates.values())
    print(lo_E, av_E, hi_E)

    # charge over energy bands
    e_step = (msout.highest_E - msout.lowest_E) / n_bands
    ticks = [msout.lowest_E + e_step * (i + 1) for i in range(n_bands - 1)]
    ms_in_bands = groupms_byenergy(msout.microstates.values(), ticks)
    for band in ms_in_bands:
        band_total_crg = 0.0
        for ms in band:
            band_total_crg += ms_charge(ms)
        print(band_total_crg / ms_counts(band))

    netural, charged = groupms_byiconf(msout.microstates.values(), [12, 13, 14, 15])
    diff_occ = whatchanged_conf(netural, charged)
    for key in diff_occ:
        print("%3d, %s: %6.3f" % (key, conformers[key].confid, diff_occ[key]))

    diff_bhd = whatchanged_res(netural, charged, msout.free_residues)
    for ir in range(len(msout.free_residues)):
        print("%s: %6.4f" % (conformers[msout.free_residues[ir][0]].resid, diff_bhd[ir]))
    charges = ms_convert2sumcrg(msout.microstates.values(), msout.free_residues)
    for ir in range(len(msout.free_residues)):
        print("%s: %6.4f" % (conformers[msout.free_residues[ir][0]].resid, charges[ir]))

    microstates = list(msout.microstates.values())
    glu35_charged, _ = groupms_byconfid(microstates, ["GLU-1A0035"])
    print(len(microstates))
    print(len(glu35_charged))

    return


if __name__ == "__main__":
    example("ms_out/pH4eH0ms.txt")
