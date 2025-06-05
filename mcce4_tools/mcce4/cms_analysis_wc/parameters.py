#!/usr/bin/env python
"""
Module: parameters.py

Functions to load the command line input parameter file, e.g. 'params.crgms'.

"""
from collections import defaultdict
from copy import deepcopy
import logging
from pathlib import Path
import re
import string
import sys
from typing import Tuple

logger = logging.getLogger(__name__)
try:
    import numpy as np

except ImportError as e:
    logger.critical("Oops! Forgot to activate an appropriate environment?\n", exc_info=e)
    sys.exit(1)

from mcce4.cms_analysis_wc import IONIZABLES, MIN_OCC, N_top 


def params_main(ph: str="7", eh: str="0") -> dict:
    """Obtain cms_analysis main parameters dict with default values for given ph, eh.
    """
    params_defaults = {
        # Most values are strings to match the values in the dicts returned by `load_param_file`.
        "mcce_dir": ".",
        "output_dir": f"crgms_corr_ph{ph}eh{eh}",
        "list_head3_ionizables": "False",
        "msout_file": f"pH{ph}eH{eh}ms.txt",
        # Do not output file 'all_crg_count_res.csv'
        # "main_csv": "all_crg_count_res.csv",
        "fixed_res_of_interest_csv": "fixed_res_of_interest.csv",
        "all_crg_count_resoi_csv": "all_crg_count_resoi.csv",
        "all_res_crg_csv": "all_res_crg_status.csv",
        "res_of_interest_data_csv": "crg_count_res_of_interest.csv",
        "n_top": str(N_top),
        "min_occ": str(MIN_OCC),
        "residue_kinds": IONIZABLES,
        "correl_resids": None,
        "corr_method": "pearson",
        "corr_cutoff": "0.02",
        "n_clusters": "5",
        "cluster_min_res_count": "6",
        "fig_show": "False",
        "energy_histogram.save_name": "enthalpy_dist.png",
        "energy_histogram.fig_size": "(8,8)",
        "corr_heatmap.save_name": "corr.png",
        "corr_heatmap.fig_size": "(20, 8)",
    }

    return params_defaults


def params_histograms() -> dict:
    """Obtain cms_analysis histogram parameters dict with default values."""
    params_defaults = {
        "charge_histogram0": {
            "bounds": "(None, None)",
            "title": "Protonation Microstates Energy",
            "save_name": "crgms_logcount_vs_E.png",
        },
        "charge_histogram1": {
            "bounds": "(Emin, Emin + 1.36)",
            "title": "Protonation Microstates Energy within 1.36 kcal/mol of Lowest",
            "save_name": "crgms_logcount_vs_lowestE.png",
        },
        "charge_histogram2": {
            "bounds": "(Eaver - 0.68, Eaver + 0.68)",
            "title": "Protonation Microstates Energy within 0.5 pH (0.68 kcal/mol) of Mean",
            "save_name": "crgms_logcount_vs_averE.png",
        },
        "charge_histogram3": {
            "bounds": "(Emax - 1.36, Emax)",
            "title": "Protonation Microstates Energy within 1.36 kcal/mol of Highest",
            "save_name": "crgms_logcount_vs_highestE.png",
        },
    }

    return params_defaults


def sort_resoi_list(resoi_list: list) -> list:
    """Return the input 'res of interest' list with ionizable residues in
    the same order as msa.IONIZABLES, i.e.:
    acid, base, polar, N term, C term, followed by user provided res, sorted.
    """
    if not resoi_list:
        return []

    userlst = [res.upper() for res in resoi_list]
    ioniz = deepcopy(IONIZABLES)

    ioniz_set = set(ioniz)
    sym_diff = ioniz_set.symmetric_difference(userlst)
    new_res = sym_diff.difference(ioniz_set)
    removal = sym_diff.difference(new_res)
    if removal:
        for res in removal:
            ioniz.pop(ioniz.index(res))

    return ioniz + sorted(new_res)


# For splitting a string with re. Remove punctuation and spaces:
re_split_pattern = re.compile(r"[\s{}]+".format(re.escape(string.punctuation)))
# for splitting lists of residues for correlation: keep underscore
correl_split_pattern = re.compile(r"[\s{}]+".format(re.escape(string.punctuation.replace("_", ""))))


def split_spunct(text, upper=True, pattern: re.Pattern = re_split_pattern) -> list:
    """Split text on space and punctuation."""
    if not text:
        return []
    if upper:
        text = text.upper()
    return re.split(pattern, text)


def load_crgms_param(filepath: str) -> Tuple[dict, dict]:
    """Load parameters file into two dicts; the second one is used for processing
    the charge histograms calculations & plotting.
    """
    fp = Path(filepath)
    if not fp.exists():
        return FileNotFoundError(fp)

    crgms_dict = {}
    correl_lines = []

    with open(filepath) as f:
        # data lines:
        lines = [line.strip() for line in f.readlines() if line.strip()
                 and not line.strip().startswith("#")]

    multi_found = False
    for line in lines:
        if not multi_found:
            try:
                key, rawval = line.split("=")
            except ValueError:
                raise ValueError("Malformed entry: single equal sign required.")

            key, rawval = key.strip(), rawval.strip()
            if rawval.startswith("("):
                # check tuple:
                if rawval.endswith(")"):
                    val = rawval
                else:
                    raise ValueError("Malformed tuple: must be (x,y) on same line.")
            elif rawval.startswith("["):
                # check list on same line:
                if rawval.endswith("]"):
                    if key == "residue_kinds":
                        val = sort_resoi_list([v for v in split_spunct(rawval[1:-1].strip()) if v])
                    else:
                        # correl_resids on same line
                        val = [
                            v for v in split_spunct(rawval[1:-1].strip(),
                                                    pattern=correl_split_pattern) if v
                        ]
                else:
                    if key == "residue_kinds":
                        sys.exit(
                            (
                                "Malformed residue_kinds entry, ']' not found: "
                                "list within square brackets must be on the same line."
                            )
                        )
                    elif key == "correl_resids":
                        multi_found = True
                        continue
            else:
                # all others: strings:
                val = rawval
            if not multi_found:
                crgms_dict[key] = val
        else:
            rawval = line.strip()
            if not rawval.endswith("]"):
                correl_lines.extend([v for v in split_spunct(rawval, pattern=correl_split_pattern) if v])
                continue
            if rawval.endswith("]"):
                multi_found = False
                continue

    if correl_lines:
        crgms_dict["correl_resids"] = correl_lines

    p, e = crgms_dict.get("msout_file", "pH7eH0ms.txt")[:-4].lower().split("eh")
    ph = p.removeprefix("ph")
    eh = e.removesuffix("ms")
    crgms_dict["ph"] = ph
    crgms_dict["eh"] = eh

    charge_histograms = defaultdict(dict)
    remove_keys = []
    for k in crgms_dict:
        if k.startswith("charge_histogram"):
            v = crgms_dict[k]
            k1, k2 = k.split(".")
            charge_histograms[k1].update({k2: v})
            remove_keys.append(k)

    if remove_keys:
        for k in remove_keys:
            crgms_dict.pop(k)

    # Add missing default params:
    main_params = params_main(ph=ph, eh=eh)
    for k in main_params:
        if crgms_dict.get(k) is None:
            crgms_dict[k] = main_params[k]

    # Add params for unbounded histogram if none were given:
    if not charge_histograms:
        charge_histograms["charge_histogram0"] = params_histograms()["charge_histogram0"]

    return crgms_dict, dict(charge_histograms)


# TODO: remove?
def get_resid2iconf_dict0(conf_info: np.ndarray) -> dict:
    """Return the mapping of conformer resid to its index.
    """
    free_ci = conf_info[np.where(conf_info[:, -3])]
    return dict((ci[0], ci[1]) for ci in free_ci[:, [1, 0]])


def check_res_list(correl_lst: list, res_lst: list = None, conf_info: np.ndarray = None) -> list:
    """Perform at most 2 checks on res_list depending on presence of the other arguments:
    - Whether items in res_list are in other_list;
    - Whether items in res_list are in the conformer space.
    """
    if not res_lst and not conf_info:
        logger.warning("Arguments 'conf_info' and 'res_lst' cannot both be None.")
        return correl_lst

    new = []
    for res in correl_lst:
        if res[:3] in res_lst:
            new.append(res)
        else:
            logger.warning(f"Ignoring {res!r} from correl_lst: {res[:3]} not in residue_kinds.")
    correl_lst = new

    if conf_info is not None:
        free_ci = conf_info[np.where(conf_info[:, -3])]  # is_free field
        correl2 = deepcopy(correl_lst)
        #res2iconf = get_resid2iconf_dict(conf_info)
        for cr in correl_lst:
            #if res2iconf.get(cr) is None:
            # check resid field:
            if not len(free_ci[np.where(free_ci[:, 1] == cr)]):
                logger.warning(f"Removing {cr!r} from correl_lst: not in conformer space.")
                correl2.remove(cr)

        return correl2

    return correl_lst
