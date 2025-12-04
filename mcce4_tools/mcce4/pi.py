#!/usr/bin/env python3

"""
Module: pi.py

Calculate the pI (isoelectric point) of a protein.
Codebase for the pI tool.
"""
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
import re
from collections import defaultdict
import sys
from typing import Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

from mcce4.constants import IONIZABLE_RES, SOLUTION_PKAS


PREC = 3


def interp(x, y) -> Tuple:
    spl = CubicSpline(x, y)
    guess = (x.max() - x.min())/2
    roots, _, found, msg = fsolve(spl, guess, full_output=True)
    if found:
        return roots, spl, guess
    else:
        return None, msg, spl, guess


def plot_pi(x, y, spl):
    xnew = np.linspace(x.min(), x.max(), num=len(x)*8)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, 'o', label='data')
    ax.plot(xnew, spl(xnew), label='fit')
    ax.legend(loc='best')
    plt.show()

    return


def run_one_pI(x: np.ndarray, y: np.ndarray, msg:str):
    """Run interp and plot."""
    result = interp(x, y)
    if result[0] is None:
        print("Interpolation not possible:", result[2])
        return
    else:
        roots = result[0]
        print(f"{msg}: fsolve-pI: {round(roots[0],2)}, n_roots: {len(roots)}")
        plot_pi(x, y, result[1])

    return


def test_pI_interp():
    """To determine minimum number of points.
    """
    x = np.array([float(ph) for ph in range(15)])
    y = np.array([float(net) for net in 
                  "18.94,18.52,16.58,13.67,11.13,9.69,8.59,7.51,7.0,6.39,4.53,1.84,-0.65,-4.74,-9.38".split(",")])    
    run_one_pI(x, y, "Full titration")
    for n in [10, 12]:
        run_one_pI(x[:n], y[:n], f"First {n} pts")
        run_one_pI(x[-n:], y[-n:], f"Last {n} pts")

    return


def pdb_res_count(pdb_path: str, ionizable: bool=True, no_CYS:bool=True) -> dict:
    """Return a dict with res name as key and total count as value.
    Args:
     - pdb_path (str): Path to input pdb, which can be a standard or mcce pdb
     - ionizable (bool): Flag to only return count of ionizable residues
     - no_CYS (bool): Flag to exclude CYS from ionizable residues
    Note:
     The no_CYS flag enables a theoretical pI calculated on a standard or mcce pdbs
     to match; this is because the standard pdb may not have identified disulfide bonds,
     but they would be present in step2_out.pdb but no longer ionizable.
    """
    pdb = Path(pdb_path)
    lines = []
    pattern = re.compile(r"^(ATOM.{9}CA.{6}|ATOM.{9}C   CTR).*", re.MULTILINE) 
    lines = pattern.findall(pdb.read_text())
    if not lines:
        sys.exit("CRTICAL: No ATOM lines found in", str(pdb))

    is_standard = pdb.name != "step2_out.pdb"
    res_cnt = defaultdict(int)
    for line in lines:
        _,_,_, res, *_ = line.split(maxsplit=4)
        if is_standard and (len(res) == 4):
            # accept if alt=='A' (as it is accepted in mcce),
            # but remove alt letter from key:
            if res[0] == "A":
                res_cnt[res[-3:]] += 1  
        else:
            res_cnt[res] += 1

    if not is_standard:
        # reset multiple CTR, NTR counts (due to nultiple conformers)
        if res_cnt.get("CTR"):
            res_cnt["CTR"] = 1
        if res_cnt.get("NTR"):
            res_cnt["NTR"] = 1
            
    if ionizable:
        if no_CYS:
            IONIZ = IONIZABLE_RES.copy()
            IONIZ.remove("CYS")
            return {ik: res_cnt[ik] for ik in IONIZ if ik in res_cnt}

        return {ik: res_cnt[ik] for ik in IONIZABLE_RES if ik in res_cnt}

    return res_cnt


def HH_crg(pH, pKa, restype:str, prec:int=PREC) -> Union[float, None]:
    if restype[0] not in "aAbB":
        print("Bad residue type: {restype!s}; must be either 'acid' or 'base' (case insensitive).")
        return None

    if restype[0].lower() == "a":
        return round(1/(1 + 10**(pKa - pH)), prec)
        
    return round(-1/(1 + 10**(pH - pKa)), prec)


def crgHH(res: str) -> Union[list, None]:
    """
    Wrapper for HH_crg function.
    Return the residue charge from Henderson-Hasselbalch function
    For acidic sites (Asp, Glu, CTR, CYS) charge = -1/(1+10^(pH-pKa))
    For basic sites (Lys, Arg, His, N terminus) charge = +1/(1+10^(pKa-pH))
    """
    pk = SOLUTION_PKAS.get(res)
    if pk is None:
        return None
    if res in ["ASP", "GLU", "CTR", "CYS"]:
        kind = "acid"
    else:
        kind = "base"

    return [HH_crg(p, pk, kind) for p in range(15)]


def theoretical_pI(pdb_path: str) -> Tuple[Union[float, None], str]:
    """Return the theoretical pI calculated using solution pKas.
    Returns a 2-tuple:
     - First item: The theoretical pI calculated using solution pKas
     - Second item: status: 'OK' or an error message.
    """
    res_cnt = pdb_res_count(pdb_path)

    theoret_crg = []
    for res, n in res_cnt.items():
        titr_vals = crgHH(res)
        if titr_vals:
            theoret_crg.append([v*n for v in titr_vals])
    if not theoret_crg:
        return None, "Could not compute the theoretical charges."

    # x = pH, y = net crg
    y = np.array(theoret_crg).sum(axis=0)
    x = np.array([float(p) for p in range(15)])
    result = interp(x, y)
    if result[0] is None:
        theoret_pI = None, f"Interpolation not possible: {result[2]}"
    else:
        roots = result[0]
        theoret_pI = round(roots[0], PREC), "OK"

    return theoret_pI


def sumcrg_pI(sumcrg_path: str="sum_crg.out") -> Tuple[Union[float, None], str]:
    """Return the pI interpolated from sum_crg.out.
    Args:
     - sumcrg_path (str, "sum_crg,out"): sum_crg.out filepath.
    Returns a 2-tuple:
     - First item: The protein pI calculated using sum_crg.out Net_Charge
     - Second item: The titration type from sum_crg.out header
    Note: if the first item is None, the second is an error message.
    """
    sumcrg_fp = Path(sumcrg_path)
    if not sumcrg_fp.exists():
        return None, f"Not found: {sumcrg_fp.name} in {sumcrg_fp.parent!s}"
    
    df = pd.read_csv(sumcrg_fp, sep="\s+")
    if len(df.columns) -1 < 10:
        return None, "Not enough titration points (<10) in sum_crg.out file."

    cols = df.columns.tolist()
    kind_col = cols[0]
    # x :: titration points
    x = np.array([float(c) for c in cols[1:]])
    if df.iloc[-3][kind_col] == "Net_Charge":
        y = df.iloc[-3].values[1:]
    else:
        return None, "Corrupted sum_crg.out? 'Net_Charge' not found in third row from bottom"

    result = interp(x, y)
    if result[0] is None:
        prot_pI = None, f"Interpolation not possible: {result[2]}"
    else:
        roots = result[0]
        # there should be only one root:
        prot_pI = round(roots[0], PREC), kind_col
    
    return prot_pI


def delta_pI(prot_pI: float, theoret_pI: float) -> Union[float, None]:
    delta = None
    try:
        delta = round(prot_pI - theoret_pI, PREC)
    except (TypeError, ValueError):
        pass

    return delta


def protein_pI(args: Namespace):
    """
    Wrapper function to calculate the pI (isoelectric point) of a protein.
    """
    pdb = Path(args.pdb)
    print(f"Protein: {pdb!s}")
    theoret_pI, t_status = theoretical_pI(pdb)

    sumcrg_fp = pdb.parent.joinpath("sum_crg.out")
    calculate_ppi = sumcrg_fp.exists()
    if calculate_ppi:
        # if prot_pI is not None, p_status is the titration type,
        # else an error message
        prot_pI, p_status = sumcrg_pI(sumcrg_fp)

    if theoret_pI is not None:
        print(f"Theoretical pI = {theoret_pI:.2f}")
    else:
        print("Theoretical pI:", t_status)

    if calculate_ppi:
        if prot_pI is not None:
            print(f"Protein pI = {prot_pI:.2f} ({p_status} titration)")
            if theoret_pI is not None:
                delta = delta_pI(prot_pI, theoret_pI)
                print(f"Delta pI (protein - theoretical) = {delta}")
        else:
            print("Protein pI:", p_status)

    return


DESC =f"""
  Calculate the theoretical pI (isoelectric point) of a protein.
  If sum_crg.out is found, then the protein pI and delta pI are also calculated.
  Solution pKas in use (mcce4_tools/mcce4/constants.py):\n  {SOLUTION_PKAS}
"""


def cli_parser():
    p = ArgumentParser(
        prog="pI",
        formatter_class=RawTextHelpFormatter,
        description=DESC
    )
    p.add_argument(
        "pdb",
        type=str,
        help="Path to input pdb, which can be a standard or mcce pdb, e.g. 4LZT.pdb, step2_out.pdb."
    )
    p.set_defaults(func=protein_pI)

    return p


def cli(argv=None):
    """
    Command line interface for pI tool.
    """
    parser = cli_parser()
    args = parser.parse_args(argv)
    args.func(args)

    return