#!/usr/bin/env python3

"""
Module: constants.py

Contains values, conversion factors, lists or dicts for use in MCCE4.
"""
NEUTRAL_RES = ["ALA", "ASN", "GLN", "GLY", "ILE", "LEU","MET", "PHE", "PRO", "SER", "THR", "TRP", "VAL"]
IONIZABLE_RES = ["ASP", "GLU", "ARG", "HIS", "LYS", "CYS", "TYR", "NTR", "CTR"]
ALL_RES = IONIZABLE_RES + NEUTRAL_RES

# Warning: in some context, 'NTG' means Guanine nucleotide, not terminal Glycine:
AA_CODES = set(ALL_RES + ["CYD","CYL","HIL","NTG"])

ACIDIC_RES = ["ASP", "GLU"]
BASIC_RES = ["ARG", "HIS", "LYS"]
POLAR_RES = ["CYS", "TYR"]

# canonical charge states of ionizable residues at pH 7 in compbio simulations:
CANONICAL = {
    "ASP": -1,
    "GLU": -1,
    "CTR": -1,
    "ARG": 1,
    "LYS": 1,
    "NTR": 1,
    "TYR": 0,
    "CYS": 0,
    }

COMMON_IONIZ_RANGES = {
    "NTR+": (5, 9),
    "ASP-": (2, 6),
    "ACY-": (2, 6),
    "GLU-": (2, 6),
    "HIS+": (5, 11),
    "CYS-": (7, 11),
    "LYS+": (8, 13),
    "TYR-": (9, 12),
    "ARG+": (10, 15),
    "CTR-": (1, 5),
    "BCL-": (35, 45),
    }

res3_to_res1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    }


COMMON_HETATMS = ["HOH", "CA", "CL", "FE", "NO3", "NA", "PO4", "SO3", "MG", ]


FLOAT_VALUES = [
    "EPSILON_PROT",
    "TITR_PH0",
    "TITR_PHD",
    "TITR_EH0",
    "TITR_EHD",
    "CLASH_DISTANCE",
    "BIG_PAIRWISE",
    "MONTE_T",
    "MONTE_REDUCE",
    "EXTRAE",
    "SCALING",
]
INT_VALUES = [
    "TITR_STEPS",
    "MONTE_RUNS",
    "MONTE_TRACE",
    "MONTE_NITER",
    "MONTE_NEQ",
    "MONTE_NSTART",
    "MONTE_FLIPS",
]


ROOMT = 298.15
PH2KCAL = ph2Kcal = ph2kcal = 1.364
KCAL2KT = Kcal2kT = kcal2kt = 1.688
# degrees to radians conversion factor
D2R = d2r = 0.017453


# VDW Cutoff
VDW_UPLIMIT = 999.0    # The upper limit value of VDW. This value allows head3.lst column align well at %8.3f format
VDW_CUTOFF_FAR = 10.0  # Set VDW to 0.0 if the atom distance is over this value
VDW_CUTOFF_NEAR = 1.0  # Set VDW to VDW_UPLIMIT if the atom distance is less than this value
VDW_SCALE14 = 0.5      # Scaling fator for 1-4 connected atom VDW


# defined by <MCCE4 clone>/name.txt:
aliphatic_groups: list = "CT1, CT2, CT3, CT4, FAR, dgd, lhg, lmg, lmt, sqd".split(", ")


pqr_frmt = "{:6s} {:>5} {:^4} {:3} {:>5} {:>8} {:>8} {:>8} {:>6} {:>6}\n"
#           rec, seq, atm, res, resnum, x, y, z, crg, rad
pqr_field_frmt_dict = {
    "rec": "{:6s}",
    "seq": "{:>5}",
    "atm": "{:^4}",
    "res": "{:3}",
    "resnum": "{:>5}",
    "x": "{:>8}",
    "y": "{:>8}",
    "z": "{:>8}",
    "crg": "{:>6}",
    "rad": "{:>6}",
    }
