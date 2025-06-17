#!/usr/bin/env python
"""
Module: constants.py

Contains values, conversion factors, lists or dicts for use in MCCE4.
"""
NEUTRAL_RES = ["ALA", "ASN", "GLN", "GLY", "ILE", "LEU","MET", "PHE", "PRO", "SER", "THR", "TRP", "VAL"]
IONIZABLE_RES = ["ASP", "GLU", "ARG", "HIS", "LYS", "CYS", "TYR", "NTR", "CTR"]
ALL_RES = IONIZABLE_RES + NEUTRAL_RES
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
    "CYS": 0
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

AA_CODES = {"ALA",
               "ARG",
               "ASN",
               "ASP",
               "CTR",
               "CYD",
               "CYL",
               "CYS",
               "GLN",
               "GLY",
               "HIL",
               "HIS",
               "ILE",
               "LEU",
               "LYS",
               "MET",
               "NTG",
               "NTR",
               "PHE",
               "PRO",
               "SER",
               "THR",
               "TRP",
               "TYR",
               "VAL"}

res3_to_res1 = {
    "ASP": "D",
    "GLU": "E",
    "ARG": "R",
    "HIS": "H",
    "LYS": "K",
    "CYS": "C",
    "TYR": "Y",
}


# VDW Cutoff
VDW_UPLIMIT = 999.0    # The upper limit value of VDW. This value allows head3.lst column align well at %8.3f format
VDW_CUTOFF_FAR = 10.0  # Set VDW to 0.0 if the atom distance is over this value
VDW_CUTOFF_NEAR = 1.0  # Set VDW to VDW_UPLIMIT if the atom distance is less than this value
VDW_SCALE14 = 0.5      # Scaling fator for 1-4 connected atom VDW
