#!/usr/bin/env python3

"""
Module: pdb_converions.py

"""
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
import re
import sys
from typing import List, Tuple


def validate_cif_convert_args(args: Namespace) -> Tuple[Path, Path]:
    """
    Returns the validated input and output file paths.

    Arguments:
      args: argparse.Namespace object
      
    Returns:
      A 2-tuple of file Path objects for the inpout and output files.
    """
    cif_fp = Path(args.cif_file)

    if not cif_fp.exists():
        sys.exit(f"Not found: {cif_fp!s}.")

    if cif_fp.name.endswith("-sf.cif"):
        sys.exit("Wrong file: Expecting a coordinate .cif file not a structure format file.")

    if cif_fp.suffix != ".cif":
        sys.exit("Wrong file: Expecting a .cif extension.")

    if not args.pdb_file:
        pdb_fp = cif_fp.with_suffix(".pdb")
    else:
        if args.pdb_file[-4:] != ".pdb":
            print("Adding '.pdb' extension to output file name.")
            pdb_fp = Path(args.pdb_file).with_suffix(".pdb")
        else:
            pdb_fp = Path(args.pdb_file)

    if pdb_fp.exists():
        if args.overwrite:
            pdb_fp.unlink()
        else:
            sys.exit("Exiting: Output file already exists & overwrite is False.")

    return cif_fp, pdb_fp


def parse_cif_line(cif_line: str) -> List:
    """Parse the cif line to return the fields needed for the pdb format.
    See: https://mmcif.wwpdb.org/docs/pdb_to_pdbx_correspondences.html#ATOMP

    CIF: 
    ATOM   27648 C  CB    . PHE N  14 33  ? -50.361  9.316    -52.409  1.00 115.78 ? 33  PHE M CB    1 
    ATOM   1020 O OXT A LEU A 1 129 ? -10.591 21.716 7.719  0.57 52.55  ? 129  LEU A OXT 1
    HETATM 1021 N N   . NO3 B 2 .   ? -8.235  -0.739 32.272 1.00 21.11  ? 201  NO3 A N   1
    1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
    0:rec 1:serial 2:elem 3:atm 4:alt 5:res 6:chn0 7:resnum0 8:inscode0 9:? 10:x 11:y 12:z 13:occ 14:b
    15:? 16:seqnum 17:res0 18:chn 19:atm0 21:unk

    PDB
    ATOM   1020  OXTALEU A 129     -10.591  21.716   7.719  0.57 52.55           O
    HETATM 1022  N   NO3 A 201      -8.235  -0.739  32.272  1.00 21.11           N
    COLUMNS        DATA  TYPE    FIELD        DEFINITION
    -------------------------------------------------------------------------------------
    1 -  6        Record name   "ATOM  "
    7 - 11        Integer       serial       Atom  serial number.
    13 - 16       Atom          name         Atom name.
    17            Character     altLoc       Alternate location indicator.
    18 - 20       Residue name  resName      Residue name.
    22            Character     chainID      Chain identifier.
    23 - 26       Integer       resSeq       Residue sequence number.
    27            AChar         iCode        Code for insertion of residues.
    31 - 38       Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
    39 - 46       Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
    47 - 54       Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
    55 - 60       Real(6.2)     occupancy    Occupancy.
    61 - 66       Real(6.2)     tempFactor   Temperature  factor.
    77 - 78       LString(2)    element      Element symbol, right-justified.
    79 - 80       LString(2)    charge       Charge  on the atom.
    """
    # ATOM   1020 O OXT A LEU A 1 129 ? -10.591 21.716 7.719  0.57 52.55  ? 129  LEU A OXT 1
    rec, serial, elem, atm, alt, res, _, _, _, _, x, y, z, occ, beta, _, seqnum, _, chn, *_ = cif_line.split()
    # return fields in pdb format order & floats :
    return [rec, serial, atm, alt, res, chn, seqnum,
            float(x), float(y), float(z), float(occ), float(beta), elem]


def get_pdb_line(cif_line: str) -> str:
    """
    Args:
      
    Returns:
      
    """
    rec, serial, atm, alt, res, chn, seqnum, x,y,z, occ, beta, elem = parse_cif_line(cif_line)
    if not alt or alt == ".":
        alt = " "

    align = "^" if len(atm) < 3 else ">"
    pdbline = (
        f"{rec:<6s}{serial:>5s} {atm:{align}4s}{alt}{res:<3s} {chn}{seqnum:>4s}{' ':4s}"
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{beta:>6.2f}{' ':11s}{elem:<2s}  \n"
    )
    return pdbline


def cif_conversion(args: Namespace):
    
    MAX_PDB_LINES = 99_999

    cif_fp, pdb_fp = validate_cif_convert_args(args)
    # extract the coordinates lines:
    pattern = re.compile(r"^[ATOM  |HETATM].*$", re.MULTILINE)
    cif_lines = pattern.findall(Path(cif_fp).read_text())
    if not cif_lines:
        sys.exit(f"No coordinate lines found in {cif_fp!s}.")

    if len(cif_lines) > MAX_PDB_LINES:
        sys.exit(f"This cif file {cif_fp!s} is to large for conversion to pdb format.")

    with open(pdb_fp, "w") as pdb:
        for line in cif_lines:
            pdbline = get_pdb_line(line)
            pdb.write(pdbline)

    print(f"{cif_fp!s} has been successfully converted to {pdb_fp!s}.")

    return


def cif_to_pdb_parser() -> ArgumentParser:
    DESC = """This program converts a cif file to a pdb file format.
    This is a very rough translation. It will not be polished. WIP.
    """
    p = ArgumentParser(prog="cif_to_pdb",
                       formatter_class=RawDescriptionHelpFormatter,
                       description=DESC,
                       usage="%(prog)s file.cif [file.pdb]",
                       )
    
    p.add_argument(
        "cif_file",
        type=str,
        help="Input cif file",
    )
    p.add_argument(
        "-pdb_file",
        type=str,
        default="",
        help="Output pdb file if different from the input file name; Default %(default)s.",
    )
    p.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite output pdb file if already exists; Default %(default)s.",
    )
    return p


def to_cif_cli(argv=None):
    cli_parser = cif_to_pdb_parser()
    args = cli_parser.parse_args(argv)
    cif_conversion(args)

    return
