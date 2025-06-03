#!/usr/bin/env python

"""
Module: pdb_converions.py

Functions for converting pdb files:
 * from .cif
 * to pse (pymol format)

"""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
import re
import sys
from typing import List, Tuple

# In case we remove the automatic installation of pymol in the user's
# active environment, this will provide a workaround:
try:
    import pymol
except ImportError:
    msg = """Oops!
You are missing pymol in your environment: you can either add the path of an already
installed pymol package to your system PATH variable, or you can add it to your current
environment:
```
 conda install -c conda-forge -c schrodinger pymol-bundle
```
"""
    sys.exit(msg)


def convert_pdbs_to_pse(pdb_paths: List, pse_name: str = None):
    """
    Converts multiple PDB files into a single PyMOL session file (.pse).

    Args:
        pdb_paths (list of str): List of paths to the PDB files to be loaded into the session.
        pse_name (str, optional): The name for the output PyMOL session file (.pse).
                                  If not provided, the name of the last PDB file is used.

    Returns:
        None
    """
    pymol.finish_launching(["pymol", "-cq"])

    object_names = []
    for pdb_path in pdb_paths:
        name = Path(pdb_path).stem
        pymol.cmd.load(pdb_path, name)
        object_names.append(name)

    # Determine .pse filename
    pse_fmt = "{}.pse"
    if pse_name is None:
        pse_filename = pse_fmt.format(object_names[-1])
    else:
        pse_filename = pse_fmt.format(Path(pse_name).stem)

    # Save the session
    pymol.cmd.save(pse_filename)
    pymol.cmd.quit()

    print(f"Saved session as {pse_filename} with objects: {', '.join(object_names)}")

    return


def to_pse_cli(argv=None):
    parser = ArgumentParser(
        prog="pdbs2pse",
        formatter_class=RawDescriptionHelpFormatter,
        description="""@author: Gehan Ranepura
Converts one or more PDB files into a single PyMOL session file (.pse).
The session file contains all the loaded PDB structures as separate objects.
The user can specify an optional output name for the .pse file, or it will default 
to the name of the last input PDB file.
""",
        usage="%(prog)s file1.pdb file2.pdb ... [--pse_name <output_name>]",
    )
    parser.add_argument(
        "pdb_files", nargs="+", help="Input PDB files")
    parser.add_argument(
        "--pse_name",
        default=None,
        help="Optional output PSE name (without .pse extension)")

    args = parser.parse_args(argv)

    # Check if the specified PDB files exist
    for pdb_file in args.pdb_files:
        if not Path(pdb_file).exists():
            print(f"File not found: {pdb_file}")
            sys.exit(1)

    # Run the conversion function
    convert_pdbs_to_pse(args.pdb_files, args.pse_name)

    return


def validate_cif_convert_args(args: Namespace) -> Tuple[Path, Path]:
    """
    Returns the validated input and output file paths.

    Arguments:
      args: argparse.Namespace object
      
    Returns:
      A 2-tuple of file Path objects for the inpout and output files.

    TODO:
    1. cif file exists? - if yes return Path
    2. user outfile given?
      No: return path with input stem & pdb extension
      Yes: if file eexists, overwrite?
    """
    cif_fp = Path(args.cif_file)

    if not cif_fp.exists():
        sys.exit("Not found:", cif_fp , "exiting.")

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

# FIX: Needs work;
#      See: https://mmcif.wwpdb.org/docs/pdb_to_pdbx_correspondences.html#ATOMP
def parse_cif_line(cif_line: str) -> List:
    """Parse the cif line to return the fields needed for the pdb format.

    This shows what each of the 14 elements of the lists mean in the cif/pdb files
    CIF:
    ATOM   27646 C  C     . PHE N  14 33  ? -49.047  10.809   -53.949  1.00 86.10  ? 33  PHE M C     1 
    ATOM   27647 O  O     . PHE N  14 33  ? -48.629  10.489   -55.065  1.00 93.13  ? 33  PHE M O     1 
    ATOM   27648 C  CB    . PHE N  14 33  ? -50.361  9.316    -52.409  1.00 115.78 ? 33  PHE M CB    1 
    0:rec 1:serial 2:elem 3:atm 4:alt 5:res 6:chn0 7:resnum0 8:inscode0 9:? 10:x 11:y 12:z 13:occ 14:b 15:?
    16:seqnum 17:res0 18:chn 19:atm0 21:unk

    ATOM   1018 C CD1 A LEU A 1 129 ? -6.199  23.869 10.557 0.57 29.33  ? 129  LEU A CD1 1
    ATOM   1019 C CD2 A LEU A 1 129 ? -6.392  21.428 10.259 0.57 24.59  ? 129  LEU A CD2 1
    ATOM   1020 O OXT A LEU A 1 129 ? -10.591 21.716 7.719  0.57 52.55  ? 129  LEU A OXT 1
    HETATM 1021 N N   . NO3 B 2 .   ? -8.235  -0.739 32.272 1.00 21.11  ? 201  NO3 A N   1
    HETATM 1022 O O1  . NO3 B 2 .   ? -8.733  0.339  32.042 1.00 20.60  ? 201  NO3 A O1  1
    HETATM 1023 O O2  . NO3 B 2 .   ? -7.019  -0.884 32.211 1.00 42.47  ? 201  NO3 A O2  1
    1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
    PDB
    ATOM   1019  CD2ALEU A 129      -6.392  21.428  10.259  0.57 24.59           C
    ATOM   1020  OXTALEU A 129     -10.591  21.716   7.719  0.57 52.55           O
    HETATM 1022  N   NO3 A 201      -8.235  -0.739  32.272  1.00 21.11           N
    HETATM 1023  O1  NO3 A 201      -8.733   0.339  32.042  1.00 20.60           O

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
    # return fields in pdb order & floats :
    return [rec, serial, atm, alt, res, chn, seqnum, float(x), float(y), float(z), float(occ), float(beta), elem]


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

    pattern = re.compile(r"^[ATOM  |HETATM].*$", re.MULTILINE)
    cif_lines = pattern.findall(Path(cif_fp).read_text())
    if not cif_lines:
        sys.exit("No coordinate lines found?!.")

    if len(cif_lines) > MAX_PDB_LINES:
        sys.exit("The cif file is to large for conversion to pdb format.")

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
