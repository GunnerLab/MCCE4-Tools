#!/usr/bin/env python

"""
Module: pymol_utils.py

"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import sys
from typing import List

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


# fIX: technically, 'convert' should be changed to 'bundle' or 'load'
def convert_pdbs_to_pse(pdb_paths: List, pse_name: str = None):
    """
    Converts multiple PDB files into a single PyMOL session file (.pse).
    @author: Gehan Ranepura

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
        description="""
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
