#!/usr/bin/env python3

"""
 Module: md_fix_psf_mdanalysis.py
"""
DESC = """
 Purpose: Test the instantiation of a MDAnalysis.Universe given a PSF topology and
          its associated DCD trajectory file.

 Outcome: Upon failure, a new psf file is created using `parmed`, which is then
          used as the topology file in a new MDAnalysis.Universe instantiation attempt.
          - If that attempt is successfull, the user is notified that the new psf file is to be used;
          - If not: the new psf file is deleted and the user is notified that the problem could not be
            resolved.
 Notes:
  1. No other combination besides PSF and DCD files is implemented.
  2. It is assumed that the problem lies with the parsing of the psf file: if the problem is
     a malformed DCD file, the final message will still be 'Could not resolved the problem.' as
     these errors are not differentiated.
  3. Packages `MDAnalysis` and `parmed` are both required in this tool.
     Activate an environment that includes them. To install both, activate a conda
     environment & run: `conda install -c conda-forge mdanalysis parmed`
"""
import argparse
from pathlib import Path
import sys
import warnings

try:
    import MDAnalysis as mda
    import parmed
except ModuleNotFoundError:
    sys.exit(
    """
    Packages `MDAnalysis` and `parmed` are both required in this tool.
    Activate an environment that includes them. To install both, activate a conda
    environment & run: `conda install -c conda-forge mdanalysis parmed`
    """
    )

# suppress some MDAnalysis warnings about PSF files
warnings.filterwarnings("ignore")


def check_input_files(psf_topology_fp: str, dcd_trajectory_fp: str):
    """Check input files for existence and correct types."""
    top_fp = Path(psf_topology_fp).resolve()
    dcd_fp = Path(dcd_trajectory_fp).resolve()
    if not top_fp.exists():
        sys.exit(f"File not found: {psf_topology_fp!r}")
    if not dcd_fp.exists():
        sys.exit(f"File not found: {dcd_trajectory_fp!r}")
    # file types:
    s1 = top_fp.suffix.lower()
    s2 = dcd_fp.suffix.lower()
    if s1 != ".psf" or s2 != ".dcd":
        sys.exit("Only '.psf' and '.dcd' file are considered in this tool.")

    return


def check_psf(psf_topology_fp: str, dcd_trajectory_fp: str):

    check_input_files(psf_topology_fp, dcd_trajectory_fp)

    top_fp = Path(psf_topology_fp)
    top_ = str(top_fp)

    try:
        u = mda.Universe(top_, dcd_trajectory_fp)
        print(u, "Trajectory length: ", len(u.trajectory))
        print("Sucess! ", top_, "is OK!")
    except ValueError:
        print("Initial MDAnalysing parsing failed, trying Parmed...")

        try:
            psf = parmed.charmm.CharmmPsfFile(top_)
        except Exception:
            sys.exit(
                "Parsing with Parmed failed: No known resolution.\n" + sys.exc_info()
            )

        new_psf = top_fp.with_name(f"{top_fp.stem}_parmed.psf")
        top_ = str(new_psf)
        psf.write_psf(top_)
        print("Trying MDAnalysing parsing with the new psf...")
        try:
            u = mda.Universe(top_, dcd_trajectory_fp)
            print(u, "Trajectory length: ", len(u.trajectory))
            print("Success! Use the new psf file: ", top_)
        except Exception:
            new_psf.unlink()
            print("Could not resolved the problem.")
            print(sys.exc_info()[1])
            sys.exit(1)

    return


def cli_parser():
    p = argparse.ArgumentParser(
        prog="md_fix_psf_mdanalysis",
        usage="%(prog) psf_file dcd_file",
        description=DESC
    )
    p.add_argument("psf_file", type=str, help="PSF filename/path.")
    p.add_argument("dcd_file", type=str, help="The DCD filename/path.")
    return p


def cli(argv=None):
    p = cli_parser()
    args = p.parse_args(argv)
    check_psf(args.psf_file, args.dcd_file)


if __name__ == "__main__":
    cli(sys.argv[:1])
