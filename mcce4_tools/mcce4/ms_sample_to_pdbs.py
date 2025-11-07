#!/usr/bin/env python3

"""
Module:
    ms_sample_to_pdbs.py

Purpose:
    To generate a random set of mcce microsates in step2_out pdb format 
    whose set size is user-defined.

Usage:
    Usage with all default arguments:
    > sampled_ms_to_pdbs.py

    Usage with different input folder and sample size:
    > sampled_ms_to_pdbs.py -mcce_dir path/to/mcce/dir -sample_size 10

    * Options default values:
    -mcce_dir:  ./
    -msout_file: pH7eH0ms.txt
    -out_dir: ms_pdb_output
    -sample_size: 100
    -sampling_kind: random
    -seed: None
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import time
from typing import TextIO

from mcce4.constants import CLI_EPILOG
from mcce4.io_utils import get_unique_filename, get_mcce_filepaths
import mcce4.ms_analysis as msa


# Residue Information:
TER = ["NTR", "CTR", "NTG"]
SPECIAL_RESIDUES = ["CLA", "CLB", "CHL", "CT1", "CT2", "CT3", "CT4"]
CAPPING_RESIDUES = ["NTR", "CTR", "NTG", "CT1", "CT2", "CT3", "CT4"]

MCCE_FIELDS = 11

R250 = """REMARK 250
REMARK 250 EXPERIMENTAL DETAILS
REMARK 250   EXPERIMENT TYPE               : MCCE simulation.
REMARK 250 MICROSTATES SPACE SIZE          : {N:,.0f}
REMARK 250 MICROSTATE INFORMATION
REMARK 250   ENERGY                        : {E:,.2f} (kcal/mol)
REMARK 250   ACCEPTANCE COUNT              : {COUNT:,.0f}
REMARK 250   INDEX                         : {IDX:,.0f}
REMARK 250
"""


def confs_to_pdb(
    step2_fh: TextIO, selected_confs: list, output_pdb: str, remark: str = None
) -> None:
    """Read step2_out coordinate line for each conformer in list `selected_confs`
    and create a pdb file.
    Args:
      step2_fh (TextIO): File handle of 'step2_out.pdb'.
      selected_confs (list): A microstate's list of conformer ids.
      output_pdb (str): Output pdb file path.

    Note: mcce (step2_out.pdb) format is as follows:
    ATOM      1  CA  NTR A0001_001   2.696   5.785  12.711   2.000       0.001      01O000M000
    ATOM     44  HZ3 LYS A0001_002  -2.590   8.781   9.007   1.000       0.330      +1O000M000
    ATOM     45  N   VAL A0002_000   4.060   7.689  12.193   1.500      -0.350      BK____M000
    """

    with open(output_pdb, "w") as out:
        if remark is not None:
            out.write(remark)

        for line in step2_fh:
            if len(line) < 82:
                continue
            if line[80:82] == "BK":
                out.write(line)
                continue
            # confID = line[17:20] + line[80:82] + line[21:26] + "_" + line[27:30]
            try:
                selected_confs.index(
                    line[17:20] + line[80:82] + line[21:26] + "_" + line[27:30]
                )  # confID)
                out.write(line)
            except ValueError:
                continue

    print(f"\tconfs_to_pdb - Written: {output_pdb}")
    return


def ms_sample_to_pdbs(
    sampled_ms: list,
    conformers: list,
    fixed_iconfs: list,
    ms_space_size: int,
    step2_fp: str,
    output_folder: str,
) -> None:
    """Obtain a sample from the microstate population held in msa.MSout.microstates,
    and for each ms in the sample gather its conformers and use them to create a pdb file.

    Args:
      sampled_ms (list): List of sampled [index, Microstate object].
      conformers (list): List of conformers from head3.lst.
      fixed_iconfs (list): List of fixed conformers.
      ms_space_size (int): Total number of conformer microstates (for pdb remark).
      step2_fp (str): Path of step2_out.pdb in mcce_dir.
      output_folder (str): Created pdb files output folder path.
    """

    step2_fh = open(step2_fp)

    for idx, sms in sampled_ms:
        pdb_out = get_unique_filename(output_folder.joinpath(f"ms_pdb_{idx}.pdb"))

        R250_txt = R250.format(N=ms_space_size, E=sms.E, COUNT=sms.count, IDX=idx)
        pdb_confs = [
            conf.confid
            for conf in conformers
            if conf.iconf in sms.state or conf.iconf in fixed_iconfs
        ]
        # write the pdb file
        confs_to_pdb(step2_fh, pdb_confs, pdb_out, remark=R250_txt)
        step2_fh.seek(0)

    step2_fh.close()
    print(
        f"\tms_sample_to_pdbs - over:\n\t{len(sampled_ms):,} (new)",
        f"sampled microstates saved to pdbs in: {output_folder!s}",
    )

    return


def do_ms_to_pdbs(args):
    """Main function to output sampled microstates to pdbs."""

    if args.mcce_dir is None:
        raise ValueError("Invalid mcce_dir (None)")

    if args.sample_size <= 0:
        raise ValueError("Sample size must be > 0.")

    mcce_dir = Path(args.mcce_dir)
    p, e = args.msout_file.removesuffix(".txt").lower().split("eh")
    ph = p.removeprefix("ph")
    eh = e.removesuffix("ms")

    head3_fp, step2_fp, msout_fp = get_mcce_filepaths(mcce_dir, ph=ph, eh=eh)

    sampled_pdbs = args.out_dir  # Default output folder name
    pdbs_dir = mcce_dir.joinpath(sampled_pdbs)
    if not pdbs_dir.exists():
        pdbs_dir.mkdir()

    sampling_kind = "random"
    if args.sampling_kind[0].lower() == "d":
        sampling_kind = "deterministic"

    print("\tGettings conformers")
    conformers = msa.read_conformers(head3_fp)

    print("\tInstantiating MSout")
    start = time.time()
    mso = msa.MSout(msout_fp)
    d = time.time() - start
    print(f"\tInstantiating `msa.MSout` took {d:.2f} seconds or {d/60:.2f} mins")

    # informational
    ms_counts = msa.ms_counts(mso.microstates)
    print(
        f"\tNumber of accepted microstates lines: {len(mso.microstates):,}",
        f"\tConformer microstates space         : {ms_counts:,}",
        f"\tSample size requested               : {args.sample_size:,}",
        sep="\n",
    )

    print("\tGetting sampled ms")
    sampled_ms = mso.get_sampled_ms(args.sample_size, kind=sampling_kind, seed=args.seed)

    print("\tCreating pdbs of sampled ms")
    ms_sample_to_pdbs(
        sampled_ms, conformers, mso.fixed_iconfs, mso.N_ms, step2_fp, pdbs_dir
    )

    return


HELP = """
    Usage with minimal number of arguments (0):
    > sampled_ms_to_pdbs.py

    * All options have their default values:
    -mcce_dir      :  ./
    -msout_file    : pH7eH0ms.txt
    -out_dir       : ms_pdb_output
    -sample_size   : 100
    -sampling_kind : random
    -seed          : None; if sampling is 'deterministic', seed = 42
"""


def cli_parser():
    """Command line arguments parser."""

    def arg_valid_dirpath(p: str):
        """Return resolved path from the command line."""
        if not len(p):
            return None
        return Path(p).resolve()

    p = ArgumentParser(
        prog="ms_sample_to_pdbs",
        description=HELP,
        formatter_class=RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG
    )
    p.add_argument(
        "-mcce_dir",
        type=arg_valid_dirpath,
        default="./",
        help="The folder with files from a MCCE simulation; required. (default: %(default)s)",
    )
    p.add_argument(
        "-msout_file",
        type=str,
        default="pH7.00eH0.00ms.txt",
        help="Name of the mcce_dir/ms_out/ microstates file, pHXeHYms.txt. (default: %(default)s)",
    )
    p.add_argument(
        "-out_dir",
        type=str,
        default="ms_pdb_output",
        help="Output folder for generated ms pdbs; required. (default: %(default)s)",
    )
    p.add_argument(
        "-sample_size",
        type=int,
        default=100,
        help="The size of the microstates sample, hence the number of pdb files to write; required. (default: %(default)s)",
    )
    p.add_argument(
        "-sampling_kind",
        type=str,
        default="r",
        choices=["d", "deterministic", "r", "random"],
        help="""The sampling kind;
             'deterministic': random number generator set with seed=42.
             'random': random indices over the microstates space. (default: %(default)s)""",
    )
    p.add_argument(
        "-seed",
        type=int,
        default=None,
        help="The seed for fixing random number generation. (default: %(default)s)",
    )
    p.set_defaults(func=do_ms_to_pdbs)

    return p


def cli(argv=None):
    clip = cli_parser()
    args = clip.parse_args(argv)
    args.func(args)

    return
