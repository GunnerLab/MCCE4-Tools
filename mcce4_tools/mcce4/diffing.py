#!/usr/bin/env python3

"""
Module: diffing.py

Code module of the command line interface for diffing two mcce output files.
The difference file is saved using the first filename prefixed with 'diff_'.
Valid files defined in mcce4.io_utils.DIFFING_FILES:
    "entropy.out",
    "fort.38",
    "head3.lst",
    "pK.out",
    "sum_crg.out",
    "vdw0.lst"
"""

import argparse
import logging
from pathlib import Path
import sys

from mcce4.io_utils import DIFFING_FILES, files_diff, MsgFmt


logging.basicConfig(format="[ %(levelname)s ] %(name)s - %(funcName)s:\n  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


TOOL = "filesdiff"
CLI_NAME = __name__ if __name__ == "__main__" else TOOL
USAGE = ("\n1. Diffing files with same filenames; save diff file in current dir:"
         f"\n\t{CLI_NAME} dir1/pK.out dir2/pK.out\n"
         "\n2. Diffing files with standard filenames in two folders; save diff file in current dir:"
         f"\n\t{CLI_NAME} dir1 dir2 --all\n"
         "\n3. Save output file with only absolute difference beyond the given threshold value (default 0):"
         f"\n\t{CLI_NAME} dir1/pK.out dir2/pK.out -threshold 2.5\n"
         "\n4. Using files with different names: need to pass their common type in the -file_type option:"
         f"\n\t{CLI_NAME} dir1/e8_pK.out dir2/pK.out -file_type pK.out\n"
         )


mf = MsgFmt  # short lowercase alias


def all_files_diff(
    p1: Path,
    p2: Path,
    out_dir: Path = Path.cwd(),
    return_df: bool = False,
    threshold: float = 0,
):
    """Compute the difference of mcce output folder files listed in DIFFING_FILES."""
    for fname in DIFFING_FILES:
        fp1 = p1.joinpath(fname)
        if not fp1.exists():
            logger.warning(f"File 1 not found: {fp1!r}.")
            continue
        fp2 = p2.joinpath(fname)
        if not fp2.exists():
            logger.warning(f"File 2 not found: {fp2!r}.")
            continue

        logger.info(mf("Diffing {} -> {}/diff_{}", fname, str(out_dir), fname))
        files_diff(fp1, fp2, out_dir=out_dir, return_df=return_df, threshold=threshold)

    return


def check_paths(args: argparse.Namespace) -> tuple:
    """Check args for consistency.
    Return:
      A 2-tuple: p1, p2
    """
    p1 = Path(args.dir_or_file_paths[0])
    p2 = Path(args.dir_or_file_paths[1])
    are_files = p1.is_file() and p2.is_file()

    if not args.all:  # paths must be files:
        if are_files:
            if p1.name != p2.name and args.file_type is None:
                logger.error(mf(
                    ("You can use files with different names only if you provide their "
                     "common format in the '-file_type' option, which can one of: {}."
                     ), DIFFING_FILES)
                )
                sys.exit(1)
            return p1, p2
        else:
            logger.error("You must give two filepaths.")
            sys.exit(1)

    if args.all:
        if are_files:
            logger.error("You must give two directory paths with --all.")
            sys.exit(1)
        return p1, p2


def diff_cli(argv=None):
    parser = argparse.ArgumentParser(prog=CLI_NAME,
                                     description=("Subtract all numerical columns from two "
                                                  "mcce run files, e.g. f1 - f2.\n"
                                                  "The output default filename is 'diff_' + f1.name."),
                                     usage=USAGE)
    parser.add_argument("dir_or_file_paths", metavar="paths", nargs=2)
    parser.add_argument(
        "-threshold",
        type=float,
        default=0,
        help="Output absolute differences beyond given value; default: %(default)s.",
    )
    parser.add_argument(
        "-odir",
        metavar="output directory",
        default=".",
        help="Output folder path; default: %(default)s.",
    )
    parser.add_argument(
        "-oname",
        metavar="output filename",
        default="",
        help="Output file name (ignored with --all); default: 'diff_ + file1.name'.",
    )
    # cannot have -file_type & --all together:
    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument(
        "-file_type",
        type=str,
        default=None,
        help=f"""If set, value is one of: {DIFFING_FILES}.
        Used to by-pass the equality check on the input filenames in case they differ,
        e.g.: fp1 :: new_sum_crg.out, fp2 :: sum_crg.out => file_type :: sum_crg.out.
        Cannot be used with '--all' (requires standard names); default: %(default)s.
        """,
    )
    mutex.add_argument(
        "--all",
        default=False,
        action="store_true",
        help=f"Process all these files: {DIFFING_FILES} (standard names); default: %(default)s.",
    )

    args = parser.parse_args(argv)
    p1, p2 = check_paths(args)
    outdir = Path(args.odir).resolve()
    if outdir.name != Path.cwd().name:
        if not outdir.is_dir():
            outdir.mkdir()

    if args.all:
        all_files_diff(p1, p2, out_dir=outdir, threshold=args.threshold)
        logger.info("Processing of files difference over.")
    else:
        files_diff(p1, p2, out_dir=outdir, out_name=args.oname, threshold=args.threshold, file_type=args.file_type)
        #if args.oname is None:
        if not args.oname:
            diff_fp = outdir.joinpath(f"diff_{p1.name}")
        else:
            diff_fp = outdir.joinpath(args.oname)
        if diff_fp.exists():
            logger.info(
                mf(("Displaying the difference file {!r} (threshold: {:.2f}),\n"
                    "  which can be loaded into a DataFrame using mcce4.io_utils.textfile2df:\n"
                    ), str(diff_fp), args.threshold)
                )
            print(diff_fp.read_text())


if __name__ == "__main__":
    diff_cli(sys.argv)
