#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path
from pprint import pprint as ptp, pformat
import sys


def count_all_pk_points(top_dir: Path, chunked_subdir_id: str = None, with_runs_dir: bool = True):
    """Iterate over mcce runs in top_dir[/chunked_subdir_id*/] to get pK point counts.
    Example of subfolder prefix string chunked_dir_id:
      apo_no_wat: identifies subfolder(s) with mcce runs.
    """
    top_dir = Path(top_dir).resolve()
    if chunked_subdir_id is None:
        if with_runs_dir:
            what = "./runs/*/pK.out"
        else:
            what = "./*/pK.out"
        out_fp = top_dir.joinpath(f"{top_dir.name}_pk_points.dict")
    else:
        if with_runs_dir:
            what =  f"{chunked_subdir_id}*/runs/*/pK.out"
        else:
            what = f"{chunked_subdir_id}*/pK.out"
        out_fp = top_dir.joinpath(f"{chunked_subdir_id}_pk_points.dict")

    counts_dict = defaultdict(int)
    for pk_fp in top_dir.glob(what):
        try:
            lines = pk_fp.read_text().splitlines()
        except UnicodeDecodeError:
            # has bytes; non utf-8' chars example: 0|��        >14.0 
            print("Corrupted pK.out in", str(pk_fp.parent))
            continue
        if len(lines) <= 1:
            continue
        for line in lines[1:]:
            # pK.out is a subset of pK_extended.out
            if len(line.split()) < 3:
               continue
            counts_dict[line[:3]] += 1
    
    counts_dict = dict(counts_dict)
    # save:
    print("Saving dict to text file:", str(out_fp))
    out_fp.write_text(pformat(counts_dict)+"\n")
    ptp(counts_dict)
    print(f"Total pKa points in {out_fp.name} runs: {sum(counts_dict.values()):,}")

    return


def cli_parser():
    p = argparse.ArgumentParser(
        prog="pk_points",
        description="Get the count of residues with pKa.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-top-dir",
        type=str,
        default=".",
        help="Path to a directory of mcce runs; default: %(default)s",
    )
    p.add_argument(
        "-subdir-id",
        type=str,
        default=None,
        help="String prefix of top-dir subfolders to process; default: %(default)s",
    )
    p.add_argument(
        "--no-runsdir",
        default=False,
        action="store_true",
        help="Proteins folders are not in a 'runs' subfolder; default: %(default)s",
    )

    return p


def cli(argv = None):
 
    p = cli_parser()
    args = p.parse_args(argv)

    count_all_pk_points(Path(args.top_dir),
                        chunked_subdir_id = args.subdir_id,
                        with_runs_dir = not args.no_runsdir)


if __name__ == "__main__":
    sys.exit(cli())
