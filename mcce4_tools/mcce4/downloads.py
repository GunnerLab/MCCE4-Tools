#!/usr/bin/env python
"""
Module: downloads.py

Provides functions to download files with the requests library.
"""
import argparse
import logging
from pathlib import Path
import requests
import shutil
import subprocess
from typing import Tuple, Union

from mcce4.io_utils import subprocess_run


logging.basicConfig(format="[ %(levelname)s ] - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# reset libs logging to higher level so that no unnecessary message is logged
# when this module is used.
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def rcsb_download(pdb_fname: str) -> requests.Response:
    url_rscb = "https://files.rcsb.org/download/" + pdb_fname
    return requests.get(url_rscb, allow_redirects=True)


# def rcsb_download_header0(pdb_fname: str) -> requests.Response:
#     """OBSOLETE: the entire file will be downloaded."""
#     url_rscb = "https://files.rcsb.org/header/" + pdb_fname
#     return requests.get(url_rscb, allow_redirects=False,
#                         headers = {"accept-encoding": "identity"})


def get_rcsb_pdb(pdbid: str,
                 get_bioassembly: bool = True,
                 bioassembly_id: int=1,
                 keep_bioassembly: bool = False) -> Union[Path, Tuple[None, str]]:
    """Given a pdb id, possibly download the pdb file containing the biological
    assembly with given bioassembly_id from rcsb.org. If get_bioassembly is False or
    the download of the bioassembly download fails, then the download of the standard
    pdb file is attempted.

    Arguments:
     - pdbid (str): The pdb id to download
     - get_bioassembly (bool, True): Whether to attempt the bioassembly or the full pdb download
     - bioassembly_id (int, 1): Which bioassembly to download if get_bioassembly is True.
     - keep_bioassembly (bool, False): Whether to retain the downloaded bioassembly file.

    Returns:
     - The path to the downloaded pdb file or a tuple signifying and error occured with values
       (None, error_message)
    """
    pdbid = pdbid.lower()
    pdb = pdbid + ".pdb"  # final pdb filename
    if not get_bioassembly:
        # get the standard, full pdb:
        r0 = rcsb_download(pdb)
        if r1.status_code < 400:
            which_ba[0] = True
            with open(pdb, "wb") as fo:
                fo.write(r0.content)
        else:
            logger.warning(f"Could not download the standard, full pdb: {r0.reason}.")
            return None, "Error: Could not download the standard, full pdb: {r0.reason}."

        return Path(pdb).resolve()

    biopdb = pdbid + f".pdb{bioassembly_id}"
    # list of bool to identify which pdb was saved:
    which_ba = [False, False]  # 0:  bio assembly, 1: pdb standard

    # try bio assembly:
    r1 = rcsb_download(biopdb)
    if r1.status_code < 400:
        which_ba[0] = True
        with open(biopdb, "wb") as fo:
            fo.write(r1.content)
    else:
        logger.warning(f"Could not download bio assembly {biopdb!r}: {r1.reason}. Trying standard pdb.")

    if not which_ba[0]:
        # try standard pdb format:
        r2 = rcsb_download(pdb)
        if r2.status_code < 400:
            which_ba[1] = True
            with open(pdb, "wb") as fo:
                fo.write(r2.content)
        else:
            logger.warning(f"Could not download the pdb file: {r2.reason}")

        if not which_ba[1]:  # both False
            logger.error("Could neither download the bio assembly or standard pdb file.")
            return None, "Error: Could neither download the bio assembly or pdb file."
    else:
        shutil.copy(biopdb, pdb)

        if not keep_bioassembly:
            Path(biopdb).unlink()

    return Path(pdb).resolve()


def cli_parser():
    p = argparse.ArgumentParser(prog="getpdb",
                                description=("Download one or more pdb files (bioassembly by default)"
                                             " from the RSCB download service."))
    p.add_argument(
        "pdbid",
        nargs="+",
        default=[],
        help="Specify the pdb ID(s), e.g.: 1ots 4lzt 1FAT",
    )
    p.add_argument(
        "-get_bioassembly",
        type=bool,
        default=True,
        help="Whether to attempt the bioassembly (default) or the full pdb download"
    )
    p.add_argument(
        "-bioassembly_id",
        type=int,
        default=1,
        help="Which bioassembly to download (default: 1)."
    )
    p.add_argument(
        "--keep_bioassembly",
        default=False,
        action="store_true",
        help="Whether to retain the downloaded bioassembly file (default: False)"
    )

    return p


def getpdb_cli(argv=None):
    """Cli function for the `getpdb` tool.
    """
    p = cli_parser()
    args = p.parse_args(argv)

    pdbids = [id.lower() for id in args.pdbid]

    for pdbid in pdbids:
        out = get_rcsb_pdb(pdbid,
                           get_bioassembly=args.get_bioassembly,
                           bioassembly_id=args.bioassembly_id,
                           keep_bioassembly=args.keep_bioassembly)
        if isinstance(out, tuple):
            logger.info(f"Download failed: {pdbid}; {out[1]}")
        else:
            logger.info(f"Download completed: {out.name}")
