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

from mcce4_tools.io_utils import subprocess_run


logging.basicConfig(format="[ %(levelname)s ] %(name)s - %(funcName)s:\n  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# reset libs logging to higher level so that no unnecessary message is logged
# when this module is used.
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def rcsb_download(pdb_fname: str) -> requests.Response:
    url_rscb = "https://files.rcsb.org/download/" + pdb_fname
    return requests.get(url_rscb, allow_redirects=True)


def rcsb_download_header(pdb_fname: str) -> requests.Response:
    url_rscb = "https://files.rcsb.org/header/" + pdb_fname
    return requests.get(url_rscb, allow_redirects=True,
                        headers = {"accept-encoding": "identity"})


def get_rcsb_pdb(pdbid: str, keep_bioassembly: bool = False) -> Union[Path, Tuple[None, str]]:
    """Given a pdb id, download the pdb file containing
    the biological assembly from rcsb.org.
    The file is downloaded with a pdb extension.

    Removed cif file download as this format cannot yet be used.
    """
    pdbid = pdbid.lower()
    pdb = pdbid + ".pdb"  # final pdb filename
    pdb1 = pdbid + ".pdb1"

    # The bioassembly and header are downloaded because the bioassembly file
    # will not have the complete remarks section, which mcce4.pdbio.py parses.

    # list of bool to identify which pdb was saved:
    which_ba = [False, False]  # 0:  bio assembly, 1: pdb standard

    # try bio assembly:
    r1 = rcsb_download(pdb1)
    if r1.status_code < 400:
        which_ba[0] = True
        with open(pdb1, "wb") as fo:
            fo.write(r1.content)
    else:
        logger.warning(f"Could not download the bio assembly: {r1.reason}")
    
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
        # get the full header to use for the bioassembly file
        r2 = rcsb_download_header(pdb)
        if r2.status_code < 400:
            with open(pdb, "w") as fo:
                fo.write(r2.text)
        else:
            logger.warning(f"Could not download the full pdb file header: bioassembly with partial header used: {r2.reason}")
            # use bioassembly as pdb file:
            shutil.move(pdb1, pdb)
            return Path(pdb).resolve()

        # check if header has a MODEL line, if not add it:
        missing_model_line_added = False
        MDL = "MODEL        1"
        cmd = f"grep '^{MDL}' {pdb}"
        try:
            o = subprocess_run(cmd, capture_output = True, check = True)
            if not o.stdout:  # no MODEL line, add it
                cmd = f"sed -i '$a\\{MDL}' {pdb}"
                o = subprocess_run(cmd, capture_output = True, check = True)
                if isinstance(o, subprocess.CompletedProcess):
                    missing_model_line_added = True
                else:
                    logger.info(f"Failed adding missing MODEL line in {pdb}.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed check on MODEL line; exit code {e.returncode}: {e.stderr}")

        # append model coordinate lines to the header file:
        cmd = "egrep '^ATOM|^HETATM' " + pdb1 + " >> " + pdb
        if missing_model_line_added:
            # need "ENDMDL" as well:
            cmd = cmd + "; echo 'ENDMDL' >> " + pdb
        o = subprocess_run(cmd, capture_output = True, check = True)
        if isinstance(o, subprocess.CalledProcessError):
            logger.warning(f"Failed adding full header for {pdb}: bioassembly with partial header used.")
            shutil.move(pdb1, pdb)

        if not keep_bioassembly:
            Path(pdb1).unlink()

    logger.info("Download completed.")

    return Path(pdb).resolve()


def getpdb_cli():
    """Cli function for the `getpdb` tool.
    """
    parser = argparse.ArgumentParser(prog="getpdb",
                                     description=("Download one or more (bioassembly) pdb files "
                                                  "from the RSCB download service."))
    parser.add_argument("pdbid", metavar="pdbid",
                        help="Specify the pdb ID(s), e.g.: 1ots 4lzt 1FAT",
                        nargs="+", default=[])
    args = parser.parse_args()

    pdbids = [id.lower() for id in args.pdbid]

    for pdbid in pdbids:
        out = get_rcsb_pdb(pdbid)
        if isinstance(out, tuple):
            msg = out[1] + "PDB id: " + pdbid
            print(msg)
