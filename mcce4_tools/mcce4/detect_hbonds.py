#!/usr/bin/env python
"""
Module: detect_hbonds.py

Description:
This module detects hydrogen bonds between conformers atoms in the input pdb file, 
step2_out.pdb by default, and outputs their list to file <pdb folder>/hah.txt.
A H-bond-blocking atoms list is also output to file <pdb folder>/blocking.txt if any
are found.
The input pdb file is first parsed for coordinate lines, which can include or exclude
backbone atoms as per users' choice passed with the --no_bk option; they are included
by default.
This script is self-contained, with numpy as only dependency and can be run independently
of the MCCE4 codebase.

Compared to step 6 hah.txt output:
  Step 6 excludes hydrogen bonds with the backbone atoms; here it's a choice.
  Step 6 doesn't check if the potential hydrogen bonds are blocked by 3rd atom.

Usage:
  detect_hbonds.py [step2_out.pdb]
  detect_hbonds.py --out_dir <path/to/location/different/from/pdb/parent/folder>
  detect_hbonds.py path/to/step2_out.pdb --no_bk
  detect_hbonds.py path/to/step2_out.pdb --out_dir <path/to/other/location>
"""
import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Tuple

import numpy as np


# The H-bond distance parameters are roughly based on https://ctlee.github.io/BioChemCoRe-2018/h-bond/
# except for ANGCUT = 100 (90 here).
# To match old program, we use the following parameters
DNEAR = 2.0     # Min Distance cutoff for hydrogen bond between heavy atoms
DFAR  = 4.0     # Max Distance cutoff for hydrogen bond between heavy atoms
ANGCUT = 90     # Angle cutoff for hydrogen bond between heavy atoms, 180 is ideal, 100 is the smallest angle allowed
ANGBLOCK = 90   # Angle cutoff for blocking atoms, 180 is ideal, 90 is the smallest angle allowed for not blocking
# Minimum charge for hydrogen bond donor/acceptor, my best guess
MIN_NCRG = -0.2 # Minimum heavy atom charge for hydrogen bond donor/acceptor
MIN_HCRG = 0.2  # Minimum H atom charge for hydrogen bond donor/acceptor
CRITERIA = f"""
Criteria:
  Donor-Acceptor (heavy atoms) Distance                : {DNEAR:1.1f} - {DFAR:1.1f}
  D-H-A Angle (Angle >= this to qualify H bond)        : {ANGCUT:3.0f}
  H--A-? Blocking Angle (Angle <= this to block H bond): {ANGBLOCK:3.0f}
  Heavy Atom Charge for H bond Donor/Acceptor          : < {MIN_NCRG:4.1f}
  H Atom Charge for H bond Donor                       : > {MIN_HCRG:3.1f}
"""

# Default input file name
in_file = "step2_out.pdb"
# Output files common endings; an actual file may be 4lzt_hah.txt
out_file = "hah.txt"
blocking_file = "blocking.txt"


class Atom:
    def __init__(self):
        self.atom_name = ""
        self.res_name = ""
        self.res_seq = 0
        self.chain_id = ""
        self.iCode = ""
        self.xyz = (0.0, 0.0, 0.0)
        self.charge = 0.0
        self.radius = 0.0
        self.confNum = 0
        self.confType = ""
        self.conn12 = []

    def loadline(self, line):
        self.atom_name = line[12:16]
        self.res_name = line[17:20]
        self.res_seq = int(line[22:26])
        self.chain_id = line[21]
        self.iCode = line[26]
        self.xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        self.charge = float(line[62:74])
        self.confNum = int(line[27:30])
        self.confType = "%3s%2s" % (self.res_name, line[80:82])
        self.confID = "%5s%c%04d%c%03d" % (self.confType, self.chain_id, self.res_seq, self.iCode, self.confNum)

    def is_H(self):
        atom = self.atom_name.strip()
        return (len(atom) < 4 and self.atom_name[1] == "H") or (len(atom) == 4 and self.atom_name[0] == "H")

    def resid(self):
        return self.confID[:3] + self.confID[5:11]

    def __str__(self):
        fmt = "{} {} {} {} {} {} {} {} {} {} {}"
        return fmt.format(self.atom_name, self.res_name, str(self.res_seq), 
                          self.chain_id, self.iCode, str(self.xyz), str(self.charge),
                          self.confNum, self.confType, self.confID)


def dist(atom1, atom2):
    """Return the distance between two atoms."""
    return ((atom1.xyz[0] - atom2.xyz[0])**2 + (atom1.xyz[1] - atom2.xyz[1])**2 + (atom1.xyz[2] - atom2.xyz[2])**2)**0.5


def vec(atom1, atom2):
    """Return the vector from atom1 to atom2."""
    return (atom2.xyz[0] - atom1.xyz[0],
            atom2.xyz[1] - atom1.xyz[1],
            atom2.xyz[2] - atom1.xyz[2])


def deg_angle(v1, v2):
    """Return the angle between two vectors in degrees."""
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi


def get_record_lines(records: list, rec_type: str=None) -> list:
    """Return the list of formated record lines for records of type rec_type,
    one of 'hbond' or 'blocking'.
    """
    lines = []
    if rec_type == "hbond":
        fmt = "{}  {}  {}~{}--{}{:5.2f} {:6.0f}\n"
        for record in records:
            donor, h, acceptor, distance, angle = record
            line = fmt.format(donor.confID, acceptor.confID,
                              donor.atom_name, h.atom_name, acceptor.atom_name, distance, angle)
            lines.append(line)
    else:
        fmt ="{}  {}  {}--{}~{} {:3.0f}\n"
        for record in records:
            h, acceptor, q, blocking_angle = record
            line = fmt.format(h.confID, acceptor.confID,
                              h.atom_name, acceptor.atom_name, q.atom_name, blocking_angle)
            lines.append(line)

    return lines


def detect_hbonds(pdb_file: str, include_bk: bool = False, no_empty_files: bool = False, out_dir: str = "") -> Tuple[int, int]:
    """
    Detect hydrogen bonds between pairs of conformers atoms in a mcce pdb file and
    save their list to file <pdb name>_hah.txt.
    Detect atoms blocking H-bonds and save their to file <pdb name>_blocking.txt.
    These output files are saved in the pdb_file parent folder or in out_dir if provided.

    Procedure:
    1. Parse the input mcce pdb file for coordinates lines with or without BK as per no_bk value
    2. Group atoms into conformers
    3. Make intra 12 connectivity of atoms based on atom distance
    4. Identify potential H bond donors and acceptors based on charge
    5. Pairwise test atoms for potential H bond donors and acceptors & detect blockers:
       For each atom pair, exclude those within the conformer
         Check distance between donor and acceptor
         Check D-H-A angle
         Check if angle between connected atoms on Acceptor intersect the D--H--A path
         If all passed, record a potential hydrogen bond
    6. Write the hydrogen bond list to  <pdb name>_hah.txt
    7. Write the blocking atoms list to  <pdb name>_blocking.txt, if found

    Returns:
      A 2-tuple indicating whether donors acceptors and blocking atoms are found; e.g.:
        results = detect_hbond(file)
        results can be
          (1,1): both found,
          (1,0): no blocking atoms found
          (0,0): no donors or acceptors found.
      Note: The return tuple enables a calling program to display appropriate messages.
            For example:
            ```
            hah, blok = detect_hbonds(inpdb, no_bk, out_dir)
            if not (hah or block):
                print("WARNING! No H-bonds in:", inpdb)
            ```
    """
    pdb = Path(pdb_file)
    if not pdb.exists():
        print("CRITICAL: pdb not found:", str(pdb))
        return 0, 0

    # save the output files in the pdb parent location if out_dir has no value:
    if out_dir:
        out_dir = Path(out_dir)
    else:
        out_dir = pdb.parent

    # delete existing output files
    hah_fp = out_dir.joinpath(f"{pdb.stem}_{out_file}")
    if hah_fp.exists():
        hah_fp.unlink()
    block_fp = out_dir.joinpath(f"{pdb.stem}_{blocking_file}")
    if block_fp.exists():
        block_fp.unlink()

    print("Detecting hydrogen bonds...")
    print(CRITERIA)
    print("Input file:    %s" % (pdb_file))
    print(f"With BK: {['No','Yes'][int(include_bk)]}\n")

    lines = []
    # Parse the input pdb file for coordinates lines, with or without BK:
    if not include_bk:
        pattern = re.compile(r"^[ATOM|HETATM].{79}(?!BK).*$", re.MULTILINE) 
    else:
        pattern = re.compile(r"^[ATOM  |HETATM].*$", re.MULTILINE)

    lines = pattern.findall(Path(pdb_file).read_text())
    if not lines:
        print("CRTICAL: pdb coordinates lines could not be parsed!")
        return 0, 0

    atoms = []
    for line in lines:
        atom = Atom()
        atom.loadline(line)
        atoms.append(atom)

    # Group atoms into conformers dict; key= confID, value= list of atoms.
    conformers = defaultdict(list)
    for atom in atoms:
        conformers[atom.confID].append(atom)

    # Make intra 12 connectivity of atoms based on atom distance
    for confID in conformers:
        conf_atoms = conformers[confID]
        for i in range(len(conf_atoms)-1):
            for j in range(i+1, len(conf_atoms)):
                atom1 = conf_atoms[i]
                atom2 = conf_atoms[j]
                if dist(atom1, atom2) < 2.0:
                    atom1.conn12.append(atom2)
                    atom2.conn12.append(atom1)

    # Identify potential hydrogen bond donors and acceptors based on charge
    donors_acceptors = []
    for confID in conformers:
        for atom in conformers[confID]:
            # Atom as H donor or acceptor => heavy atom with a charge <-0.2
            if not atom.is_H() and atom.charge < MIN_NCRG:
                donors_acceptors.append(atom)
    if not donors_acceptors:
        if not no_empty_files:
            hah_fp.touch()
        return 0, 0

    # Pick two atoms as potential hydrogen bond donors and acceptors
    hbond_records = []
    blocking_records = []
    for donor in donors_acceptors:
        for acceptor in donors_acceptors:
            # exclude atoms from the same residue
            if donor.resid() == acceptor.resid():
                continue
            if DNEAR < dist(donor, acceptor) < DFAR:
                for h in donor.conn12:
                    if h.is_H() and h.charge > MIN_HCRG:
                        Vhd = vec(h, donor)
                        Vha = vec(h, acceptor)
                        angle = deg_angle(Vhd, Vha)
                        if angle > ANGCUT:  # good D--H--A angle
                            blocking = False
                            for q in acceptor.conn12:
                                # donor and acceptor are in different conformers: no need to check if q is donor
                                Vqa = vec(q, acceptor)
                                blocking_angle = deg_angle(Vqa, Vha)
                                # connected atoms on acceptor block the D--H--A path if their angle
                                # with the acceptor is < ANGBLOCK
                                if blocking_angle < ANGBLOCK:
                                    blocking = True
                                    blocking_records.append((h, acceptor, q, blocking_angle))
                                    break
                            if not blocking:
                                distance_h2a = dist(h, acceptor)
                                hbond_records.append((donor, h, acceptor, distance_h2a, angle))

    with open(hah_fp, "w") as hah:
        hah.writelines(get_record_lines(hbond_records, rec_type="hbond"))
    print(f"Output file: {hah_fp!s}")

    if blocking_records:
        with open(block_fp, "w") as block:
            block.writelines(get_record_lines(blocking_records, rec_type="blocking"))
        print(f"Blocking file: {block_fp!s}")
    else:
        return 1, 0

    return 1,1


def cli_parser():
    p = argparse.ArgumentParser(prog="detect_hbonds",
        description="Detect hydrogen bonds between conformer atoms."
    )
    p.add_argument("inpdb",
                    metavar="inpdb",
                    default=in_file,
                    nargs="?",
                    help="Input pdb file in mcce format. Default: %(default)s.",
                    )
    p.add_argument("--include_bk",
                    default=False,
                    action="store_true",
                    help="Include backbone atoms? default: %(default)s."
                    )
    p.add_argument("--no_empty_files",
                    default=False,
                    action="store_true",
                    help="Don't create an empty file when no H-bonds are found; default: %(default)s."
                    )
    p.add_argument("--out_dir",
                    default="",
                    help="Optional output dir. Default: the pdb parent folder."
                    )
    return p


def cli(argv=None):
    p = cli_parser()
    args = p.parse_args(argv)

    pdb = Path(args.inpdb)
    if not pdb.exists():
        sys.exit("CRITICAL: pdb not found:", str(pdb))

    # delete existing output files
    hah_fp = pdb.parent.joinpath(f"{pdb.stem}_{out_file}")
    if hah_fp.exists():
        hah_fp.unlink()
    block_fp = pdb.parent.joinpath(f"{pdb.stem}_{blocking_file}")
    if block_fp.exists():
        block_fp.unlink()

    result = detect_hbonds(args.inpdb, args.include_bk, args.no_empty_files, args.out_dir)
    if result[0]:
        print("hbonds collection over.")


if __name__ == "__main__":
    cli(sys.argv[:1])
