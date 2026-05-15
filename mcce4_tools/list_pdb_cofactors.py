#!/usr/bin/env python

import argparse
import sys
import os
from collections import OrderedDict
from Bio.PDB import PDBParser

def find_cofactors(pdb_file, chains=None, include_water=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    cofactors = OrderedDict()

    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            if chains and chain_id not in chains:
                continue
            for residue in chain:
                het_field, res_num, _ = residue.get_id()

                if het_field == " ":
                    continue
                if het_field == "W" and not include_water:
                    continue

                res_name = residue.get_resname().strip()
                n_atoms = len(list(residue.get_atoms()))
                location = f"{chain_id}{res_num:04d}"

                if res_name not in cofactors:
                    cofactors[res_name] = []
                cofactors[res_name].append((location, n_atoms))

    return cofactors


def print_table(cofactors, pdb_file):
    if not cofactors:
        print("No cofactors found.")
        return

    total = sum(len(locs) for locs in cofactors.values())
    print(f"PDB file : {os.path.basename(pdb_file)}")
    print(f"Cofactors: {len(cofactors)} types, {total} instances\n")

    loc_width = max(
        len(", ".join(loc for loc, _ in locs))
        for locs in cofactors.values()
    )
    loc_width = max(loc_width, 9)

    header = f"{'Cofactor':<10} {'Count':>5}   {'Atoms':>5}   {'Locations':<{loc_width}}"
    print(header)
    print("-" * len(header))
    for name, entries in cofactors.items():
        locations = ", ".join(loc for loc, _ in entries)
        atoms_per = entries[0][1]
        print(f"{name:<10} {len(entries):>5}   {atoms_per:>5}   {locations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List cofactors and ligands in a PDB file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python list_pdb_cofactors.py structure.pdb
  python list_pdb_cofactors.py structure.pdb -c A B
  python list_pdb_cofactors.py structure.pdb --water
""",
    )
    parser.add_argument("pdb", help="Path to the PDB file")
    parser.add_argument("-c", "--chains", nargs="+", metavar="CHAIN",
                        help="Only list cofactors in these chain(s)")
    parser.add_argument("--water", action="store_true",
                        help="Include water molecules (HOH) in the output")
    args = parser.parse_args()

    if not os.path.exists(args.pdb):
        print(f"Error: {args.pdb} not found.", file=sys.stderr)
        sys.exit(1)

    cofactors = find_cofactors(args.pdb, chains=args.chains, include_water=args.water)
    print_table(cofactors, args.pdb)
