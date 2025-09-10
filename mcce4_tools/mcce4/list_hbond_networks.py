#!/usr/bin/env python3

"""
Created on Mar 16 06:00:00 2025

@author: Gehan Ranepura
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys

import networkx as nx


def parse_entryexit_info(residue_info):
    """Parse residue information (ENTRY/EXIT residues)"""
    residue_name = residue_info[:3]  # Charcters  1-3 are residue name
    chain_id = residue_info[3]  # Character  4 is the chain ID
    residue_number = residue_info[4:9]  # Characters 5-8 are the residue number
    return residue_name, chain_id, residue_number


def parse_donoracceptor_info(residue_info):
    """Parse residue information (DONOR/ACCEPTOR residues)"""
    residue_name = residue_info[:3]  # Characters 1-3 are residue name
    chain_id = residue_info[5]  # Character  6 is the chain ID
    residue_number = residue_info[6:10]  # Characters 7-10 are the residue number
    return residue_name, chain_id, residue_number


def read_residue_pairs(file_path):
    """parse ENTRY/EXIT residues in the resi_list"""
    entry_residues_i = set()
    exit_residues_i = set()

    with open(file_path) as file:
        lines = file.readlines()

    # Skip the header row (first line)
    for line in lines[1:]:  # Start from the second line
        # Extract the entry and exit residues based on character positions

        # ENTRY occupies characters 1-8 (0-7 in 0-indexed)
        entry_residue = line[:8].strip()
        # EXIT occupies characters 12-19 (11-18 in 0-indexed)
        exit_residue = line[11:19].strip()  

        # If both residues are not empty, add them to respective sets
        if entry_residue:
            entry_residues_i.add(entry_residue)
        if exit_residue:
            exit_residues_i.add(exit_residue)

    # Parse entry and exit residues (if any parsing is required)
    entry_residues = {parse_entryexit_info(residue) for residue in entry_residues_i}
    exit_residues = {parse_entryexit_info(residue) for residue in exit_residues_i}

    return entry_residues, exit_residues


def process_hbond_graph(file_path, entry_residues, exit_residues, node_min):
    """process an individual hydrogen bond network file"""
    with open(file_path) as file:
        lines = file.readlines()

    nodes = set()
    edges = []

    ENTRY = set()
    EXIT = set()

    # Process each line and build graph edges based on donor/acceptor pairs
    for line in lines:
        parts = line.split()
        if len(parts) < node_min:
            continue  # Skip malformed lines

        donor, acceptor = parts[0], parts[1]
        nodes.update([donor, acceptor])  # Create nodes
        edges.append((donor, acceptor))  # Create edges

        # Extract the part before the conformer info
        donor_residue_info = donor.rsplit("_", 1)[0]
        # Extract the part before the conformer info 
        acceptor_residue_info = acceptor.rsplit("_", 1)[0]

        donor_residue_name, donor_chain_id, donor_residue_number = (
            parse_donoracceptor_info(donor_residue_info)
        )
        acceptor_residue_name, acceptor_chain_id, acceptor_residue_number = (
            parse_donoracceptor_info(acceptor_residue_info)
        )

        # Match donor and acceptor residues against entry/exit residues
        if (donor_residue_name, donor_chain_id, donor_residue_number) in entry_residues:
            ENTRY.add(donor)
        if (
            acceptor_residue_name,
            acceptor_chain_id,
            acceptor_residue_number,
        ) in entry_residues:
            ENTRY.add(acceptor)

        if (donor_residue_name, donor_chain_id, donor_residue_number) in exit_residues:
            EXIT.add(donor)
        if (
            acceptor_residue_name,
            acceptor_chain_id,
            acceptor_residue_number,
        ) in exit_residues:
            EXIT.add(acceptor)

    if len(nodes) < node_min:
        # Skip this network if it has fewer than 2 nodes
        return None, None, None, None

    return list(nodes), list(edges), list(ENTRY), list(EXIT)


def natural_sort_key(s):
    """Extracts numeric and non-numeric parts to sort alphanumerically using regex."""
    return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)]


def build_graph(ENTRY, EXIT, nodes, edges, output_file, gro, node_min):
    """build graph and save paths between entry and exit residues"""
    # Create the graph object and choose type based on --gro flag
    G = nx.DiGraph() if gro else nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    all_paths = set()  # Dictionary to store paths for each node

    # Check if the graph has at least 2 nodes
    if len(G.nodes) < node_min:
        print(f"Skipping network with fewer than node_min.")
        return  # Exit the function if the graph has fewer than 2 nodes

    all_paths = set()  # Dictionary to store paths for each node

    # Find all simple paths between specified entry and exit residues
    for entry_resi in ENTRY:
        for exit_resi in EXIT:
            print(f"Entry/Exit: {entry_resi} <--> {exit_resi}")
            # Find all simple paths starting from entry_residue and ending at exit_residue
            try:
                paths = list(nx.all_simple_paths(G, source=entry_resi, target=exit_resi))
                for path in paths:
                    if len(path) > 1:  # Only add paths that have at least two nodes
                        all_paths.add(tuple(path))  # Convert list to tuple for uniqueness
            except nx.NetworkXNoPath:
                continue  # Skip if no path exists

    # Find all simple paths between entry resi GLU-1P0041_005 and exit resi GLU-1P0050_005
    # GLU-1P0041_005 -> ARG+1P0048_006 -> GLU-1P0050_005
    # try:
    #    paths = list(nx.all_simple_paths(G, source="GLU-1P0041_005", target="GLU-1P0050_005"))
    #    for path in paths:
    #         all_paths.add(tuple(path))  # Convert list to tuple for uniqueness
    #     except nx.NetworkXNoPath:
    #            continue  # Skip if no path exists

    # Find all simple paths leading to specified exit residues
    # for node in nodes:
    #    for exit_resi in EXIT:
    #        print(f"Exit: {exit_resi}")
    #        try:
    #            paths = list(nx.all_simple_paths(G, source=node, target=exit_resi))
    #            for path in paths:
    #                all_paths.add(tuple(path))  # Convert list to tuple for uniqueness
    #        except nx.NetworkXNoPath:
    #               continue  # Skip if no path exists

    # Save unique and alphanumercally sorted paths to output file
    sorted_paths = sorted(
        all_paths, key=lambda path: [natural_sort_key(node) for node in path]
    )
    try:
        with open(output_file, "w") as file:
            for path in sorted_paths:
                file.write(" -> ".join(path) + "\n")
        print(f"Unique sorted paths saved to {output_file}")
    except Exception as e:
        print(f"Error writing to file {output_file}: {e}")

    return


def process_directory(directory, entry_residues, exit_residues, output_dir, gro, node_min):
    """Main function to process all files and build graphs
    """
    # filter dir for files ending with _hah.txt:
    files = list(Path(directory).glob("*_hah.txt"))
    if not files:
        print(f"No valid hydrogen bond files found in '{directory}'.")
        return

    for file_path in files:
        print(f"Processing file {file_path.stem} ...")

        # Extract nodes and edges from the hydrogen bond graph file
        nodes, edges, ENTRY, EXIT = process_hbond_graph(
            file_path, entry_residues, exit_residues, node_min
        )
        print(f"NODES: {nodes}\n")
        print(f"EDGES: {edges}\n")

        print(f"ENTRY: {ENTRY}")
        print(f"EXIT:  {EXIT}")

        # Build and save the graph for this file
        output_file = Path(output_dir).joinpath(f"Paths_{file_path.stem}_Resi-EntryToExit.txt")
        build_graph(ENTRY, EXIT, nodes, edges, output_file, gro, node_min)
        print("-" * 105)

    return


def argparser():
    """Argument parser for input directory and residue pair file
    """
    parser = argparse.ArgumentParser(
        description="Computes hydrogen bond graph networks across a collection of hah.txt files."
    )
    parser.add_argument(
        "-i",
        type=str,
        default="ms_pdb_output_hbonds",
        help="Input directory containing hbond text files. (default: %(default)s)",
    )
    parser.add_argument(
        "-resi_list",
        type=str,
        default="resi_list.txt",
        help="File containing entry and exit residues of interest (format: XXXCYYYY). (default: %(default)s)",
    )
    parser.add_argument(
        "-node_min",
        type=int,
        default=2,
        help="Minimum number of nodes in the graph to process. (default: %(default)s)",
    )
    parser.add_argument(
        "--gro",
        action="store_true",
        help="Enable computing of Grotthuss competent networks only.",
    )
    return parser


def cli(argv=None):

    parser = argparser()
    args = parser.parse_args(argv)

    # Get the input directory and check if it exists
    input_dir = args.i
    if not Path(input_dir).exists():
        sys.exit(f"Error: Input directory {input_dir!s} does not exist. Please check the path.")

    # Check if output directory exists and prompt the user
    output_dir = Path(f"{input_dir}_gronets") if args.gro else Path(f"{input_dir}_nets")
    if not output_dir.exists():
        output_dir.mkdir()
    else:
        user_input = (
            input(
                f"The output directory {output_dir!s} already exists. Do you want to delete and remake it? (y/n): "
            )
            .strip()
            .lower()
        )
        if user_input == "y":
            shutil.rmtree(output_dir)  # Delete directory
            output_dir.mkdir()  # Remake directory
        else:
            sys.exit(f"Please rename or remove {output_dir!s} before running the script again.")

    # Ensure residue pairs file is provided and exists
    if not args.resi_list or not Path(args.resi_list).exists():
        sys.exit("Error: Residue pair list file is missing or does not exist.")

    # Read the residue pairs from the file
    entry_residues, exit_residues = read_residue_pairs(args.resi_list)

    # Run the script
    process_directory(input_dir, entry_residues, exit_residues, output_dir, args.gro, args.node_min)

    return


if __name__ == "__main__":
    cli(sys.argv[:1])
