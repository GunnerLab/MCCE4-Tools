#!/usr/bin/env python
"""
Module: stats_hbond_networks.py

Created on Apr 09 05:20:00 2025

@author: Gehan Ranepura
"""
import argparse
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pymol


def parse_network_file(filepath, node_min):
    """
    Parses a text file containing directed network paths and returns a list of node sequences.

    Each line in the file is expected to contain nodes connected by "->", representing a path
    (e.g., "A -> B -> C"). Lines that do not contain "->" or do not meet the minimum number of 
    nodes specified by `node_min` are ignored.

    Parameters:
        filepath (str): Path to the text file containing the network data.
        node_min (int): Minimum number of nodes required for a path to be included.

    Returns:
        List[List[str]]: A list of node sequences, where each sequence is a list of node names (str).
    """
    networks = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            nodes = [n.strip() for n in line.split("->")]
            if len(nodes) >= node_min:
                networks.append(nodes)

    return networks


def write_pymol_session_file(pdb_paths, pse_name=None, delete_pml=False):
    """
    Bundles a list of PDB files into a PyMOL session (.pse) file.
    Displays water molecules (HOH) as small spheres, and creates a blue mesh object for the 
    last PDB loaded (including waters). The session is saved as a .pse file.
    
    Arguments:
    -----------
    pdb_paths : list of str
        Paths to the PDB files to be loaded into PyMOL.
        Each file will be loaded as a separate object.

    pse_name : str, optional
        Desired name of the output PyMOL session (.pse file).
        If not provided, the base name of the last PDB file with a ".pse"
        extension will be used.
    """
    pml_script = Path("pymol_load.pml")
    with open(pml_script, "w") as script:
        for pdb_path in pdb_paths:
            name = Path(pdb_path).stem
            script.write(f"load {pdb_path}, {name}\n")

        # Add the desired display settings
        script.write("bg_color white\n")
        script.write("show spheres, resn HOH+WAT\n")
        script.write("set sphere_scale, 0.2, resn HOH+WAT\n")

        # Create a mesh for residues in the last PDB file object (non-water atoms)
        script.write(f"create {name}_mesh, {name}\n")
        script.write(f"show mesh, {name}_mesh\n")
        script.write(f"color blue, {name}_mesh\n")  # Comment to turn off the single mesh color

        # Create a mesh for waters in the last PDB file object (HOH)
        script.write(f"create {name}_waters, ({name} and resn HOH+WAT)\n")
        script.write(f"map_new {name}_watermap, gaussian, 1.0, {name}_waters, 5\n")
        script.write(f"isomesh {name}_watermesh, {name}_watermap, 1.0\n")
        script.write(f"color blue, {name}_watermesh\n")

        # Save and quit
        script.write(f"save {pse_name}\n")
        script.write("quit\n")

    # Run PyMOL silently without printing subprocess commands
    script_path = str(pml_script)
    subprocess.run(["pymol", "-cq", script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if delete_pml:
        pml_script.unlink()

    return


def convert_pdbs_to_pse(pdb_paths, pse_name=None):
    """
    Convert a list of PDB files into a PyMOL session (.pse) file.

    This function loads multiple PDB files into a PyMOL session, displays water 
    molecules (HOH) as small spheres, and creates a blue mesh object for the 
    last PDB loaded (including waters). The session is saved as a .pse file.

    Arguments:
    -----------
    pdb_paths : list of str
        Paths to the PDB files to be loaded into PyMOL.
        Each file will be loaded as a separate object.

    pse_name : str, optional
        Desired name of the output PyMOL session (.pse file).
        If not provided, the base name of the last PDB file with a ".pse"
        extension will be used.

    Output:
    -------
    Creates a PyMOL session file (.pse) containing:
      - All loaded PDB structures.
      - Water molecules shown as spheres with a small scale.
      - A mesh object colored blue for the last PDB structure including waters.
    """
    if pse_name is None:
        pse_name = os.path.splitext(os.path.basename(pdb_paths[-1]))[0]
        pse_name += ".pse"
    elif not pse_name.endswith(".pse"):
        pse_name += ".pse"

    # Create temporary PyMOL script
    with tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False) as temp_script:
        for pdb_path in pdb_paths:
            name = os.path.splitext(os.path.basename(pdb_path))[0]
            temp_script.write(f"load {pdb_path}, {name}\n")

        # Add the desired display settings
        temp_script.write("bg_color white\n")
        temp_script.write("show spheres, resn HOH+WAT\n")
        temp_script.write("set sphere_scale, 0.2, resn HOH+WAT\n")

        # Last PDB file object name
        last_name = os.path.splitext(os.path.basename(pdb_paths[-1]))[0]

        # Create a mesh for residues in the last PDB file object (non-water atoms)
        temp_script.write(f"create {last_name}_mesh, {last_name}\n")
        temp_script.write(f"show mesh, {last_name}_mesh\n")
        temp_script.write(f"color blue, {last_name}_mesh\n")  # Comment to turn off the single mesh color

        # Create a mesh for waters in the last PDB file object (HOH)
        temp_script.write(f"create {last_name}_waters, ({last_name} and resn HOH+WAT)\n")
        temp_script.write(f"map_new {last_name}_watermap, gaussian, 1.0, {last_name}_waters, 5\n")
        temp_script.write(f"isomesh {last_name}_watermesh, {last_name}_watermap, 1.0\n")
        temp_script.write(f"color blue, {last_name}_watermesh\n")

        # Save and quit
        temp_script.write(f"save {pse_name}\n")
        temp_script.write("quit\n")
        script_path = temp_script.name

    # Run PyMOL silently without printing subprocess commands
    subprocess.run(["pymol", "-cq", script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #print(f"Saved session as {pse_name} with objects: {", ".join([os.path.splitext(os.path.basename(p))[0] for p in pdb_paths])}")

    return


def print_and_write(output_file, text):
    print(text)
    output_file.write(text + "\n")


def main(input_dir, topnets, node_min, A, output_file):
    all_networks = []
    all_pairwise_presence = defaultdict(set)
    network_file_map = defaultdict(set)
    file_networks_dict = {}
    output_dir = Path(output_file).parent  # -> f"{input_dir}_stats"

    filenames = [f for f in Path(input_dir).iterdir()
                 if f.name.startswith("Paths_ms_pdb_") and f.name.endswith("_hah_Resi-EntryToExit.txt")]
    total_files = len(filenames)

    with open(output_file, "w") as out_file:
        print_and_write(out_file, f"\nTotal microstate PDBs analyzed: {total_files}\n")
        print_and_write(out_file, f"Top {topnets} networks (with at least {node_min} nodes):\n")

        for filename in filenames:
            full_path = str(filename)
            file_networks = parse_network_file(full_path, node_min)

            file_networks_strs = []
            for path_nodes in file_networks:
                network_str = " -> ".join(path_nodes)
                file_networks_strs.append(network_str)
                all_networks.append(network_str)
                network_file_map[network_str].add(filename.name)

                for i in range(len(path_nodes) - 1):
                    pair = (path_nodes[i], path_nodes[i + 1])
                    all_pairwise_presence[pair].add(filename.name)

            file_networks_dict[filename.name] = file_networks_strs

        network_counts = Counter(all_networks)
        top_networks = network_counts.most_common(topnets)

        for idx, (network, count) in enumerate(top_networks, start=1):
            nodes = network.split(" -> ")
            network_str = " -> ".join(nodes)
            network_percentage = (count / total_files)
            ratio = count / total_files
            if ratio == 1:
                E = 0
                k = A
            else:
                E = -1.364 * math.log10(ratio)
                k = A * 10 ** (-E / 1.364)

            print_and_write(out_file,
                            (f"{idx}. Count: {count} microstate PDBs ({network_percentage:6.2%}) "
                            "have this fully connected network.")
            )
            print_and_write(out_file, f"Network Rate & Energy: k = {k:.2e} /sec, E = {E:.2e} kcal/mol")
            print_and_write(out_file, f"Network: {network_str}")

            # Subnet %
            subnet_percents = []
            for i in range(1, len(nodes) + 1):
                subnet_str = " -> ".join(nodes[:i])
                count_subnet = sum(
                    any(subnet_str in net for net in nets)
                    for nets in file_networks_dict.values()
                )
                percent = count_subnet / total_files
                subnet_percents.append(f"{percent:6.2%}")

            # Build Subnet % aligned to the right under each final node (excluding first)
            node_positions = []
            cursor = 0
            for i, node in enumerate(nodes):
                start_idx = network_str.find(node, cursor)
                end_idx = start_idx + len(node)
                node_positions.append(end_idx)
                cursor = end_idx

            subnet_line = [" "] * len(network_str)
            for i in range(1, len(nodes)):  # skip the first node
                subnet_str = subnet_percents[i]
                # Align the subnet % to the right under the current node
                end = node_positions[i]
                pos = end  # right-aligned at the end of the node
                pos = max(0, min(pos - len(subnet_str), len(subnet_line) - len(subnet_str)))
                for j, ch in enumerate(subnet_str):
                    subnet_line[pos + j] = ch

            print_and_write(out_file, "Subnet %:" + "".join(subnet_line))

            # PW %
            print_and_write(out_file, "PW %:")
            for i in range(len(nodes) - 1):
                pair = (nodes[i], nodes[i + 1])
                percent = len(all_pairwise_presence[pair]) / total_files
                print_and_write(out_file, f"      {nodes[i]} -> {nodes[i + 1]} : {percent:6.2%}")

            print_and_write(out_file, f"\nNetwork {idx} is found in files of directory: {input_dir}")
            print_and_write(out_file, f"Directory files: {', '.join(sorted(network_file_map[network]))}")
            #print_and_write(out_file, f"Directory files: {", ".join(sorted(network_file_map[network]))}")
            print_and_write(out_file, f"{'-' * 200}\n")
            #print_and_write(out_file, f"{"-" * 200}\n")

    print(f"Results saved to: {output_file}\n")

    # --- Create directory for network-specific PDB files ---
    #os.path.join("ms_pdb_output_hbonds_stats")
    #os.makedirs(output_dir, exist_ok=True)

    print(f"Preparing Network Specific PDBs and PSEs files...")
    for idx, (network, count) in enumerate(top_networks, start=1):
        pdb_residues = network.split(" -> ")
        print(f"Network {idx}: Count {count}: {pdb_residues}")
        found_files = sorted(network_file_map[network])

        for found_file in found_files:
            ms_id = found_file.split("Paths_ms_pdb_")[1].split("_hah_Resi-EntryToExit.txt")[0]
            pdb_filename = f"ms_pdb_{ms_id}.pdb"

            print("\nFIX: directory 'ms_pdb_output' is hard-coded: should rather be 'input_dir' agument?\n")
            pdb_path = Path("ms_pdb_output").joinpath(pdb_filename)
            if not pdb_path.exists():
                print(f"Warning: Missing PDB file {pdb_path!s}")
                continue

            selected_lines = []
            with open(pdb_path) as pdb_file:
                pdb_lines = pdb_file.readlines()

            for res in pdb_residues:
                resname   = res[0:3]    # e.g., HOH or ARG
                coordinfo = res[5:14]   # e.g., A0534_003
                #print(f"resname: {resname}, coordinfo: {coordinfo}")

                for line in pdb_lines:
                    if line.startswith(("ATOM", "HETATM")):
                        line_resname = line[17:20].strip()
                        line_coord =   line[21:30].strip()
                        #print(f"line_resname: {line_resname}, line_coord: {line_coord}")
                        if line_resname == resname and line_coord == coordinfo:
                            selected_lines.append(line)

            output_pdb_name = f"Network{idx}_ms_pdb_{ms_id}.pdb"
            output_pdb_path = output_dir.joinpath(output_pdb_name)
            with open(output_pdb_path, "w") as out_pdb:
                out_pdb.writelines(selected_lines)

            # Full path to PSE file
            output_pse_path = str(output_dir.joinpath(f"Network{idx}_ms_pdb_{ms_id}.pse"))
            #convert_pdbs_to_pse(["prot.pdb", output_pdb_path], pse_name=output_pse_path)
            write_pymol_session_file(["prot.pdb", output_pdb_path], pse_name=output_pse_path)

    print("\nNetwork-specific PDBs and PSEs with objects (prot, NetworkID_ms_pdb_msID)",
          f"are saved in: {output_dir!s}\n")

    return


def hbstats_parser():
    parser = argparse.ArgumentParser(
        prog="stats_hbond_networks",
        description="Analyze network similarities/statistics across a directory of microstate associated network files."
    )
    parser.add_argument(
        "-i",
        type=str,
        default="ms_pdb_output_hbonds_nets",
        help="Directory containing network files (default: %(default)s)"
    )
    parser.add_argument(
        "-topnets",
        type=int,
        default=5,
        help="Number of top networks to display (default: %(default)s)"
    )
    parser.add_argument(
        "-node_min", 
        type=int,
        default=5,
        help="Minimum number of nodes for networks to be considered (default: %(default)s)"
    )
    parser.add_argument(
        "-A",
        type=float,
        default=10**13,
        help="Pre-exponential factor (default: %(default)s)"
    )
    parser.add_argument(
        "--gro",
        action="store_true",
        help="Enable computing of Grotthuss competent networks only."
    )

    return parser


def cli(argv=None):
    p = hbstats_parser()
    args = p.parse_args()

    if not Path("prot.pdb").exists():
        raise FileNotFoundError("The file 'prot.pdb' is missing.")

    input_dir = args.i.replace("nets", "gronets") if args.gro else args.i
    output_dir = Path(f"{input_dir}_stats")
    output_dir.mkdir(exist_ok=True)
    output_file_name = f"MCCE-ms_hbond_network_stats_nodemin{args.node_min}.txt"
    output_file_path = output_dir.joinpath(output_file_name)

    main(input_dir, args.topnets, args.node_min, args.A, output_file_path)

    return


if __name__ == "__main__":
    cli(sys.argv[:1])
