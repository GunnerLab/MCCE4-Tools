#!/usr/bin/env python3

"""
Module: md_clustering.py

"""
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
import sys
from typing import Union
import warnings
warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
except ModuleNotFoundError:
    sys.exit(("Package `MDAnalysis` is required in this module.\n"
              "Activate an environment that includes it & try again."))

import MDAnalysis.analysis.encore as mda_encore



def example():
    text = """
    Given:
    ```
    > select_res="(backbone and (resid 22 173 194 226) and segid NDHA) or (backbone and (resid 87) and segid NDHC)"
    ```
    Commands:
    ```
    > cd my_md_files_dir  # optional
    > md_clustering some.psf traj1.dcd traj2.dcd -labels DCD1 DCD2 -selection $select_res
    ```
    Outcome:
    The pdbs of the cluster elements will be saved in the topology file's parent directory.
    """
    print(text)


def output_cluster_elements(args: Union[dict, Namespace]):
    """Main function.
    """
    if isinstance(args, dict):
        args = Namespace(**args)

    top_fp = Path(args.topology_file)
    if not top_fp.exists():
        sys.exit(f"\nTopology file not found: {str(top_fp)}")

    n_trajs = len(args.trajectory_files)
    n_labels = len(args.labels)

    if len(args.trajectory_files) > 1:
        # validate labels if any:
        if n_labels and n_labels != n_trajs:
            sys.exit("\nThe number of labels must match the number of trajectories.")
    # check traj files found:
    for traj in args.trajectory_files:
        if not Path(traj).exists():
            sys.exit(f"\nTrajectory file not found: {traj}")

    out_dir = top_fp.parent
    universes = []
    cluster_collections = []

    for i, traj in enumerate(args.trajectory_files):
        uni = mda.Universe(str(top_fp), traj)
        universes.append(uni)

        traj_lbl = args.labels[i]
        print(f"{traj_lbl} universe: ", uni, len(uni.trajectory))

        if args.selection:
            clui = mda_encore.cluster(uni, selection=args.selection)
        else:
            clui = mda_encore.cluster(uni)

        print(f"{traj_lbl} clusters: ", clui)
        cluster_collections.append(clui)

    for i, cluster_sets in enumerate(cluster_collections):
        traj_lbl = args.labels[i]
        clust_name = out_dir.joinpath(f"cluster_info_{traj_lbl}.txt")

        with open(clust_name, "w") as fo:
            fo.write(f"\n{traj_lbl}:\n")
            for cluster in cluster_sets:
                line = (f"{cluster.id}:: centroid: {cluster.centroid}, "
                        f"size: {cluster.size:,}, elements: {cluster.elements}\n")
                print(line)
                fo.write(line)
                for elem in cluster.elements:
                    universes[i].trajectory[elem]
                    pdb_fp = out_dir.joinpath(f"{traj_lbl}_clus_{cluster.id}_{elem}_frame.pdb")
                    with mda.Writer(pdb_fp) as pdb:
                        pdb.write(universes[i])

    print("Cluster analysis over.")

    return


DESC = "Cluster one or more MD trajectories and\nreturn the elements in each cluster as pdb files."
USAGE = "\n> md_clustering some.psf trj1.dcd trj2.dcd -labels DCD1 DCD2 -selection my_selection"


def cli_parser():
    p = ArgumentParser(prog="md_clustering",
                       description=DESC,
                       formatter_class=RawDescriptionHelpFormatter,
                       usage=USAGE,
    )
    p.add_argument(
        "topology_file",
        type=str,
        help="Topology filename/path."
        )
    p.add_argument(
        "trajectory_files",
        nargs="+",
        help="Trajectory filepath or list thereof."
        )
    p.add_argument(
        "-labels",
        nargs="+",
        help="Labels for the cluster info to return."
        )
    p.add_argument(
        "-selection",
        required=False,
        default="",
        help="Atom selection over all the trajectories."
        )
    p.add_argument(
        "--show_example",
        default=False,
        action="store_true",
        help="Just show a call example.")

    return p


def cli(argv=None):
    p = cli_parser()
    args = p.parse_args(argv)
    if args.show_example:
        example()
    else:
        output_cluster_elements(args)

    return


if __name__ == "__main__":
    cli(sys.argv[:1])
