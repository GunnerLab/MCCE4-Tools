 #!/usr/bin/env python3

"""
Module: md_extract_frames.py

Purpose:
  Extract a trajectory's frames with the given indices into pdb files with options
  to select segment and cofactors, and print the structure segments parsed by MDAnalysis.

Usage:
   md_extract_frames topo_file traj_file --output_segments
   md_extract_frames topo_file traj_file -frames_indices 10 20 30
   md_extract_frames topo_file traj_file -frames_indices 10 -select_segments NDHA NDHC
   md_extract_frames topo_file traj_file -frames_indices 10 -select_cofactors HEM

"""
import argparse
from pathlib import Path
import sys
from typing import List
import warnings

import numpy as np

try:
    import MDAnalysis as mda
except ModuleNotFoundError:
    msg = """
    Package `MDAnalysis` is required in this module. Activate an environment that includes it & try again."
    To install, activate a conda environment & run: `conda install -c conda-forge mdanalysis`
    """
    sys.exit(msg)


# suppress some MDAnalysis warnings about PSF files
warnings.filterwarnings("ignore")


def validate_selected_frames(frames_indices: List[int], n_frames: int) -> list:
    """Validate selected frames_indices viz n_frames, the total number of frames.

    Args:
    -----
    frames_indices (list of ints): selected frames indices.
    n_frames (int): Total number of frames in the trajectory.

    Returns:
    --------
    The list of frame_indices excluding out-of-bound indices if found.

    Note:
    -----
    Selected frame_indices are assumed to be sorted ascendingly.
    """
    arr = np.array(frames_indices)
    if arr[arr.argmax()] > n_frames:
        frames_within_range = list(np.where(arr < n_frames, arr, -1))
        bad_idx = frames_within_range.index(-1)
        return frames_indices[:bad_idx]

    return frames_indices


def extract_mdtrajectory_frames(
    topology_fp: str,
    trajectory_fp: str,
    frames_indices: list,
    select_segments: list = None,
    select_cofactors: list = None,
    output_segments: bool = False,
):

    top_fp = Path(topology_fp).resolve()
    if not top_fp.exists():
        sys.exit(f"File not found: {topology_fp!r}")
    if not Path(trajectory_fp).resolve().exists():
        sys.exit(f"File not found: {trajectory_fp!r}")

    try:
        u = mda.Universe(topology_fp, trajectory_fp)
        traj_frames = u.trajectory.n_frames
        print(
            "Trajectory info:",
            f" - Frames  : {traj_frames}",
            f" - Segments: {u.segments}",
            f" - Residues: {u.residues}",
            f" - Atoms   : {u.atoms}",
            sep="\n",
        )
        if output_segments:
            for seg in u.segments:
                print(seg.segid)

    except ValueError:
        print(
            "MDAnalysis could not parse the file(s).",
            "If files are PSF & DCD run this command to try to fix the psf file:",
            f"  md_fix_psf_mdanalysis {topology_fp} {trajectory_fp}",
            sep="\n",
        )
        sys.exit(1)

    if not frames_indices:
        print("No frame indices given.")
        sys.exit(1)
    else:
        selected_frames = validate_selected_frames(frames_indices, traj_frames)
        if not selected_frames:
            print("Invalid frame indices.")
            sys.exit(1)

    selection_str = ""
    has_seg = False

    has_selection = select_segments is not None or select_cofactors is not None
    if has_selection:
        # build the user selection str:
        # ex: "segid NDHA NDHC NDHG NDHE NDHL NDHH or resname PL9"
        if select_segments is not None:
            has_seg = True
            selection_str = "segid " + " ".join(s for s in select_segments)
        if select_cofactors is not None:
            if has_seg:
                selection_str = (
                    selection_str
                    + " or resname "
                    + " ".join(s for s in select_cofactors)
                )
            else:
                selection_str = "resname " + " ".join(s for s in select_cofactors)

    info_msg = (
        "MD trajectory frames extraction.\n"
        f"- Trajectory: {trajectory_fp!r}\n"
        f"- Frames to extract: {len(selected_frames)}\n"
    )
    if selection_str:
        info_msg += f"- Selections: {selection_str!r}\n"
    else:
        info_msg += "- Selection: 'protein'\n"
    print(info_msg)

    # Output frames
    traj_dir = top_fp.parent
    for frame in selected_frames:
        output_pdb = traj_dir.joinpath(f"frame{frame}.pdb")
        u.trajectory[frame]
        if has_selection:
            try:
                subunits = u.select_atoms(selection_str)
            except Exception:
                print("Failed on `u.select_atoms`: bad selection?")
                print(sys.exc_info()[1])
                sys.exit(1)
        else:
            subunits = u.select_atoms("protein")

        subunits.write(filename=output_pdb, frames=frame)
        print(f"Frame {frame} exracted into {output_pdb}.")

    return


def cli_parser():
    p = argparse.ArgumentParser(
        description="Extract the MD trajectory frames with the given indices into pdb files.",
        add_help=True,
        usage=(
            "\n md_extract_frames topo_file traj_file --output_segments"
            " md_extract_frames topo_file traj_file -frames_indices 10 20 30\n"
            " md_extract_frames topo_file traj_file -frames_indices 10 -select_segments NDHA NDHC\n"
            " md_extract_frames topo_file traj_file -frames_indices 10 -select_cofactors HEM\n"
        ),
    )
    p.add_argument("topology_file", type=str, help="Topology filename/path.")
    p.add_argument("trajectory_file", type=str, help="Trajectory filename/path.")
    p.add_argument(
        "-frames_indices",
        required=True,
        type=int,
        nargs="+",
        help="Indices of the trajectory frames to return.",
    )
    p.add_argument(
        "-select_segments",
        required=False,
        nargs="+",
        help="Molecule segments to return."
    )
    p.add_argument(
        "-select_cofactors",
        required=False,
        nargs="+",
        help="Cofactors to return."
    )
    p.add_argument(
        "--output_segments",
        default=False,
        action="store_true",
        help="Print the structure segments parsed by MDAnalysis.",
    )
    return p


def cli(argv=None):
    p = cli_parser()
    args = p.parse_args(argv)
    # list of ints from cli is still list of strings:
    frames_idx = sorted([int(i) for i in args.frames_indices])

    extract_mdtrajectory_frames(
        args.topology_file,
        args.trajectory_file,
        frames_idx,
        args.select_segments,
        args.select_cofactors,
        args.output_segments,
    )

    return


if __name__ == "__main__":
    cli(sys.argv[:1])
