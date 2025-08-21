#!/usr/bin/env python
"""
Module: ms_plot_energies.py

Plot the microstate energies from microstates PDB files created with `ms_sampled_ms_to_pdbs.py`.

Created on Apr 01 09:00:00 2025

@author: Gehan Ranepura
"""
import argparse
import logging
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_microstate_energies(pdb_dir: Path):
    """Extracts microstate energies from microstates PDB files created with
    `ms_sampled_ms_to_pdbs.py` in the given directory and counts the files.
    """
    energy_pattern = re.compile(r"REMARK 250\s+ENERGY\s+:\s+(-?\d+\.\d+)")
    energies = []
    pdb_count = 0  # To count the number of PDB files processed

    for pdb in Path(pdb_dir).glob("*.pdb"):
        pdb_count += 1
        with open(pdb) as file:
            for line in file:
                match = energy_pattern.search(line)
                if match:
                    energies.append(float(match.group(1)))
                    break  # Stop after finding the first energy in the file

    return np.array(energies), pdb_count


def plot_microstate_energy_histogram(
    energies,
    out_dir: Path,
    pdb_count: int,
    save_name="enthalpy_dist.png",
    show=True,
    fig_size=(8, 8),
):
    """Plots a histogram of microstate energies with a fitted skewed normal distribution."""
    if len(energies) == 0:
        logger.warning("No energies to plot.")
        return

    # Fit skewed normal distribution
    skewness, mean, std = skewnorm.fit(energies)

    # Set up figure
    fig, ax = plt.subplots(figsize=fig_size)
    fs = 14  # Font size

    # Histogram: Count instead of density
    graph_hist = ax.hist(energies, bins=100, alpha=0.6, edgecolor="black")

    # Fitted skew normal distribution
    x_range = np.linspace(min(energies), max(energies), 1000)
    y_fit = skewnorm.pdf(x_range, skewness, mean, std)
    y_fit_scaled = (
        y_fit * max(graph_hist[0]) / max(y_fit)
    )  # Scale to match histogram height
    ax.plot(x_range, y_fit_scaled, label="Fitted SkewNorm", color="k")

    # Labels and styling
    ax.set_title(
        f"Skewness: {skewness:.2f}, Mean: {mean:.2f}, Std Dev: {std:.2f}, Files: {pdb_count}",
        fontsize=fs,
    )
    ax.set_xlabel("Microstate Energy (kcal/mol)", fontsize=fs)
    ax.set_ylabel("Count", fontsize=fs)
    ax.tick_params(axis="x", direction="out", length=8, width=2)
    ax.tick_params(axis="y", direction="out", length=8, width=2)
    ax.legend()

    # Save and show
    fig_fp = out_dir / save_name
    fig.savefig(fig_fp, dpi=300, bbox_inches="tight")
    logger.info(f"Histogram figure saved as {fig_fp}")

    if show:
        plt.show()  # Show the plot interactively
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot microstate energy distribution from MCCE PDB files."
    )
    parser.add_argument(
        "pdb_dir",
        nargs="?",
        default="ms_pdb_output",
        type=Path,
        help="Directory containing PDB files (default: pdb_output_mc)",
    )
    parser.add_argument(
        "-out_dir",
        default=".",
        type=Path,
        help="Directory to save the plot (default: current directory)",
    )
    parser.add_argument(
        "--noshow", action="store_true", help="Do not show plot, only save"
    )

    args = parser.parse_args()

    if not args.pdb_dir.exists():
        logger.error(f"Error: Directory '{args.pdb_dir}' does not exist.")
    else:
        energies, pdb_count = parse_microstate_energies(args.pdb_dir)
        if len(energies) == 0:
            logger.warning("No energies found in the specified directory.")
        else:
            # By default, show the plot. Use --noshow to suppress it.
            plot_microstate_energy_histogram(
                energies, args.out_dir, pdb_count, show=not args.noshow
            )
