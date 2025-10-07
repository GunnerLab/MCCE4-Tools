#!/usr/bin/env python3

"""
Module: plots.py

  Plots for Protonation microstate analysis with weighted correlation.
"""
import logging
from pathlib import Path
import sys
from typing import Tuple

logger = logging.getLogger(__name__)
try:
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import skewnorm
except ImportError as e:
    logger.critical("Oops! Forgot to activate an appropriate environment?\n", exc_info=e)
    sys.exit(1)


plt.ioff()


HEATMAP_SIZE = (20, 8)


def energy_distribution(
    input_arr: np.ndarray,
    out_dir: Path,
    kind: str,
    mc_method: str = "MONTERUNS",
    save_name: str = "enthalpy_dist.png",
    show: bool = False,
    fig_size=(7,7),
):
    """
    Plot the histogram and distribution fit of the conformer or charge microstates energies.
    Arguments:
      - input_arr (np.ndarray): Array of conformer or charge microstates data returned by MSout_np.
      - kind (str): Input array ms kind, either 'ms' (conformer ms) or 'cms' (charge microstates)
    """
    kind = kind.lower()
    if kind not in ("ms", "cms"):
        raise ValueError("Argument 'kind' must be either 'ms' or 'cms'.")
    kind_lbl = "Conformer" if kind == "ms" else "Charge"

    if mc_method == "MONTERUNS":
        if kind == "cms":
            # sort by average energy
            energies = np.array(sorted(input_arr[:, -2]), dtype=float)
        else:
            energies = np.array(sorted(input_arr[:, -1]), dtype=float)
    else:
        if kind == "cms":
            # sort by energy
            energies = np.array(sorted(input_arr[:, -3]), dtype=float)
        else:
            energies = np.array(sorted(input_arr[:, -2]), dtype=float)

    skewness, mean, std = skewnorm.fit(energies)
    y = skewnorm.pdf(energies, skewness, mean, std)

    fig = plt.figure(figsize=fig_size)
    fs = 12  # fontsize

    graph_hist = plt.hist(energies, bins=100, alpha=0.6)
    Y = graph_hist[0]
    pdf_data = Y.max() / max(y) * y
    plt.plot(energies, pdf_data, label="approx. skewnorm", color="indigo")
    plt.title(f"{skewness= :.2f} {mean= :.2f} {std= :.2f}", fontsize=fs)
    plt.xlabel(f"{kind_lbl} Microstate Energy (Kcal/mol)", fontsize=fs)
    plt.ylabel("Count", fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.tick_params(axis="x", direction="out", length=8, width=2)
    plt.tick_params(axis="y", direction="out", length=8, width=2)
    plt.legend()
    fig_fp = out_dir.joinpath(save_name)
    fig.savefig(fig_fp, dpi=300, bbox_inches="tight")
    print(f"Microstate energy distribution figure saved: {fig_fp!s}")

    if show:
        plt.show()
    else:
        plt.close()

    return


def crgms_energy_histogram(
    top_cms: list,
    background_crg: int,
    fig_title: str,
    out_dir: Path,
    save_name: str,
    show: bool = False,
):
    """Plot charge microstates average energies vs the state protein charge,
    along with a marginal histogram.
    """
    state_idx = 1 if len(top_cms[0]) == 6 else 0
    data = np.array([[sum(arr[state_idx]) + background_crg, arr[-1]] for arr in top_cms])
    net_crg = data[:, 0]

    fs = 12  # font size for axes labels and title
    
    g1 = sns.JointGrid(marginal_ticks=True, height=6)
    ax = sns.scatterplot(
        x=net_crg,
        y=data[:, 1],
        size=data[:, 1],
        legend="brief",
        ax=g1.ax_joint,
    )
    plt.yscale("log")

    ax.set_xticks(range(int(min(net_crg)), int(max(net_crg)) + 1))
    ax.set_xlabel("Charge", fontsize=fs)
    ax.set_ylabel("log$_{10}$(Count)", fontsize=fs)
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0.0)

    ax2 = sns.histplot(x=net_crg, linewidth=2, discrete=True, ax=g1.ax_marg_x)
    ax2.set_ylabel(None, fontsize=fs)
    g1.ax_marg_y.set_axis_off()
    g1.fig.subplots_adjust(top=0.9)
    if fig_title:
        g1.fig.suptitle(fig_title, fontsize=fs)
    fig_fp = out_dir.joinpath(save_name)
    g1.savefig(fig_fp, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {fig_fp}")

    if show:
        plt.show()
    else:
        plt.close()

    return


def corr_heatmap(
    df_corr: pd.DataFrame,
    out_dir: Path = None,
    save_name: str = "corr.png",
    check_allzeros: bool = True,
    show: bool = False,
    lower_tri=False,
    fig_size: Tuple[float, float] = HEATMAP_SIZE,
):
    """Produce a heatmap from a correlation matrix.
    Args:
     - df_corr (pd.DataFrame): Correlation matrix as a pandas.DataFrame,
     - out_dir (Path, None): Output directory for saving the figure,
     - save_name (str, "corr.png"): The name of the output file,
     - check_allzeros (bool, True): Notify if all off-diagonal entries are 0,
     - show (bool, False): Whether to display the figure.
     - lower_tri (bool, False): Return only the lower triangular matrix,
     - figsize (float 2-tuple, (25,8)): figure size in inches, (width, height).
     Note:
      If check_allzeros is True & the check returns True, there is no plotting.
    """
    df_corr = df_corr.round(2)
    if check_allzeros:
        corr_array = df_corr.values
        # Create a mask for off-diagonal elements
        off_diag_mask = ~np.eye(corr_array.shape[0], dtype=bool)
        # Check if all off-diagonal elements are zero
        if np.all(corr_array[off_diag_mask] == 0):
            logging.warning("All off-diagonal correlation values are 0.00: not plotting.")
            return

    if df_corr.shape[0] > 14 and fig_size == HEATMAP_SIZE:
        logger.warning(
            ("With a matrix size > 14 x 14, the fig_size argument" f" should be > {HEATMAP_SIZE}.")
        )

    n_resample = 8
    top = mpl.colormaps["Reds_r"].resampled(n_resample)
    bottom = mpl.colormaps["Blues"].resampled(n_resample)
    newcolors = np.vstack((top(np.linspace(0, 1, n_resample)), bottom(np.linspace(0, 1, n_resample))))
    cmap = ListedColormap(newcolors, name="RB")
    norm = BoundaryNorm([-1.0, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1.0], cmap.N)

    if lower_tri:
        # mask to get lower triangular matrix w diagonal:
        msk = np.triu(np.ones_like(df_corr), k=1)
    else:
        msk = None

    fig = plt.figure(figsize=fig_size)

    fs = 12  # font size
    ax = sns.heatmap(
        df_corr,
        mask=msk,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        norm=norm,
        square=True,
        linecolor="white",
        linewidths=0.01,
        fmt=".2f",
        annot=True,
        annot_kws={"fontsize": 10},
    )
    ax.set(xlabel="", ylabel="")
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)  # rotation=90)
    plt.tight_layout()

    if save_name:
        if out_dir is not None:
            fig_fp = out_dir.joinpath(save_name)
        else:
            fig_fp = Path(save_name)

        fig.savefig(fig_fp, dpi=300, bbox_inches="tight")
        logger.info(f"Correlation heat map saved as {fig_fp}")

    if show:
        plt.show()
    else:
        plt.close()

    return
