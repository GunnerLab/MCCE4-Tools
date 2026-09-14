#!/usr/bin/env python3

"""
Module: ms_hbnets_heatmaps.py
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Union

try:
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Oops! Forgot to activate an appropriate environment?\n{e}")
    sys.exit(1)


def despine(ax = None, which=['top','right']):
    """which ([str])): 'left','top','right','bottom'."""
    if ax is None:
        ax = plt.gca()
    for side in which:
        ax.spines[side].set_visible(False)
    return


def axis_ticklabels_overlap(labels: list) -> bool:
    """Return a boolean for whether the list of ticklabels have overlaps.
    """
    if not labels:
        return False
    try:
        bboxes = [lb.get_window_extent() for lb in labels]
        overlaps = [bx.count_overlaps(bboxes) for bx in bboxes]
        return max(overlaps) > 1
    except RuntimeError:
        # Issue on macos backend raises an error in the above code
        return False


def plot_heatmap(df, cmap, kws, ax=None, fig=None):
    """Draw the heatmap on the provided Axes."""
    # Remove all the Axes spines
    despine(ax=ax)
    
    plot_data = df.values
    vmin, vmax = plot_data.min(), plot_data.max()
    # setting vmin/vmax in addition to norm is deprecated
    # so avoid setting if norm is set
    if kws.get("norm") is None:
        kws.setdefault("vmin", vmin)
        kws.setdefault("vmax", vmax)
   
    # Draw the heatmap
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    mesh = ax.pcolormesh(plot_data, cmap=cmap, **kws)

    # Set the axis limits
    ax.set(xlim=(0, df.shape[1]), ylim=(0, df.shape[0]))
    # Invert the y axis to show the plot in matrix form
    ax.invert_yaxis()

    # Center the ticks in the middle of each box
    xticklabels = df.columns.tolist()
    yticklabels = df.index.tolist()
    ax.set_xticks(np.arange(df.shape[1]) + 0.5)
    ax.set_yticks(np.arange(df.shape[0]) + 0.5)
    ax.tick_params(axis='both', direction='out', length=6) #, width=1.5)

    xtl = ax.set_xticklabels(xticklabels)
    ytl = ax.set_yticklabels(yticklabels, rotation="vertical")
    plt.setp(ytl, va="center")  # GH2484

    norm = kws.get("norm")
    cb = ax.figure.colorbar(mesh,
                            ax=ax, 
                            boundaries=norm, 
                            orientation='vertical',
                            pad=0.02,    # closer to plot
                            shrink=0.8,  # Shrinks the height to 70% of the axis height
                            aspect=25,   # Higher number makes the width thinner (default is 20)
    )
    cb.outline.set_linewidth(0)
    cb.set_label('occ', rotation=270, labelpad=5, weight="bold")    
    # If rasterized is passed to pcolormesh, also rasterize the
    # colorbar to avoid white lines on the PDF rendering
    if kws.get('rasterized', False):
        cb.solids.set_rasterized(True)

    # Draw fig
    fig.canvas.draw()
    if fig.stale:
        try:
            fig.draw(fig.canvas.get_renderer())
        except AttributeError:
            pass

    # Possibly rotate them if they overlap
    if axis_ticklabels_overlap(xtl):
        plt.setp(xtl, rotation="vertical")
    if axis_ticklabels_overlap(ytl):
        plt.setp(ytl, rotation="horizontal")

    # Add the axis labels
    ax.set(xlabel=df.columns.name, ylabel=df.index.name)
    ax.set_aspect("equal")

    return ax


OCC_MIN = 0.01
RESMAP_FIGSIZE = (12,10)


def get_res_heatmap(pairs_res_csv: Path,
                    occ_cutoff: float = OCC_MIN,
                    fig_size: tuple = RESMAP_FIGSIZE,
                    save_as: str = "res_da_co_occurence.png",
                   ):

    res_pairs_df = pd.read_csv(pairs_res_csv)
    msk = res_pairs_df["occ"].ge(occ_cutoff)
    res_pairs_df = res_pairs_df.loc[msk]
    
    def get_resnum(val: str) -> int:
        return int(val.split("_")[1][1:])
        
    res_pairs_df["di"] = res_pairs_df["res_d"].apply(get_resnum)
    res_pairs_df["ai"] = res_pairs_df["res_a"].apply(get_resnum)
    res_pairs_df = res_pairs_df.sort_values(by=["di","ai"])
    res_pairs_df = res_pairs_df.rename(columns={"res_d":"Donor",
                                                "res_a":"Acceptor"})

    matrix_df = res_pairs_df.pivot(index="Donor",
                                   columns="Acceptor",
                                   values="occ").fillna(0)
    matrix_df = matrix_df.reindex(index=res_pairs_df["Donor"].unique(),
                                  columns=res_pairs_df["Acceptor"].unique()
                                  )
    #n_rows, n_cols = matrix_df.shape
    n_resample = 10
    blu = mpl.colormaps["Blues"].resampled(n_resample)
    newcolors = blu(np.linspace(0, 1, n_resample))
    cmap = ListedColormap(newcolors, name="B")
    kws={'rasterized':True,
         'norm':BoundaryNorm([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0],
                             cmap.N)
        }

    fig, ax = plt.subplots(1, 1, figsize=fig_size, layout='constrained')
    plot_heatmap(matrix_df, cmap, kws, ax=ax, fig=fig)
    ax.set_title("Donors - Acceptors co-occurences", fontdict={'weight':'bold'});
    plt.savefig(pairs_res_csv.with_name(save_as))

    plt.show()


def cli_parser():
    def parse_tuple(arg):
        # Splits the string by commas and converts each element to an integer
        return tuple(map(int, arg.split(',')))

    p = ArgumentParser(prog="ms_hbnets_heatmaps",
        description="""
Gather the H-bonding conformer pairs and states occupancies 
from the microstates file given a mcce dir, pH & Eh.""",
    usage="""ms_hbnets_heatmaps
       ms_hbnets_heatmaps -mcce-dir <dirpath>
       ms_hbnets_heatmaps -ph 5
       ms_hbnets_heatmaps -fig-size 12,12
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    p.add_argument("-mcce-dir",
                    default=".",
                    type=str,
                    help="MCCE run directory; Default: %(default)s",
                    )
    p.add_argument("-map-kind",
                    default="co_occurence",
                    type=str,
                    help="Kind of heatmap (currently only one choice); Default: %(default)s",
                    )
    # ph, eh: as strings to easily determine the precision
    p.add_argument("-ph",
                    default=7,
                    type=float,
                    help="Titration pH; Default: %(default)s"
                    )
    p.add_argument("-eh",
                    default=0,
                    type=float,
                    help="Titration Eh; Default: %(default)s"
                    )
    p.add_argument("-fig-size",
                   type=parse_tuple,
                   default=f"{RESMAP_FIGSIZE[0]},{RESMAP_FIGSIZE[1]}",
                   help="Figure size for the donor/acceptor co-occurence heatmap; Default: %(default)s"
                    )
    p.add_argument("-occ-min",
                   type=float,
                   default=OCC_MIN,
                   help="Minimum occupancy to retain; Default: %(default)s"
                    )
    return p


def cli(argv=None):
    p = cli_parser()
    args = p.parse_args(argv)
    # print(f" cli args = \n{args}\n")

    mcce_dir = Path(args.mcce_dir).resolve()

    if args.map_kind != "co_occurence":
        print("Only one kind of heatmap is currently implemented: keep the default value.")
        return

    # look for the file in the calling dir: hb_pairs_res_pH7.00eH0.00.csv
    # most likely format: float:
    pairs_res_csv = mcce_dir.joinpath(f"hb_pairs_res_pH{args.ph:.2f}eH{args.eh:.2f}.csv")
    if not pairs_res_csv.exists():
        pairs_res_csv = mcce_dir.joinpath(f"hb_pairs_res_pH{args.ph:.0f}eH{args.eh:.0f}.csv")
        if not pairs_res_csv.exists():
            print(f"Found no csv file starting with 'hb_pairs_res' for the given ph, eh in {mcce_dir!s}.")
            return

    get_res_heatmap(pairs_res_csv,
                    occ_cutoff=args.occ_min,
                    fig_size=args.fig_size,
    )

    return
