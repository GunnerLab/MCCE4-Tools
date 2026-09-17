#!/usr/bin/env python3

"""
Module: ms_hbnets_heatmaps.py
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Union

from mcce4.constants import res3_to_res1


try:
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Oops! Forgot to activate an appropriate environment?\n{e}")
    sys.exit(1)


OCC_MIN = 0.01
RESMAP_FIGSIZE = (6,6)  # uniq donors, uniq acceptors counts < 30
FIG_CO_OCCURRENCE_RES = "hb_res_co_occurrence.png"
FIG_CO_OCCURRENCE_BK = "hb_bk_co_occurrence.png"
RESMAP_TITLE = "Donors - Acceptors co-occurences"


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


def plot_heatmap(df, ax=None, fig=None):
    """Draw the heatmap on the provided Axis."""
    # get color map & boundaries
    n_resample = 10
    blu = mpl.colormaps["Blues"].resampled(n_resample)
    newcolors = blu(np.linspace(0, 1, n_resample))
    cmap = ListedColormap(newcolors, name="B10")
    bnorm = BoundaryNorm([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                          0.6, 0.7, 0.8, 0.9, 1.0], cmap.N)
    kws={'rasterized':True, 'norm':bnorm}
   
    # Draw the heatmap
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # Remove all the Axes spines
    despine(ax=ax)
    
    plot_data = df.values
    nR, nC = df.shape
    # Center the ticks in the middle of each box
    x = np.arange(nC) + 0.5
    y = np.arange(nR) + 0.5
    mesh = ax.pcolormesh(x, y, plot_data, cmap=cmap, **kws)

    # Set the axis limits
    ax.set(xlim=(0, nC), ylim=(0, nR))
    # Invert the y axis to show the plot in matrix form
    ax.invert_yaxis()

    # use labels from df
    ax.set_xticks(x, df.columns.tolist())
    ax.set_yticks(y, df.index.tolist(), rotation="horizontal", va="center")
    ax.tick_params(axis='both', direction='out', length=6) #, width=1.5)

    cb = ax.figure.colorbar(mesh,
                            ax=ax, 
                            boundaries=bnorm, 
                            orientation='vertical',
                            pad=0.02,    # closer to plot
                            shrink=0.8,  # Shrinks the height to 70% of the axis height
                            aspect=25,   # Higher number makes the width thinner (default is 20)
    )
    cb.outline.set_linewidth(0)
    cb.set_label('occ', rotation=270, labelpad=5, weight="bold")    
    # also rasterize the colorbar to avoid white lines on the PDF rendering
    cb.solids.set_rasterized(True)

    # Draw fig
    fig.canvas.draw()
    if fig.stale:
        try:
            fig.draw(fig.canvas.get_renderer())
        except AttributeError:
            pass

    # Possibly rotate them if they overlap
    xtl = ax.get_xticklabels()
    if axis_ticklabels_overlap(xtl):
        plt.setp(xtl, rotation="vertical")
    #if axis_ticklabels_overlap(ytl):
    #    plt.setp(ytl, rotation="horizontal")

    # Add the axis labels
    ax.set(xlabel=df.columns.name, ylabel=df.index.name)
    ax.set_aspect("equal")

    return ax


def get_resid(confid:str) -> str:
    id1 = res3_to_res1.get(confid[:3], confid[:3])
    return f"{id1}_" + confid[5] + str(int(confid[6:-4]))


def convert_pairs_res_with_bk(pairs_fp: Path):
    pheh = pairs_fp.name.rsplit("_", maxsplit=1)[1]
    new_hb_respairs_fp = pairs_fp.with_name("hb_pairs_res_" + pheh)
    
    pairs_df = pd.read_csv(pairs_fp)
    pairs_df[["res_d","res_a"]] = pairs_df.apply(
        lambda x: pd.Series([get_resid(x["donor"]), get_resid(x["acceptor"])]),
        axis=1)
    
    # add with_bk flag:
    pairs_df["with_bk"] = pairs_df.apply(
        lambda row: row["donor"][3:5]=="BK" or row["acceptor"][3:5]=="BK",
        axis=1)

    # collapse confs to res
    respairs_df = pairs_df.groupby(["res_d","res_a"],
                                   as_index=False).agg({"count": "max", "occ": "max", "with_bk": "max"})
    respairs_df["count"] = respairs_df["count"].astype("int32")
    respairs_df = respairs_df.sort_values(by="count", ascending=False)
    respairs_df.to_csv(new_hb_respairs_fp, index=False)

    return


def get_resnum(val: str):
    """Get the res num from the res id in hb_pairs_res_pH*.csv file.
    Examples:
      HOH_W123 -> 123; _S1_O1 ->   1
    """
    if val.startswith("_"):
        return int(val[1:].split("_")[1][1:])
    else:
        return int(val.split("_")[1][1:])


def heat_map_from_df(df: pd.DataFrame,
                     df_uniq_donors: list,
                     df_uniq_acceptors: list,
                     fig_size: tuple,
                     fig_save_fp: Path=None,
                     title: str = RESMAP_TITLE):

    matrix_df = df.pivot(index="Donor", columns="Acceptor", values="occ").fillna(0)
    matrix_df = matrix_df.reindex(index=df_uniq_donors, columns=df_uniq_acceptors)
    # save as hb_pairs.cooccurrence_matrix.csv?

    fig, ax = plt.subplots(1,1, figsize=fig_size, layout='constrained')
    plot_heatmap(matrix_df, ax=ax, fig=fig)
    ax.set_title(title, fontdict={'weight':'bold', "size":10});
    if fig_save_fp is not None:
        plt.savefig(fig_save_fp)
        print(f"Map saved to: {fig_save_fp!s}")

    plt.close()

    return


def get_res_heatmap(pairs_res_csv: Path,
                    occ_cutoff: float = OCC_MIN,
                    fig_size: tuple = RESMAP_FIGSIZE,
                   ):

    res_pairs_df = pd.read_csv(pairs_res_csv)
    # check format:
    if "with_bk" not in res_pairs_df.columns:
        print(f" Converting {pairs_res_csv.name} to new format (+ bk flag).")
        pairs_fp = pairs_res_csv.with_name(pairs_res_csv.name.replace("res_",""))
        convert_pairs_res_with_bk(pairs_fp)
        # reload:
        res_pairs_df = pd.read_csv(pairs_res_csv)

    # 1. apply occ mask:
    occ_msk = res_pairs_df["occ"].ge(occ_cutoff)
    if occ_msk.any():
        res_pairs_df = res_pairs_df.loc[occ_msk]
    else:
        print(f"No res hb-pairs data at this occ cutoff: {occ_cutoff:.2f}")
        return

    # 2. get res num for sorting:
    res_pairs_df["di"] = res_pairs_df["res_d"].apply(get_resnum)
    res_pairs_df["ai"] = res_pairs_df["res_a"].apply(get_resnum)
    res_pairs_df = res_pairs_df.sort_values(by=["di","ai"])
    res_pairs_df = res_pairs_df.drop(columns=["di","ai"])
    res_pairs_df = res_pairs_df.rename(columns={"res_d":"Donor",
                                                "res_a":"Acceptor"})

    # 3. split res-res and res-bk pairs: up to 2 maps
    res_msk = res_pairs_df["with_bk"].eq(False)
    res_respairs_df =  res_pairs_df.loc[res_msk, ["Donor", "Acceptor", "count", "occ"]]
    # plot:
    if res_respairs_df.shape[0]:
        uniq_donors = res_respairs_df["Donor"].unique()
        n_res_ud = len(uniq_donors)
        uniq_acceptors = res_respairs_df["Acceptor"].unique()
        n_res_ua = len(uniq_acceptors)

        print(f" res_respairs, uniq donors: {n_res_ud}, uniq acceptors: {n_res_ua}", sep="\n")
        heat_map_from_df(res_respairs_df,
                         uniq_donors,
                         uniq_acceptors,
                         fig_size=fig_size,
                         fig_save_fp=pairs_res_csv.with_name(FIG_CO_OCCURRENCE_RES),
                         title=f"Residues {RESMAP_TITLE}\n({pairs_res_csv.name})"
                        )
    else:
        print("No hb pairs of residue-residue kind.")
    
    bk_respairs_df = res_pairs_df.loc[~res_msk, ["Donor", "Acceptor", "count", "occ"]]
    # plot:
    if bk_respairs_df.shape[0]:
        uniq_donors = bk_respairs_df["Donor"].unique()
        n_bk_ud = len(uniq_donors)
        uniq_acceptors = bk_respairs_df["Acceptor"].unique()
        n_bk_ua = len(uniq_acceptors)
        print(f" bk_respairs, uniq donors: {n_bk_ud} uniq acceptors: {n_bk_ua}", sep="\n")
        bk_size = fig_size[0] + 3, fig_size[1] + 3
        heat_map_from_df(bk_respairs_df,
                         uniq_donors,
                         uniq_acceptors,
                         fig_size=bk_size,
                         fig_save_fp=pairs_res_csv.with_name(FIG_CO_OCCURRENCE_BK),
                         title=f"Backbone {RESMAP_TITLE}\n({pairs_res_csv.name})"
                        )
    else:
        print("No hb pairs of residue-backbone kind.")
    return

    
def cli_parser():
    def parse_tuple(arg):
        # Splits the string by commas and converts each element to an integer
        return tuple(map(int, arg.split(',')))

    p = ArgumentParser(prog="ms_hbnets_heatmaps",
        description="""
Produces heatmaps from 'ms_hbnets' output files. Heatmaps implemented:
 - Residue donor/acceptor pairs co-occurrence map (visualization of hb_pairs_res_pH*.csv file).
""",
    usage="""ms_hbnets_heatmaps
       ms_hbnets_heatmaps -mcce-dir <dirpath>
       ms_hbnets_heatmaps -ph 5
       ms_hbnets_heatmaps -fig-size 8,8
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    p.add_argument("-mcce-dir",
                    default=".",
                    type=str,
                    help="MCCE run directory; Default: %(default)s",
                    )
    p.add_argument("-map-kind",
                    default="co_occurrence",
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
                   help="""
    Figure size for the residue donor/acceptor co-occurrence heatmap;
    The default size is ok for counts of uniq donors, acceptors < 30; Default: %(default)s
    """
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

    mcce_dir = Path(args.mcce_dir).resolve()

    if args.map_kind != "co_occurrence":
        print("Only one kind of heatmap is currently implemented: keep the default value.")
        return

    # Look for the file in the calling dir: hb_pairs_res_pH7.00eH0.00.csv
    # using the most likely format first (floats):
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
