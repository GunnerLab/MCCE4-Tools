#!/usr/bin/env python3

"""
Module: plotting.py

Codebase: mcce4/ms_hbnets/plotting.py
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FIGSIZE = (6,6)  # uniq donors, uniq acceptors counts < 30


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


COLOR_GRADIENTS = ['Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys',
                   'OrRd', 'Oranges', 'PuBu', 'PuBuGn', 'PuRd', 'Purples',
                   'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']


def get_cmap_bnorm(map_kind: str = "data",
                   min_bound: float = 0.0,
                   color: str = "Blues") -> tuple:
    """
    Return the ListedColormap and BoundaryNorm objects depending
    on map_kind and min_bound and color with the later two arguments
    only applicable to map_kind='data'.
    Value of min_bound reset to 0 if negative or > 0.8.
    Value of color reset to 'Blues' if not found in in COLOR_GRADIENTS
    (valid matplotlib colormap names).
    """
    if map_kind == "data":
        if color not in COLOR_GRADIENTS:
            print("Unknown colormap name, reset to 'Blues'")
            color = "Blues"

        # maybe reset range:
        if min_bound < 0:
            print("Invalid negative min bound, reset to 0.0")
            min_bound = 0.0

        if min_bound > 0.8:
            print("Excessive min bound, reset to 0.0")
            min_bound = 0.0

        n_resample = 10
        bounds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 1.0]

        if min_bound > 0:
            bounds = [n for n in bounds if n >= min_bound]
            n_resample = len(bounds)

        gradcols = mpl.colormaps[color].resampled(n_resample)
        newcolors = gradcols(np.linspace(0, 1, n_resample))
        cmap = ListedColormap(newcolors, name="Dat")
        bnorm = BoundaryNorm(bounds, cmap.N)
    else:
        n_resample = 8
        top = mpl.colormaps["Reds_r"].resampled(n_resample)
        bottom = mpl.colormaps["Blues"].resampled(n_resample)
        newcolors = np.vstack((top(np.linspace(0, 1, n_resample)),
                               bottom(np.linspace(0, 1, n_resample))))
        cmap = ListedColormap(newcolors, name="RB")
        bnorm = BoundaryNorm([-1.0, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1.0], cmap.N)

    return cmap, bnorm


def plot_heatmap(df: pd.DataFrame,
                 ax=None, fig=None,
                 map_kind: str = "data",
                 min_bound: float = 0.0,
                 color: str = "Blues"   # data heatmap
                 ):
    """Draw the pcolormesh heatmap on the provided Axis.
    pcolormesh can handle unevenly spaced or skewed (non-rectilinear) grids.
    """
    # get color map & boundaries
    cmap, bnorm = get_cmap_bnorm(map_kind=map_kind,
                                 min_bound=min_bound,
                                 color=color)
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # Remove the top & right Axes spines
    despine(ax=ax)
    
    plot_data = df.values
    nR, nC = df.shape
    # Center the ticks in the middle of each box
    x = np.arange(nC) + 0.5
    y = np.arange(nR) + 0.5

    kws={'rasterized': True, 'norm': bnorm}
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
                            shrink=0.75,  # Shrinks the height to 80% of the axis height
                            aspect=30,   # Higher number makes the width thinner (default is 20)
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


def maybe_resize(figsize: tuple, shape: tuple, incr:int=1) -> tuple:
    """Applies to default sizes.
    """
    if figsize != DEFAULT_FIGSIZE:
        return figsize

    nR, nC = shape
    if nR > 30:
        w = DEFAULT_FIGSIZE[0] + incr
    else:
        w = figsize[0]

    if nC > 30:
        h = DEFAULT_FIGSIZE[1] + incr
    else:
        h = figsize[1]

    if len(figsize) == 3:
        return w, h, figsize[2]
    else:
        return w, h


def heatmap_from_df(df: pd.DataFrame,
                    fig_size: tuple,
                    fig_save_fp: Path=None,
                    title: str = "",
                    map_kind: str = "data",
                    min_bound: float = 0.0,
                    color: str = "Blues"   # data heatmap
                    ):
    """Wrapper function to plot_heatmap: Creates fig & ax prior to call;
    defines and sets title, decides if fig is to be saved.
    """
    figsize = maybe_resize(fig_size, df.shape)
    fig, ax = plt.subplots(1,1, figsize=figsize, layout='constrained')
    plot_heatmap(df, ax=ax, fig=fig,
                 map_kind=map_kind,
                 min_bound=min_bound,
                 color=color
                 )

    ax.set_title(title, fontdict={'weight':'bold', "size":10});
    if fig_save_fp is not None:
        plt.savefig(fig_save_fp)
        print(f"   Figure: {fig_save_fp!s}; Size: {figsize}")

    plt.draw()
    
    return
