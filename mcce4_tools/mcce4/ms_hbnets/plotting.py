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


def plot_heatmap(df: pd.DataFrame, ax=None, fig=None):
    """Draw the pcolormesh heatmap on the provided Axis.
    pcolormesh can handle unevenly spaced or skewed (non-rectilinear) grids.
    """
    # get color map & boundaries
    n_resample = 10
    blu = mpl.colormaps["Blues"].resampled(n_resample)
    newcolors = blu(np.linspace(0, 1, n_resample))
    cmap = ListedColormap(newcolors, name="B10")
    bnorm = BoundaryNorm([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                          0.6, 0.7, 0.8, 0.9, 1.0], cmap.N)
    kws={'rasterized':True, 'norm':bnorm}

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
                            shrink=0.8,  # Shrinks the height to 80% of the axis height
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
                    title: str = "",):
    """Wrapper function to plot_heatmap: Creates fig & ax prior to call;
    defines and sets title, decides if fig is to be saved.
    """
    figsize = maybe_resize(fig_size, df.shape)
    fig, ax = plt.subplots(1,1, figsize=figsize, layout='constrained')
    plot_heatmap(df, ax=ax, fig=fig)
    ax.set_title(title, fontdict={'weight':'bold', "size":10});
    if fig_save_fp is not None:
        plt.savefig(fig_save_fp)
        print(f"   Figure: {fig_save_fp!s}; Size: {figsize}")

    plt.draw()
    
    return
