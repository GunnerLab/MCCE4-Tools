#!/usr/bin/env python3

"""
Module: cli.py

Codebase: mcce4/ms_hbnets/cli.py

Command line interface for the ms_hbnets tool.

Gathers the H-bonding conformer pairs and states occupancies from the
microstates file specified by a mcce dir, pH & Eh (if the corresponding
output files are not found), and saves several heatmaps:
 - The Donors correlation and the Acceptors correlation for both hb pairs
   and hb states
 - The 'co-occurence heatmap', which is the hb_pairs_res_pH*.csv or 
   hb_states_pairs_pH*.csv data as a figure

 Requires the structural H-bonds data output by the `detect_hbonds` tool.
"""
from argparse import ArgumentParser
from argparse import Namespace
from argparse import RawTextHelpFormatter
from inspect import CO_ASYNC_GENERATOR
from pathlib import Path
from pprint import pformat
from time import time
from typing import Union

import matplotlib.pyplot as plt

from mcce4.io_utils import N_STATES
from mcce4.io_utils import show_elapsed_time
from mcce4.ms_hbnets.correlator import HbCorrelator
from mcce4.ms_hbnets.msout_hb import do_checks
from mcce4.ms_hbnets.msout_hb import is_int
from mcce4.ms_hbnets.msout_hb import min_occ as MIN_OCC, printed_occ
from mcce4.ms_hbnets.msout_hb import MSout_hb
from mcce4.ms_hbnets.plotting import DEFAULT_FIGSIZE, AVAIL_COLORS


APP = "ms_hbnets"
PLOTTING_ARGS = [
    'data_cbar_min',
    'data_map_color',
    'figsize_data',
    'figsize_data_bk',
    'figsize_corr',
    'figsize_corr_bk',
    'figsize_donor_corr',
    'figsize_donor_corr_bk',
    'figsize_acceptor_corr',
    'figsize_acceptor_corr_bk',
]


def process_pairs(args: Union[dict, Namespace]):
    if isinstance(args, dict):
        args = Namespace(**args)

    print("Processing pairs...")
    process_start = time()
    mshb = MSout_hb(args.mcce_dir, args.ph, args.eh,
                    load_states=False,
                    min_occ=float(args.min_occ),
                    reload = args.reload,
                    verbose=args.verbose)

    if not mshb.proceed:
        print(f"[STOP]: The H-bond collection pipeline cannot be run in {Path(args.mcce_dir).resolve()!s}")
        show_elapsed_time(process_start, info="Loading MSout_hb data")
        return
 
    mshb.run_ms_pipeline(load_states=False)
    print("Microstates H_bonds collection for pairs over.")

    if args.run_checks:
        status = do_checks(args.mcce_dir, args.ph, args.eh)
        if status:
            print("Pairs H_bonds checks: passed.")
        else:
            print("Pairs H_bonds checks: failed.")

    figs_args = {k: v for k, v in vars(args).items() if k in PLOTTING_ARGS}

    HbCorr = HbCorrelator(mshb,
                          hb_kind="pairs",
                          pairs_of_interest_fp=args.pairs_of_interest_csv,
                          min_occ=float(args.min_occ),
                          split_BK=args.split_BK,
                          figs_args=figs_args,
    )
    HbCorr.process_pairs()
    show_elapsed_time(process_start, info="H-bond pairs processing")
    if HbCorr.ok:
        plt.show()

    return


def process_states(args: Union[dict, Namespace]):

    if isinstance(args, dict):
        args = Namespace(**args)

    process_start = time()
    mshb = MSout_hb(args.mcce_dir, args.ph, args.eh,
                    n_target_states=args.n_states,
                    load_states=True,
                    min_occ=float(args.min_occ),
                    reload = args.reload,
                    verbose=args.verbose)

    if not mshb.proceed:
        print(f"[STOP]: The H-bond collection pipeline cannot be run in {Path(args.mcce_dir).resolve()!s}")
        return

    mshb.run_ms_pipeline(load_states=True)
    print("Microstates H_bonds collection over.")

    if args.run_checks:
        status = do_checks(args.mcce_dir, args.ph, args.eh)
        if status:
            print("Microstates H_bonds checks: passed.")
        else:
            print("Microstates H_bonds checks: failed.")

    figs_args = {k: v for k, v in vars(args).items() if k in PLOTTING_ARGS}
    HbCorr = HbCorrelator(mshb,
                          hb_kind="states",
                          pairs_of_interest_fp=args.pairs_of_interest_csv,
                          min_occ=float(args.min_occ),
                          include_states_da_corr=args.do_states_da_corr,
                          split_BK=args.split_BK,
                          figs_args=figs_args,
    )
    HbCorr.process_states()
    show_elapsed_time(process_start, info="H-bond states processing")
    if HbCorr.ok:
        plt.show()

    return


def to_figsize(strtpl: str):
    """Process option as figure size, (width, height, unit).
    One of width or height may be None; the respective value is 
    taken from the app default figsize (rcParams["figure.figsize"]
    default would be [6.4, 4.8]).
    If third item: must be one of "in", "cm" (removed "px" unit:
    would make no sense with default values).
    Option value examples:
      ",3,cm" -> 6,3,'cm' :: change default H, change unit to cm;
      ",,cm" -> 6,6,'cm' :: default size but in cm.
    """
    # Split by comma and convert individual items
    nums = strtpl.split(',')
    if len(nums) < 2:
        print("Option value: figsize must evaluate to at least a 2-tuple: 'w,h' or ',h' or 'w,'; default used.")
        return DEFAULT_FIGSIZE

    if not nums[0]:
        nums[0] = DEFAULT_FIGSIZE[0]  # None would mean 6.4
    if not nums[1]:
        nums[1] = DEFAULT_FIGSIZE[1]  # None would mean 4.8

    unit = None
    if len(nums) == 3 and isinstance(nums[2], str):
        if nums[2] in ["in", "cm"]:
            unit = nums[2]
            # else ignored;
    w = int(nums[0]) if is_int(nums[0]) else round(float(nums[0]),2)
    h = int(nums[1]) if is_int(nums[1]) else round(float(nums[1]),2)

    if unit is None:
        return w, h
    else:
        return w, h, unit


def cli_parser():
    p = ArgumentParser(prog=APP,
        usage="""ms_hbnets pairs [+ options if non-default]; see ms_hbnets pairs -h
       ms_hbnets states [+ options if non-default]; see ms_hbnets pairs -h""",
        description="""
Gathers the H-bonding conformer pairs and states occupancies from the
microstates file specified by a mcce dir, pH & Eh (if the corresponding
output files are not found or --reload is true), and saves several heatmaps:
  - The hb_states_pairs or hb_pairs_res heatmaps plot the data as a figure
  - The Donors and the Acceptors correlation heatmaps for both hb pairs and hb states
The default heatmaps size is (6,6) with implied unit 'in'.
Requires the structural H-bonds data output by the `detect_hbonds` tool.""",
    formatter_class=RawTextHelpFormatter,
    )
    # COMMON parser: for pairs and states processing
    cp = ArgumentParser(add_help=False)
    cp.add_argument(
        "-mcce-dir",
        default=".",
        type=str,
        help="MCCE run directory; Default: %(default)s",
    )
    # ph, eh: as strings to easily determine the precision
    cp.add_argument("-ph",
                    default="7",
                    type=str,
                    help="Titration pH; Default: %(default)s"
                    )
    cp.add_argument("-eh",
                    default="0",
                    type=str,
                    help="Titration Eh; Default: %(default)s"
                    )
    cp.add_argument("-min-occ",
                    type=float,
                    default=MIN_OCC,
                    help="To return data with occ >= min-occ; Default: " + printed_occ
                    )
    cp.add_argument("-pairs-of-interest-csv",
                    default=None,
                    help="""
File listing comma-separated H-bonding pairs that will filter the main data file
prior to producing the heatmaps; Default: %(default)s"""
                    )
    cp.add_argument("-data-cbar-min",
                    type=float,
                    default=0.0,
                    help="Lower bound of the colorbar in the data heatmaps; Default: %(default)s"
                    )
    cp.add_argument("-data-map-color",
                    type=str,
                    default="Reds",
                    help=f"Color of the data heatmaps, one of: {AVAIL_COLORS}; Default: %(default)s"
                    )
    cp.add_argument("--run-checks",
                   action="store_true",
                   default=False,
                   help="Perform checks on main outputs; Default: %(default)s"
                   )
    cp.add_argument("--reload",
                   action="store_true",
                   default=False,
                   help="Reload the microstates data (overwrites existing output files); Default: %(default)s"
                   )
    cp.add_argument("-v", "--verbose",
                   action="store_true",
                   default=False,
                   help="Output more details and save 'dropped_fixedoff_confs.tsv' during reduction; Default: %(default)s"
                   )

    subparsers = p.add_subparsers(
        required=True,
        title=f"{APP} sub-commands",
        dest="subparser_name",
    )
    sub1 = subparsers.add_parser("pairs",
        usage="""ms_hbnets pairs -ph 5                # get hb pairs at a non-default pH (--reload needed if outputs exist)
       ms_hbnets pairs -figsize-data 8,8    # change the data heatmap size;
       ms_hbnets pairs -figsize-data ,,cm   # change the unit of the data heatmap to cm
       ms_hbnets pairs -data-cbar-min 0.3   # change the cbar lowest bound of the data heatmap
       ms_hbnets pairs -data-map-color Reds # change the color of the data heatmap""",
        description="""
Gathers the H-bond pairs data (the occupancy of hb pairs over all microstates), if needed or --reload is true,
then outputs the data and correlation heatmaps.
""",
        formatter_class=RawTextHelpFormatter,
        parents=[cp],
    )
    sub1.add_argument("-figsize-data",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
Data as figure; Size of the pairs data heatmap; 
The default size is ok for ~30 residues; Default: %(default)s"""
    )
    sub1.add_argument("-figsize-data-bk",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
If --split-BK is used or no filtering pairs are provided, the data is divided into res-res and
res-bk pairs. Size of the res-bk pairs data heatmap; Default: %(default)s"""
    )
    sub1.add_argument("-figsize-donor-corr",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="Size of the Donors correlation heatmap; Default: %(default)s"
    )
    sub1.add_argument("-figsize-donor-corr-bk",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
If --split-BK is used or no filtering pairs are provided, the data is divided into res-res and
res-bk pairs. Size of the res-bk Donors correlation heatmap; Default: %(default)s"""
    )
    sub1.add_argument("-figsize-acceptor-corr",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="Size of the Acceptors correlation heatmap; Default: %(default)s"
    )
    sub1.add_argument("-figsize-acceptor-corr-bk",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
If --split-BK is used or no filtering pairs are provided, the data is divided into res-res and
res-bk pairs. Size of the res-bk Acceptors correlation heatmap; Default: %(default)s"""
    )
    sub1.add_argument("--split-BK",
                      action="store_true",
                      default=False,
                      help="""To divide the pairs into res-res and res-bk groups and 
compute each group correlation. Always true without filtering with pairs of interest; Default: %(default)s""",
    )
    sub1.set_defaults(func=process_pairs)

    sub2 = subparsers.add_parser("states",
    usage="""ms_hbnets states -ph 5                # get hb states at a non-default pH (--reload needed if outputs exist)
       ms_hbnets states -n-states 50000      # change the target number of output hb microstates (--reload needed if outputs exist)
       ms_hbnets states -figsize-data ,7     # get hb states and change the data heatmap height
       ms_hbnets states -figsize-data ,,cm   # change the unit of the data heatmap to cm
       ms_hbnets states -data-cbar-min 0.3   # change the cbar lowest bound of the data heatmap
       ms_hbnets states -data-map-color Reds # change the color of the data heatmap""",
        description="""
Gathers the H-bond microstates data if needed or --reload is true, then outputs the data and correlation heatmaps.
WARNING: The states correlation is NOT currently split into res-res, res-bk subsets: without filtering,
""",
        formatter_class=RawTextHelpFormatter,
        parents=[cp],
    )
    sub2.add_argument("-n-states",
                      default=N_STATES,
                      type=int,
                      help="""
Number of H-bonding states to return, possibly. Consult the header row of hb_states_pH*.csv
to find out if it is adequate (`head -n1 hb_states_pH*.csv`); Default: %(default)s"""
    )
    sub2.add_argument("-figsize-data",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
Data as figure; Size of the states pairs data heatmap; 
The default size is ok for ~30 residues; Default: %(default)s"""
    )
    sub2.add_argument("-figsize-data-bk",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
If --split-BK is used or no filtering pairs are provided, the data is divided into res-res and
res-bk pairs. Size of the states res-bk pairs data heatmap; Default: %(default)s"""
    )
    sub2.add_argument("-figsize-corr",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
Size of the states pairs correlation heatmap; Default: %(default)s"""
    )
    sub2.add_argument("-figsize-corr-bk",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
If --split-BK is used or no filtering pairs are provided, the data is divided into res-res and
res-bk pairs. Size of the states res-bk pairs correlation heatmap; Default: %(default)s"""
    )  
    sub2.add_argument("-figsize-donor-corr",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
If --do-states-da-corr is used, the state Donors and Acceptors correlation is included.
Size of the states Donors correlation heatmap; Default: %(default)s"""
    )
    sub2.add_argument("-figsize-donor-corr_bk",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
If --do-states-da-corr and --split-BK are used, the state Donors and Acceptors correlation is included.
Size of the states Donors correlation heatmap; Default: %(default)s"""
    )
    sub2.add_argument("-figsize-acceptor-corr",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
If --do-states-da-corr is used, the state Donors and Acceptors correlation is included.
Size of the states Acceptors correlation heatmap; Default: %(default)s"""
    )
    sub2.add_argument("-figsize-acceptor-corr-bk",
                      type=to_figsize,
                      default=DEFAULT_FIGSIZE,
                      help="""
If --do-states-da-corr and --split-BK are used, the state Donors and Acceptors correlation is included.
Size of the states Acceptors correlation heatmap; Default: %(default)s"""
    )
    sub2.add_argument("--split-BK",
                      action="store_true",
                      default=False,
                      help="""To divide the pairs into res-res and res-bk groups and 
compute each group correlation. Always true without filtering with pairs of interest; Default: %(default)s""",
    )
    sub2.add_argument("--do-states-da-corr",
                      action="store_true",
                      default=False,
                      help="Include the correlation of states Donors and Acceptors; Default: %(default)s",
    )
    sub2.set_defaults(func=process_states)

    return p


def cli(argv=None):
    p = cli_parser()
    args = p.parse_args(argv)
    print("CLI options used ::",
          pformat(args.__dict__, sort_dicts=False) + "\n" + "-"*60,
          sep="\n")

    args.func(args)

    return
