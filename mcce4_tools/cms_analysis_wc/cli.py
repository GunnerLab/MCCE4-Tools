#!/usr/bin/env python

"""
Module: cli.py

  Command line interfac for Protonation microstate analysis with weighted correlation.

"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import logging
from pathlib import Path
from shutil import copyfile
import sys

from mcce4_tools.cms_analysis_wc import APP_NAME, IONIZABLES
from mcce4_tools.cms_analysis_wc import analysis as msa
from mcce4_tools.cms_analysis_wc import parameters as prm


logger = logging.getLogger(__name__)


DEF_PARAMS = "default_params.crgms"


def list_head3_ionizables(h3_fp: Path, as_string: bool = True) -> list:
    """Return the list of ionizable resids from head3.lst.
    When argument 'as_string' is True, the output ia a 'ready-to-paste'
    string that can be pasted into the input parameter file to populate the
    argument 'correl_resids'.
    """
    h3_fp = Path(h3_fp)
    if h3_fp.name != "head3.lst":
        logger.error(f"File name not 'head3.lst': {h3_fp!s}")
        return []
    if not h3_fp.exists():
        logger.error(f"Not found: {h3_fp!s}")
        return []

    h3_lines = [line.split()[1] for line in h3_fp.read_text().splitlines()[1:]]
    h3_ioniz_res = list(
        dict((f"{res[:3]}{res[5:11]}", "") for res in h3_lines if res[:3] in IONIZABLES).keys()
    )
    if not as_string:
        return h3_ioniz_res

    res_lst = "[\n"
    for res in h3_ioniz_res:
        res_lst += f"{res!r},\n"

    return res_lst + "]"


def crgmsa_parser() -> ArgumentParser:

    USAGE = f"""
CALL EXAMPLE:
  {APP_NAME} params.crgms

The input parameter file must be found at the location where the command is run.

Notes:
1. If you add this line in your input parameter file:
     list_head3_ionizables = true
   the program will list the resids in head3.lst and exit;
   The list or a portion thereof can then be used as values to
   the 'correl_resids' identifier in the parameter file.
2. Text files with an extension of '.crgms' in MCCE4/runprms can be copied, then modified
   for use by this tool.
"""
    p = ArgumentParser(
        prog=APP_NAME,
        description="Protonation Microstates Analysis with Weighted Correlation",
        usage=USAGE,
        formatter_class=RawDescriptionHelpFormatter,
    )
    p.add_argument("params_file", type=str, help="The input parameters file.")

    return p


def crgmswc_cli(argv=None):

    parser = crgmsa_parser()
    args = parser.parse_args(argv)

    params = Path.cwd().joinpath(args.params_file)
    logger.info(f"params_file: {params!s}")
    if not params.exists():
        sys.exit("Parameters file not found.")

    # load the parameters from the input file into 2 dicts:
    main_d, crg_histo_d = prm.load_crgms_param(params)

    # Optional: List ionizables, drop a copy of 'default_params.crgms' if not found and exit
    list_ionizables = main_d.get("list_head3_ionizables")
    if list_ionizables is not None:
        if list_ionizables.capitalize() == "True":
            logger.info("List of ionizable residues in head3.lst:")
            print(list_head3_ionizables(main_d.get("mcce_dir", ".") + "/head3.lst"))

            dest_fp = Path.cwd().joinpath(DEF_PARAMS)
            if not dest_fp.exists():
                copyfile(Path(__file__).parent.joinpath(DEF_PARAMS), dest_fp)
            sys.exit()

    # Instantiate and run the pipeline
    try:
        pipeline = msa.CMSWC_Pipeline(main_d, crg_histo_d)
        pipeline.run()
    except Exception as e:
         # Log traceback
         logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
         sys.exit(1)

    logger.info("Pipeline execution completed.")
    return
