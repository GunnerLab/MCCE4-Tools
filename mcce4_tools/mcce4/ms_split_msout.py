#!/usr/bin/env python3

"""
Module: ms_split_msout.py

Description:
  Cli-enabled module to split a msout file into a smaller one that contains the data
  from the nth MC run, or from a range of (consecutive) mc runs.

Update: 11-13-2025
  Purpose: Modifications to only keep reduced msout files directly under the ms_out folder
           as step6.py reads all '.txt' files in that folder.

  - Added option --do_all (default: True) to process all msout files in the ms_out folder.
  - Added 'splits' folder to store each mc split subset for each msout file in 'ms_out'
"""
from argparse import ArgumentParser, Namespace
from itertools import islice
import logging
from pathlib import Path
import shutil
import sys
from typing import List, Union

from mcce4.constants import CLI_EPILOG
from mcce4.io_utils import get_mcce_filepaths
from mcce4.io_utils import subprocess_run, CalledProcessError


TOOL_NAME = "ms_split_msout"
NEW_EXT = ".split"
HDR_FILE = "header.txt" + NEW_EXT
MASTER_FORMAT = "all_{}{}"
VALID_MC = ["MONTERUNS",]   # valid methods/format
MC_RUNS = 6  # default number of MC runs
LOG = "msout_split.log"


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def run_process(cmd: str):
    """Wrapper for catching failed subprocess_run function
    using all defaults.
    Use for creating/processing files with system calls, not
    when return data is needed.
    """
    out = subprocess_run(cmd, check=True)
    if isinstance(out, CalledProcessError):
        sys.exit(out)

    return


def get_msout_path(mcce_dir: Path, ph: str = "7", eh: str = "0") -> Path:
    """Constructs the 'msout file' path in mcce_dir/ms_out folder.

    Args:
    - mcce_dir (Path): The MCCE run directory Path.
    - ph (str, "7"): The pH value for the desired msout file.
    - eh (str, "0"): The Eh value for the desired msout file.

    Returns:
      The path (Path) to the msout file, e.g.: mcce_dir/ms_out/pH7eH0ms.txt.

    Call example with the default ph, eh:
        msout_fp = get_msout_path(mcce_dir)
    """
    dp = Path(mcce_dir)
    if dp.name != "ms_out":
        dp = dp.joinpath("ms_out")
        if not dp.is_dir():
            sys.exit(f"Not found: ms_out dir: {dp!s}")
    dp = dp.relative_to(Path(mcce_dir))

    ph = float(ph)
    eh = float(eh)
    # test msout filename with fractional ph, eh:
    msout_fp = dp.joinpath(f"pH{ph:.2f}eH{eh:.2f}ms.txt")
    if msout_fp.exists():
        return msout_fp

    # filename with integers
    msout_fp = dp.joinpath(f"pH{ph:.0f}eH{eh:.0f}ms.txt")
    if not msout_fp.exists():
        sys.exit(f"Not found: msout file for pH={ph:.0f}, eH={eh:.0f}")

    return msout_fp


def msout_with_mc_lines(mcce_dir: str, ph: str, eh: str):
    """
    Create smaller msout files with these number of accepted state lines:
    10, 20, 50, 100, 200, 500, 1000.
    The smaller files have the count in their extension: ".txt.sm<target_mc>",
    i.e. pH7eH0ms.txt.sm10, and are save in the parent file location.
    Use case:
      For esting the loading of a msout file with a tractable number of mc lines.
      Not a cli function.
    """
    mcce_dir = Path(mcce_dir)

    # h3_fp & s2_fp output paths not used here:
    _, _, msout_fp = get_mcce_filepaths(mcce_dir, ph, eh)

    method = get_mc_method(msout_fp)
    if method == "invalid":
        logger.error(f"Not Applicable: The msout file method is not in {VALID_MC}.")
        return
        
    N = 23  # N :: number of msout file lines to get 10 mc accepted state lines

    # how many mc_lines to output; will be used in smaller file extension:
    target_mc = [10, 20, 50, 100, 200, 500, 1000]
    # head_n :: integers to pass to `head` command -n option:
    # sorted in descending order so that the largest small file wil be reused as
    # input for the others:
    head_n = sorted([N + x for x in target_mc], reverse=True)

    largest_fp = str(msout_fp.with_suffix(f".txt.sm{target_mc[-1]}"))

    # create smaller msout files
    for i, hn in enumerate(head_n, start=1):
        msout_in = largest_fp
        if i == 1:
            msout_in = str(msout_fp)

        outname = str(msout_fp.with_suffix(f".txt.sm{target_mc[-i]}"))
        # create sed command:
        cmd_str = "sed '/^MC\:1/q'" + f" {msout_in} | head -n{hn} > {outname}"
        #print(cmd_str)
        run_process(cmd_str, cwd=mcce_dir)

    return


def get_mc_method(msout_fp: Path) -> str:
    """
    MC method must be MONTERUNS.
    """
    with open(msout_fp) as fh:
        head = list(islice(fh, 2))  # first 2 lines

    key, value = head[1].split(":")
    method = value.strip()
    if key.strip() != "METHOD" or method not in VALID_MC:
        # print(f"Invalid msout file: {msout_fp!s}; method: {method!s}.")
        method = "invalid"

    return method  


def validate_mc_range(mc: list):
    if not mc:
        sys.exit(f"Error: Empty mc range.")

    if len(mc) > 2:
        sys.exit(f"Error: Argument mc (list) can have 2 items at most.")
    
    if mc[0] == 0:
        sys.exit("Error: Use natural number, i.e. 1 for the 1st run, etc.")

    if mc[0] > MC_RUNS:
        sys.exit(f"Error: The maximum number of MC runs is {MC_RUNS}.")

    if len(mc) == 2:
        if mc == [1, 6]:
            sys.exit(f"Error: Nothing to extract (all {MC_RUNS} selected).")

        if (mc[0] > MC_RUNS) or (mc[1] > MC_RUNS):
            sys.exit(f"Error: The maximum number of MC runs is {MC_RUNS}.")

        if mc[0] > mc[1]:
            sys.exit("Error: The range start must be lower than the range end")

    return

    
def preserve_msout_file(msout_fp: Path, master_fp: Path):
    msout_fp = Path(msout_fp)
    master_fp = Path(master_fp)
    # case output from previous version: master_fp w/o '.split' extension
    prev_master_fp = msout_fp.parent.joinpath(MASTER_FORMAT.format(msout_fp.name, ""))
    if prev_master_fp.exists():
        prev_master_fp.unlink()

    if not master_fp.exists():
        logger.info(f"Preserving {msout_fp.name} as master file: {master_fp!s}; {msout_fp!s} will be overwritten.")
        shutil.copyfile(msout_fp, master_fp, follow_symlinks=False)
 
    return


def extract_msout_header(master_fp: Path, hdr_fp: Path):
    prev_hdr_fp = hdr_fp.with_suffix("")
    if prev_hdr_fp.exists():
        prev_hdr_fp.unlink()

    if not Path(hdr_fp).exists():
        cmd = f"sed -n '/^MC\:0/q;p' {str(master_fp)} > {str(hdr_fp)}"
        run_process(cmd)
        logger.info(f"Saved msout file 'header' as {str(master_fp)}")

    return


def extract_mc_data(master_fp: Path, mc: List[int]) -> Path:
    """
    Args:
      - master_fp (Path): The preserved msout file: 'all_<msout_file>.txt.split'
      - mc (list): The given MC runs range
    Returns:
      - Path object of the extracted mc data file
    """
    master_fp = Path(master_fp)
    # remove extracted files with previous format:
    for fp in master_fp.parent.glob("pH7eH0ms_mc*.txt"):
        fp.unlink()

    # original name, i.e.: <msout_file>.txt
    msout = master_fp.stem.removeprefix("all_").removesuffix(".txt")

    end_pat = None
    # get mc as a range of 0-based indices bc 1st MC run is tagged 'MC:0' in msout
    if len(mc) == 1:
        # if mc==[2] :: "2nd run", then the exclusive range is imc_range = (1, 2):
        imc = (mc[0]-1, mc[0])
        if imc[1] != MC_RUNS:
            end_pat = f"^MC\:{imc[1]}"
        logger.info(f"Single MC selected: {mc} => exclusive, 0-based range: {imc}")
        mc_out = f"{msout}_mc{imc[0]}.txt{NEW_EXT}"
    else:
        # get multiple mc runs data, including last bound, mc==[2,6] -> (1,5)
        imc = (mc[0]-1, mc[1]-1)
        if imc[1] != MC_RUNS - 1:
            end_pat = f"^MC\:{imc[1]+1}"
        logger.info(f"Multiple MC selected: {mc} => inclusive, 0-based range: {imc}")
        mc_out = f"{msout}_mc{imc[0]}{imc[1]}.txt{NEW_EXT}"
    
    mc_out_fp = master_fp.parent.joinpath(mc_out)
    if mc_out_fp.exists():
        logger.info(f"Extracted file already exists: {mc_out_fp!s}")
        return mc_out_fp

    # build the sed cmd:
    start_pat = f"^MC\:{imc[0]}"
    if end_pat is None:
        # from nth run to EOF:
        cmd = f"sed -n '/{start_pat}/,$p' {str(master_fp)} > {str(mc_out_fp)}"
    else:
        cmd = (f"sed -n '/{start_pat}/,/{end_pat}/ "
            "{/"
            f"{end_pat}/!p"
            "}' "
            f"{str(master_fp)} > {str(mc_out_fp)}"
            )
    run_process(cmd)
    logger.info(f"Extracted file is: {mc_out_fp!s}")

    return mc_out_fp


def check_size_consistency(header_fp: Path, mc_data_fp: Path, msout_fp: Path):
    """
    Simple check on correct file sizes.
    40169 ms_out/pH7eH0ms.txt
       11 ms_out/pH7eH0ms_header.txt
    40158 ms_out/pH7eH0ms_mc4.txt
    """
    cmd = f"wc -l {str(msout_fp)} {str(header_fp)} {str(mc_data_fp)}"
    out = subprocess_run(cmd)
    if isinstance(out, CalledProcessError):
        logger.error("Could not size the files.\n%s\n", out.stderr)
        return

    lines = out.stdout.splitlines()[:-1]  # remove total
    sizes = [int(line.split()[0].strip()) for line in lines]
    if sizes[0] != sizes[1] + sizes[2]:
        msg = ("Inconsistent file sizes: "
               f"msout file ({sizes[0]}) != header ({sizes[1]}) + mc_data ({sizes[2]})")
        logger.error(msg)

    return


def create_reduced_msout(mc_data_fp: Path, msout_fp: Path):
    hdr = msout_fp.parent.joinpath(f"{msout_fp.stem}_{HDR_FILE}")
    if not hdr.exists():
        sys.exit("Error: Did not find the extracted msout file header.")

    cmd = f"cat {str(hdr)} {str(mc_data_fp)} > {str(msout_fp)}"
    run_process(cmd)
    logger.info(f"Created the reduced msout file: {msout_fp!s}")
    check_size_consistency(hdr, mc_data_fp, msout_fp)

    return


def reset_master_file(master_fp: Path, msout_fp: Path):
    """
    Reset the all_<msout_file>.txt[.split] to <msout_file>.txt
    """
    master_fp = Path(master_fp)
    msout_fp = Path(msout_fp)
    # case output from previous version: master_fp w/o '.split' extension
    prev_master_fp = msout_fp.parent.joinpath(MASTER_FORMAT.format(msout_fp.name, ""))
    if prev_master_fp.exists():
        master_fp = prev_master_fp

    if not master_fp.exists():
        logger.info(f"Nothing to reset. No corresponding master file found for {msout_fp!s}.")
        return
    shutil.move(master_fp, msout_fp)
    logger.info(f"Reverted {master_fp!s} to {msout_fp!s}.")

    return


def split_mc_file(msout_fp: Path, mc_range: List[int], reset_master: bool):
    """
    Split the method-validated 'msout file' according to the range of MC runs given in mc_range.
    The original filename will be copied as 'all_<same name>.txt.split' to preserve the intact
    master file, and the new, reduced file will have the name of the original file.
    Args:
     - msout_fp (Path): Path to a specific msout file.
     - mc_range (tuple, (2,)): A tuple of up to two integers indicating the range of MC
                        runs to return; a single value means a single mc, e.g. (2,)
                        run will be returned, while two indicate a true range, e.g. with
                        mc=(4,6) mc runs 4, 5 and 6 will be returned.
     - reset_master (bool): If True, the preserved msout file (all_<msout_file>.txt.split)
                        is renamed to the original <msout_file>.
    Notes:
      Initial implementation: no check on what the max mc run should be (this mcce
      parameter is rarely changed) so max = 6 is assumed.
    
    Returns:
      None
    """
    validate_mc_range(mc_range)

    msout_dir = msout_fp.parent
    master_fp = msout_dir.joinpath(MASTER_FORMAT.format(msout_fp.name, NEW_EXT))
    if reset_master:
        reset_master_file(master_fp, msout_fp)
        return

    preserve_msout_file(msout_fp, master_fp)
    hdr_fp = msout_dir.joinpath(f"{msout_fp.stem}_{HDR_FILE}")
    extract_msout_header(master_fp, hdr_fp)
    mc_data_fp = extract_mc_data(master_fp, mc_range)
    create_reduced_msout(mc_data_fp, msout_fp)

    return


def split_all(args: Namespace):
    """
    Process all msout files in the ms_out folder in either 'split' or 'reset' mode.
    """
    if isinstance(args.mc_range, int):
        args.mc_range = [args.mc_range]

    if Path(args.mcce_dir).name != "ms_out":
        msout_dir = Path(args.mcce_dir).joinpath("ms_out")
        if not msout_dir.is_dir():
            logger.error(f"Not found: ms_out dir: {msout_dir!s}")
            sys.exit(1)
    else:
        msout_dir = Path(args.mcce_dir)
        
    # glob msout files with pattern pHXeHYms.txt
    for msout_fp in msout_dir.glob("pH*eH*ms.txt"):
        logger.info(f"Processing msout file: {msout_fp.name!s}")
        method = get_mc_method(msout_fp)
        if method == "invalid":
            logger.error(f"Not Applicable: The msout file method is not in {VALID_MC}.")
            break
        split_mc_file(msout_fp, args.mc_range, args.reset_master)
   
    return


def split_pipeline(args: Union[Namespace, dict]):
    """
    Process the msout file(s) depending on the value of the --all option:
      - True: Split all msout files or reset all preserved masters
      - False (default): Process single file given by ph/eh values.
    """
    if isinstance(args, dict):
        args = Namespace(**args)

    if isinstance(args.mc_range, int):
        args.mc_range = [args.mc_range]
    
    if args.all:
        split_all(args)
    else:
        # get the msout with standard name:
        msout_fp = get_msout_path(args.mcce_dir, ph=args.ph, eh=args.eh)
        method = get_mc_method(msout_fp)
        if method == "invalid":
            logger.error(f"Not Applicable: The msout file method is not in {VALID_MC}.")
            return
        split_mc_file(msout_fp, args.mc_range, args.reset_master)
  
    logger.info("Done.")

    return


def cli_parser():
    """Command line arguments parser."""

    def arg_valid_dirpath(p: str):
        """Return resolved path from the command line."""
        if not len(p):
            return None
        return Path(p).resolve()

    p = ArgumentParser(
        prog=TOOL_NAME,
        epilog=CLI_EPILOG
    )
    p.add_argument(
        "-mcce_dir",
        type=arg_valid_dirpath,
        default="./",
        help="The mcce run or ms_out folder. Default: %(default)s",
    )
    # ph & eh: parser will keep string input; easier to determine if int or float later.
    p.add_argument(
        "-ph",
        default="7",
        type=str,
        help="pH point (e.g.: 7, 7.5) of the msout file. Default: %(default)s",
    )
    p.add_argument(
        "-eh",
        default="0",
        type=str,
        help="pH point (e.g.: 200, 200.5) of the msout file. Default: %(default)s",
    )
    p.add_argument(
        "-mc_range",
        type=int,
        nargs='*',
        default=2,  # 2nd run
        help="""Which MC run data to extract.
        One integer for a single run, e.g. 1 for the 1st MC run, or two for an inclusive range, 
        e.g. 3 6. Default: %(default)s
        """,
    )
    p.add_argument(
        "--reset_master",
        default=False,
        action="store_true",
        help="Rename 'all_<msout_file>.txt.split' (preserved master file) to the original name. Default: %(default)s",
    )
    p.add_argument(
        "--all",
        default=False,
        action="store_true",
        help="""To process all msout files in ms_out when splitting or resetting  
        (ignoring ph & eh). Default: %(default)s""",
    )
    p.set_defaults(func=split_pipeline)
 
    return p


def cli(argv=None):
    
    clip = cli_parser()
    args = clip.parse_args(argv)
    if sys.argv[-1] in ["-h", "--help"]:
        return

    # conditionally create a log file
    log_path = ""
    if Path.cwd().name == "ms_out":
        log_path = LOG
    elif Path.cwd().joinpath("ms_out").is_dir():
        log_path = Path.cwd().joinpath("ms_out", LOG)
    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)s: %(message)s")
        )
        logger.addHandler(fh)
        opts = "-" * 80 + "\n" + TOOL_NAME + f" options:\n{args}"
        logger.info(opts)

    args.func(args)

    return
