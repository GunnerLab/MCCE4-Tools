#!/usr/bin/env python3

"""
Module: io_utils.py

Provides these helper functions:

  * `mcfile2df`: Loads mcce output files into pandas.DataFrames.
    Numerical columns are floats or integers;
    File pK.out df gets an extra 'note' column that retains the out-of-bound
    or bad curve info from the original 'pKa/Em' column, which now holds
    float values or null for out-of-bound.

    Files supported (listed in DF_READY_FILES):
       acc.atm
       acc.res
       entropy.out
       fort.38
       head3.lst
       pK.out
       respair.lst
       sum_crg.out
       vdw0.lst

 * `files_diff`: Subtracts the numerical columns of two files (file1 - file2).
   By default, the difference file is saved using the first filename prefixed
   with 'diff_'.

 * `textfile2df`: Loads most output files into a pandas.DataFrame without any 
   additional processing.
   This is the function to use to load 'diff_' files.

 * show_elapsed_time: To display the length of time from given time to time of call.
   Additional info can be given as prepend to 'Elapsed time' message;
   The default 'writer' function is print, but could be logger.info.

 * class MsgFmt: Callable class to preclude eager execution of f-strings in logging.
"""
import logging
from pathlib import Path
import subprocess
from subprocess import CompletedProcess, CalledProcessError
import sys
import time
from typing import Callable, List, Tuple, Union

import pandas as pd
from pandas.api.types import is_object_dtype


logging.basicConfig(format="[ %(levelname)s ] %(funcName)s:\n  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def table_to_df(file_fp: str) -> pd.DataFrame:
    """Load a fixed-width table in to a pandas.DataFrame."""
    return pd.read_fwf(file_fp)


def reader_gen(fpath: Path):
    """Generator function yielding a file line."""
    with open(fpath) as fh:
        for line in fh:
            yield line


def show_elapsed_time(start_t: time, info: str = None, writer: Callable = print):
    """If specific info is given, it follows the elapsed time marker, e.g.:
    'Elapsed time - <info>:'.
    Argument 'writer' is an output function, e.g. print.
    If writer is None, writer is set to logger.info if a logger is configured,
    otherwise to print.
    """
    elapsed = time.time() - start_t
    if info is None:
        msg = f"Elapsed time: {elapsed:,.2f} s ({elapsed/60:,.2f} min).\n"
    else:
        msg = f"Elapsed time - {info}: {elapsed:,.2f} s ({elapsed/60:,.2f} min).\n"

    writer(msg)

    return


def subprocess_run(
    cmd: str,
    capture_output: bool = True,
    check: bool = False,
    text: bool = True,
    shell: bool = True,
) -> Union[CompletedProcess, CalledProcessError]:
    """Wraps subprocess.run. Return CompletedProcess or err obj."""
    try:
        data = subprocess.run(
            cmd, capture_output=capture_output, check=check, text=text, shell=shell
        )
    except CalledProcessError as e:
        data = e

    return data


def mccepdbline_positions() -> List[Tuple]:
    """
    Returns a list of character indices ranges for each items in an extended mcce
    pdb line.
    """
    #      [rec, seq, atm, alt, res, conf, x, y, z, rad, crg, hist]
    return [
            (0, 5),(6, 10),(11, 15),(16, 16),(17, 19),(21, 29),
            (30, 37),(38, 45),(46, 53),(54, 61),(62, 73),(80, 89),
            ]


def parse_mcce_line(pdb_coord_line: str) -> List:
    """
    Parses a mcce pdb coordinates line into elements based on format positions.

    Args:
        pdb_coord_line: The line to parse.
    Returns:
        A list of extracted elements, i.e.:
        [rec, seq, atm, alt, res, conf, x, y, z, rad, crg, hist]
    """
    return [
        pdb_coord_line[start : end + 1].strip()
        for start, end in mccepdbline_positions()
        ]


def get_mcce_filepaths(mcce_dir: Path, ph: str = "7", eh: str = "0") -> Tuple:
    """Constructs and validates paths for these mcce output files:
    head3.lst, step2_out.pdb and the 'msout file' in the ms_out subfolder.

    Args:
        mcce_dir (Path): The MCCE run directory Path.
        ph (str, "7"): The pH value for the desired msout file.
        eh (str, "0"): The Eh value for the desired msout file.

    Returns:
        A 3-tuple containing Path objects for head3.lst, step2_out.pdb,
        and the located msout file.
        Note: The msout filename will have one of the two possible formats
        depending on the MCCE version used when the ms_out/ file(s) were generated;
        it could be pH7.00eH0.00ms.txt or pH7eH0ms.txt.

    Raises:
        SystemExit: If any of the required files are not found.

    Call example with the default ph, eh:
        h3_fp, step2_fp, msout_fp = get_mcce_filepaths(mcce_dir)
    """
    ok = True
    out = []
    tried = []
    for fname in ["head3.lst", "step2_out.pdb", "ms_out"]:
        fp = mcce_dir.joinpath(fname)
        ok = ok and fp.exists()
        if not ok:
            sys.exit(f"Missing: {fp!s}")

        if fname == "ms_out":
            ph = float(ph)
            eh = float(eh)
            # msout filename can have either 2 variants: with integers for pH/Eh,
            # or with float with a precision of 2.
            #
            # test msout filename with fractional, precision 2, ph, eh:
            msout_fp = fp.joinpath(f"pH{ph:.2f}eH{eh:.2f}ms.txt")
            if msout_fp.exists():
                out.append(msout_fp)
            else:
                tried.append(msout_fp.name)
                # filename with integers
                msout_fp = fp.joinpath(f"pH{ph:.0f}eH{eh:.0f}ms.txt")
                if msout_fp.exists():
                    out.append(msout_fp)
                else:
                    tried.append(msout_fp.name)
                    msg = (f"Not found: msout file for pH={ph:.0f}, eH={eh:.0f} "
                           f"in either of these formats: {tried}.")
                    sys.exit(msg)
        else:
            out.append(fp)

    return tuple(out)


def get_unique_filename(filepath: str) -> str:
    """
    Generates a unique filename by appending an integer suffix (_1, _2, etc.)
    to the original filename's stem if the file already exists.

    Args:
        filepath (str): The desired filepath, e.g., "path/to/file.txt".

    Returns:
        A unique filepath string.
    """
    p = Path(filepath)
    if not p.exists():
        return filepath  # Original path is already unique

    parent_dir = p.parent
    file_stem = p.stem
    file_suffix = p.suffix  # Includes the dot, e.g., ".txt"
    n = 1
    new_path = parent_dir / f"{file_stem}_{n}{file_suffix}"
    while new_path.exists():
        n += 1
        new_path = parent_dir / f"{file_stem}_{n}{file_suffix}"

    return str(new_path)


class MsgFmt:
    """Preclude eagerly exec of f-strings."""

    def __init__(self, fmt, /, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)

    def __call__(self, fmt, /, *args, **kwargs) -> str:
        return fmt.format(*args, **kwargs)


mf = MsgFmt  # short lowercase alias


# mcce output files that can be loaded into a pandas.dataframe using mcfile2df:
DF_READY_FILES = [
    "acc.atm",
    "acc.res",
    "all_pK.out",
    "all_pK.out.txt",
    "all_sum_crg.out",
    "all_sum_crg.out.txt",
    "entropy.out",
    "fort.38",
    "head3.lst",
    "pK.out",
    "residues_stats.txt",
    "respair.lst",
    "sum_crg.out",
    "vdw0.lst",
]

DIFFING_FILES = [
    "all_pK.out",
    "all_sum_crg.out",
    "entropy.out",
    "fort.38",
    "head3.lst",
    "pK.out",
    "residues_stats.txt",
    "sum_crg.out",
    "vdw0.lst",
]

# files listed have a variable column 1 header depending on titr type:
titr_type_files = ["entropy.out", "fort.38", "pK.out", "sum_crg.out"]


# files with data over a titration, hdr: <ph> 0 1 2 3 4...
titration_files = ["entropy.out", "fort.38", "sum_crg.out"]


class ToDf:
    """Class attribute: headers_dict.
    This dict is used to (re)set the header of the file parsed by pandas
    for some files: head3.lst and the acc files.
    """

    headers_dict = {
        "head3.lst":
        # last colname 'm' ('mark') added for correct column parsing/splitting
        "iConf CONFORMER FL occ crg Em0 pKa0 ne nH vdw0 vdw1 tors epol dsolv extra history m",
        "acc.atm": "kind atom res resseq sasa",
        "acc.res": "kind res resseq sasa pct",
    }

    @staticmethod
    def reset_pk_oob(row):
        """Argument 'row' is from a pd.df row loaded from pK.out, after addition
        of the 'note' column.
        """
        pk = row.loc["pKa/Em"]
        if isinstance(pk, str):
            if pk[0].isdigit():
                row.loc["pKa/Em"] = float(pk)
                return row

        ro = row.copy()
        cols = ro.index.to_list()
        vals = list(ro.values)
        pk = ro["pKa/Em"]
        if isinstance(pk, str):
            if pk[0] in "<>":
                # shift next 2 columns values
                ro.loc[cols[:2]] = vals[:2]
                ro.loc[cols[2:5]] = [None, None, None]
                ro.loc[cols[5:]] = vals[3:-2]
                ro.loc["pKa/Em"] = float(pk[1:])
            else:
                ro.loc[cols[0]] = vals[0]
                # titration curve too sharp -> info = "curve"
                ro.loc[cols[1]] = "curve"
                # keep until example shows which subsequent columns have data:
                ro.loc[cols[2:]] = [None] * len(vals[2:])
            return ro

        return row

    def __call__(self, fp: Path, file_type: str = None) -> Union[pd.DataFrame, None]:
        """Load 'df ready files' into pandas.DataFrame.
        If the path filename is not a canonical mcce output filename, then file_type
        must be set with its 'parent file', e.g.: if fp="/path/to/pK.out_new", then
        file_typle="pK.out".
        Call method to enable the use of an instance as a function.
        """
        fname = fp.name
        if file_type is None:
            which = fname
        else:
            which = file_type
        if which not in DF_READY_FILES:
            logger.error(mf("File {!r} cannot be loaded into a pandas.DataFrame.", which))
            return None

        df = pd.read_csv(fp, sep=r"\s+")  # noqa: W605
        if fname not in ToDf.headers_dict:
            if (fname in ["pK.out", "all_pK.out"]) or which == "pK.out":
                if "note" not in df.columns:
                    # preset new 'note' column
                    # ('info' is a df method: not a good name)
                    df["note"] = "-"
                    # maybe update:
                    if is_object_dtype(df["pKa/Em"].dtype):
                        df["note"] = df["pKa/Em"].apply(
                            lambda x: "-" if x[0].isdigit() else x
                        )

                    # reorder cols:
                    cols0 = df.columns.to_list()
                    cols = [cols0[0]] + ["note"] + cols0[1:-1]
                    df = df[cols]
                    # update 'note' column with oob and bad fit:
                    df = df.apply(
                        self.reset_pk_oob,
                        axis=1,
                    )

            if (fname in ["pK.out", "sum_crg.out"]) or (
                which in ["pK.out", "sum_crg.out"]
            ):
                # replace -0. -> 0.
                try:
                    df.update(
                        df.select_dtypes(include="float64").apply(lambda x: x + 0.0)
                    )
                except AttributeError:
                    df.update(df[df.apply(type) == "float64"].apply(lambda x: x + 0.0))

            return df

        cols, types = None, None
        hdr = ToDf.headers_dict.get(fname)
        if hdr is not None:
            cols = hdr.split()
        if fname == "head3.lst":
            types = {"iConf": str}

        return pd.read_csv(
            fp,
            sep=r"\s+",  # noqa: W605
            header=None,
            skiprows=1,
            names=cols,
            dtype=types,
        )


def mcfile2df(fp: Path, file_type: str = None) -> Union[pd.DataFrame, None]:
    """Friendly wrapper for ToDf()(fp)."""
    return ToDf()(fp, file_type=file_type)


# FIX: Will fail absent a space separating (some) columns
def textfile2df(fp: Path) -> pd.DataFrame:
    """Load a mcce output file without any modifications."""
    convert = None
    if "head3" in fp.name:
        convert = {"iConf": "{:>05}".format}
    return pd.read_csv(fp, sep=r"\s+", converters=convert)  # noqa: W605


def read_titr_type(fpath: Path) -> str:
    with open(fpath) as fp:
        hdr = fp.readline()
    return hdr.split()[0].strip()


def drop_unpaired_cols(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the input dfs with matching titration columns."""
    s1 = set(df1.columns.to_list())
    s2 = set(df2.columns.to_list())
    intr1 = s1.difference(s2)  # elements only in df1
    intr2 = s2.difference(s1)  # elements only in df2
    if intr1:
        df1 = df1.drop(columns=list(intr1))
        logger.warning(mf("Unmatched cols in df1 dropped:\n\t{}\n", list(intr1)))
    if intr2:
        df2 = df2.drop(columns=list(intr2))
        logger.warning(mf("Unmatched cols in df2 dropped:\n\t{}\n", list(intr2)))

    return df1, df2


def diffing_ready(fp1: Path, fp2: Path) -> bool:
    s = set([fp1.name, fp2.name]).difference(set(DIFFING_FILES))
    return len(s) == 0


def files_diff(
    fp1: Path,
    fp2: Path,
    out_dir: Path = Path.cwd(),
    out_name: str = None,
    return_df: bool = False,
    threshold: float = None,
    file_type: str = None,
    collated: bool = False,
) -> Union[None, pd.DataFrame]:
    """Output a file with the difference of each common numerical columns (fp1 - fp2).
    Args:
      fp1 (Path), fp2 (Path): The filepaths for diffing.
      out_dir (Path, cwd): Output path, default to current working dir; the file name
        is fp1 name prefixed with 'diff_'.
      out_name (str, None): User defined output filename or 'diff_' + fp1.name if None.
      return_df (bool, False): Return the pandas.DataFrame if True.
      threshold (float, 0): Keep only absolute differences greater than this value.
      file_type (str, None): If set, value is one of DIFFING_FILES. Used to bypass the
        equality check on the input filenames in case they differ, e.g. fp1="new_sum_crg.out",
        fp2="sum_crg.out" => file_type="sum_crg.out".
    Note:
      - threshold is None by default instead of 0: it's up to the caller to apply it.
        If given, rows below threshold are drop and this may affect downstream processing
        that relies on fixed totals rows as in bench_compare diffing of collated sum_crg files.
    """
    fn1 = fp1.name
    fn2 = fp2.name
    which_name = fn1

    if file_type is None:
        if fn1 != fn2:
            logger.error(
                (
                    f"Can't diff {fn1} viz {fn2}: different files.\n"
                    "If they have a common file type, pass it in file_type."
                )
            )
            return

        if not diffing_ready(fp1, fp2):
            logger.error(mf("Diffing not supported; files not in {}.", DIFFING_FILES))
            return
    else:
        which_name = file_type
        if which_name not in DIFFING_FILES:
            logger.error(mf("Diffing not supported; files not in {}.", DIFFING_FILES))
            return

    same_type = True
    titr1 = titr2 = None
    if which_name in titr_type_files:
        titr1 = read_titr_type(fp1)
        titr2 = read_titr_type(fp2)
        same_type = titr1 == titr2

    if not same_type:
        logger.error(
            mf("Can't diff {!r} files with different titration types.", which_name)
        )
        return

    df1 = mcfile2df(fp1, file_type=file_type)
    df2 = mcfile2df(fp2, file_type=file_type)

    ph = ""
    # for getting matched_colname from column index:
    matchcol = 0
    if which_name == "head3.lst":
        matchcol = 1

    match_colname = df1.columns.to_list()[matchcol]

    if which_name != "head3.lst":
        ph = match_colname[:2]

    if which_name in titration_files:
        if df1.shape[1] != df2.shape[1]:
            df1, df2 = drop_unpaired_cols(df1, df2)

    diff_df = df1.select_dtypes(exclude="object").sub(
        df2.select_dtypes(exclude="object"), axis="index"
    )
    if threshold is not None:
        msk = diff_df.select_dtypes(exclude="object").abs() > threshold
        diff_df = diff_df.where(msk, other=pd.NA)

    # add the names column values back:
    diff_df[match_colname] = df1[match_colname].values

    if which_name == "head3.lst":
        # also add back iConf col
        diff_df["iConf"] = [f"{c:>05}" for c in df1["iConf"]]
        cols = ["iConf", match_colname] + diff_df.columns.to_list()[:-2]

    elif which_name == "pK.out":
        # add back new note column
        diff_df["note"] = [
            f"{n1}|{n2}" if ((n1 != "-") or (n2 != "-")) else pd.NA
            for (n1, n2) in zip(df1["note"], df2["note"])
        ]
        cols = [match_colname, "note"] + diff_df.columns.to_list()[:-2]

    else:
        cols = [match_colname] + diff_df.columns.to_list()[:-1]

    # reset df with all columns
    diff_df = diff_df[cols]
    newname = "Diff"
    if collated:
        if ":" in match_colname:  # PDB:pH
            _, s2 = match_colname.split(":")
            newname = f"{newname}:{s2}"
    else:
        newname = f"{newname}:{ph}"
    diff_df.rename(columns={match_colname: newname}, inplace=True)

    # remove all-NA columns:
    diff_df.dropna(how="all", inplace=True, axis=1)
    # remove all-NA rows:
    diff_df.set_index(newname, inplace=True)
    diff_df.dropna(how="all", inplace=True, axis=0)
    diff_df.reset_index(drop=False, inplace=True)
    # replace remaining NAs to 0:
    # diff_df.where(~diff_df.isna(), other=0, inplace=True)

    # save diff file:
    if not out_name:
        diff_fp = out_dir.joinpath(f"diff_{fn1}")
    else:
        print("out_name is not None", out_name)
        diff_fp = out_dir.joinpath(out_name)
    diff_df.to_string(diff_fp, float_format="{:>5.2f}".format, index=False)

    if return_df:
        return diff_df

    return
