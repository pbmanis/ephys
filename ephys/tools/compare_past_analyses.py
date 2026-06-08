"""Compare spike shape data across analysis runs"""

import argparse
from datetime import datetime
import re
from ast import literal_eval
from pathlib import Path
from typing import Union

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from ephys.tools import get_configuration
from pylibrary.tools import cprint

expts = get_configuration.get_configuration("config/experiments.cfg")

CP = cprint.cprint
longid = re.compile(r"(?P<date>\d{4}\.\d{2}\.\d{2}_000)/slice_(?P<sn>\d{3})/cell_(?P<cn>\d{3})")
shortid = re.compile(r"(?P<date>\d{4}\.\d{2}\.\d{2}_000)_S(?P<sn>\d{1,3})C(?P<cn>\d{1,3})")
re_list = r"^\[([0-9.,\s]+)\]$"  # matches a list of numbers, with surrounded by brackets
re_float = r"^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$"  # matches a float number, including scientific notation, no brackets

rpath = Path("R_statistics_summaries")


def re_id(row):
    cell_id = row["cell_id"]
    m = re.match(longid, cell_id)
    n = re.match(shortid, cell_id)

    if n:
        cell_idn = f"{n.group('date')}/slice_{int(n.group('sn')):03d}/cell_{int(n.group('cn')):03d}"
        row["cell_id"] = cell_idn
    return row


def check_verbose(flags: argparse.Namespace, col: str, skip: list) -> bool:
    return flags.verbose and not any(col.startswith(s) for s in skip)


def convert_string_to_array(val):
    """Convert a string representation of a list of numbers to a numpy array.
    If the string is not in the correct format, return the original value.

    Parameters
    ----------
    val : str

    Returns
    -------
    np.ndarray or original value if is float
    """
    if isinstance(val, str):
        val = val.replace("np.float64(", "").replace(")", "")
        res = re.match(re_list, val)
        if res:
            return np.array([literal_eval(res.group(0))])
        else:
            res = re.match(re_float, val)
            if res:
                return np.array([literal_eval(res.group(0))])
    elif isinstance(val, (int, float)):
        return np.array([val])
    return val


def clean_string(val):
    if isinstance(val, str) and val.startswith("[") and not val.endswith("]"):
        if "," in val:
            val = val + "]"  # add missing closing bracket if the list was truncated in the CSV
        else:
            val = val[1:]
    return val


def compare_one_measure(
    row1,
    row2,
    col: str,
    cell_id: str,
    cell_id_ok: bool,
    flags: argparse.Namespace,
    skip: list,
    ndiff: int,
    diff_cells: set,
    diff_measures: set,
    date1: str,
    date2: str,
) -> tuple[set, set]:
    val1 = row1[col]
    val2 = row2[col]
    if pd.isna(val1) and pd.isna(val2):
        return diff_cells, diff_measures  # both are NaN, consider them equal for this check
    ignore_subject = False
    # if col == "Subject" and pd.isna(val1) or pd.isna(val2):
    #     ignore_subject = True  # if Subject is missing in one dataset, we can't compare protocols, so skip this check
    #     CP(
    #         "y",
    #         f"Cell {cell_id} has missing Subject value in one dataset, skipping protocol comparison for this cell.",
    #     )
    #     CP("y", f"        {date1} Subject: {val1}, {date2} Subject: {val2}")
    # check if protocol list is different
    val1 = clean_string(val1)
    val2 = clean_string(val2)
    if col == "protocols_used" and flags.diffprotocols:
        if val1 != val2:
            if cell_id_ok:
                if verbose_flag:
                    CP(
                        "y",
                        f"\nCell {cell_id} {row1['age_category']} has differences in protocols used:",
                    )
                cell_id_ok = False
            diff_cells.add(cell_id)
            diff_measures.add(col)
            CP("y", f"        protocols_used: {val1} vs {val2}")
        return (
            diff_cells,
            diff_measures,
        )  # skip further checks for this measure since it's just a protocol difference

    # convert strings of numbers or lists to np arrays
    if isinstance(val1, str) or isinstance(val2, str):
        if check_verbose(flags, col, skip):
            print(
                "str found: col: ",
                col,
                "\n   val1: ",
                val1,
                type(val1),
                "\n   val2: ",
                val2,
                type(val2),
            )
        val1 = convert_string_to_array(val1) if isinstance(val1, str) else val1
        val2 = convert_string_to_array(val2) if isinstance(val2, str) else val2
        #     if res1 is None:
        #         res1 = re.match(re_float, val1)
        #         val1 = f"[{res1.group(0)}]" if res1 else None  # if it's a single float, convert to list format for consistency
        #         val1 = val1.replace("np.float64(", "").replace(")", "")
        # if isinstance(val2, str):
        #     val2 = val2.replace("np.float64(", "").replace(")", "")
        #     res2 = re.match(re_list, val2)
        #     if res2 is None:
        #         res2 = re.match(re_float, val2)
        #         val2 = f"[{res2.group(0)}]" if res2 else None  # if it's a single float, convert to list format for consistency
        #         val2 = val2.replace("np.float64(", "").replace(")", "")
        # print("After regex check: res1: ", res1, "res2: ", res2)
        # if res1 and res2:
        #     val1 = np.array(literal_eval(val1)) if res1 else val1
        #     val1 = np.array([float(x) for x in val1]) if isinstance(val1, np.ndarray) else val1

        #     val2 = np.array(literal_eval(val2)) if res2 else val2
        #     val2 = np.array([float(x) for x in val2]) if isinstance(val2, np.ndarray) else val2
        print("val1: ", type(val1), val1)
        print("val2: ", type(val2), val2)
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(np.sort(val1), np.sort(val2)):
                if cell_id_ok:
                    if check_verbose(flags, col, skip):
                        CP("y", f"\nCell {cell_id} {row1['age_category']} has differences:")
                        CP("y", f"         Rs: {float(row1['Rs'])} vs {float(row2['Rs'])}")
                    cell_id_ok = False
                diff_cells.add(cell_id)
                diff_measures.add(col)

                try:
                    if check_verbose(flags, col, skip):
                        CP(
                            "r",
                            f"         {col}: {val1} vs {val2}\n              (diff = {val1-val2})",
                        )
                except Exception as e:
                    CP(
                        "r",
                        f"         {col}: {val1} vs {val2}\n              (diff = error: {e})",
                    )
            else:
                if check_verbose(flags, col, skip):
                    CP(
                        "g",
                        f"Cell {cell_id} {row1['age_category']} {col}: values are equal",
                    )
                    prot1 = row1["used_protocols"] if "used_protocols" in row1 else "unknown"
                    prot2 = row2["used_protocols"] if "used_protocols" in row2 else "unknown"
                    CP(
                        "g",
                        f"        Protocols used:\n     {date1}:  {prot1}\n     {date2}:  {prot2}",
                    )

        # print(f"   Age Group: ", row1["age_category"], " vs ", row2["age_category"])

    # compare numbers, with possible NaNs
    elif np.isnan(val1) or np.isnan(val2):
        if isinstance(val2, str) and val2 == "nan":
            val2 = np.nan
        elif isinstance(val2, str):
            val2 = float(val2)
        # if check_verbose(flags, col, skip):
        #     print("\nval1, val2: ", val1, val2, type(val1), type(val2))
        if not (np.isnan(val1) and np.isnan(val2)):
            if cell_id_ok:
                if not col in ['dvdt_ratio_bestRs', 'AP_max_V_bestRs']:  # we know about these and they can be NaN in one analysis but not the other without it being a real difference
                    if check_verbose(flags, col, skip):
                        CP(
                            "y",
                            f"\nCell {cell_id} {row1['age_category']} has differences in {col}",
                        )
                        # CP("y", f"        {col}: {val1} vs {val2}")
                        CP(
                            "y",
                            f"\nCell {cell_id} {row1['age_category']} has differences in {col}:\n" +
                            f"    One value is OK and the other is NaN",
                        )
                cell_id_ok = False
            diff_cells.add(cell_id)
            diff_measures.add(col)
            if check_verbose(flags, col, skip) and col not in ['dvdt_ratio_bestRs', 'AP_max_V_bestRs']:
                CP("y", f"        {col}: {date1}: {val1} vs {date2}: {val2}")
    # compare numbers/arrays
    else:
        # print(val1, val2)
        # if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
        #     print(val1, val2)
        #     raise Exception("Debugging: stop here to check values")
        val1 = np.array([float(x) for x in val1]) if isinstance(val1, np.ndarray) else val1
        val2 = np.array([float(x) for x in val2]) if isinstance(val2, np.ndarray) else val2
        if not np.allclose(val1, val2, 1e-6, equal_nan=True):
            if col.startswith("Rs"):
                if np.allclose(val1, 1e-6 * val2, 1e-6, equal_nan=True):
                    if check_verbose(flags, col, skip):
                        CP(
                            "g",
                            f"\nCell {cell_id} {row1['age_category']} has differences in {col} but they are likely due to a unit conversion (1e6 factor)",
                        )
                        CP("g", f"        {col}: {val1} vs {val2} (val2: {val2*1e-6})")
                    val2 = val2 * 1e-6
                    # skip to next column without marking this as a difference
            if cell_id_ok:
                if check_verbose(flags, col, skip):
                    CP("y", f"\nCell {cell_id} {row1['age_category']} has differences:")
                    row1["Rs"] = clean_string(row1["Rs"])
                    row2["Rs"] = clean_string(row2["Rs"])
                    CP(
                        "y",
                        f"        Rs: {float(row1['Rs']):.2f} vs {float(row2['Rs']):.2f} close: {np.allclose(float(row1['Rs']), float(row2['Rs']), 1e-6, equal_nan=True)}",
                    )
                    prot1 = row1["used_protocols"] if "used_protocols" in row1 else "unknown"
                    prot2 = row2["used_protocols"] if "used_protocols" in row2 else "unknown"
                    CP(
                        "y",
                        f"        Protocols used:\n     {date1}:  {prot1}\n     {date2}:  {prot2}",
                    )
                    ndiff += 1
                cell_id_ok = False
            diff_cells.add(cell_id)
            diff_measures.add(col)
            if check_verbose(flags, col, skip):
                CP("c", f"        {col}: {val1} vs {val2}")

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                try:
                    if check_verbose(flags, col, skip):
                        CP("y", f"             (diff = {val1-val2})")
                except Exception as e:
                    CP("y", f"             {col}: {val1} vs {val2} (diff = error: {e})")
            else:
                if check_verbose(flags, col, skip):
                    CP("y", f"             (diff = not numeric)")
            if check_verbose(flags, col, skip) and col.startswith("Rin"):
                CP("m", f"    Rin = 1: {row1['Rin']}\n         2: {row2['Rin']}")
                CP("m", f"    RMP = 1: {row1['RMP']}\n          2: {row2['RMP']}")
                CP("m", f"    Rs = 1: {row1['Rs']}\n         2: {row2['Rs']}")
                CP("m", f"    taum = 1: {row1['taum']}\n           2: {row2['taum']}")
            # print(f"         Age Group: ", row1["age_category"], " vs ", row2["age_category"])

    return diff_cells, diff_measures


def compare_cell_data(
    cell_id: str,
    d1: pd.DataFrame,
    d2: pd.DataFrame,
    skip: list,
    flags: argparse.Namespace,
    diff_cells: set,
    diff_measures: set,
    date1: str,
    date2: str,
) -> int:
    """compare_cell_data Compare the data between 2 cells in different analysis tables.

    Parameters
    ----------
    cell_id : str
        cell_id to compare between the two datasets, in long or short formats
    d1 : pd.DataFrame
        DataFrame containing the cell data for the first dataset.
    d2 : pd.DataFrame
        DataFrame containing the cell data for the second dataset.
    skip : list
        List of column name prefixes to skip during comparison.
    flags : argparse.Namespace
        Command-line flags controlling the comparison behavior.
    diff_cells : set
        Set holding the cell IDs with differences.
    diff_measures : set
        Set holding the measure names with differences.
    date1 : str
        Date of the first dataset (dd-mm-yyyy format).
    date2 : str
        Date of the second dataset (dd-mm-yyyy format).

    Returns
    -------
    int
        Number of cells not found in both datasets.

    Raises
    ------
    ValueError
        Raised if a required column is not found in both datasets.
    """

    not_found:int = 0
    ndiff:int = 0
    diff_cells = set()
    diff_measures = set()
    # print("flags: ", flags)

    row1 = d1[d1["cell_id"] == cell_id]
    row2 = d2[d2["cell_id"] == cell_id]
    if len(row1) == 0 or len(row2) == 0:
        CP("r", f"Cell {cell_id} not found in both datasets")
        if len(row1) == 0:
            CP("r", f"    Cell in file 2 {date2} not in file 1 {date1}: {row2.cell_id.values[0]}")
        if len(row2) == 0:
            CP("r", f"    Cell in file 1 {date1} not in file 2 {date2}: {row1.cell_id.values[0]}")
        # print(f"    Cells in file 2: {set(d2['cell_id'])}")
        # raise ValueError(f"Cell {cell_id} not found in both datasets")
        not_found += 1
        return not_found
    row1 = row1.iloc[0]
    row2 = row2.iloc[0]
    cell_id_ok = True
    for col in d1.columns:

        # print(col)
        if not col.endswith("_bestRs"):
            continue
        # CP("c", f"   Checking column: {col}")
        if any(col.startswith(s) for s in skip):
            continue
        if col not in d2.columns or col not in d1.columns:
            # CP("y", f"Column {col} not found in both datasets")
            if col in ["AP_max_V_bestRs", "dvdt_ratio_bestRs"]:  # we know about these
                continue           
            if col in d1.columns:
                CP("y", f"\nColumn {col} found in file 1: {date1} but not in file 2: {date2}")
            else:
                CP("y", f"\nColumn {col} found in file 2: {date2} but not in file 1: {date1}")
            # CP("y", f"\nColumns in file 1: {date1}: {set(d1.columns)}")
            # CP("y", f"\nColumns in file 2: {date2}: {set(d2.columns)}")
            print(f"{'-'*40}")

            raise ValueError(f"Column {col} not found in both datasets.")
        diff_cells, diff_measures = compare_one_measure(
            row1,
            row2,
            col,
            cell_id=cell_id,
            cell_id_ok= cell_id_ok,
            flags=flags,
            skip=skip,
            ndiff=ndiff,
            diff_cells=diff_cells,
            diff_measures=diff_measures,
            date1=date1,
            date2=date2,
        )
    return not_found


def compare_analyses(
    compare_type: str,
    files_1: Union[str, Path] = None,
    files_2: Union[str, Path] = None,
    flags: argparse.Namespace = None,
) -> dict:

    fn1 = Path(files_1)
    fn2 = Path(files_2)
    date1 = (
        re.search(r"\d{2}-\w{3}-\d{4}", fn1.name).group(0)
        if re.search(r"\d{2}-\w{3}-\d{4}", fn1.name)
        else "unknown"
    )
    date2 = (
        re.search(r"\d{2}-\w{3}-\d{4}", fn2.name).group(0)
        if re.search(r"\d{2}-\w{3}-\d{4}", fn2.name)
        else "unknown"
    )
    if date1 == "unknown" or date2 == "unknown":
        CP("r", f"Could not extract date from file names: {fn1.name}, {fn2.name}")
        raise ValueError(f"Could not extract date from file names: {fn1.name}, {fn2.name}")

    d1 = pd.read_csv(fn1)
    d2 = pd.read_csv(fn2)
    if not flags.summary:
        print(
            f"Comparing {compare_type} data from:\n    {str(fn1)}\n   {'and':^24s}\n    {str(fn2)}"
        )
    # print(d1.columns)
    # print(d2.columns)

    # Accumulate some summary information about differences between the datasets.
    n_diffs = 0
    diff_cells = set()
    diff_measures = set()
    skip = ["AdaptIndex2", "Subject"]
    ndiff = 0

    if flags.verbose:
        print(f"{'='*80:s}")

    # make cell id format the same for both datasets, in case they are different (e.g. long vs short format)
    d1 = d1.apply(re_id, axis=1)
    d2 = d2.apply(re_id, axis=1)
    not_found = 0  # count missing cells in one dataset vs the other.
    print(f"Entries in d1, d2: {len(d1)}, {len(d2)}")
    not_found = 0
    for icell, cell_id in enumerate(d1["cell_id"]):
        not_found += compare_cell_data(
            cell_id,
            d1=d1,
            d2=d2,
            skip=skip,
            flags=flags,
            diff_cells=diff_cells,
            diff_measures=diff_measures,
            date1=date1,
            date2=date2,
        )

    if not_found > 0:
        CP("m", f"Total cells not found in both datasets: {not_found}")

    if flags.summary:
        CP("b", f"\n{'='*40}\nSummary of differences between datasets:\n{'='*40}")
        print(
            f"    {fn1}  (n1={len(set(d1['cell_id']))}), vs {fn2}  (n2={len(set(d2['cell_id']))})"
        )
        missing_in_set_1 = set(d2["cell_id"]) - set(d1["cell_id"])
        missing_in_set_2 = set(d1["cell_id"]) - set(d2["cell_id"])
        if len(missing_in_set_1) > 0:
            CP("y", f"    Cells missing in file 1 present in file 2: {missing_in_set_1}")
        else:
            CP("g", f"    No cells missing in file 1 that are present in file 2.")
        if len(missing_in_set_2) > 0:
            CP("y", f"    Cells missing in file 2 present in file 1: {missing_in_set_2}")
        else:
            CP("g", f"    No cells missing in file 2 that are present in file 1.")

        # if check_verbose(flags, d2.columns, skip):
        #     CP("y", f"    Cells with differences: {diff_cells}\n")
        print(
            f"    Total cells with differences: {len(diff_cells)}, cells with no differences: {len(set(d1['cell_id']).union(set(d2['cell_id']))) - len(diff_cells)}"
        )
        print(f"    Measures with differences: {diff_measures}")
        print(f"    Total measures with differences: {len(diff_measures)}")
        print(
            f"    Measures with NO differences: ",
            [col for col in d1.columns if col not in diff_measures and col in d2.columns],
        )
        for i in range(len(diff_cells)):
            cell_id = list(diff_cells)[i]
            row1 = d1[d1["cell_id"] == cell_id].iloc[0]
            row2 = d2[d2["cell_id"] == cell_id].iloc[0]
            CP("w", f"\n#{i+1:4d}    Cell {cell_id} ({row1['age_category']}):")
            if len(diff_measures) > 0:
                uprot1 = (
                    [p.strip() for p in row1["used_protocols"].split(",")]
                    if "used_protocols" in row1 and isinstance(row1["used_protocols"], str)
                    else []
                )
                uprot2 = (
                    [p.strip() for p in row2["used_protocols"].split(",")]
                    if "used_protocols" in row2 and isinstance(row2["used_protocols"], str)
                    else []
                )
                uprot1 = sorted([p for p in uprot1 if len(p) > 0])
                uprot2 = sorted([p for p in uprot2 if len(p) > 0])
                if set(uprot1) != set(uprot2):
                    print(f"   Protocols 1: {uprot1 if 'used_protocols' in row1 else 'unknown'}")
                    print(f"   Protocols 2: {uprot2 if 'used_protocols' in row2 else 'unknown'}")
                else:
                    CP("g", f"           Protocols used are the same in both analyses")
            diff_msg = {}
            for col in diff_measures:
                if col in d1.columns and col in d2.columns:
                    val1 = row1[col]
                    val2 = row2[col]
                    same = (
                        np.allclose(val1, val2, 1e-6, equal_nan=True)
                        if isinstance(val1, (int, float, np.ndarray))
                        and isinstance(val2, (int, float, np.ndarray))
                        else val1 == val2
                    )
                    if same:
                        label = "same"
                        color = "g"
                    else:
                        label = "different"
                        color = "r"
                        diff_msg[col] = (
                            f"           {col}: {val1} vs {val2} ({label})\n",
                            color,
                            label,
                        )
            if len(diff_msg) > 0:
                if len(diff_msg) == 1 and list(diff_msg.keys())[0].startswith(
                    "Rs"
                ):  # only note Rs if nothing else is different
                    CP("y", f"           Cell {cell_id} has differences in Rs only")
                    CP("m", diff_msg[list(diff_msg.keys())[0]][0])
                else:  # list all the differences
                    CP(
                        "y",
                        f"           Cell {cell_id} has differences in measures: {list(diff_msg.keys())}",
                    )
                    if len(diff_msg) > 1:
                        for col, (msg, color, label) in diff_msg.items():
                            if label != "same":
                                CP(color, msg)
                    else:
                        if not list(diff_msg.keys())[0].startswith("Rs"):
                            CP("r", diff_msg[list(diff_msg.keys())[0]][0])

    date_1 = (
        re.search(r"\d{2}-\w{3}-\d{4}", fn1.name).group(0)
        if re.search(r"\d{2}-\w{3}-\d{4}", fn1.name)
        else "unknown"
    )
    date_2 = (
        re.search(r"\d{2}-\w{3}-\d{4}", fn2.name).group(0)
        if re.search(r"\d{2}-\w{3}-\d{4}", fn2.name)
        else "unknown"
    )
    # check for valid values in date
    if date_1 == "unknown" or date_2 == "unknown":
        CP("r", f"Could not extract date from file names: {fn1.name}, {fn2.name}")
        raise ValueError(f"Could not extract date from file names: {fn1.name}, {fn2.name}")

    summary_data = {
        "file_1": str(fn1),
        "file_2": str(fn2),
        "date_1": date_1,
        "date_2": date_2,
        "d1": datetime.strptime(date_1, "%d-%b-%Y") if date_1 != "unknown" else None,
        "d2": datetime.strptime(date_2, "%d-%b-%Y") if date_2 != "unknown" else None,
        "n_cells_1": len(set(d1["cell_id"])),
        "n_cells_2": len(set(d2["cell_id"])),
        "n_cells_with_diffs": len(diff_cells),
        "n_cells_no_diffs": len(set(d1["cell_id"]).union(set(d2["cell_id"]))) - len(diff_cells),
        "diff_cells": list(diff_cells),
        "diff_measures": list(diff_measures),
        "n_diff_measures": len(diff_measures),
    }
    if len(diff_cells) == 0:
        CP("g", f"\nNo differences found between datasets {fn1} and {fn2}!")
    print(summary_data["n_cells_no_diffs"], "cells with no differences, out of total ", summary_data["n_cells_1"], " and ", summary_data["n_cells_2"])
    return summary_data


def sort_by_date(files: Union[str, Path]) -> list:
    """Sort a list of files by date, with the most recent first"""
    if isinstance(files, str):
        files = [files]
    files = [Path(f) for f in files]
    files_with_dates = []
    for f in files:
        try:
            date_str = re.search(r"\d{2}-\w{3}-\d{4}", f.name).group(0)
            date = datetime.strptime(date_str, "%d-%b-%Y")
            files_with_dates.append((f, date))
        except Exception as e:
            CP("r", f"Could not extract date from file name {f}: {e}")
    sorted_files = sorted(files_with_dates, key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_files]


def main():
    parser = argparse.ArgumentParser(
        description="Compare spike shape analyses across different runs"
    )
    parser.add_argument(
        "--experiment",
        dest="experiment",
        type=str,
        help="Experiment to compare analyses for",
    )
    parser.add_argument(
        "--compare_type",
        dest="compare_type",
        type=str,
        choices=["spike_shapes", "rmtau", "firing_parameters"],
        help="Type of comparison to perform",
    )

    parser.add_argument("--date_1", type=str, help="Date of first CSV file to compare")
    parser.add_argument("--date_2", type=str, help="Date of second CSV file to compare")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information about differences"
    )
    parser.add_argument("--summary", action="store_true", help="Print a summary of differences")
    parser.add_argument("--plot", action="store_true", help="Generate plots comparing the datasets")
    parser.add_argument(
        "--diffprotocols",
        action="store_true",
        help="Compare datasets and report whether different protocols were used in a cell's analysis",
    )
    args = parser.parse_args()
    # check for experiment in the config file
    if args.experiment not in expts[1]:
        CP("r", f"Experiment {args.experiment} not found in config file")
        print(f"Available experiments: {list(expts[1].keys())}")
        exit()
    expt = expts[1][args.experiment]
    files_1 = [f"{args.compare_type}_{args.experiment}_{args.date_1}.csv"] if args.date_1 else []
    files_2 = [f"{args.compare_type}_{args.experiment}_{args.date_2}.csv"] if args.date_2 else []
    # print(args.files_1, args.files_2)
    # files_1 = sort_by_date(args.files_1) if args.files_1 else []
    # files_2 = sort_by_date(args.files_2) if args.files_2 else []
    # print(args.summary, args.verbose)
    summary_data = []
    if len(files_1) == 1 and len(files_2) == 1:
        sd = compare_analyses(
            args.compare_type, Path(rpath, files_1[0]), Path(rpath, files_2[0]), flags=args
        )
        summary_data.append(sd)
    elif len(files_1) == 1 and len(files_2) == 0:
        files_2 = list(Path(rpath).glob(f"{args.compare_type}_CBA_Age*.csv"))
        print(files_2)
        for file in files_2:
            if file.name == files_1[0].name:
                continue  # skip comparing the file to itself
            # if file.name != "rmtau_CBA_Age_27-Aug-2025.csv":
            #     continue  # skip this file which has known differences in protocols used
            sd = compare_analyses(args.compare_type, Path(rpath, files_1[0]), file, flags=args)
            summary_data.append(sd)
        if args.plot:
            # Generate plots comparing the datasets
            fig, ax = mpl.subplots()
            dates = [sd["d2"] for sd in summary_data if sd["d2"] is not None]
            ax.plot(
                dates,
                [sd["n_cells_with_diffs"] for sd in summary_data if sd["d2"] is not None],
                "o",
                color="red",
                label="Cells with differences",
            )
            # ax.plot(dates, [sd["n_cells_no_diffs"] for sd in summary_data if sd["d2"] is not None], label="Cells with no differences")
            ax.plot(
                dates,
                [sd["n_cells_2"] for sd in summary_data if sd["d2"] is not None],
                "s",
                color="b",
                label="Total cells in file 2",
            )

            ax.set_title(f"Comparison of {args.compare_type} data across analyses")
            ax.set_xlabel("Date of analysis")
            ax.set_ylabel("Number of cells")
            ax.legend()
            mpl.show()

    exit()


if __name__ == "__main__":
    main()
