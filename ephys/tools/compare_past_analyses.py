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


# fn = "CBA_Age/CBA_Age_combined_by_cell.pkl"
# date = datetime.now().strftime("%d-%m-%Y")
# fn_no_ext = Path(fn).stem
# fn_parent = Path(fn).parent
# fn_out = Path(fn_parent, f"{fn_no_ext}_{date}.pkl")



CP = cprint.cprint

re_list = r"^\[([0-9.,\s]+)\]$"  # matches a list of numbers, with surrounded by brackets

rpath = Path("R_statistics_summaries")

def check_verbose(flags: argparse.Namespace, col: str, skip: list) -> bool:
        return flags.verbose and not any(col.startswith(s) for s in skip)


def compare_analyses(compare_type: str, files_1: Union[str, Path] = None, files_2: Union[str, Path] = None,
                     flags: argparse.Namespace = None) -> dict:


    fn1 = Path(files_1)
    fn2 = Path(files_2)
    date1 = re.search(r"\d{2}-\w{3}-\d{4}", fn1.name).group(0) if re.search(r"\d{2}-\w{3}-\d{4}", fn1.name) else "unknown"
    date2 = re.search(r"\d{2}-\w{3}-\d{4}", fn2.name).group(0) if re.search(r"\d{2}-\w{3}-\d{4}", fn2.name) else "unknown"
    if date1 == "unknown" or date2 == "unknown":
        CP("r", f"Could not extract date from file names: {fn1.name}, {fn2.name}")
        raise ValueError(f"Could not extract date from file names: {fn1.name}, {fn2.name}")

    d1 = pd.read_csv(fn1)
    d2 = pd.read_csv(fn2)
    if not flags.summary:
        print(f"Comparing {compare_type} data from:\n    {str(fn1)}\n   {'and':^24s}\n    {str(fn2)}")
    # print(d1.columns)
    # print(d2.columns)

    # Accumulate some summary information about differences between the datasets.
    n_diffs = 0
    diff_cells = set()
    diff_measures = set()
    skip = ["AdaptIndex2", "Subject"]

    if flags.verbose:
        print(f"{'='*80:s}")
    for cell_id in d1["cell_id"]:
        # if cell_id != "2023.01.23_000_S1C1":
        #     continue
        row1 = d1[d1["cell_id"] == cell_id]
        row2 = d2[d2["cell_id"] == cell_id]
        if len(row1) == 0 or len(row2) == 0:
            CP('m', f"Cell {cell_id} not found in both datasets")
            continue
        row1 = row1.iloc[0]
        row2 = row2.iloc[0]
        cell_id_ok = True
        for col in d1.columns:
            if any(col.startswith(s) for s in skip):
                continue
            if col in d2.columns:
                val1 = row1[col]
                val2 = row2[col]
                if pd.isna(val1) and pd.isna(val2):
                    continue
                ignore_subject = False
                if col == "Subject" and pd.isna(val1) or pd.isna(val2):
                    ignore_subject = True  # if Subject is missing in one dataset, we can't compare protocols, so skip this check
                # check if protocol list is different
                if col == "protocols_used" and flags.diffprotocols:
                    if val1 != val2:
                        if cell_id_ok:
                            if verbose_flag:
                                CP("y", f"\nCell {cell_id} {row1['age_category']} has differences in protocols used:")
                            cell_id_ok = False
                        diff_cells.add(cell_id)
                        diff_measures.add(col)
                        CP("y", f"        protocols_used: {val1} vs {val2}")
                    continue
                # print("col:, ", col, "val1: ", val1, type(val1), "val2: ", val2, type(val2))
                # convert strings of numbers or lists to np arrays
                if isinstance(val1, str) and isinstance(val2, str):
                    val1 = val1.replace("np.float64(", "").replace(")", "")
                    val2 = val2.replace("np.float64(", "").replace(")", "")
                    res1 = re.match(re_list, val1)
                    res2 = re.match(re_list, val2)
                    if res1 and res2:
                        val1 = np.array(literal_eval(val1)) if res1 else val1
                        val2 = np.array(literal_eval(val2)) if res2 else val2
                    # print('val1, val2: ', val1, val2)
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
                                CP("g", f"        Protocols used:\n     {date1}:  {prot1}\n     {date2}:  {prot2}")

                    # print(f"   Age Group: ", row1["age_category"], " vs ", row2["age_category"])
                
                # compare numbers, with possible NaNs
                elif np.isnan(val1) or np.isnan(val2):
                    if isinstance(val2, str) and val2 == 'nan':
                        val2 = np.nan
                    elif isinstance(val2, str):
                        val2 = float(val2)
                    print("val1, val2: ", val1, val2, type(val1), type(val2))
                    if not (np.isnan(val1) and np.isnan(val2)):
                        if cell_id_ok:
                            if check_verbose(flags, col, skip):
                                CP("y", f"\nCell {cell_id} {row1['age_category']} has differences in {col}: one value is NaN and the other is not")
                            cell_id_ok = False
                        diff_cells.add(cell_id)
                        diff_measures.add(col)
                        if check_verbose(flags, col, skip):
                            CP("y", f"        {col}: {val1} vs {val2}")
                # compare numbers/arrays 
                else:
                    # print(val1, val2)
                    if not np.allclose(val1, val2, 1e-6, equal_nan=True):
                        if col.startswith("Rs"):
                            if np.allclose(val1, 1e-6*val2, 1e-6, equal_nan=True):
                                if check_verbose(flags, col, skip):
                                    CP("g", f"\nCell {cell_id} {row1['age_category']} has differences in {col} but they are likely due to a unit conversion (1e6 factor)")
                                    CP("g", f"        {col}: {val1} vs {val2} (val2: {val2*1e-6})")
                                val2 = val2*1e-6
                                continue  # skip to next column without marking this as a difference
                        if cell_id_ok:
                            if check_verbose(flags, col, skip):
                                CP("y", f"\nCell {cell_id} {row1['age_category']} has differences:")
                                CP("y", f"        Rs: {float(row1['Rs']):.2f} vs {float(row2['Rs']):.2f} close: {np.allclose(float(row1['Rs']), float(row2['Rs']), 1e-6, equal_nan=True)}")
                                prot1 = row1["used_protocols"] if "used_protocols" in row1 else "unknown"
                                prot2 = row2["used_protocols"] if "used_protocols" in row2 else "unknown"
                                CP("y", f"        Protocols used:\n     {date1}:  {prot1}\n     {date2}:  {prot2}")
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

        # print summary
    if flags.summary:
        CP("b", f"\n{'='*40}\nSummary of differences between datasets:\n{'='*40}")
        print(f"    {fn1}  (n1={len(set(d1['cell_id']))}), vs {fn2}  (n2={len(set(d2['cell_id']))})")

        if check_verbose(flags, col, skip):
            print(f"    Cells with differences: {diff_cells}\n")
        print(
            f"    Total cells with differences: {len(diff_cells)}, cells with no differences: {len(set(d1['cell_id']).union(set(d2['cell_id']))) - len(diff_cells)}"
        )
        print(f"    Measures with differences: {diff_measures}")
        print(f"    Total measures with differences: {len(diff_measures)}")
        print(f"    Measures with NO differences: ", 
               [col for col in d1.columns if col not in diff_measures and col in d2.columns])
        for i in range(len(diff_cells)):
            cell_id = list(diff_cells)[i]
            row1 = d1[d1["cell_id"] == cell_id].iloc[0]
            row2 = d2[d2["cell_id"] == cell_id].iloc[0]
            CP("w",f"\n#{i+1:4d}    Cell {cell_id} ({row1['age_category']}):")
            if len(diff_measures) > 0:
                uprot1 = [p.strip() for p in row1['used_protocols'].split(",")] if 'used_protocols' in row1 and isinstance(row1['used_protocols'], str) else [  ]
                uprot2 = [p.strip() for p in row2['used_protocols'].split(",")] if 'used_protocols' in row2 and isinstance(row2['used_protocols'], str) else [  ]
                uprot1 = sorted([p for p in uprot1 if len(p) > 0])
                uprot2 = sorted([p for p in uprot2 if len(p) > 0])
                print(f"   Protocols 1: {uprot1 if 'used_protocols' in row1 else 'unknown'}")
                print(f"   Protocols 2: {uprot2 if 'used_protocols' in row2 else 'unknown'}")
            for col in diff_measures:
                
                if col in d1.columns and col in d2.columns:
                    print(col)
                    val1 = row1[col]
                    val2 = row2[col]
                    same = np.allclose(val1, val2, 1e-6, equal_nan=True) if isinstance(val1, (int, float, np.ndarray)) and isinstance(val2, (int, float, np.ndarray)) else val1 == val2
                    if same:
                        label = "same"
                        color = 'g'
                    else:
                        label = "different"
                        color = 'r'
                    CP("r", f"        {col}: {val1} vs {val2} ({label})")
    
    date_1 = re.search(r"\d{2}-\w{3}-\d{4}", fn1.name).group(0) if re.search(r"\d{2}-\w{3}-\d{4}", fn1.name) else "unknown"
    date_2 = re.search(r"\d{2}-\w{3}-\d{4}", fn2.name).group(0) if re.search(r"\d{2}-\w{3}-\d{4}", fn2.name) else "unknown"
    # check for valid values in date
    if date_1 == "unknown" or date_2 == "unknown":
        CP("r", f"Could not extract date from file names: {fn1.name}, {fn2.name}")
        raise ValueError(f"Could not extract date from file names: {fn1.name}, {fn2.name}")
    
    summary_data = {
        "file_1": str(fn1),
        "file_2": str(fn2),
        "date_1": date_1,
        "date_2": date_2,
        "d1": datetime.strptime(date_1, '%d-%b-%Y') if date_1 != "unknown" else None,
        "d2": datetime.strptime(date_2, '%d-%b-%Y') if date_2 != "unknown" else None,
        "n_cells_1": len(set(d1["cell_id"])),
        "n_cells_2": len(set(d2["cell_id"])),
        "n_cells_with_diffs": len(diff_cells),
        "n_cells_no_diffs": len(set(d1["cell_id"]).union(set(d2["cell_id"]))) - len(diff_cells),
        "diff_cells": list(diff_cells),
        "diff_measures": list(diff_measures),
        "n_diff_measures": len(diff_measures),
    }

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

if __name__ == "__main__":
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

    parser.add_argument('--files_1', type=str, help="Path to the first CSV file to compare")
    parser.add_argument('--files_2', type=str, help="Path to the second CSV file to compare")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information about differences"
    )
    parser.add_argument("--summary", action="store_true", help="Print a summary of differences")
    parser.add_argument("--plot", action="store_true", help="Generate plots comparing the datasets")
    parser.add_argument("--diffprotocols", action="store_true", 
                        help="Compare datasets and report whether different protocols were used in a cell's analysis")
    args = parser.parse_args()
    # check for experiment in the config file
    if args.experiment not in expts:
        CP("r", f"Experiment {args.experiment} not found in config file")
        exit()
    expt = expts[args.experiment]
    # print(args.files_1, args.files_2)
    files_1 = sort_by_date(args.files_1) if args.files_1 else []
    files_2 = sort_by_date(args.files_2) if args.files_2 else []
    # print(args.summary, args.verbose)
    summary_data = []
    if len(files_1) == 1 and len(files_2) == 1:
        sd = compare_analyses(args.compare_type, Path(rpath, files_1[0]), Path(rpath, files_2[0]),
                              flags = args)
        summary_data.append(sd)
    elif len(files_1) == 1 and len(files_2) == 0:
        files_2 = list(Path(rpath).glob(f"{args.compare_type}_CBA_Age*.csv"))
        print(files_2)
        for file in files_2:
            if file.name == files_1[0].name:
                continue  # skip comparing the file to itself
            # if file.name != "rmtau_CBA_Age_27-Aug-2025.csv":    
            #     continue  # skip this file which has known differences in protocols used
            sd = compare_analyses(args.compare_type, Path(rpath, files_1[0]), file, flags = args)
            summary_data.append(sd)
        if args.plot:
            # Generate plots comparing the datasets
            fig, ax = mpl.subplots()
            dates = [sd["d2"] for sd in summary_data if sd["d2"] is not None]
            ax.plot(dates, [sd["n_cells_with_diffs"] for sd in summary_data if sd["d2"] is not None],
                     'o', color='red',label="Cells with differences")
            # ax.plot(dates, [sd["n_cells_no_diffs"] for sd in summary_data if sd["d2"] is not None], label="Cells with no differences")
            ax.plot(dates, [sd["n_cells_2"] for sd in summary_data if sd["d2"] is not None], 's', color='b', label="Total cells in file 2")

            ax.set_title(f"Comparison of {args.compare_type} data across analyses")
            ax.set_xlabel("Date of analysis")
            ax.set_ylabel("Number of cells")
            ax.legend()
            mpl.show()
                            
 
    exit()

