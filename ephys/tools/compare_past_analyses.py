"""Compare spike shape data across analysis runs"""

import argparse
import re
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Union

import colorama
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import tabulate
from pylibrary.tools import cprint

from ephys.tools import get_configuration

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


def _scalar_rs(row: pd.Series) -> float:
    """Extract a scalar Rs value from a row Series.

    Claude fixed 2026-06-10: handles both 'Rs_bestRs' and 'Rs' column names, duplicate
    index entries (which return a Series instead of a scalar), and non-numeric values.
    """
    for key in ("Rs_bestRs", "Rs"):
        if key not in row.index:
            continue
        val = row[key]
        if isinstance(val, pd.Series):
            val = val.iloc[0]  # duplicate column names produce a Series; take first
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        try:
            return float(clean_string(val))
        except (TypeError, ValueError):
            pass
    return np.nan


def compare_measures(
    measures: list,
    row1: pd.Series,
    row2: pd.Series,
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

    table_data = {}
    # check if protocol list is different
    protocols_1 = clean_string(row1["used_protocols"])
    protocols_2 = clean_string(row2["used_protocols"])
    if flags.diffprotocols:
        if protocols_1 != protocols_2:
            if cell_id_ok:
                if verbose_flag:
                    CP(
                        "y",
                        f"\nCell {cell_id} {row1['age_category']} has differences in protocols used:",
                    )
                cell_id_ok = False
            diff_cells.add(cell_id)
            diff_measures.add("used_protocols")
            CP("y", f"        used_protocols: {protocols_1} vs {protocols_2}")
        # return (
        #     diff_cells,
        #     diff_measures,
        # )  # skip further checks for this measure since it's just a protocol difference

    # convert strings of numbers or lists to np arrays
    for measure in measures:
        if measure.startswith("post_"):  # skip post current step spike counts
            continue
        val1 = row1[measure] if measure in row1 else None
        val2 = row2[measure] if measure in row2 else None

        if isinstance(val1, str) or isinstance(val2, str):
            val1 = convert_string_to_array(val1) if isinstance(val1, str) else val1
            val2 = convert_string_to_array(val2) if isinstance(val2, str) else val2
            print("val1: ", type(val1), val1)
            print("val2: ", type(val2), val2)
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.array_equal(np.sort(val1), np.sort(val2)):
                    if cell_id_ok:
                        if check_verbose(flags, measure, skip):
                            CP("y", f"\nCell {cell_id} {row1['age_category']} has differences:")
                            # Claude fixed 2026-06-10: old code: float(row1['Rs']) / float(row2['Rs'])
                            CP(
                                "y",
                                f"         Rs: {_scalar_rs(row1):.2f} vs {_scalar_rs(row2):.2f}",
                            )
                        cell_id_ok = False
                    diff_cells.add(cell_id)
                    diff_measures.add(measure)

                    try:
                        if check_verbose(flags, measure, skip):
                            CP(
                                "r",
                                f"         {measure}: {val1} vs {val2}\n              (diff = {val1-val2})",
                            )
                    except Exception as e:
                        CP(
                            "r",
                            f"         {measure}: {val1} vs {val2}\n              (diff = error: {e})",
                        )
                else:
                    if check_verbose(flags, measure, skip):
                        CP(
                            "g",
                            f"Cell {cell_id} {row1['age_category']} {measure}: values are equal",
                        )
                        prot1 = row1["used_protocols"] if "used_protocols" in row1 else "unknown"
                        prot2 = row2["used_protocols"] if "used_protocols" in row2 else "unknown"
                        CP(
                            "g",
                            f"        Protocols used:\n     {date1}:  {prot1}\n     {date2}:  {prot2}",
                        )

            # print(f"   Age Group: ", row1["age_category"], " vs ", row2["age_category"])

        # compare numbers, with possible NaNs
        # elif np.isnan(val1) or np.isnan(val2):
        #     if isinstance(val2, str) and val2 == "nan":
        #         val2 = np.nan
        #     elif isinstance(val2, str):
        #         val2 = float(val2)
        #     # if check_verbose(flags, col, skip):
        #     #     print("\nval1, val2: ", val1, val2, type(val1), type(val2))
        #     if not (np.isnan(val1) and np.isnan(val2)):
        #         if cell_id_ok:
        #             if not measure in ['dvdt_ratio_bestRs', 'AP_max_V_bestRs']:  # we know about these and they can be NaN in one analysis but not the other without it being a real difference
        #                 if check_verbose(flags, measure, skip):
        #                     CP(
        #                         "y",
        #                         f"\nCell {cell_id} {row1['age_category']} has differences in {measure}",
        #                     )
        #                     # CP("y", f"        {measure}: {val1} vs {val2}")
        #                     CP(
        #                         "y",
        #                         f"\nCell {cell_id} {row1['age_category']} has differences in {measure}:\n" +
        #                         f"    One value is OK and the other is NaN",
        #                     )
        #             cell_id_ok = False
        #         diff_cells.add(cell_id)
        #         diff_measures.add(measure)
        #         if check_verbose(flags, measure, skip) and measure not in ['dvdt_ratio_bestRs', 'AP_max_V_bestRs']:
        #             CP("y", f"        {measure}: {date1}: {val1} vs {date2}: {val2}")
        # compare numbers/arrays
        else:
            # print(val1, val2)
            # if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            #     print(val1, val2)
            #     raise Exception("Debugging: stop here to check values")

            val1 = np.array([float(x) for x in val1]) if isinstance(val1, np.ndarray) else val1
            val2 = np.array([float(x) for x in val2]) if isinstance(val2, np.ndarray) else val2

            # Normalize None / pandas NA types to np.nan so np.allclose/np.isnan don't crash.
            def _to_numeric(v):
                if isinstance(v, np.ndarray):
                    return v
                try:
                    return np.nan if (v is None or pd.isna(v)) else float(v)
                except (TypeError, ValueError):
                    return np.nan

            val1 = _to_numeric(val1)
            val2 = _to_numeric(val2)
            if not np.allclose(val1, val2, 1e-6, equal_nan=True):
                if measure.startswith("Rs"):
                    if np.allclose(val1, 1e-6 * val2, 1e-6, equal_nan=True):
                        if check_verbose(flags, measure, skip):
                            CP(
                                "g",
                                f"\nCell {cell_id} {row1['age_category']} has differences in {measure} but they are likely due to a unit conversion (1e6 factor)",
                            )
                            CP("g", f"        {measure}: {val1} vs {val2} (val2: {val2*1e-6})")
                        val2 = val2 * 1e-6
                        # skip to next column without marking this as a difference
                if cell_id_ok:
                    if check_verbose(flags, measure, skip):
                        CP("y", f"\nCell {cell_id} {row1['age_category']} has differences:")
                        # Claude fixed 2026-06-10: removed row2.rename({'Rs': 'Rs_bestRs'}) —
                        # normalization is now done at the start of compare_cell_data.
                        # old code: rs1 = clean_string(row1["Rs_bestRs"]) / float(rs1) — crashed
                        # when row1["Rs_bestRs"] returned a Series due to duplicate column names.
                        rs1_val = _scalar_rs(row1)
                        rs2_val = _scalar_rs(row2)
                        CP(
                            "y",
                            f"        Rs: {rs1_val:.2f} vs {rs2_val:.2f} close: {np.allclose(rs1_val, rs2_val, 1e-6, equal_nan=True)}",
                        )
                        prot1 = row1["used_protocols"] if "used_protocols" in row1 else "unknown"
                        prot2 = row2["used_protocols"] if "used_protocols" in row2 else "unknown"
                        print("prot1: ", prot1)
                        print("prot2: ", prot2)
                        if pd.isna(prot1) or pd.isna(prot2):
                            prot1 = prot1 if not pd.isna(prot1) else "unknown"
                            prot2 = prot2 if not pd.isna(prot2) else "unknown"
                            CP(
                                "y",
                                f"        Missing or excluded analysis, no protocols:\n     {date1}:  {prot1}\n     {date2}:  {prot2}",
                            )
                            continue
                        prot1 = [p.strip() for p in prot1.split(",") if p.strip()]
                        prot2 = [p.strip() for p in prot2.split(",") if p.strip()]
                        # convert used protocol list to dict of measures: protocols
                        measure_keys = [x.split(":")[0] + "_bestRs" for x in prot1]
                        measure_vals = [x.split(":")[1] if ":" in x else "" for x in prot1]
                        p1 = dict(zip(measure_keys, measure_vals))
                        p1v = dict(
                            zip(
                                measure_keys,
                                [row1[key] if key in row1 else "missing" for key in measure_keys],
                            )
                        )
                        measure_keys = [
                            x.split(":")[0] + "_bestRs" for x in prot2 if not x.startswith("CC")
                        ]
                        print(f"{date2} measure keys: ", measure_keys)
                        measure_vals = [
                            x.split(":")[1] if ":" in x else ""
                            for x in prot2
                            if not x.startswith("CC")
                        ]
                        p2 = dict(zip(measure_keys, measure_vals))
                        p2v = dict(
                            zip(
                                measure_keys,
                                [row2[key] if key in row2 else "missing" for key in measure_keys],
                            )
                        )
                        # Combine the dictionaries into rows
                        table_data = [
                            [
                                key,
                                p1[key],
                                p2[key],
                                p1v[key],
                                p2v[key],
                                (
                                    p1v[key] - p2v[key]
                                    if isinstance(p1v[key], (int, float))
                                    and isinstance(p2v[key], (int, float))
                                    else "N/A"
                                ),
                            ]
                            for key in p2 if not key.startswith('post_')
                        ]

                        # Print using a grid layout
                        # print(tabulate.tabulate(table_data, headers=["Measure", date1, date2, f"{date1} Value", f"{date2} Value", "Difference"], tablefmt="grid"))
                        ndiff += 1
                    cell_id_ok = False
                diff_cells.add(cell_id)
                diff_measures.add(measure)
            # if check_verbose(flags, col, skip):
            #     CP("c", f"        {col}: {val1} vs {val2}")

            # if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            #     try:
            #         if check_verbose(flags, col, skip):
            #             CP("y", f"             (diff = {val1-val2})")
            #     except Exception as e:
            #         CP("y", f"             {col}: {val1} vs {val2} (diff = error: {e})")
            # else:
            #     if check_verbose(flags, col, skip):
            #         CP("y", f"             (diff = not numeric)")
            # if check_verbose(flags, col, skip) and col.startswith("Rin"):
            #     CP("m", f"    Rin = 1: {row1['Rin']}\n         2: {row2['Rin']}")
            #     CP("m", f"    RMP = 1: {row1['RMP']}\n          2: {row2['RMP']}")
            #     CP("m", f"    Rs = 1: {row1['Rs']}\n         2: {row2['Rs']}")
            #     CP("m", f"    taum = 1: {row1['taum']}\n           2: {row2['taum']}")
            # print(f"         Age Group: ", row1["age_category"], " vs ", row2["age_category"])

    return diff_cells, diff_measures, table_data


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
) -> tuple[int, set, set]:
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
    tuple[int, set, set]
        Number of cells not found in both datasets, set of cell IDs with differences,
        set of measure names with differences.

    Raises
    ------
    ValueError
        Raised if a required column is not found in both datasets.
    """

    not_found: int = 0
    ndiff: int = 0
    # diff_cells and diff_measures are local per-cell sets; caller accumulates them.
    # Old code reassigned the passed-in parameters here, then never returned them —
    # so compare_analyses always saw empty sets and reported "no differences found".
    diff_cells = set()
    diff_measures = set()

    # Claude fixed 2026-06-10: normalize Rs column name before extracting rows.
    # New CSVs have "Rs" (renamed from Rs_bestRs in export_r); old CSVs may have "Rs_bestRs".
    # Renaming after row extraction (the old approach at line ~369 below) could not affect
    # already-extracted row Series, and also created duplicate columns when d1 already had
    # Rs_bestRs, causing row["Rs_bestRs"] to return a Series instead of a scalar.
    for _df in [d1, d2]:
        if "Rs" in _df.columns and "Rs_bestRs" not in _df.columns:
            _df.rename(columns={"Rs": "Rs_bestRs"}, inplace=True)

    # match the cells in the two data sets by cell_id
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
        return not_found, diff_cells, diff_measures

    row1_x = row1.iloc[0]
    row2_x = row2.iloc[0]
    cell_id_ok = True
    table_data = {}
    measures = []
    for col in d1.columns:
        if not col.endswith("_bestRs"):  # only compare data for bestRs
            continue
        # CP("c", f"   Checking column: {col}")
        if any(
            col.startswith(s) for s in skip
        ):  # skip the columns that we are not interested in comparing
            continue
        if col in [
            "AP_max_V_bestRs",
            "dvdt_ratio_bestRs",
            "AP_SS_HW_bestRs",
        ]:  # we know about these (old, recalc, other analysis)
            continue
        if col not in d2.columns or col not in d1.columns:  # check columns.
            if col == "Rs_bestRs":
                # Claude fixed 2026-06-10: normalization is now done at the top of this function
                # (before row extraction), so no rename is needed here.
                # old code: d1.rename(columns={"Rs": "Rs_bestRs"}, inplace=True)
                pass
            elif col in d1.columns:
                CP("y", f"\nColumn {col} found in file 1: {date1} but not in file 2: {date2}")
            else:
                CP("y", f"\nColumn {col} found in file 2: {date2} but not in file 1: {date1}")
            print(f"{'-'*40}")
        if col not in d2.columns and col not in d1.columns:
            raise ValueError(f"Column {col} not found in both datasets.")
        measures.append(col)

    diff_cells, diff_measures, table_data = compare_measures(
        measures,
        row1_x,
        row2_x,
        cell_id=cell_id,
        cell_id_ok=cell_id_ok,
        flags=flags,
        skip=skip,
        ndiff=ndiff,
        diff_cells=diff_cells,
        diff_measures=diff_measures,
        date1=date1,
        date2=date2,
    )
    if len(table_data) > 0:
        print(
            tabulate.tabulate(
                table_data,
                headers=["Measure", date1, date2, f"{date1} Value", f"{date2} Value", "Difference"],
                tablefmt="grid",
            )
        )
    return not_found, diff_cells, diff_measures


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
        n, cell_diffs, measure_diffs = compare_cell_data(
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
        not_found += n
        diff_cells.update(cell_diffs)
        diff_measures.update(measure_diffs)

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
                    CP("m", f"Protocols are not matched in the two analyses!")
                    print(
                        f"   Protocols 1 ({date_1}): {uprot1 if 'used_protocols' in row1 else 'unknown'}"
                    )
                    print(
                        f"   Protocols 2 ({date_2}): {uprot2 if 'used_protocols' in row2 else 'unknown'}"
                    )
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
    print(
        summary_data["n_cells_no_diffs"],
        "cells with no differences, out of total ",
        summary_data["n_cells_1"],
        " and ",
        summary_data["n_cells_2"],
    )
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
