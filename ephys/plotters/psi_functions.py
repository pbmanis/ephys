from pathlib import Path
import datetime
import numpy as np
import pandas as pd


def get_plot_order(experiment):
    """get_plot_order get the order of the groups to plot

    Parameters
    ----------
    experiment : dict
        experiment dictionary

    Returns
    -------
    list
        list of groups in order to plot
    """
    if "plot_order" in experiment.keys():
        plot_order = experiment["plot_order"]
    else:
        raise ValueError("No Plot Order is defined in the configuration file")
    return plot_order


def get_protodurs(experiment):
    if "protocol_durations" in experiment.keys():
        protodurs = experiment["protocol_durations"]
    else:
        raise ValueError("No protocol durations are defined in the configuration file")
    return protodurs


def get_plot_colors(experiment):
    """get_plot_colors get the colors to use for the groups

    Parameters
    ----------
    experiment : dict
        experiment dictionary

    Returns
    -------
    dict
        dictionary of colors
    """
    if "plot_colors" in experiment.keys():
        plot_colors = experiment["plot_colors"]
    else:
        raise ValueError("No Plot Colors are defined in the configuration file")
    return plot_colors


def rename_groups(row, experiment):
    # print("renaming row group: ", row.Group)
    if row.Group in list(experiment["group_map"].keys()):
        row.groupname = experiment["group_map"][row.Group]
    else:
        row.groupname = np.nan  # deassign the group
    return row.groupname


def get_datasummary(experiment):
    datasummary = Path(
        experiment["analyzeddatapath"],
        experiment["directory"],
        experiment["datasummaryFilename"],
    )
    if not datasummary.exists():
        raise ValueError(f"Data summary file {datasummary!s} does not exist")
    df_summary = pd.read_pickle(datasummary)
    df_summary.rename(
        {"Subject": "animal_identifier", "animal identifier": "animal_identifier"},
        axis=1,
        inplace=True,
    )
    return df_summary


def setup(experiment):
    excelsheet = Path(
        experiment["analyzeddatapath"],
        experiment["directory"],
        experiment["datasummaryFilename"],
    )
    analysis_cell_types = experiment["celltypes"]
    adddata = Path(
        experiment["analyzeddatapath"],
        experiment["directory"],
        experiment["adddata"],
    )

    return excelsheet, analysis_cell_types, adddata


def make_cell_id(row):
    sliceno = int(row["slice_slice"][-3:])
    cellno = int(row["cell_cell"][-3:])
    cell_id = f"{row['date']:s}_S{sliceno:d}C{cellno:d}"
    row["cell_id"] = cell_id
    return row


def set_subject(row):
    """set_subject if subject is empty, set to date name

    Parameters
    ----------
    row : pandas dataframe row
        _description_

    Returns
    -------
    _type_pandas_dataframe_row
        _description_
    """
    # print("row subj: ", row["Subject"])
    if row["Subject"] in ["", " ", None]:
        subj = Path(row.cell_id).name
        # print("   subj: ", subj, subj[:10])
        row["Subject"] = subj[:10]
    if row["Subject"] is None:
        row["Subject"] = "NoID"
    return row["Subject"]


def get_age(age_value):
    if isinstance(age_value, pd.Series):
        age = age_value.values[0]
    else:
        age = age_value
    if isinstance(age, (float, np.float64)):
        age = int(age)
    elif isinstance(age, str):
        age = int("".join(filter(str.isdigit, age)))
        if isinstance(age, float):
            age = int(age)
    else:
        raise ValueError(f"age is not a pd.Series, float or string: {age!s}")
    return age


def numeric_age(row):
    """numeric_age convert age to numeric for pandas row.apply

    Parameters
    ----------
    row : pd.row_

    Returns
    -------
    value for row entry
    """
    if isinstance(row.age, float):
        return row.age
    if isinstance(row.age, str):
        if len(row.age) == 0:
            row.age = np.nan
        else:
            row.age = int("".join(filter(str.isdigit, row.age)))
        return float(row.age)
    else:
        raise ValueError(f"age is not a float or string: {row.age!s}")


def make_datetime_date(row, colname="date"):
    if colname == "date" and "Date" in row.keys():
        colname = "Date"
    if pd.isnull(row[colname]) or row[colname] == "nan":
        row.shortdate = 0
        return row.shortdate

    date = str(Path(row[colname]).name)
    date = date.split("_", maxsplit=1)[0]
    shortdate = datetime.datetime.strptime(date, "%Y.%m.%d")
    shortdate = datetime.datetime.timestamp(shortdate)
    st = datetime.datetime.timestamp(
        datetime.datetime.strptime("1970-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
    )
    row.shortdate = shortdate - st
    if pd.isnull(row.shortdate):
        raise ValueError("row.shortdate is null ... in make_datetime_date")

    return row.shortdate


def clean_sex_column(row):
    if row.sex not in ["F", "M"]:
        row.sex = "U"
    return row.sex


def categorize_ages(row, experiment: dict):
    age = int(numeric_age(row))
    if "age_categories" not in experiment.keys():
        row.age_category = "ND"
        return row.age_category
    for k in experiment["age_categories"].keys():
        if (age >= experiment["age_categories"][k][0]) and (
            age <= experiment["age_categories"][k][1]
        ):
            row.age_category = k
            break
        else:
            row.age_category = "ND"
    return row.age_category


def clean_rin(row, experiment: dict):
    min_Rin = 6.0
    if "data_inclusion_criteria" in experiment.keys():
        if row.cell_type in experiment["data_inclusion_criteria"].keys():
            min_Rin = experiment["data_inclusion_criteria"][row.cell_type]["Rin_min"]
        else:
            min_Rin = experiment["data_inclusion_criteria"]["default"]["Rin_min"]
    # print("rowrin: ", row.Rin)
    if isinstance(row.Rin, float):
        row.Rin = [row.Rin]
    for i, rin in enumerate(row.Rin):
        # print("rin: ", rin)
        if row.Rin[i] < min_Rin:
            row.Rin[i] = np.nan
    return row.Rin


def adjust_AHP_depth_V(row, experiment: dict):
    """adjust_AHP_relative_depth_V adjust the AHP depth voltage measurement
    for the junction potential. This does not change the value
    Parameters
    ----------
    row : pandas series
        data row
    Returns
    -------
    AHP_depth_V : list
        adjusted AHP depth
    """
    # print("adjust AHP Depth ", row.AHP_depth_V)
    if isinstance(row.AHP_depth_V, float):
        row.AHP_depth_V = [row.AHP_depth_V + 1e-3 * experiment["junction_potential"]]
    else:
        row.AHP_depth_V = [ap + 1e-3 * experiment["junction_potential"] for ap in row.AHP_depth_V]
    return row.AHP_depth_V


def compute_ap_relative_depth_v(row, measure: str):
    if isinstance(row["AHP_relative_depth_V_bestRs"], (list, np.ndarray)):
        val = 1e3 * float(row["AHP_relative_depth_V_bestRs"][0])
        # print(thrv, df.iloc[index]["AP_peak_V"][0], val)
    else:
        val = np.nan
    row[measure] = val
    return row


def adjust_AHP_trough_V(row, experiment: dict):
    # print("adjust AHP Trough: ", row.AHP_trough_V)
    if isinstance(row.AHP_trough_V, float):
        row.AHP_trough_V = [row.AHP_trough_V + 1e-3 * experiment["junction_potential"]]
    else:
        row.AHP_trough_V = [ap + 1e-3 * experiment["junction_potential"] for ap in row.AHP_trough_V]

    return row.AHP_trough_V

def compute_AHP_relative_depth(self, row):
    # Calculate the AHP relative depth, as the voltage between the the AP threshold and the AHP trough
    # if the depth is positive, then the trough is above threshold, so set to nan.
    # this creates a AHP_rel_depth_V column.
    # Usually we want to take this from the spike evoked by the lowest-current level (e.g., near rheobase)
    # as the analysis of that spike is what is used for the dvdt/hw measures and threshold,
    # and is an isolated spike (minimum default distance to next spike is 25 ms)

    # print("row.keys: ", row.keys())
    if "AHP_relative_depth_V" in row.keys():
        lcs_depth = row.get("AHP_relative_depth_V", np.nan)
        # print("lcs depth: ", lcs_depth)
        # CP("c", f"LowestCurrentSpike in row keys, AHP = {row["AHP_relative_depth_V"]}")
        row.AHP_depth_measure = "Lowest Current Spike"
        row.AHP_relative_depth_V = -1 * np.array(row.AHP_relative_depth_V)
    else:
        # This is the first assignment/caluclation of AHP_depth_V, so we need to make sure
        # it is a list of the right length
        if isinstance(row.AP_thr_V, float):
            row.AP_thr_V = [row.AP_thr_V]
        rel_depth_V = [np.nan] * len(row.AHP_depth_V)
        for i, apv in enumerate(row.AHP_trough_V):
            rel_depth_V[i] = (
                row.AP_thr_V[i] - row.AHP_depth_V[i]
            )  # note sign is positive ... consistent with LCS in spike analysis
            # but rescale and change sign for plotting
            rel_depth_V[i] = -1.0 * rel_depth_V[i] * 1e3  # convert to mV
            if rel_depth_V[i] > 0:
                rel_depth_V[i] = np.nan
        row.AHP_depth_measure = "Multiple spikes"
        row.AHP_relative_depth_V = rel_depth_V
    return row  # single measure


def compute_AHP_trough_time(row):
    # RE-Calculate the AHP trough time, as the time between the AP threshold and the AHP trough
    # if the depth is positive, then the trough is above threshold, so set to nan.

    if isinstance(row.AP_thr_T, float):
        row.AP_thr_T = [row.AP_thr_T]
    if isinstance(row.AHP_trough_T, float):
        if np.isnan(row.AHP_trough_T):
            return row.AHP_trough_T
        row.AHP_trough_T = [row.AHP_trough_T]
    for i, att in enumerate(row.AP_thr_T):  # base index on threshold measures
        # print("AP_thr_T: ", row.AP_thr_T[i], row.AHP_trough_T)
        if np.isnan(row.AHP_trough_T[i]):
            return row.AHP_trough_T[i]
        # print("trought_t, thrt: ", row.AHP_trough_T[i], row.AP_thr_T[i])  # note AP_thr_t is in ms, AHP_trough_T is in s
        if not np.isnan(row.AHP_trough_T[i]):
            row.AHP_trough_T[i] = row.AHP_trough_T[i] - row.AP_thr_T[i] * 1e-3
            if row.AHP_trough_T[i] < 0:
                row.AHP_trough_T[i] = np.nan
    return row.AHP_trough_T


def adjust_AP_thr_V(row, experiment: dict):
    if isinstance(row.AP_thr_V, float):
        row.AP_thr_V = [row.AP_thr_V + 1e-3 * experiment["junction_potential"]]
    else:
        row.AP_thr_V = [ap + 1e-3 * experiment["junction_potential"] for ap in row.AP_thr_V]
    return row.AP_thr_V


def clean_rmp(row, experiment: dict):
    """clean_rmp check that the RMP is in range (from the config file),
    and adjust for the junction potential

    Parameters
    ----------
    row : pandas series
        data row

    Returns
    -------
    RMP values: list
        _description_
    """
    jp = experiment.get("junction_potential", 0.0)
    if isinstance(row.RMP, float):
        rmp = [row.RMP + jp]
    else:
        rmp = [rmpi + experiment["junction_potential"] for rmpi in row.RMP]
    if "data_inclusion_criteria" in experiment.keys():
        # print("RMP clean: ", row.cell_type)
        # print("RMPS: ", row.RMP)
        # print("inclusion limits: ", experiment["data_inclusion_criteria"])
        if row.cell_type in experiment["data_inclusion_criteria"].keys():
            min_RMP = experiment["data_inclusion_criteria"][row.cell_type]["RMP_min"]
        else:
            min_RMP = experiment["data_inclusion_criteria"]["default"]["RMP_min"]

    for i, rmpi in enumerate(rmp):
        if rmpi > min_RMP:
            row.RMP[i] = np.nan
        else:
            row.RMP[i] = float(rmpi)
    return row.RMP  # returns the original value.


def clean_rmp_zero(row, experiment: dict):
    """clean_rmp_zero adjust RMP measured at zero current, much like clean_rmp.

    Parameters
    ----------
    row : pandas series
        cell data row

    Returns
    -------
    row.RMP_Zero : list
        adjusted RMP_zeros.
    """
    min_RMP = None
    if "data_inclusion_criteria" in experiment.keys():
        if row.cell_type in experiment["data_inclusion_criteria"].keys():
            min_RMP = experiment["data_inclusion_criteria"][row.cell_type]["RMP_min"]
        else:
            min_RMP = experiment["data_inclusion_criteria"]["default"]["RMP_min"]
    if isinstance(row.RMP_Zero, float):
        r0 = [
            row.RMP_Zero + experiment["junction_potential"]
        ]  # handle case where there is only one float value
    else:
        r0 = [rmp_0 + experiment["junction_potential"] for rmp_0 in row.RMP_Zero]
    for i, r0 in enumerate(r0):
        if min_RMP is not None and r0 > min_RMP:
            row.RMP_Zero[i] = np.nan
    return row.RMP_Zero
