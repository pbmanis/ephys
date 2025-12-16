# show assembed file data
import datetime
from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import pylibrary.tools.cprint as CP
import seaborn as sns
from pandas import read_pickle
from typing import Union

import ephys.tools.categorize_ages as CatAge
import ephys.tools.filename_tools as FT
from ephys.tools.get_configuration import get_configuration


def transfer_cc_taum(row, excludes: list):
    """
    transfer_cc_taum : This routine is used to transfer the taum value from the
    CC_taum protocol to the taum column. This is done to allow the selection
    of the best value for taum from the CC_taum protocol, if it exists.
    """
    assert "CC_taum" in row.keys()
    assert "taum" in row.keys()
    row["CC_taum"] = [np.nan] * len(row["protocols"])

    n = 0
    for i, p in enumerate(row["protocols"]):
        proto = Path(p)
        if str(proto) in excludes:
            CP.cprint("y", f"       Excluding: {proto!s}")
            continue
        if proto.name == "all":
            return row
        pn = str(Path(p).name)

        if pn.startswith("CC_taum"):
            if isinstance(row["taum"], list):
                if n < len(row["taum"]):
                    row["CC_taum"][n] = row["taum"][n]
                else:
                    row["CC_taum"][n] = row["taum"][0]
            else:
                row["CC_taum"][n] = row["taum"]
        else:
            row["CC_taum"][n] = np.nan
        n += 1
    row["CC_taum"] = row["CC_taum"][:n]  # trim to the correct length
    assert isinstance(row["CC_taum"], list)
    return row


def apply_select_by(row, parameter: str, select_by: str, select_limits: list):
    """
    apply_select_by : Here we filter the data from the different protocols
    in the measurements for this row on the "select_by" criteria (measurement type, limits).
    We then select the "best" value to report for the selected parameter based on the filtered data.

    Usually, this will be when the select_by parameter with the lowest value(s),
    where the parameter value is not nan, and the select_by value is not nan.
    Typically, the selection will be for the protocol with the lowest Rs that
    has a valid measurement for the parameter.

    This routine sets the following columns in the row:
        'parameter'_mean : the mean of the parameter values
        'parameter'_bestselect_by : the best value of the parameter
        If the parameter is "CC_taum", then the value is taken only from the
        CC_taum protocols, if they exist.
        Otherwise the value is taken from the named parameter.
        'used_protocols' : a list of the protocols used to select the best value,
            in the form 'parameter:protocol'
    """

    # handle the case where there is no data, which may be indicated by a single [nan]
    # representing all protocols:
    verbose = False
    verbose_selects = [None] # ["RMP", "taum", "Rs", "Rin"]
    if verbose and (select_by in verbose_selects):
        print("\n", "="*60)
        print("Selector to use: ", select_by)
        print("Parameter to test, value: ", parameter, row[parameter])
    if parameter not in row.keys():
        CP.cprint("y", f"Parameter {parameter:s} is not in current data row")
        raise ValueError
        print("     row keys: ", row.keys())
        return row
    if parameter == "used_protocols":
        return row

    # first, convert the measurements to a *list* if they are not already
    if verbose and (select_by in verbose_selects):
        print("type of row[parameter]: ", type(row[parameter]), row[parameter])
    if isinstance(row[parameter], (float, np.float64)):
        row[parameter] = [row[parameter]]
    # print("row par: ", row[parameter], type(row[parameter]))
    if isinstance(row[parameter], np.ndarray):
        # print("ndim: ", row[parameter].ndim)
        if row[parameter].ndim == 1:
            if len(row[parameter]) == 0:
                # row[parameter] = [row[parameter][0]] * len(row["protocols"])
            # else:
                row[parameter] = [np.nan] * len(row["protocols"])
        elif row[parameter].ndim == 0:
            row[parameter] = [row[parameter]] * len(row["protocols"])
    if len(row[parameter]) > 0:
        row[parameter] = [float(x) for x in row[parameter]]
    else:
        row[parameter] = [np.nan]
    if verbose and (select_by in verbose_selects):
        print("converted: type of row[parameter]: ", type(row[parameter]), row[parameter])
    # if isinstance(row[parameter], np.ndarray) and row[parameter].shape == (0,):
    #     row[parameter] = [np.nan] * len(row["protocols"])
    # elif isinstance(row[parameter], (float, np.float64)) or not hasattr(row[parameter], "__iter__"):
    #     row[parameter] = [row[parameter]] * len(row["protocols"])
    # elif isinstance(row[parameter], list):
    #     if len(row[parameter]) > 0:
    #         row[parameter] = row[parameter][0] * len(row["protocols"])
    #     else:
    #         row[parameter] = [np.nan] * len(row["protocols"])
    # elif isinstance(row[parameter], np.ndarray) and row[parameter].ndim == 1:
    #     row[parameter] = [row[parameter][0]] * len(row["protocols"])
    # elif isinstance(row[parameter], np.ndarray) and row[parameter].ndim == 0:
    #     row[parameter] = [row[parameter]] * len(row["protocols"])
    # if len(row[parameter]) == 1 and np.isnan(row[parameter][0]):
    #     row[parameter] = [row[parameter][0]] * len(row["protocols"])
    # if no valid measurements, just return the row
    if verbose and (select_by in verbose_selects):
        print("row par: ", row[parameter])
        print("type: ", type(row[parameter]))
        print("row parameter: ", row[parameter])
    if not isinstance(row[parameter], str):
        if np.all(np.isnan(row[parameter])):
            # CP.cprint("r", f"Parameter {parameter:s} is all nan")
            row[parameter] = np.nan
            return row

    # Standard measurements: the mean of ALL of the measurements
    # collected across all protocols (data in a new column, representing the
    # mean)

    # now do the selection. Find out which protocols have the
    # lowest select_by measurement
    # these arrays are the same length as the number of protocols
    # also, if the selection value is out of range, set
    # the parameter to nan, to remove it
    select_limits = np.array(select_limits) * 1e-6
    if isinstance(row[select_by], float):
        selector_vals = np.array([row[select_by]])
    else:
        selector_vals = np.array(row[select_by])
    selector_values = np.array(
        [np.nan if v < select_limits[0] or v > select_limits[1] else v for v in selector_vals]
    )

    # print(" selected values:", selector_values, print(select_limits))
    # print("parametre, row[parameter]: ", parameter, row[parameter])
    params = np.array(row[parameter])
    prots = row["protocols"]
    if verbose and (select_by in verbose_selects):
        print("selector_values: , ", selector_values)
        print(
            "   # measures, selectors, protocols : ",
            params.shape,
            selector_values.shape,
            len(prots),
        )
        print("   row[parameter], selector values: ", params, selector_values)

    valid_measures = np.argwhere(~np.isnan(params)).ravel()
    if verbose and (select_by in verbose_selects):
        print("Params: ", params)
        print("selector values: ", selector_values)
        print("valid indices: ", valid_measures)

    if len(valid_measures) == 0:  # no matching values available, set nans.
        CP.cprint("r", f"No valid values for {parameter:s} in {row.cell_id:s}")
        # row[parameter + f"_best{select_by:s}"] = np.nan
        # row[parameter + f"_mean"] = np.nan

        row["used_protocols"] = " ".join((row["used_protocols"], f"{parameter:s}:None"))
        return row

    # Here we find the value(s) for different measurements that
    # are derived from the protocols with the
    # minimum Rs that are in the valid IV list
    # If there is more than one measurement at the minimum Rs,
    # we average them.
    # We do this by looping through the protocols.
    min_val = np.max(select_limits)
    iprots = []  # indicies to list of protocol names
    equal_mins = []  # indicices to equal minimum values (to be averaged)
    values = []  # measurement value
    taums = []  # taum values for CC_taum protocol
    if verbose and (select_by in verbose_selects):
        print("prots: ", prots)
    for i, prot in enumerate(prots):
        if i not in valid_measures:  # no measure for this protocol, so move on
            continue
        if verbose and (select_by in verbose_selects):
            print("prot: ", prot)
        p = str(Path(prot).name)  # get the name of the protocol
        # skip specific protocols
        if p.startswith(
            "analysistime"
        ):  # where did this come from? I don't see where it is written.
            continue
        # if p.startswith("CC_taum"):
        #     continue
        if verbose and (select_by in verbose_selects):
            print("select by: ", select_by)
            print("index: ", i)
            print("row selection data: ", row[select_by])

        select_value = np.nan
        if isinstance(row[select_by], float):
            select_value = row[select_by]
        else:
            # print("I, selectby: ", i, row[select_by])
            if i < len(row[select_by]):
                select_value = row[select_by][i]

        value = params[i]
        if np.isnan(value):
            continue
        # this will be set on the first run if min_val is correctly initialized
        # print(select_by, "select_value: ", select_value, "min val: ", min_val)
        if select_value < min_val:
            min_val = value  # set the value
            equal_mins = [i]  # reset the list of equal minima
            iprots = [i]  # set the index of the protocol with the minimum value
            values.append(value)  # set the value

        elif select_value == min_val:
            if i not in equal_mins:
                equal_mins.append(i)
                values.append(value)
                iprots.append(i)
            # equal_mins.append(i)
            if not p.startswith("CC_taum"):
                values.append(params[i])
            else:
                taums.append(params[i])

    if verbose  and (select_by in verbose_selects):
        print("iprot, eq_mins, values, taums: ", iprots, equal_mins, values, taums)
    if len(iprots) == 0:
        CP.cprint(
            "r",
            f"No minimum value found for: {row.cell_id!s}, {select_by:s} {params!s}, {valid_measures!s}, {iprots!s}, {equal_mins!s}",
        )
        if verbose and (select_by in verbose_selects):
            print("    protocols: ", row[select_by])
            print("row[parameter]: ", row[parameter])
        return row

    if len(iprots) == 1:
        if isinstance(values, list):
            values = values[0]
        # CP.cprint("g", f"{parameter:s} Single value: {prots[iprots[0]]!s}, value={values}")
        row[parameter + f"_best{select_by:s}"] = values
        used_prots = f"{parameter:s}:{str(Path(prots[iprots[0]]).name):s}"
    elif len(iprots) > 1:
        used = ",".join([str(Path(prots[i]).name) for i in equal_mins])
        # CP.cprint("c", f"{parameter:s} Multiple averaged from: {used!s}")
        row[parameter + f"_best{select_by:s}"] = np.mean(values)
        used_prots = f"{parameter:s}:{used:s}"

    # global means (indpendent of select_by)
    if not parameter.startswith("CC_taum"):  # standard IV protocols
        # print("Params: ", params)
        # print("prots: ", prots)
        p_mean = [
            v
            for i, v in enumerate(params)
            if i < len(prots) and not Path(prots[i]).name.startswith("CC_taum")
        ]
        if len(p_mean) > 0:
            row[parameter + f"_mean"] = np.nanmean(p_mean)
    else:  # specific to CC_taum protocol
        row[parameter + f"_mean"] = np.nanmean(
            [
                v
                for i, v in enumerate(params)
                if i < len(prots) and Path(prots[i]).name.startswith("CC_taum")
            ]
        )

    row["used_protocols"] = ",".join((row["used_protocols"], used_prots))
    return row


def innermost(datalist):
    """innermost For a nested set of lists, return the innermost list

    Parameters
    ----------
    data :list
        a potentially nested set of lists

    Returns
    -------
    list
        Innermost list
    """
    for element in datalist:
        if isinstance(element, list):
            return innermost(element)
        else:
            continue
    return datalist  # no inner list found


def filter_rs(row, maxRs, axis=1):
    """filter_rs : Filter the Rs values to remove those that are too high

    Parameters
    ----------
    row : pandas Series
        The row of data to filter
    maxRs : float
        The maximum Rs value allowed

    Returns
    -------
    pandas Series
        The filtered row of data
    """
    if isinstance(row["Rs"], float):
        if row["Rs"] > maxRs:
            row["Rs"] = np.nan
            return row
        else:
            return row
    else:
        return row


def populate_columns(
    data: pd.DataFrame,
    configuration: dict,
    parameters: list,
    select_by: str = "Rs",
    select_limits: list = [0, 1e9],
):
    datap = data.copy(deep=True)  # defrag dataframe.
    # print("populate columns (show assemb data): ", datap.columns)
    # populate the new columns for each parameter
    if "taums" not in datap.columns:
        datap["taums"] = np.nan
    for p in parameters:
        if p not in datap.columns:
            CP.cprint("c", f"ADDING {p:s} to data columns")
            datap[p] = np.nan
        m_str = p + "_mean"
        if m_str not in datap.columns:
            datap[p + "_mean"] = np.nan
        b_str = p + f"_best{select_by:s}"
        if b_str not in datap.columns:
            datap[b_str] = np.nan
    if "age_category" not in datap.columns:
        datap["age_category"] = None
    age_cats = None
    if "age_categories" in configuration.keys():
        age_cats = configuration["age_categories"]
    
    # generate list of excluded protocols:
    # ones ending in "all" mean exclude everything
    excludes = []
    if configuration["excludeIVs"] is not None:
        for cellid in configuration["excludeIVs"]:
            for protos in configuration["excludeIVs"][cellid]["protocols"]:
                excludes.append(str(Path(cellid, protos)))
   
    datap = datap.apply(
        filter_rs, maxRs=select_limits[1] * 1e-6, axis=1
    )  # (data["Rs"].values[0] <= select_limits[1]*1e-6)
    datap.dropna(subset=["Rs"], inplace=True)  # trim out the max Rs data
    if "taum" not in data.columns:
        datap["taum"] = {}
    if "CC_taum" not in data.columns:
        datap["CC_taum"] = {}
    datap = datap.apply(transfer_cc_taum, excludes=excludes, axis=1)
   
    assert isinstance(datap["CC_taum"], pd.Series)
    datap["used_protocols"] = ""
    datap["age_category"] = datap.apply(lambda row: CatAge.categorize_ages(row, age_cats), axis=1)
    return datap


def check_types(data1, data2):
    # Compare 2 pandas series, element by element (e.g., rows)
    # print("Checking types between data1 and data2")
    pars = [
        "taum",
        "AP_thr_V",
        "AP_HW",
        "AdaptRatio",
        "AHP_trough_V",
        "AHP_trough_T",
        "AHP_depth_V",
        "AP_peak_V",
        "dvdt_rising",
    ]
    # define types suitable as float values
    floats = [np.float64, float]
    for d in data1.index:
        d2 = data2.get(d)
        d1 = data1.get(d)
        # print("looking at d, d1, d2: ",d,  type(d1), d1, type(d2), d2)
        if d in ["pars", "fit"]:
            data1[d] = innermost(data1[d])
            data2[d] = innermost(data2[d])
            d2 = data2.get(d)
            d1 = data1.get(d)
            # print("re-looking at d, d1, d2: ",d,  type(d1), d1, type(d2), d2)

        original_data = f"checktypes: {d!s}, {type(d1)}, {d1!s}, {type(d2)}, {d2!s}, floats? : {type(d1) in floats}, {type(d2) in floats}"

        if type(d1) != type(d2):
            if type(d1) in floats and type(d2) in floats:
                pass
            else:
                CP.cprint(
                    "r",
                    f"Types do not match before conversion: {d}, d1: {type(d1)}, d2: {type(d2)}",
                )
                # print(d, d in pars)
                if d in pars:
                    assert type(data1.get(d) == type(data2.get(d)))
                    if isinstance(data1[d], list):
                        data1[d] = float(innermost(data1[d])[0])  # convert to float
                    else:
                        data1[d] = float(data1[d])
                    if d == "AHP_trough_V":
                        CP.cprint("y", f"AHP_trough_V:  {data1[d]}")

                d1 = data1.get(d)  # get the new type
                if type(d1) != type(d2) and (type(d1) not in floats or type(d2) not in floats):
                    if type(d1) in floats and type(d2) is None:
                        # print("    converted d2 to float nan")
                        data2[d] = np.nan
                        d2 = data2.get(d)
                    if type(d2) in floats and type(d1) is None:
                        # print("    converted d1 to float nan")
                        data1[d] = np.nan
                        d1 = data1.get(d)
                    else:
                        CP.cprint(
                            "r", f"Types do not match after conversion: {d}, {type(d1)}, {type(d2)}"
                        )
                        print(data1.get(d), data2.get(d))
                else:
                    CP.cprint("g", f"Types match after conversion: {d}, {type(d1)}, {type(d2)}")
        if isinstance(d1, list) and isinstance(d2, list):
            # print(d1, d2)
            # print(type(d1), type(d2))
            for d1i, d2i in zip(d1, d2):
                # print("d1i, d2i: ", d1i, d2i)
                if isinstance(d1i, list) and isinstance(d2i, list):
                    # print("d1i: ", d, d1i)
                    if len(set(d1i) - set(d2i)) > 0 or len(set(d2i) - set(d1i)) > 0:
                        CP.cprint("r", f"lists are not matched {d1}, {d2}")
        elif type(d1) in floats and type(d2) in floats:
            if not np.equal(d1, d2):
                if np.isnan(d1) and np.isnan(d2):
                    break
                CP.cprint("r", f"floats are not matched, {d1}, {d2}")
        elif isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
            if not np.array_equal(d1, d2):
                CP.cprint("r", f"np arrays are not matched:  {d1}, {d2}")
        elif isinstance(d1, str) and isinstance(d2, str):
            if d1 != d2:
                CP.cprint("r", f"Strings are not matched: \n>>>{d1}\n>>>{d2}\n")
        elif d1 is None or d2 is None:
            if d1 != d2:
                CP.cprint("r", f"One of the types is 'None'; not matched: d1={d1}, d2={d2}")
        else:
            CP.cprint(
                "r", f"show_assembled_datafile: check_types: Uncaught comparision for variable: {d}"
            )
            CP.cprint("r", f"original data: {original_data}")

    # print("Done checking types")
    return data1


def perform_selection(
    data: pd.DataFrame,
    parameters: list,
    configuration: dict,
    select_by: str = "Rs",
    select_limits: list = [0, 1e9],

):
    assert configuration is not None
    assert data is not None
    # fn = experiment['databasepath']
    # print(fn.is_file())
    # select_by = "Rs"
    # cfg, d = get_configuration(str(fn))

    expts = configuration
    data = populate_columns(
        data,
        configuration=expts,
        parameters=parameters,
        select_by=select_by,
        select_limits=select_limits,
    )
    # print("In perform selection, did populate columns again")
    # check_values(data, halt=False)

    for parameter in parameters:
        # CP.cprint("c", f"**** PROCESSING*** : {parameter:s}, len: {len(data[parameter])}")
        try:
            data = data.apply(
                apply_select_by,
                parameter=parameter,
                select_by=select_by,
                select_limits=select_limits,
                axis=1,
            )
        except:
            print("Error in apply_select_by on key=", parameter)
            raise ValueError

    # check_values(data, halt=True)

    return data


parameters = [
    "Rs",
    "Rin",
    "CNeut",
    "taum",
    "CC_taum",  # CC_taum protocol
    "AP_thr_V",
    "AP_thr_T",
    "AP_HW",
    "AdaptRatio",
    "AdaptIndex",
    "AHP_trough_V",
    "AHP_trough_T",
    "AHP_depth_V",
    "AP_peak_V",
    "dvdt_rising",
    "dvdt_falling",
    "tauh",
    "Gh",
    "used_protocols",
]

def check_values(df, halt:bool=False):
    print("*"*80)
    for index in df.index[:20]:
        row = df.loc[index]
        print(row['cell_id'], row['AdaptIndex2'])
    if halt:
        print("stop, debugging")
        raise ValueError("Debugging")
        

def get_best_and_mean(
    data: pd.DataFrame, experiment: dict, parameters: list, select_by: str, select_limits: list
):
    """get_best_and_mean Taking the basic data frame that has
    all the measures (as lists per protocol), we then generate
    new columns for the measures associated with the best Rs
    and the mean.

    Parameters
    ----------
    data : pandas dataframe
        data table (this is the "assembled" data table)
    expts : the configuration dictionary
        all the various analysis control parameters for this experiment
    parameters : a list of the parameters to do these computations on
        list of strings
    select_by : str
        The parameter to select by (usually "Rs")
    select_limits : list
        values of of the select_by parameter that are acceptable.

    Returns
    -------
    pandas dataframe
        the updated data frame
    """
    # print("get_best_and_mean: 1", data["Group"].unique())
    # check_values(data, halt=False)
    data = populate_columns(
        data,
        configuration=experiment,
        parameters=parameters,
        select_by=select_by,
        select_limits=select_limits,
    )
    # print("get_best_and_mean: 2", data["Group"].unique())
    # print("after populate columns: ")
    # check_values(data, halt=False)
    data = perform_selection(
        data=data,
        configuration=experiment,
        parameters=parameters,
        select_by=select_by,
        select_limits=select_limits,
    )
    # print("after perform selection:")
    # check_values(data, halt=True)
    # print("get_best_and_mean: 3", data["Group"].unique())
    return data


def show_best_rs_data(data, experiment, select_limits):
    print("Parameters: ", parameters, "select_by", select_by)
    print("Data columns: ", data.columns)
    # for index in data.index:
    #     print("index: ", index, data.loc[index]['AP_thr_V'])
    # exit()
    data = get_best_and_mean(
        data=data,
        experiment=experiment,
        parameters=parameters,
        select_by=select_by,
        select_limits=select_limits,
    )

    #     data = populate_columns(
    #     data,
    #     configuration=expts,
    #     parameters=parameters,
    #     select_by=select_by,
    #     select_limits=select_limits,
    # )
    # # print("populated data columns: ", data.columns)
    # data = perform_selection(
    #     select_by=select_by,
    #     select_limits=select_limits,
    #     data=data,
    #     parameters=parameters,
    #     configuration=expts,
    # )
    # print("# Data columns: ", data.columns)
    # print(data['dvdt_rising_bestRs'])
    # exit()
    # print(data["age_category"])
    # for idx, row in data.iterrows():
    #     mpl.plot(data.iloc[idx].FI_Curve1[0], data.iloc[idx].FI_Curve1[1])
    #     mpl.plot(data.iloc[idx].FI_Curve4[0], data.iloc[idx].FI_Curve4[1])
    # mpl.show()
    yx = ["taum_bestRs", "taum_mean", "CC_taum_bestRs"]
    yx = [
        "AP_thr_V_bestRs",
        "AP_thr_V_mean",
        "AP_peak_V",
        "dvdt_rising_bestRs",
        "dvdt_rising_mean",
        "AP_peak_V_bestRs",
    ]
    f, ax = mpl.subplots(1, len(yx), figsize=(12, 6))
    for i, axi in enumerate(ax):
        sns.boxplot(
            x="age_category",
            y=yx[i],
            data=data,
            hue="age_category",
            order=expts["age_categories"],
            palette=expts["plot_colors"],
            ax=axi,
        )
        sns.swarmplot(
            x="age_category",
            y=yx[i],
            data=data,
            hue="age_category",
            order=expts["age_categories"],
            palette=expts["plot_colors"],
            edgecolor="k",
            linewidth=0.5,
            ax=axi,
            # dodge=True,
        )
        axi.set_title(f"{yx[i]:s}")
        axi.set_ylabel(f"taum (s)")
        # axi.set_ylim(0, 0.08)
        axi.set_xticklabels(axi.get_xticklabels(), rotation=45)
    mpl.tight_layout()
    for idx, row in data.iterrows():
        if idx >= 0 and idx < 500:
            # if data.iloc[idx].taum_bestRs > 1.0 or data.iloc[idx].taum_bestRs < 0.0 or data.iloc[idx].taum_mean > 1.0 or data.iloc[idx].taum_mean < 0.0:
            #     print("taum OUT OF BOUNDS: ", data.iloc[idx].cell_id)
            #     print("     taum mean: ", data.iloc[idx].taum_mean, "taum best rs: ", data.iloc[idx].taum_bestRs)
            #     continue
            # else:
            # mpl.plot([idx, idx], [data.iloc[idx].taum_bestRs, data.iloc[idx].taum_bestRs], 'o-')

            # print(" plot?    taum mean: ", data.iloc[idx].taum_mean, "taum best rs: ", data.iloc[idx].taum_bestRs)
            # if data.iloc[idx].taum_bestRs > 0.02:
            #     print("\ntaum too high: ", data.iloc[idx].cell_id, data.iloc[idx].taum_bestRs, data.iloc[idx].taum)
            #     print("               ", data.iloc[idx].Rs)
            #     print("       ", data.iloc[idx].protocols)
            pass

    mpl.show()
    # print("taum mean: ", data["taum_mean"].values)
    # print("taum best rs: ", data["taum_bestRs"].values)
    # print("taum raw: ", data["taum"].values)
    # print("rs best rs: ", data["Rs_bestRs"].values)
    # for p in parameters:
    #     print(f"{p} # best Rs: ", data[f"{p:s}_bestRs"].values, "mean: ", data[f"{p:s}_mean"].values)
    #     print(f"    f{p}:  {data[f'{p:s}'].values!s}")

    # for u in data["used_protocols"].values:
    #     print(f"used protocols: , {u:s}")
    # print("Rin mean: ", data["Rin_mean"].values)
    # print("Rin raw: ", data["Rin"].values)


def categorize_ages(row, experiment: dict):
    age_category = "NA"
    if row.age == "P0D ?":
        return np.nan
    intage = parse_ages.age_as_int(parse_ages.ISO8601_age(row.age))
    for k in experiment["age_categories"].keys():
        if (
            intage >= experiment["age_categories"][k][0]
            and intage <= experiment["age_categories"][k][1]
        ):
            age_category = k
    return age_category


def mean_adaptation(row):
    if row.AdaptIndex is not None:
        row.ADI = np.nanmean(row.AdaptIndex)
    if row.AdaptRatio is not None:
        row.ADR = np.nanmean(row.AdaptRates)
    return row


if __name__ == "__main__":
    # print(data.head(10))
    import matplotlib.pyplot as mpl

    import ephys.tools.parse_ages as parse_ages

    # fn = Path("/Users/pbmanis/Desktop/Python/mrk-nf107/config/experiments.cfg")
    # fn = Path("/Users/pbmanis/Desktop/Python/Maness_ANK2_nex/config/experiments.cfg")
    fn  = Path("/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg")
    # fn = Path("./config/experiments.cfg")

    select_by = "Rs"
    cfg, d = get_configuration(str(fn))
    # exptname = "VM_Dentate"
    exptname = "GlyT2_NIHL"
    print(cfg)
    experiment = d[exptname]
    expts = experiment

    assembled_filename = Path(
        expts["analyzeddatapath"], expts["directory"], expts["assembled_filename"]
    )
    print(assembled_filename)
    data = read_pickle(assembled_filename, compression="gzip")
    assembled_time = assembled_filename.stat().st_mtime
    print("assembled data columns: ", data.columns)
    print("assembled data: ", data.iloc[0].keys())
    # print(data["post_durations"].values)
    # print(data["post_rates"].values)
    # print(data["post_spike_counts"].values)
    # print(data["FI_Curve1"][0][0]*1e9)
    for i, row in data.iterrows():
        pkl = Path(FT.get_cell_pkl_filename(experiment=experiment, df=data, cell_id=row.cell_id))
        pkl_time = pkl.stat().st_mtime

        print(
            i, 
            "cell id: ",
            pkl,
            pkl.is_file(),
            datetime.datetime.fromtimestamp(pkl.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )
        if pkl_time > assembled_time:
            pkl_d = datetime.datetime.fromtimestamp(pkl_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            ass_d = datetime.datetime.fromtimestamp(assembled_time).strftime("%Y-%m-%d %H:%M:%S")
            CP.cprint(
                "r",
                f"     *pkl file: {pkl.name} at {pkl_d} is newer than assembled data file: {assembled_filename.name} at {ass_d}",
            )


    # print("AP Peak, thr, subject, protocol: ", data.AP_peak_V, data.AP_thr_V, data.Subject, data.protocol)
    # exit()
    for index in data.index:
        print(
            "index: ",
            index,
            data.loc[index]["Subject"],
            data.loc[index]["protocol"],
            "AI2: ",
            data.iloc[index].AdaptIndex2,
            "FSL: ", 
            data.iloc[index].post_latencies,
        )
        # print("     values: ", data.iloc[index])

        # print(
        #     "Peak V: ",
        #     data.iloc[index].AP_peak_V,
        #     "thr: ",
        #     data.iloc[index].AP_thr_V,
        #     "trough: ",
        #     data.iloc[index].AHP_trough_V,
        # )  # , data.iloc[index].AP_thr_V)
        # print(
        #     "    AHP depth: ",
        #     data.iloc[index].AHP_depth_V,
        #     "Relative depth: ",
        #     data.iloc[index].AHP_relative_depth_V,
        # )
        # print(
        #     data.loc[index]['post_latencies']
        # )
        # print(data.iloc[index].AdaptRatio2) # , data.iloc[index].protocol)
        # print(data.iloc[index].Subject, data.iloc[index].protocol)
        # print(data.iloc[index].cell_id)
        # print(data.iloc[index].FI_Curve1)
        # print(data.iloc[index].FI_Curve4)
        # print(np.array(data.AP_peak_V.values) - np.array(data.AP_thr_V.values))
    exit()

    df_summary_filename = Path(
        expts["analyzeddatapath"], expts["directory"], expts["datasummaryFilename"]
    )
    df_summary = read_pickle(df_summary_filename, compression="infer")

    data["ADI"] = {}
    data["ADR"] = {}
    data["age_category"] = None
    data["age_category"] = data.apply(categorize_ages, experiment=experiment, axis=1)

    data = data.apply(mean_adaptation, axis=1)
    data.dropna(subset=["age"], inplace=True)
    # print(experiment["plot_order"]["age_category"])
    # f, ax = mpl.subplots(1,1)
    # sns.boxplot(
    #     x="age_category",
    #     y="ADI",
    #     data=data,
    #     hue="age_category",
    #     palette=experiment["plot_colors"],
    #     order=experiment["plot_order"]["age_category"],
    #     # edgecolor="black",
    #     # size=2.5,
    #     linewidth=0.5,
    #     zorder=50,
    #     ax=ax,
    #     saturation=0.5,
    # )
    # sns.swarmplot(
    #         x="age_category",
    #         y="ADI",
    #         data=data,
    #         hue="age_category",
    #         palette=experiment["plot_colors"],
    #         hue_order=experiment["plot_order"]["age_category"],
    #         edgecolor="black",
    #         size=2.5,
    #         linewidth=0.5,
    #         zorder=100,
    #         ax=ax,
    #         alpha=0.9,
    #     )

    # ax.set_ylim(-1, 1)

    f, ax = mpl.subplots(2, 2)
    ax = ax.ravel()
    cells = ["pyramidal"]  # , 'tuberculoventral', 'cartwheel']

    # groups = ['B', 'A', 'AA', "AAA"]
    groups = ["Pubescent", "Young Adult", "Mature Adult"]
    for i, cell in enumerate(cells):
        dfn = data[data.cell_type == cell]
        # for ix in dfn.index:
        #         print(dfn.loc[ix].cell_type, dfn.loc[ix].cell_id, FT.make_cellid_from_slicecell(dfn.loc[ix].cell_id))
        #         # continue
        # continue
        ax[i].set_title(f"{cell:s}")
        for j, group in enumerate(groups):
            dfg = dfn[dfn.Group == group]
            # print("ct: ", cell, "group: ", group, "len: ", len(dfg))
            if i == 0 and j == 0:
                print("dataframe grouping columns: \n", dfg.columns)
            # just to keep it simple, sort so we can compare
            dfg.sort_values(
                by=["cell_id"], ignore_index=False, inplace=True
            )  # sort by date within the group
            for cellidx in dfg.index:
                dc = dfg.loc[cellidx]
                # print(cellidx, cell, dc.cell_id)
                # print(dir(dc))
                if cell == "tuberculoventral":
                    proper_cellid = FT.make_cellid_from_slicecell(dc.cell_id)
                    print(
                        f"\n{group:3s}: Cell index: : {cellidx:4d}, {dc.cell_id:32s} ProperID: {proper_cellid:s}"
                    )
                    print(f"      analyzed Protocols: {dc.protocols}")
                    print(
                        "       complete Protocols: ",
                        df_summary[df_summary.cell_id == proper_cellid]["data_complete"].values,
                    )
                    print("       len fi curve: ", len(dc.FI_Curve1), len(dc.FI_Curve1[0]))
                    if len(dc.FI_Curve1) == 0:
                        print("Nothing in FI_Curve1 for: ", dc.cell_id)
                        continue
                color = experiment["plot_colors"][group]
                lw = 0.5

                if cell == "tuberculoventral" and proper_cellid in [
                    "2018.07.27_000/slice_001/cell_000"
                ]:
                    color = "k"
                    lw = 1.5
                    print("*****")
                    print(dc.FI_Curve1)
                fi = dc.FI_Curve1
                # print("FI is: ", fi)
                ax[i].plot(fi[0], fi[1], label=f"{cell:s}, {group:s}", color=color, lw=lw)
            ax[i].legend()

    mpl.show()
