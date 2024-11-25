# show assembed file data

from ephys.tools.get_configuration import get_configuration
from pathlib import Path
from pandas import read_pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import seaborn as sns
import ephys.tools.categorize_ages as CatAge
import pylibrary.tools.cprint as CP


def transfer_cc_taum(row, excludes: list):
    """
    transfer_cc_taum : This routine is used to transfer the taum value from the
    CC_taum protocol to the taum column. This is done to allow the selection
    of the best value for taum from the CC_taum protocol, if it exists.
    """
    row["CC_taum"] = [np.nan] * len(row["protocols"])
    if "taum" not in row.keys():
        row["taum"] = row["CC_taum"]
    n = 0
    for i, p in enumerate(row["protocols"]):
        proto = Path(p)
        if str(proto) in excludes:
            CP.cprint("y", f"       Excluding: {proto!s}")
            continue
        if proto.name == "all":
            return row["CC_taum"]
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
    return row["CC_taum"]


def apply_select_by(row, parameter: str, select_by: str, select_limits: list):
    """
    apply_select_by : Here we filter the data from the different protocols
    in the measurements for this row on the "select_by" criteria (measurement type, limits).
    We then select the "best" value to report for the selected parameter based on the filtered data.

    Usually, this will be when the select_by parameter with the lowest value(s),
    where the parameter value is not nan, and the select_by value is not nan.
    Typically, the selection will be for the protocol with the lowest Rs that
    has a valid measurement for the parameter.

    This routine stes the following columns in the row:
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
    if verbose:
        print("Selector to use: ", select_by)
        print("Parameter to test, value: ", parameter, row[parameter])
    # first, convert the measurements to a list if they are not already
    if parameter not in row.keys():
        CP.cprint("y", f"Parameter {parameter:s} is not in current data row")
        raise ValueError
        print("     row keys: ", row.keys())
        return row
    if parameter == "used_protocols":
        return row
    if isinstance(row[parameter], float):
        row[parameter] = [row[parameter]]
    # if there is just one value, propagate it to all protocols
    if len(row[parameter]) == 1 and np.isnan(row[parameter][0]):
        row[parameter] = [row[parameter][0]] * len(row["protocols"])
    # if no valid measurements, just return the row
    if verbose:
        print("row par: ", row[parameter])
        print("type: ", type(row[parameter]))
        print("row parameter: ", row[parameter])
    if not isinstance(row[parameter], str):
        if np.all(np.isnan(row[parameter])):
            CP.cprint("r", f"Parameter {parameter:s} is all nan")
            row[parameter] = np.nan
            return row

    # Standard measurements: the mean of ALL of the measurements
    # collected across all protocols (data in a new column, representing the
    # mean)

    # now do the selection. Find out which protocols have the
    # lowest select_by measurement
    # these arrays are the same length as the number of protocols
    selector_values = np.array(row[select_by])
    # print("parametre, row[parameter]: ", parameter, row[parameter])
    params = np.array(row[parameter])
    prots = row["protocols"]
    if verbose:
        print("selector_values: , ", selector_values)
        print(
            "   # measures, selectors, protocols : ",
            params.shape,
            selector_values.shape,
            len(prots),
        )
        print("   row[parameter], selector values: ", params, selector_values)

    valid_measures = np.argwhere(~np.isnan(params)).ravel()
    if verbose:
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
    if verbose:
        print("prots: ", prots)
    for i, prot in enumerate(prots):
        if i not in valid_measures:  # no measure for this protocol, so move on
            continue
        if verbose:
            print("prot: ", prot)
        p = str(Path(prot).name)  # get the name of the protocol
        # skip specific protocols
        if p.startswith(
            "analysistime"
        ):  # where did this come from? I don't see where it is written.
            continue
        # if p.startswith("CC_taum"):
        #     continue
        if verbose:
            print("select by: ", select_by)
            print("index: ", i)
            print("row selection data: ", row[select_by])

        if isinstance(row[select_by], float):
            select_value = row[select_by]
        else:
            select_value = row[select_by][i]
        value = params[i]
        if np.isnan(value):
            continue
        # this will be set on the first run if min_val is correctly initialized
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

    if verbose:
        print("iprot, eq_mins, values, taums: ", iprots, equal_mins, values, taums)
    if len(iprots) == 0:
        CP.cprint(
            "r",
            f"No minimum value found for: {row.cell_id!s}, {params!s}, {valid_measures!s}, {iprots!s}, {equal_mins!s}",
        )
        if verbose:
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
        row[parameter + f"_mean"] = np.nanmean(
            [v for i, v in enumerate(params) if not prots[i].startswith("CC_taum")]
        )
    else:  # specific to CC_taum protocol
        row[parameter + f"_mean"] = np.nanmean(
            [v for i, v in enumerate(params) if prots[i].startswith("CC_taum")]
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


def populate_columns(
    data: pd.DataFrame,
    configuration: dict = None,
    parameters: list = None,
    select_by: str = "Rs",
    select_limits: list = [0, 1e9],
):
    data = data.copy() # defrag dataframe.
    # populate the new columns for each parameter
    if "taums" not in data.columns:
        data["taums"] = np.nan
    for p in parameters:
        if p not in data.columns:
            CP.cprint("c", f"ADDING {p:s} to data columns")
            data[p] = np.nan
        m_str = p + "_mean"
        if m_str not in data.columns:
            data[p + "_mean"] = np.nan
        b_str = p + f"_best{select_by:s}"
        if b_str not in data.columns:
            data[b_str] = np.nan
    if "age_category" not in data.columns:
        data["age_category"] = None
    age_cats = configuration["age_categories"]

    # generate list of excluded protocols:
    # ones ending in "all" mean exclude everything
    excludes = []
    for cellid in configuration["excludeIVs"]:
        for protos in configuration["excludeIVs"][cellid]["protocols"]:
            excludes.append(str(Path(cellid, protos)))

    data["CC_taum"] = data.apply(transfer_cc_taum, excludes=excludes, axis=1)
    assert isinstance(data["CC_taum"], pd.Series)
    data["used_protocols"] = ""
    data["age_category"] = data.apply(lambda row: CatAge.categorize_ages(row, age_cats), axis=1)
    return data


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
            CP.cprint("r", f"show_combined: check_types: Uncaught comparision for variable: {d}")
            CP.cprint("r", f"original data: {original_data}")

    # print("Done checking types")
    return data1


def perform_selection(
    select_by: str = "Rs",
    select_limits: list = [0, 1e10],
    data: pd.DataFrame = None,
    parameters: list = None,
    configuration: dict = None,
):
    assert configuration is not None
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
    for parameter in parameters:
        CP.cprint("r", f"**** PROCESSING*** : {parameter:s}, len: {len(data[parameter])}")
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

    # for idx, row in data.iterrows():
    #     for parameter in parameters:
    #         print("param, idx: ", parameter, idx)
    #         rowdat = apply_select_by(
    #             row,
    #             parameter=parameter,
    #             select_by=select_by,
    #             select_limits=select_limits,
    #         )
    #         data.loc[idx] = check_types(rowdat, data.loc[idx])
    #         #check_types(rowdat, data.loc[idx])

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
    data = populate_columns(
        data,
        configuration=experiment,
        parameters=parameters,
        select_by=select_by,
        select_limits=select_limits,
    )

    data = perform_selection(
        data=data,
        configuration=experiment,
        parameters=parameters,
        select_by=select_by,
        select_limits=select_limits,
    )
    return data


if __name__ == "__main__":
    # print(data.head(10))
    import matplotlib.pyplot as mpl

    fn = Path("/Users/pbmanis/Desktop/Python/RE_CBA//config/experiments.cfg")
    fn = Path("/Users/pbmanis/Desktop/Python/Maness_ANK2_nex/config/experiments.cfg")
    print(fn.is_file())
    select_by = "Rs"
    cfg, d = get_configuration(str(fn))

    experiment = d[cfg[0]]
    expts = experiment

    assembled_filename = Path(expts["analyzeddatapath"], expts['directory'], expts["assembled_filename"])
    print(assembled_filename)
    data = read_pickle(assembled_filename)
    select_limits = [0, 1e9]
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
    yx = ["AP_thr_V_bestRs", "AP_thr_V_mean", "dvdt_rising_bestRs", "dvdt_rising_mean"]
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
