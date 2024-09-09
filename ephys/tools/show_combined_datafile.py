# show assembed file data

from ephys.tools.get_configuration import get_configuration
from pathlib import Path
from pandas import read_pickle
import pandas as pd
import numpy as np

fn = Path("/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg")
print(fn.is_file())

cfg, d = get_configuration(str(fn))

expts = d[cfg[0]]

assembled_filename = Path(expts["analyzeddatapath"], cfg[0], expts["assembled_filename"])
print(assembled_filename)
data = read_pickle(assembled_filename)
# print(data.head())
# print(data["Rs"].values)
# print(data["protocols"].values)
# print(data["AP_thr_V"].values)
select_by = "protocols"

values = "mean"  # or "lowest_Rs"
parameters = [
    "Rs",
    "Rin",
    "CNeut",
    "taum",
    # "taum_2",  # data from the "taum" protocol
    "AP_thr_V",
    "AP_thr_T",
    "AP_HW",
    "AdaptRatio",
    "AHP_trough_V",
    "AHP_trough_T",
    "AHP_depth_V",
    "tauh",
    "Gh",
]


def _apply_select_by(row, parameter: str, select_by: str, select_limits: list):
    """
    _apply_select_by : apply the select_by criteria to the data, and return the selected values

    We select the measurement based on the select_by parameter with the lowest value(s),
    where the parameter value is not nan, and the select_by value is not nan.
    
    This routine stes the following columns in the row:
        'parameter'_mean : the mean of the parameter values
        'parameter'_bestselect_by : the best value of the parameter
        'used_protocols' : a list of the protocols used to select the best value,
            in the form 'parameter:protocol'
    """

    # handle the case where there is no data, which may be indicated by a single [nan]
    # representing all protocols:
    verbose = False
    # if parameter in ["AP_thr_V"]: # , "AHP_depth_V"]:

    if len(row[parameter]) == 1 and np.isnan(row[parameter][0]):
        row[parameter] = [row[parameter][0]] * len(row["protocols"])


    if np.all(np.isnan(row[parameter])):
        return row
    # ow[parameter + f"_best{select_by:s}"] = np.nan
    row[parameter + "_mean"] = np.nanmean(row[parameter])
    selector = np.array(row[select_by])
    params = np.array(row[parameter])
    if verbose:
        print("   # measures : ", params.shape)
        print("   # select_by: ", selector.shape)
        print("   row[parameter]: ", params, selector)

    # if params.shape != selector.shape:
    #     print(
    #         "Shapes do not match: ", row.cell_id, parameter, select_by, params.shape, selector.shape
    #     )
    #     print("    protocols: ", row["protocols"])
    #     print("************************************************")
    #     return row
    valids = np.argwhere(~np.isnan(params)).ravel() #  & ~np.isnan(selector)).ravel()
    if verbose:
        print("Params: ", params)
        print("selects: ", selector)
        print("valid indices: ", valids)

    if len(valids) == 0:  # no matching values available, set nans.
        row[parameter + f"_best{select_by:s}"] = np.nan
        row[parameter + f"_mean"] = np.nan
        row["used_protocols"] = " ".join((row["used_protocols"], f"{parameter:s}:None"))
        return row
    
    prots = row["protocols"]
    # brute force
    # find value(s) associated with the minimum Rs that are in the valid IV list
    min_val = np.inf
    iprot = None
    equal_mins = []
    values = []
    for i, s in enumerate(selector):
        if i not in valids:
            continue
        p = str(Path(row.protocols[i]).name)
        # skip some protocols
        if p.startswith("CC_taum"):
            continue
        if p.startswith("analysistime"):  # where did this come from? I don't see where it is written.
            continue
        print(parameter, row.cell_id, str(Path(prots[i]).name))
        print(i, s)
        if s < min_val and i in valids:
            min_val = s  # set the value
            iprot = i  # mark the index
            equal_mins = []  # reset the list of equal minima
            values = [params[i]]
            continue
        if s == min_val and i in valids:
            if iprot not in equal_mins:
                equal_mins.append(iprot)
                values = [params[iprot]]
            equal_mins.append(i)
            values.append(params[i])
    print("iprot, eq: ", iprot, equal_mins, values)
    if iprot is None or len(equal_mins) == 0:
        print("No minimum value found for: ", row.cell_id, params, valids)
        print("    protocols: ", row[select_by])
        print(row[parameter])
        return row

    if len(equal_mins) == 0:
        print("Single value: ")
        row[parameter + f"_best{select_by:s}"] = params[iprot]
        aprots = f"{parameter:s}:{str(Path(prots[iprot]).name):s}"
    else:
        print("Multiple averaged")
        row[parameter + f"_best{select_by:s}"] = np.mean(params[equal_mins])
        aprots = f"{parameter:s}:{','.join([str(Path(prots[i]).name) for i in equal_mins]):s}"
    row["used_protocols"] = ",".join(
        (row["used_protocols"], aprots)
    )

    return row


select_by = "Rs"
select_limits = sorted([0, 1e9])  # Rs limits for selection

# populate the new columns for each parameter
for p in parameters:
    data[p + "_mean"] = np.nan
    data[p + f"_best{select_by:s}"] = np.nan
data["used_protocols"] = ""


print("data columns: ", data.columns)
for idx, row in data.iterrows():

    for parameter in parameters:
        data.iloc[idx] = _apply_select_by(
            row, parameter, select_by="Rs", select_limits=select_limits
        )

# print(data.head(10))
import matplotlib.pyplot as mpl

# for idx, row in data.iterrows():
#     mpl.plot(data.iloc[idx].FI_Curve1[0], data.iloc[idx].FI_Curve1[1])
#     mpl.plot(data.iloc[idx].FI_Curve4[0], data.iloc[idx].FI_Curve4[1])
# mpl.show()
for idx, row in data.iterrows():
    if idx >= 0 and idx < 500:
        if data.iloc[idx].taum_bestRs > 1.0 or data.iloc[idx].taum_bestRs < 0.0 or data.iloc[idx].taum_mean > 1.0 or data.iloc[idx].taum_mean < 0.0:
            print("taum OUT OF BOUNDS: ", data.iloc[idx].cell_id)
            print("     taum mean: ", data.iloc[idx].taum_mean, "taum best rs: ", data.iloc[idx].taum_bestRs)
            continue
        else:
            mpl.plot([idx, idx], [data.iloc[idx].taum_bestRs, data.iloc[idx].taum_bestRs], 'o-')
            # print(" plot?    taum mean: ", data.iloc[idx].taum_mean, "taum best rs: ", data.iloc[idx].taum_bestRs)
        if data.iloc[idx].taum_bestRs > 0.02:
            print("\ntaum too high: ", data.iloc[idx].cell_id, data.iloc[idx].taum_bestRs, data.iloc[idx].taum)
            print("               ", data.iloc[idx].Rs)
            print("       ", data.iloc[idx].protocols)


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
