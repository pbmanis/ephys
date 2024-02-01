"""
Filter_data: filter ephys tables according to some criteria. 
Criteria include RMP, Rin, taum, group, internal solution, temperature and
cells in the experiment exclusion list.

Provides utilities to list which cells are accepted by the filters, 
and which cells are rejected.
Filtering may be done at the prototol level as well.

The filter is applied to the dataframe, and the dataframe is returned.

"""
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List
from dataclasses import dataclass, field, asdict
from pylibrary.tools import cprint as CP


def def_empty_list():
    return []


def def_RMP():
    return [-87.5, -55.0]


def def_Rin():
    return [5, 1000]


def def_taum():
    return [0.25e-3, 35e-3]


@dataclass
class FilterSettings:
    RMP: List = field(default_factory=def_RMP)  # min and max RMP, V
    SDRMP: float=0.0025  # SD of RMP, V
    Rin: List = field(default_factory=def_Rin)  # min and max Rin, Ohm
    taum: List = field(
        default_factory=def_taum
    )  # min and max membrane time constant, seconds
    AP_Min_Abs_Ht: float = 0.0  # minimum absolute height of an action potential, in V
    AP_Min_Rel_Ht: float = (
        0.020  # minimum absolute heighe of action potential, from RMP, in V
    )
    min_sample_rate: float = 1e4  # minimum sampling rate, in Hz (points per second)
    junctionpotential: float = -12  # junction potential, in mV


class FilterDataset:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.FS = FilterSettings()

    def set_RMP(self, rmplimits: list):
        assert isinstance(rmplimits, list)
        assert len(rmplimits) == 2
        self.FS.RMP = sorted(rmplimits)
    
    def set_SDRMP(self, sdrmp: float):
        assert isinstance(sdrmp, float)
        assert sdrmp >= 0.0
        self.FS.SDRMP = sdrmp

    def set_Rin(self, rinlimits: list):
        assert isinstance(rinlimits, list)
        assert len(rinlimits) == 2
        self.FS.Rin = sorted(rinlimits)

    def set_taum(self, taulimits: list):
        assert isinstance(taulimits, list)
        assert len(taulimits) == 2
        self.FS.taum = sorted(taulimits)

    def set_min_sample_rate(self, minrate: float):
        assert isinstance(minrate, float)
        assert minrate > 0.0
        self.FS.min_sample_rate = minrate

    def make_cell_id2(self, row):
        """make_cell_id2 make short cell id in format:
        yyyy.mm.dd_000_S0C0

        Parameters
        ----------
        row : _type_
            pandas row
        """
        full_id = row["cell_id"]
        id_parts = Path(full_id).parts
        row.cell_id2 = id_parts[-1]

        return row.cell_id2

    def filter_data_entries(
        self,
        df: pd.DataFrame,
        remove_groups: list = ["C", "D", "?", "X", "30D", "nan"],
        excludeIVs: list = [],
        exclude_internals: list = ["cesium", "Cesium"],
        exclude_temperatures: list = ["25C", "room temp"],
        exclude_unimportant: bool = False,
        verbose: bool = True,
    ):
        """clean_data_files Remove cells by group, exclusion,
        internal solution, temperature, empty cell_id.

        Parameters
        ----------
        df : pandas dataframe
            dataframe to clean

        remove_groups : list, optional, default=[]
            list of groups to remove from the analysis

        excludeIVs : list, optional, default=[]
            list of IVs to exclude from the analysis in format:
                "cellid...: reason for exclusion"
            where the reason for exclusion is a string that will be printed



        Returns
        -------
        _type_
            _description_
        """
        CP.cprint("y", "Filtering data entries")
        if remove_groups is not None:
            for group in remove_groups:
                CP.cprint("y", f"    Filtering out group: {group:s}")
                df.drop(df.loc[df["Group"] == group].index, inplace=True)

        df["cell_id2"] = df.apply(self.make_cell_id2, axis=1)
        if excludeIVs is None:
            excludeIVs = []
        for fns in excludeIVs:
            fn = fns.split(":")[0]  # remove the comment
            if fn in list(df.cell_id2.values):
                df.drop(df.loc[df.cell_id2 == fn].index, inplace=True)
                CP.cprint(
                    "r",
                    f"Excluding {fn:s} from analysis, reason = {fns.split(':')[1]:s}",
                )
        # exclude certain internal solutions
        if exclude_internals is not None:
            CP.cprint("y", f"    Starting with Internal Solutions: {df['internal'].unique()!s}")
            for internal in exclude_internals:
                CP.cprint("y", f"    Filtering out recordings with internal: {internal:s}")
                df = df[df["internal"] != internal]
            CP.cprint("y", f"   Remaining Internal Solutions: {df['internal'].unique()!s}")
 
        # exclude certain temperatures
        if exclude_temperatures is not None:
            CP.cprint("y", f"    Starting with Temperatures: {df['temperature'].unique()!s}")
            for temperature in exclude_temperatures:
                CP.cprint("y", f"    Filtering out recordings with temperature: {temperature:s}")
                df = df[df["temperature"] != temperature]
            CP.cprint("y", f"    Remaining temperatures: {df['temperature'].unique()!s}")
 
        # exclude "not" important
        if exclude_unimportant:
            CP.cprint("y", f"    Starting with Important: {df['important'].unique()!s}")
            print(len(df))
            df = df[df["important"] == True]
            print(len(df))
            CP.cprint("y", f"    Remaining important: {len(df['important'])!s}")
        else:  #
            CP.cprint("c", f"    not excluding the unimportant entries - Remaining entries: {len(df['important'])!s}")


        
        if verbose:
            print("Groups Accepted: ", list(set(df.Group)))
            print("Internal Solutions Accepted: ", list(set(df.internal)))
            print("Temperatures Accepted: ", list(set(df.temperature)))
            print("Dates in combined data after culling: ", sorted(list(set(df.date))))
            print("   # days: ", len(list(set(df.date))))
            print("   # selected: ", len(df))

        print("Unique protocols: ", list(set(df.protocol)))
        print("Unique groups: ", list(set(df.Group)))
        return df

    def do_filters(self, row):
        row.data_OK = True
        any_rej = False
        rowIV = row
        # rowIV = row[k]
        # print("rowIV: ", rowIV, k)
        # # print("\nsk: ", k, "rowIV: ", row[k])
        # if len(list(rowIV.keys())) == 0:
        #     rowIV = {}
        #     rowIV['RMP'] = np.nan
        #     rowIV['Rin'] = np.nan
        #     rowIV['taum'] = np.nan
        #     row.dataOK = False
        #     return row
        any_rej = {"RMP": False, "Rin": False, "taum": False, "RMP_SD": False}
        if (
            rowIV["RMP"] + self.FS.junctionpotential < self.FS.RMP[0]
            or rowIV["RMP"] + self.FS.junctionpotential > self.FS.RMP[1]
        ):
            any_rej["RMP"] = True
        if rowIV["RMP_SD"] > self.FS.SDRMP:
            any_rej["RMP_SD"] = True
        if rowIV["Rin"] < self.FS.Rin[0] or rowIV["Rin"] > self.FS.Rin[1]:
            any_rej["Rin"] = True
        if rowIV["taum"] < self.FS.taum[0] or rowIV["taum"] > self.FS.taum[1]:
            any_rej["taum"] = True

        row.data_OK = True
        fs = asdict(self.FS)
        icol = {"RMP": "r", "Rin": "m", "taum": "b"}
        for rd in ["RMP", "SD_RMP", "Rin", "taum"]:
            if any_rej[rd]:
                prots = [Path(p).name for p in rowIV["protocols"]]
                plist = ", ".join(prots)
                if rd == "RMP":
                    jp = self.FS.junctionpotential
                else:
                    jp = 0.0
                CP.cprint(
                    icol[rd],
                    f"   Removing datasets {rowIV['ID']:s} ({rowIV['cell_type']:>12s}) {plist:s}",
                )
                CP.cprint(
                    icol[rd],
                    f"           {rd:s} = {jp+rowIV[rd]:.3f} {rowIV[rd]:.3f} not in range {fs[rd][0]:.3f} - {fs[rd][1]:.3f}",
                )
                row.data_OK = False
            rd = np.nan
        return row

    def filter_data(self):
        """filter_data apply the filter to the dataframe"""
        newdf = self.df.copy()
        newdf["data_OK"] = True
        newdf = newdf.sort_values("cell_type")
        newdf = newdf.apply(self.do_filters, axis=1)

        newdf.drop(newdf.loc[newdf["data_OK"] == False].index, inplace=True)
        return newdf

    def show_rejected_data(self):
        newdf = self.df.copy()


if __name__ == "__main__":
    """test..."""
    fn = Path(
        "/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets/NF107Ai32_NIHL/NIHL_combined_by_cell.pkl"
    )
    fn_excel = Path(
        "/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets/NF107Ai32_NIHL/NIHL_combined_by_cell.xlsx"
    )
    if not fn.is_file():
        raise FileNotFoundError(f"Could not find file: {str(fn):s}")
    df = pd.read_pickle(fn,  compression="gzip")
    df.to_excel(fn_excel)
    exit()

    print("Columns: ", df.columns)
    ivs = df.keys()
    print("ivs: ", ivs)
    print(df.index)
    # fiv = list(df[ivs[0]].keys())
    # print("fiv: ", fiv)
    # print(df[ivs][0][fiv[0]].keys())
    FD = FilterDataset(df)
    newdf = FD.filter_data()
    print("old: ", len(df), "new: ", len(newdf))
