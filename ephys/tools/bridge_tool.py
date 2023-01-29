import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pylibrary.tools import cprint as CP
from pyqtgraph import configfile
import ephys.datareaders as DR

re_ccdata = re.compile("^(CCIV_|Ic_)", re.IGNORECASE)
re_mapdata = re.compile("^(Map_|LED)", re.IGNORECASE)
re_vcdata = re.compile("^(VC|vc)", re.IGNORECASE)
main_df_name = "../mrk-nf107-data/datasets/NF107Ai32_Het/NF107Ai32_Het_26Jan2023.pkl"

old_bridge_name = "../mrk-nf107-data/datasets/NF107Ai32_Het/NF107Ai32_Het_bcorr2.pkl"
new_bridge_name = (
    "../mrk-nf107-data/datasets/NF107Ai32_Het/NF107Ai32_Het_BridgeCorrections.xlsx"
)

def highlight_by_status(row):
    if not row.BridgeEnabled:
            style = [f"background-color: magenta" for s in range(len(row))] #  for s in range(len(row))]
    elif not np.isnan(row["BridgeAdjust(ohm)"]):
            style = [f"background-color: lightgreen" for s in range(len(row))] # for s in range(len(row))]
    else:
         style = [None for s in range(len(row))]
    return style

def save_excel(df, outfile):
        writer = pd.ExcelWriter(outfile, engine='xlsxwriter')

        df.to_excel(writer, sheet_name = "Sheet1")
        # df = self.organize_columns(df)
        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]
        fdot3 = workbook.add_format({'num_format': '####0.000'})
        df.to_excel(writer, sheet_name = "Sheet1")

        resultno = []
        # resultno = ['holding', 'RMP', 'Rin', 'taum', 'dvdt_rising', 'dvdt_falling', 
        #     'AP_thr_V', 'AP_HW', "AP15Rate", "AdaptRatio", "AP_begin_V", "AHP_trough_V", "AHP_depth_V"]
        # df[resultno] = df[resultno].apply(pd.to_numeric)    
        for i, column in enumerate(df):
            # print('column: ', column)
            if column in resultno:
                writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1,  cell_format=fdot3)
            if column not in ['notes', 'description', 'OriginalTable', 'FI_Curve']:
                coltxt = df[column].astype(str)
                coltxt = coltxt.map(str.rstrip)
                maxcol = coltxt.map(len).max()
                column_width = np.max([maxcol, len(column)]) # make sure the title fits
                if column_width > 50:
                    column_width = 50 # but also no super long ones
                #column_width = max(df_new[column].astype(str).map(len).max(), len(column))
            else:
                column_width = 25
            if column_width < 8:
                column_width = 8
            if column in resultno:
                writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, cell_format=fdot3, width=column_width) # column_dimensions[str(column.title())].width = column_width
                # print(f"formatted {column:s} with {str(fdot3):s}")
            else:
                writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width

        df = df.style.apply(highlight_by_status, axis=1)
        df.to_excel(writer, sheet_name = "Sheet1")
        writer.close()

def recolor(filename):
    print('filename: ', filename)
    df = pd.read_excel(filename, engine="openpyxl")
    save_excel(df, filename)

def _add_cell_id(row):
    row.cell_id = str(Path(row.date, row.slice_slice, row.cell_cell))
    return row


def make_new_bridge_corr(old_bridge_name, new_bridge_name, main_df_name):
    """make the new bridge file format (just Cell_ID/protocol)"""
    main_df = pd.read_pickle(main_df_name)
    bridge_data = []
    old_bridge_df = pd.read_pickle(old_bridge_name)
    # print(old_bridge_df.columns)
    old_bridge_df["cell_id"] = ""
    old_bridge_df = old_bridge_df.apply(_add_cell_id, axis=1)
    # print(old_bridge_df.columns)
    for idx in main_df.index:
        # if idx > 90:
        #     break
        cell_id = main_df.iloc[idx]["cell_id"]
        # print(cell_id)
        prots = main_df.iloc[idx]["data_complete"].replace(" ", "").split(",")
        oldrowidx = old_bridge_df.index[old_bridge_df["cell_id"] == cell_id].to_list()
        if len(oldrowidx) > 0:  # check to see if cell id is in the old bridge database
            olddata = old_bridge_df.iloc[oldrowidx[0]]
            for (
                p
            ) in (
                prots
            ):  # check to see if the protocol was done and the BridgeAdjust parameter was added
                oldvalue = np.nan
                bridge_enabled = False
                bridge_resistance = np.nan

                if re_ccdata.match(p) is not None:  # but only for protocols in CC
                    # CP.cprint("y", f"{cell_id:s}/{p:s}")  # list the protocol here
                    # look in the old bridge datafile for the bridge value
                    oldkey = Path(olddata.cell_id, p)
                    if (
                        oldkey in list(olddata.IV.keys())
                        and "BridgeAdjust" in olddata.IV[oldkey]
                    ):
                        # print(olddata.IV[oldkey].keys())
                        oldvalue = olddata.IV[oldkey]["BridgeAdjust"]
                        ccinfo = olddata.IV[oldkey]['CCComp']
                        bridge_enabled = ccinfo['CCBridgeEnable'] == 1
                        bridge_resistance = ccinfo['CCBridgeResistance']
                    CP.cprint("g", f"{str(Path(cell_id,p)):>72s}  : {oldvalue/1e6:6.3f}")  # list the protocol and value here
 
                    bridge_data.append(
                        {
                            "Cell_ID": cell_id,
                            "Protocol": p,
                            "BridgeEnabled": bridge_enabled,
                            "BridgeResistance": bridge_resistance,
                            "BridgeAdjust(ohm)": oldvalue,
                            "TrueResistance": bridge_resistance + oldvalue, 
                            "update_date": "",
                            "data_directory": main_df.iloc[idx]["data_directory"],
                        }
                    )
            # if re_mapdata.match(p) is not None:
            #     CP.cprint("b", f"{cell_id:s}/{p:s}")
            # if re_vcdata.match(p) is not None:
            #     CP.cprint("m", f"{cell_id:s}/{p:s}")
        else:  # add a 0 to the end
            # olddata is missing (not in table), so read from the original data
            
            for (
                p
            ) in (
                prots
            ): 
                if re_ccdata.match(p) is not None:  # but only for protocols in CC
                    # CP.cprint("y", f"{cell_id:s}/{p:s}")  # list the protocol here
                    # read data from original source
                    protoPath = Path(main_df.iloc[idx]["data_directory"], cell_id, p)
                    AR = DR.acq4_reader.acq4_reader(protoPath)
                    AR.getData()
                    clampInfo = AR.getDataInfo(Path(AR.clampInfo["dirs"][0], AR.dataname))
                    clamp_params = clampInfo[1]['ClampState']['ClampParams']
                    oldvalue = np.nan
                    bridge_enabled = clamp_params['BridgeBalEnable'] == 1
                    bridge_resistance = clamp_params['BridgeBalResist']
                    bridge_data.append(
                        {
                            "Cell_ID": cell_id,
                            "Protocol": p,
                            "BridgeEnabled": bridge_enabled,
                            "BridgeResistance": bridge_resistance,
                            "BridgeAdjust(ohm)": oldvalue,
                            "TrueResistance": bridge_resistance + oldvalue, 
                            "data_directory": main_df.iloc[idx]["data_directory"],
                        }
                    )
                    CP.cprint("y", f"{str(Path(cell_id,p)):>72s}  : {oldvalue/1e6:6.3f}")  # list the protocol and value here

    bridge_df = pd.DataFrame(bridge_data)
    save_excel(bridge_df, "testbridge.xlsx")


def list_old_bridge_corr(bridge_file_old: Union[str, Path]):
    fnp = Path(bridge_file_old)
    print(fnp.is_file())
    df = pd.read_pickle(fnp)
    print(df.columns)
    return
    for idx in df.index:
        cellid = f"{df.iloc[idx].date:s}/{df.iloc[idx].slice_slice:s}/{df.iloc[idx].cell_cell:s}"
        #  print(list(df.iloc[idx].IV.keys()))
        for iv in df.iloc[idx].IV:
            print(f"{cellid:s}  {str(iv.name):>32s} ", end=" ")
            if "BridgeAdjust" in list(df.iloc[idx].IV[iv].keys()):
                print(f"Bridge Value: {df.iloc[idx].IV[iv]['BridgeAdjust']/1e6:.2f}")
            else:
                print(" NO BRIDGE ADJUST")
        # print()


if __name__ == "__main__":
    # list_old_bridge_corr(old_bridge_name)
    make_new_bridge_corr(
        old_bridge_name=old_bridge_name,
        new_bridge_name=new_bridge_name,
        main_df_name=main_df_name,
    )
    # recolor("testbridge.xlsx")