
from pathlib import Path

import numpy as np
import pandas as pd
from typing import List
from pylibrary.tools import cprint as CP

"""write to an excel sheet with a standard set of colors for the cell types
in the sheet. Also formats the column widths.
"""

class ColorExcel():
    def __init__(self):
        self.last_cell = None  # last cell name in row sequence
        self.last_ctype = ""  # last cell type in row sequence

        # for each cell color combination, provide a light and dark version
        # to allow alternation of colors in the same cell type

        self.colors = {"pyramidal": ["#c5d7a5", "#a5b785"], #"darkseagreen",
            "fusiform": ["#c5d7a5", "#a5b785"],
            "pyr": ["#c5d7a5", "#a5b785"],
            "cartwheel": ["skyblue", "deepskyblue"],
            "tuberculoventral": ["lightpink", "hotpink"],
            "granule": ["linen", "bisque"],
            "golgi": ["yellow", "khaki"],
            "unipolar brush cell": ["sienna", "chocolate"],
            "chestnut": ["peru", "saddlebrown"],
            "giant": ["sandybrown", "peachpuff"],
            "giant?": ["sandybrown", "peachpuff"],
            "giant cell": ["sandybrown", "peachpuff"],
            "Unknown": ["white", "gainsboro"],
            "unknown": ["white", "gainsboro"],

            "bushy": ["lightslategray", "darkslategray"],
            "t-stellate": ["thistle", "plum"],
            "l-stellate": ["darkcyan", "darkturquoise"],
            "d-stellate": ["skyblue", "deepskyblue"],
            "stellate": ["thistle", "plum"],
            "octopus": ["goldenrod", "darkgoldenrod"],
            
            # cortical (uses some of the same colors)
            "basket": ["lightpink", "deeppink"],
            "chandelier": ["sienna", "chocolate"],
            "fast spiking":["salmon" ,"darksalmon"],
            "fs":["salmon", "darksalmon"],
            "RS": ["lightgreen", "darkgreen"],
            "LTS":[ "paleturquoise", "darkturquoise"],
            "rs": ["lightgreen", "darkgreen"],
            "lts":[ "paleturquoise", "darkturquoise"],

            # thalamic neurons
            "thalamic": ["yellow", "khaki"],

            # cerebellar
            "Purkinje": ["mediumorchid", "darkorchid"],
            "purkinje": ["mediumorchid", "darkorchid"],
            "purk": ["mediumorchid", "darkorchid"],
            
            # not neurons
            'glia': ['lightslategray', 'darkslategray'],
            'Glia': ['lightslategray', 'darkslategray'],
        }

    def _find_celltype_in_field(self, celltype:str, cellkeys:list):
        for k in cellkeys:
            if celltype.find(k) >= 0:
                return k
        return None

    def _highlight_by_cell_type(self, row):

        cellkeys = list(self.colors.keys())
        celltype = row.cell_type.lower()
        ctype = self._find_celltype_in_field(celltype, cellkeys) 
        CP.cprint("y", f"celltype: {celltype:s},  ctype: {str(ctype):s}")
        if ctype is not None and ctype != "":
            if self.last_cell is None:
                self.last_cell = f"{row.date:s}~{row.slice_slice:s}~{row.cell_cell:s}"
                color = self.colors[ctype][0]
                self.last_ctype = ctype
                self.dark = False
                return [f"background-color: {color:s}" for s in range(len(row))]

            else:
                this_cell = f"{row.date:s}~{row.slice_slice:s}~{row.cell_cell:s}"
                if this_cell == self.last_cell:  # continue with this color
                    if self.dark:
                        color = self.colors[ctype][1]
                    else:
                        color = self.colors[ctype][0]
                    return [f"background-color: {color:s}" for s in range(len(row))]
                elif ctype == self.last_ctype and this_cell != self.last_cell: # different cell, same type, alternate colors
                    self.last_cell = this_cell
                    if self.dark: # alternate colors light/dark
                        color =   self.colors[ctype][0]  # set to lighter
                        self.dark = False
                    else:
                        color = self.colors[ctype][1]  # set to darker
                        self.dark = True
                    return [f"background-color: {color:s}" for s in range(len(row))]

                else: # not same cell and not same cell type - start over with light color
                    color = self.colors[ctype][0]  # different cell type, revert to light
                    self.dark = False
                    self.last_ctype = ctype
                    self.last_cell = this_cell
                    return [f"background-color: {color:s}" for s in range(len(row))]
        else:
            self.last_ctype = ""
            self.ark = False
            return [f"background-color: red" for s in range(len(row))]



    def _organize_columns(self, df, columns = None):
        if columns is None:
            return df
        # cols = ['ID', 'Group', 'Date', 'slice_slice','cell_cell', 'cell_type', 
        #     'iv_name', 'holding', 'RMP', 'RMP_SD', 'Rin', 'taum',
        #     'dvdt_rising', 'dvdt_falling', 'AP_thr_V', 'AP_HW', "AP15Rate", "AdaptRatio", 
        #     "AP_begin_V", "AHP_trough_V", "AHP_depth_V", "tauh", "Gh", "FiringRate",
        #     "FI_Curve",
        #     'date']
        df = df[columns + [c for c in df.columns if c not in columns]]
        return df

    def make_excel(self, df:pd.DataFrame, outfile:Path, sheetname:str="Sheet1", columns:list=None):
        """cleanup: reorganize columns in spreadsheet, set column widths
        set row colors by cell type

        Args:
            df: object
                Pandas dataframe object
            excelsheet (_type_): _description_
        """
        if outfile.suffix != '.xlsx':
            outfile = outfile.with_suffix('.xlsx')

        writer = pd.ExcelWriter(outfile, engine='xlsxwriter')

        df.to_excel(writer, sheet_name = sheetname)
        # df = self._organize_columns(df, columns=columns)
        workbook = writer.book
        worksheet = writer.sheets[sheetname]
        fdot3 = workbook.add_format({'num_format': '####0.000'})
        df.to_excel(writer, sheet_name = sheetname)

        resultno:List = []
        # resultno = ['holding', 'RMP', 'Rin', 'taum', 'dvdt_rising', 'dvdt_falling', 
        #     'AP_thr_V', 'AP_HW', "AP15Rate", "AdaptRatio", "AP_begin_V", "AHP_trough_V", "AHP_depth_V"]
        # df[resultno] = df[resultno].apply(pd.to_numeric)    
        for i, column in enumerate(df):
            # print('column: ', column)
            if column in resultno:
                writer.sheets[sheetname].set_column(first_col=i+1, last_col=i+1,  cell_format=fdot3)
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
                writer.sheets[sheetname].set_column(first_col=i+1, last_col=i+1, cell_format=fdot3, width=column_width) # column_dimensions[str(column.title())].width = column_width
                # print(f"formatted {column:s} with {str(fdot3):s}")
            else:
                writer.sheets[sheetname].set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width

        df = df.style.apply(self._highlight_by_cell_type, axis=1)
        try:
            df.to_excel(writer, sheet_name = sheetname, columns=columns)  # organize columns at the end
            writer.close()
        except:
            print(df.columns)
            print(columns)
            raise ValueError


if __name__ == "__main__":
    celltype =  "layer vi lts"
    ce = ColorExcel()
    ct = ce._find_celltype_in_field(celltype, list(ce.colors.keys()))
    print(ct)