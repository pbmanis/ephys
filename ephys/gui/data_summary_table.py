"""Class to handle the data summary table, and some actions from the 
table in the gui.
"""
import datetime
import pprint
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from pylibrary.tools import cprint as CP
from pyqtgraph.Qt import QtGui, QtWidgets, QtCore

import ephys
from ephys.tools import map_cell_types as MCT
from ephys.gui import data_table_functions as functions

FUNCS = functions.Functions()
# import vcnmodel.util.fixpicklemodule as FPM


cprint = CP.cprint
PP = pprint.PrettyPrinter(indent=8, width=80)
process = subprocess.Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE)
git_head_hash = process.communicate()[0].strip()

ephyspath = Path(ephys.__file__).parent
process = subprocess.Popen(
    ["git", "-C", str(ephyspath), "rev-parse", "HEAD"],
    shell=False,
    stdout=subprocess.PIPE,
)
ephys_git_hash = process.communicate()[0].strip()


""" The Data summary database has the following:
Index(['date', 'description', 'notes', 'species', 'strain', 'genotype',
    'reporters', 'age', 'subject', 'sex', 'weight', 'reporters.1',
    'solution', 'internal', 'temperature', 'important', 'expUnit',
    'slice_slice', 'slice_notes', 'slice_location', 'slice_orientation',
    'important.1', 'cell_cell', 'cell_notes', 'cell_type', 'cell_location',
    'cell_layer', 'cell_expression', 'cell_important', 'cell_id',
    'data_incomplete', 'data_complete', 'data_images', 'annotated',
    'data_directory'],



"""


def defemptylist():
    """defemptylist
    make an empty list for the dataclass.
    Yeah, I know this is silly, but this is what you
    have to do to instantiate an empty list in a dataclass.

    Returns
    -------
    list
        empty list
    """
    return []


#
# dataclass for the datasummary file.
#
@dataclass
class IndexData:
    project_code_hash: str = git_head_hash  # this repository!
    ephys_hash: str = ephys_git_hash  # save hash for the model code
    date: str = ""
    cell_id: str = ""
    cell_type: str = ""
    important: str = ""
    description: str = ""
    notes: str = ""
    species: str = ""
    strain: str = ""
    genotype: str = ""
    solution: str = ""
    internal: str = ""
    subject: str = ""
    sex: str = ""
    age: str = ""
    weight: str = ""
    temperature: str = ""
    slice_orientation: str = ""
    cell_cell: str = ""
    slice_slice: str = ""
    cell_location: str = ""
    cell_layer: str = ""
    data_complete: List = field(default_factory=defemptylist)
    data_directory: str = ""
    flag: bool = False


class TableManager:
    def __init__(
        self,
        parent=None,
        table: object = None,
        experiment: dict = None,
        selvals: Union[dict, None] = None,
        altcolormethod: object = None,
    ):
        self.table = table
        assert parent is not None
        self.parent = parent

        self.experiment = experiment
        self.selvals = selvals
        self.alt_colors = altcolormethod
        self.data = []
        self.table_data = []
        self.current_table_data = None
        self.keep_index = set()

    def make_indexdata(self, row):
        """
        Load up the index data class with selected information from the datasummary
        """
        if pd.isnull(row.cell_id):
            return None
        Index_data = IndexData()
        # print("row: ", row.keys())
        Index_data.ephys_hash = ephys_git_hash  # save hash for the model code
        Index_data.project_code_hash = git_head_hash  # this repository!
        Index_data.date = str(row.date)
        Index_data.cell_id = str(row.cell_id)
        Index_data.cell_type = str(row.cell_type)
        Index_data.important = str(row.important)
        Index_data.description = textwrap.fill(str(row.description), width=40)
        Index_data.notes = textwrap.fill(str(row.notes), width=40)
        Index_data.species = str(row.species)
        Index_data.strain = str(row.strain)
        Index_data.genotype = str(row.genotype)
        Index_data.solution = str(row.solution)
        Index_data.internal = str(row.internal)
        if "animal_identifier" in row.keys():
            Index_data.subject = str(row["animal_identifier"])
        elif "animal identifier" in row.keys():
            Index_data.subject = str(row["animal identifier"])
        Index_data.sex = str(row.sex)
        Index_data.age = str(row.age)
        Index_data.weight = str(row.weight)
        Index_data.temperature = str(row.temperature)
        Index_data.slice_orientation = textwrap.fill(str(row.slice_orientation), width=15)
        Index_data.cell_cell = str(row.cell_cell)
        Index_data.slice_slice = str(row.slice_slice)
        Index_data.cell_location = textwrap.fill(str(row.cell_location), width=10)
        Index_data.cell_layer = str(row.cell_layer)
        Index_data.data_complete = textwrap.fill(str(row.data_complete), width=40)
        Index_data.data_directory = textwrap.fill(str(row.data_directory), width=40)
        Index_data.flag = False
        return Index_data

    def build_table(self, dataframe, mode="scan"):
        """build_table Create the table from the dataframe

        Parameters
        ----------
        dataframe : pandas dataframe
            datasummary table as a pd.dataframe
        mode : str, optional
            How the dataframe is created, by default "scan"
            Unused argument.
        """
        self.dataframe = dataframe  # save pointer to the dataframe
        self.table.setData(self.data)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        indxs = []
        for i, dfindex in enumerate(dataframe.index):
            index_file_data = self.make_indexdata(dataframe.loc[dfindex])
            if index_file_data is not None:
                indxs.append(index_file_data)
        self.table_data = indxs

        # transfer to the data array for the table
        self.data = np.array(
            [
                (
                    indxs[i].cell_id,
                    indxs[i].cell_type,
                    indxs[i].important,
                    indxs[i].description,
                    indxs[i].notes,
                    indxs[i].species,
                    indxs[i].strain,
                    indxs[i].genotype,
                    indxs[i].solution,
                    indxs[i].internal,
                    indxs[i].subject,
                    indxs[i].sex,
                    indxs[i].age,
                    indxs[i].weight,
                    indxs[i].temperature,
                     indxs[i].slice_orientation,
                    indxs[i].cell_cell,
                    indxs[i].slice_slice,
                    indxs[i].cell_location,
                    indxs[i].cell_layer,
                    indxs[i].data_complete,
                    indxs[i].data_directory,
                    indxs[i].flag,
                )
                for i in range(len(indxs))
            ],
            dtype=[
                ("cell_id", object),  # 0
                ("cell_type", object),  # 1
                ("important", object),  # 2
                ("description", object),  # 3
                ("notes", object),  # 4
                ("species", object),  # 5
                ("strain", object),  # 6
                ("genotype", object),  # 7
                ("solution", object),  # 8
                ("internal", object),  # 9
                ("subject", object),  # 10
                ("sex", object),  # 11  $$$$$
                ("age", object),  # 12
                ("weight", object),  # 13
                ("temperature", object),  # 14
                ("slice_orientation", object),  # 15
                ("cell_cell", object),  # 16
                ("slice_slice", object),  # 17
                ("cell_location", object),  # 18
                ("cell_layer", object),  # 19
                ("data_complete", object),  # 20
                ("data_directory", object),  # 21
                ("flag", bool),  # 22
            ],
        )
        self.update_table(self.data)
        self.alt_colors(self.table)  # reset the coloring for alternate lines
        if QtGui is None:
            return
        for i in range(self.table.rowCount()):
            if self.table_data[i].flag:
                self.set_color_to_row(i, QtGui.QColor(0xFF, 0xEF, 0x00, 0xEE))
                self.set_color_to_row_text(i, QtGui.QColor(0x00, 0x00, 0x00))
            else:
                if i % 2:
                    self.set_color_to_row(i, QtGui.QColor(0xCA, 0xFB, 0xF4, 0x66))
                else:
                    self.set_color_to_row(i, QtGui.QColor(0x33, 0x33, 0x33))
                self.set_color_to_row_text(i, QtGui.QColor(0xFF, 0xFF, 0xFF))
        cprint("g", "Finished updating index files")

    def update_table(self, data):
        """update_table 
        WHen changes are made to the data, update the table

        Parameters
        ----------
        data : data to go in the table
            _description_
    
        """
        cprint("g",f"Updating data table")
        # print("data for update: ", data)
        # print("data complete: ", data[:]['data_complete'])
        self.table.setData(data)
        style = "section:: {font-size: 4pt; color:black; font:TimesRoman;}"
        self.table.setStyleSheet(style)
        # if QtCore is not None:
        #     # print('sorting by a column')
        #     self.table.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        # self.table.setStyle(QtGui.QFont('Arial', 6))
        # self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        self.current_table_data = data
        self.alt_colors(self.table)  # reset the coloring for alternate lines
        # if QtGui is not None:
        # if self.table_data[i].flag:
        #     self.set_color_to_row(i, QtGui.QColor(0x88, 0x00, 0x00))
        # else:
        #     self.set_color_to_row(i, QtGui.QColor(0x00, 0x00, 0xf00))

    def set_color_to_row(self, rowIndex, color):
        """set_color_to_row Set the color of the row

        Parameters
        ----------
        rowIndex : int
            row to change
        color : QtColor
            Color for the row
        """
        for j in range(self.table.columnCount()):
            if self.table.item(rowIndex, j) is not None:
                self.table.item(rowIndex, j).setBackground(color)

    def set_color_to_row_text(self, rowIndex, color):
        """set_color_to_row_text set the foreground color for a row

        Parameters
        ----------
        rowIndex : int
            row to change
        color : QtColor
            Color for the row
        """
        for j in range(self.table.columnCount()):
            if self.table.item(rowIndex, j) is not None:
                self.table.item(rowIndex, j).setForeground(color)

    def get_table_data_index(self, index_row, use_sibling=False) -> int:
        """get_table_data_index

        Parameters
        ----------
        index_row : int
            row in the table
        use_sibling : bool, optional
            whether to use the sibling table, by default False

        Returns
        -------
        int
            row in the table
        """
        if use_sibling:
            value = index_row.sibling(index_row.row(), 1).data()
            for i, d in enumerate(self.data):
                if self.data[i][1] == value:
                    return i
            return None

        elif isinstance(index_row, IndexData):
            value = index_row
        else:
            value = index_row.row()+1
            return value


    def get_selected_cellid_from_table(self, selected_row):
        """
        Regardless of the sort, read the current index row and map it back to
        the data in the table.
        We do this because the visible table might be sorted,
        but the pandas database is not necessarily, so we just
        look it up.

        """
        # print("get_table_data")
        # print("  index row: ", index_row)
        ind = self.get_table_data_index(selected_row)
        if ind is None:
            print("ind is none")
            return None
        cell_id = self.table.item(ind-1, 0).text()
        if cell_id is None:
            print("cell is is none")
            return None
        return cell_id

    def get_table_data(self, selected_row):
        """
        Regardless of the sort, read the current index row and map it back to
        the data in the table.
        We do this because the visible table might be sorted,
        but the pandas database is not necessarily, so we just
        look it up.

        """
        ind = self.get_table_data_index(selected_row)
        
        for i, td in enumerate(self.table_data):
            if self.table_data[ind-1].cell_id == td.cell_id:
                # print("  found: ", i, ind, self.table_data[i].cell_id, td.cell_id)
                return self.table_data[ind-1]
        return None

        if ind is not None:
            return self.table_data[ind]
        else:
            return None

    def get_table_data_by_cell_id(self, cell_id):
        """get_table_data_by_cell_id

        Parameters
        ----------
        cell_id : string
            cell id in the form 'yyyy.mm.dd_nnn/slice_mmm/cell_xxx'

        Returns
        -------
        IndexData
            data for the cell_id
        """
        for i, td in enumerate(self.table_data):
            if cell_id == td.cell_id:
                return self.table_data[i]
        return None
    
    def select_row_by_cell_id(self, cell_id):
        """select_row_by_cell_id Select a row by the cell_id

        Parameters
        ----------
        cell_id : string
            cell id in the form 'yyyy.mm.dd_nnn/slice_mmm/cell_xxx'

        Returns
        -------
        int
            row, or None
        """
        for i, td in enumerate(self.table_data):
            if cell_id == td[i].cell_id:
                self.table.selectRow(i)
                return i
        return None

    def select_row_by_row(self, irow: int):
        """select_row_by_row select row given row number 
        just make sure the row exists...
        Parameters
        ----------
        irow : int
            row
        Returns
        -------
        row or None
            
        """
        try:
            self.table.selectRow(irow)
            return irow
        except:
            return None

    def apply_filter(self):
        """
        self.filters = {'Use Filter': False, 'dBspl': None, 'nReps': None,
        'Protocol': None,
        'Experiment': None, 'modelName': None, 'dendMode': None,
        "dataTable": None,}
        """
        if not self.parent.filters["Use Filter"]:  # no filter, so update to show whole table
            self.update_table(self.data)

        else:
            self.filter_table(self.parent.filters, QtCore=QtCore, QtGui=QtGui)

    def filter_table(self, filters):
        """filter_table _summary_

        Parameters
        ----------
        filters : _type_
            _description_
        """
        coldict = {  # numbers must match column in the table.
            "flag": 21,
            "cell_type": 16,
            "age": 10,
            "sex": 9,
            # "Group": 6,
            # "DataTable": 18,
        }
        self.parent.doing_reload = True
        filtered_table = self.data.copy()
        matchsets = dict([(x, set()) for x in filters.keys() if x != "Use Filter"])
        for k, d in enumerate(self.data):
            for f, v in filters.items():
                if v is None or v == "None":
                    continue
                if f not in coldict.keys():
                    continue
                if f == "cell_type":
                    print("old: ", v)
                    v = MCT.map_cell_type(v)  # convert various names to a consistent one
                    print("new: ", v)
                    if v is None:
                        continue
                if (
                    not isinstance(v, list)
                    and (coldict.get(f, False))
                    and (self.data[k][coldict[f]] == v)
                ):
                    # and f in list(coldict.keys())) and if
                    # (self.data[k][coldict[f]] == v): print("f: ", f, "
                    # v: ", v)
                    matchsets[f].add(k)
                elif isinstance(v, list) and (coldict.get(f, False)):
                    v = sorted(v)  # be sure order is ok for comparisions
                    if f == "age":
                        age = ephys.tools.parse_ages.age_as_int(self.data[k][coldict[f]])
                        if age >= v[0] and age <= v[1]:
                            matchsets[f].add(k)
                    else:
                        if self.data[k][coldict[f]] >= v[0] and self.data[k][coldict[f]] <= v[1]:
                            matchsets[f].add(k)

        baseset = set()
        for k, v in matchsets.items():
            if len(v) > 0:
                baseset = v
                break
        # and logic:
        finds = [v for k, v in matchsets.items() if len(v) > 0]
        keep_index = baseset.intersection(*finds)
        self.keep_index = keep_index  # we might want this later!
        # print('Filter index: ', keep_index)
        filtered_table = [filtered_table[ft] for ft in keep_index]
        self.update_table(filtered_table)
        self.parent.doing_reload = False

    def export_brief_table(self, textbox, dataframe:pd.DataFrame):
        #  table for coding stuff
        FUNCS.textbox_setup(textbox)
        FUNCS.textclear()
        FUNCS.textappend("date\tStrain\tGroup\treporters\tage\tdob\tAnimal_ID\tsex\tslice_slice\tcell_cell\tcell_expression")
        for i, dfindex in enumerate(dataframe.index):
            row = dataframe.loc[dfindex]
            if i == 0:
                print(row)

            cell_id = row.date
            Group = row.genotype
            strain = row.strain
            reporters = row.reporters
            age = row.age
            dob = "" # row.dob
            animal_id = row['subject']  # animal_id
            sex = row.sex
            slice_slice = row.slice_slice
            cell_cell = row.cell_cell
            cell_expression = row.cell_expression
            msg = f"{cell_id:s}\t{strain:s}\t{Group:s}\t{reporters:s}\t{age:s}\t{dob:s}\t{animal_id:s}\t{sex:s}\t{slice_slice:s}\t{cell_cell:s}\t{cell_expression:s}"
            FUNCS.textappend(msg)
        print("Table exported in Report")
    
    def print_indexfile(self, dataframe:pd.DataFrame, indexrow:IndexData):
        """
        Print the values in the index file
        """
        return  # this is broken.
        print("=" * 80)
        print("\nIndex file and data file params")
        cprint("c", f"Index row: {str(indexrow.row):s}")
        data = self.get_table_data(indexrow)
        print(data)
        print(self.parent.datasummary.cell_id)
        d = filename_tools.get_cell(self.experiment,  df=self.parent.datasummary,  cell_id = data.cell_id)
        print("Cell ID: ", data.cell_id)
        print("d: ", d)
        taum = []
        rin = []
        rmp = []

        for k, v in d["IV"].items():
            # print(k, v)
            print("k: ", k)
            if isinstance(v, dict):
                print(v.keys())
                print("taum: ", v["taum"])
                print("taupars: ", v["taupars"])
                taum.append(v["taum"])
                rin.append(v["Rin"])
                rmp.append(v["RMP"])
                if 'analysistimestamp' in v.keys():
                    print("analysistime: ", v["analysistimestamp"])
        print(taum, np.nanmean(taum))
        print(rin, np.nanmean(rin))
        print(rmp, np.nanmean(rmp))
