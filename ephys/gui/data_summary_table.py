
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph import Qt
import pandas as pd
import datetime as datetime
import subprocess
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import pprint
from typing import List, Union
import ephys
from ephys.tools import win_print as WP
import pandas as pd
import textwrap
import numpy as np
from pylibrary.tools import cprint as CP
from ephys.tools import map_cell_types as MCT

# import vcnmodel.util.fixpicklemodule as FPM


cprint = CP.cprint
PP = pprint.PrettyPrinter(indent=8, width=80)
process = subprocess.Popen(
    ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
)
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
    'reporters', 'age', 'animal identifier', 'sex', 'weight', 'reporters.1',
    'solution', 'internal', 'temperature', 'important', 'expUnit',
    'slice_slice', 'slice_notes', 'slice_location', 'slice_orientation',
    'important.1', 'cell_cell', 'cell_notes', 'cell_type', 'cell_location',
    'cell_layer', 'cell_expression', 'cell_important', 'cell_id',
    'data_incomplete', 'data_complete', 'data_images', 'annotated',
    'data_directory'],



"""

def defemptylist():
    return []


#
# dataclass for the datasummary file.
#
@dataclass
class IndexData:
    project_code_hash: str = git_head_hash  # this repository!
    ephys_hash: str = ephys_git_hash  # save hash for the model code
    date: str = ""
    cell_id: str=""
    important: str=""
    description: str=""
    notes: str=""
    species: str=""
    strain: str=""
    genotype: str=""
    solution: str=""
    internal: str=""
    sex: str=""
    age: str=""
    weight: str=""
    temperature: str=""
    slice_orientation: str=""
    cell_cell: str=""
    slice_slice: str=""
    cell_type: str = ""
    cell_location: str=""
    cell_layer: str=""
    data_complete: List = field(default_factory=defemptylist)
    data_directory: str=""
    flag: bool = False



class TableManager:
    def __init__(
        self,
        parent = None,
        table: object = None,
        experiment: dict = None,
        selvals: dict = {},
        altcolormethod: object = None,
    ):
        self.table = table
        assert parent is not None
        self.parent = parent

        self.experiment = experiment
        self.selvals = selvals
        self.altColors = altcolormethod

    
    def make_indexdata(self, row):
        """
        Load up the index data class with selected information from the datasummary
        """
        if pd.isnull(row.cell_id):
            return None
        Index_data = IndexData()
        Index_data.ephys_hash = ephys_git_hash  # save hash for the model code
        Index_data.project_code_hash = git_head_hash  # this repository!
        Index_data.date = str(row.date)
        Index_data.cell_id = str(row.cell_id)
        Index_data.important = str(row.important)
        Index_data.description = textwrap.fill(str(row.description), width=40)
        Index_data.notes = textwrap.fill(str(row.notes), width=40)
        Index_data.species = str(row.species)
        Index_data.strain = str(row.strain)
        Index_data.genotype = str(row.genotype)
        Index_data.solution = str(row.solution)
        Index_data.internal = str(row.internal)
        Index_data.sex = str(row.sex)
        Index_data.age = str(row.age)
        Index_data.weight = str(row.weight)
        Index_data.temperature = str(row.temperature)
        Index_data.slice_orientation = textwrap.fill(str(row.slice_orientation), width=15)
        Index_data.cell_cell = str(row.cell_cell)
        Index_data.slice_slice = str(row.slice_slice)
        Index_data.cell_type = str(row.cell_type)
        Index_data.cell_location = str(row.cell_location)
        Index_data.cell_layer = str(row.cell_layer)
        Index_data.data_complete = str(row.data_complete) # textwrap.fill(str(row.data_complete), width=40)
        Index_data.data_directory = str(row.data_directory)
        Index_data.flag = False
        return Index_data

    def build_table(self, dataframe, mode="scan"):
        if mode == "scan":
            force = False
        if mode == "update":
            force = True
        self.data = []
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
                indxs[i].important,
                indxs[i].description,
                indxs[i].notes,
                indxs[i].species,
                indxs[i].strain,
                indxs[i].genotype,
                indxs[i].solution,
                indxs[i].internal,
                indxs[i].sex,
                indxs[i].age,
                indxs[i].weight,
                indxs[i].temperature,
                indxs[i].slice_orientation,
                indxs[i].cell_cell,
                indxs[i].slice_slice,
                indxs[i].cell_type,
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
                ("important", object),  # 1
                ("description", object),  # 2
                ("notes", object),  # 3
                ("species", object),  # 4
                ("strain", object),  # 5
                ("genotype", object),  # 6
                ("solution", object),  # 7
                ("internal", object),  # 8
                ("sex", object),  # 9
                ("age", object),  # 10
                ("weight", object),  # 11
                ("temperature", object),  # 12
                ("slice_orientation", object),  # 13
                ("cell_cell", object),  # 14
                ("slice_slice", object),  # 15
                ("cell_type", object),  # 16
                ("cell_location", object),  # 17
                ("cell_layer", object),  # 18
                ("data_complete", object),  # 19
                ("data_directory", object),  # 20
                ("flag", bool),  # 21
            ],
        )
        self.update_table(self.data)
        self.altColors(self.table)  # reset the coloring for alternate lines
        if QtGui is None:
            return
        for i in range(self.table.rowCount()):
            if self.table_data[i].flag:
                self.setColortoRow(i, QtGui.QColor(0xff, 0xef, 0x00, 0xee))
                self.setColortoRowText(i, QtGui.QColor(0x00, 0x00, 0x00))
            else:
                if i % 2:
                    self.setColortoRow(i, QtGui.QColor(0xCA, 0xFB, 0xF4, 0x66))
                else:
                    self.setColortoRow(i, QtGui.QColor(0x33, 0x33, 0x33))
                self.setColortoRowText(i, QtGui.QColor(0xff, 0xff, 0xff))
        cprint("g", "Finished updating index files")

    def update_table(self, data, QtCore=None, QtGui=None):
        cprint("g", "Updating data table")
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
        self.altColors(self.table)  # reset the coloring for alternate lines
        # if QtGui is not None:
        # if self.table_data[i].flag:
        #     self.setColortoRow(i, QtGui.QColor(0x88, 0x00, 0x00))
        # else:
        #     self.setColortoRow(i, QtGui.QColor(0x00, 0x00, 0xf00))


    def setColortoRow(self, rowIndex, color):
        for j in range(self.table.columnCount()):
            if self.table.item(rowIndex, j) is not None:
                self.table.item(rowIndex, j).setBackground(color)
    
    def setColortoRowText(self, rowIndex, color):
        for j in range(self.table.columnCount()):
            if self.table.item(rowIndex, j) is not None:
                self.table.item(rowIndex, j).setForeground(color)

    def get_table_data_index(self, index_row, use_sibling=False) -> int:
        if use_sibling:
            value = index_row.sibling(index_row.row(), 1).data()
            for i, d in enumerate(self.data):
                if self.data[i][1] == value:
                    return i
            return None
    
        elif isinstance(index_row, IndexData):
            value = index_row
        else:
            value = index_row.row()
            return value
        

    def get_table_data(self, selected_row):
        """
        Regardless of the sort, read the current index row and map it back to
        the data in the table.
        This is because the table might be sorted, but the data itself is not.

        """
        # print("get_table_data")
        # print("  index row: ", index_row)

        ind = self.get_table_data_index(selected_row)
        # print("  ind: ", ind)
        for i in range(len(self.table_data)):
            if self.table_data[ind].cell_id == self.table_data[i].cell_id:
                # print("  found: ", i, self.table_data[i].cell_id)
                return self.table_data[ind]
        return None
        
        # if ind is not None:
        #     return self.table_data[ind]
        # else:
        #     return None

    def select_row_by_cell_id(self, cell_id):
        for i in range(len(self.table_data)):
            if cell_id == self.table_data[i].cell_id:
                self.table.selectRow(i)
                return i
        return None

    def apply_filter(self, QtCore=None, QtGui=None):
        """
        self.filters = {'Use Filter': False, 'dBspl': None, 'nReps': None,
        'Protocol': None,
        'Experiment': None, 'modelName': None, 'dendMode': None,
        "dataTable": None,}
        """
        if not self.parent.filters["Use Filter"]:  # no filter, so update to show whole table
            self.update_table(self.data, QtCore=QtCore, QtGui=QtGui)

        else:
            self.filter_table(self.parent.filters, QtCore=QtCore, QtGui=QtGui)

    def filter_table(self, filters, QtCore=None, QtGui=None):
        coldict = {  # numbers must match column in the table.
            "flag": 21,
            "cell_type": 16,
            "age": 10,
            "sex": 9,
            #"Group": 6,
            #"DataTable": 18,
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
        self.update_table(filtered_table, QtCore=QtCore, QtGui=QtGui)
        self.parent.doing_reload = False

