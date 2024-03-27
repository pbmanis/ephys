
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



class TableManager:
    def __init__(
        self,
        table: object = None,
        experiment: dict = None,
        selvals: dict = {},
        altcolormethod: object = None,
    ):
        self.table = table

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
                    
                )

                for i in range(len(indxs))
            ],
            dtype=[
                ("cell_id", object),  # 0
                ("important", object),  # 1
                ("description", object),  # 1
                ("notes", object),  # 2
                ("species", object),  # 3
                ("strain", object),  # 4
                ("genotype", object),  # 5
                ("solution", object),  # 6
                ("internal", object),  # 7
                ("sex", object),  # 8
                ("age", object),  # 9
                ("weight", object),  # 10
                ("temperature", object),  # 11
                ("slice_orientation", object),  # 12
                ("cell_cell", object),  # 13
                ("slice_slice", object),  # 14
                ("cell_type", object),  # 15
                ("cell_location", object),  # 16
                ("cell_layer", object),  # 17
                ("data_complete", object),  # 18
                ("data_directory", object),  # 19
            ],
        )
        self.update_table(self.data)
        cprint("g", "Finished updating index files")

    def update_table(self, data):
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

