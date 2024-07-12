"""
Fill in a table using pyqtgrph that summarizes the data in a directory.
Generates index files (pickled python dataclass) that summarize the results.

Set the force flag true to rebuild the pkl files.

This is meant to be called from data_tables.py

This module is derived from table_manager in *vcnmodel*.

Support::

    NIH grants:
    DC R01 DC015901 (Spirou, Manis, Ellisman),
    DC R01 DC004551 (Manis, 2013-2019, Early development)
    DC R01 DC019053 (Manis, 2020-2025, Later development)

Copyright 2014- Paul B. Manis
Distributed under MIT/X11 license. See license.txt for more infomation. 
"""
import datetime
import pickle
import pprint
import subprocess
import textwrap
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from ephys.gui import data_table_functions as functions
import ephys
import numpy as np
import pandas as pd
from pylibrary.tools import cprint as CP

from ephys.tools import win_print as WP

# import vcnmodel.util.fixpicklemodule as FPM
FUNCS = functions.Functions()

cprint = CP.cprint
PP = pprint.PrettyPrinter(indent=8, width=80)

# Get the git hash of the repositories so we know exactly what code was run.
# The primary important hash is from the ephys module,
# but we also want to get the hash from the project repository.
# (assumed the repo was updated and committed ahead of the run)

# print("Current path: ", Path().absolute())

git_hashes = functions.get_git_hashes()


def defemptylist():
    return []


#
# dataclass for the pkl file. Abstracted information from the main data files to
# speed up access.
#
@dataclass
class IndexData:
    project_code_hash: str = git_hashes["project"]  # current project
    ephys_hash: str = git_hashes["ephys"]  # ephys
    date: str = ""
    cell_id: str = ""
    sex: str = ""
    age: str = ""
    weight: str = ""
    temperature: str = ""
    slice_slice: str = ""
    slice_mosaic: str = ""
    cell_cell: str = ""
    cell_type: str = ""
    cell_mosaic: str = ""
    data_complete: List = field(default_factory=defemptylist)
    Group: str = ""
    protocols = ""
    RMP: str = ""
    RMP_SD: str = ""
    Rin: str = ""
    taum: str = ""
    holding: str = ""
    flag: bool = False


class TableManager:
    def __init__(
        self,
        parent=None,
        table: object = None,
        experiment: dict = None,
        selvals: dict = {},
        altcolormethod: object = None,
    ):
        assert parent is not None
        self.parent = parent
        self.table = table
        self.experiment = experiment
        self.selvals = selvals
        self.alt_colors = altcolormethod
        self.data = []

    def textclear(self):
        if self.parent is None:
            print("Parent module is not set (None)")
            raise ValueError()
        else:
            self.parent.textbox.clear()

    def textappend(self, text, color="white"):
        if self.parent is None:
            cprint(color, text)  # just go straight to the terminal
        else:
            self.parent.textbox.setTextColor(self.parent.QColor(color))
            self.parent.textbox.append(text)
            self.parent.textbox.setTextColor(self.parent.QColor("white"))

    def force_suffix(self, filename, suffix=".pkl"):
        fn = Path(filename)
        if fn.suffix != suffix:
            fn = str(fn)
            fn = fn + suffix
            fn = Path(fn)
        return fn

    # def find_build_indexfiles(self, indexdir: Union[str, Path], force=False):
    #     """
    #     Given the indexdir, determine: Whether an index file exists for this
    #     directory

    #         If it does, read it If it does not, then make one from the data in
    #         the directory.

    #     Parameters
    #     ----------
    #     indexdir : (str or Path)
    #         The directory to check.

    #     Returns
    #     ------
    #     Contents of the index file.
    #     """

    #     # cprint('b', indexdir)
    #     indexfile = self.force_suffix(indexdir)
    #     # print("indexfile: ", indexfile) cprint("c", f"Checking for index file:
    #     # {str(indexfile):s}")
    #     if indexfile.is_file() and not force:
    #         # cprint("g", f"    Found index file, reading") print('
    #         # indexfile: ', indexfile)
    #         try:
    #             with open(indexfile, "rb") as fh:
    #                 d = pickle.load(fh, fix_imports=True)  # , encoding="latin1")
    #         except:
    #             # try:
    #             #     # import vcnmodel.util.fixpicklemodule as FPM

    #             #     with open(indexfile, "rb") as fh:
    #             #         d = FPM.pickle_load(
    #             #             fh
    #             #         )  # , fix_imports=True) # , encoding="latin1")
    #             # except:
    #                 raise ValueError(f"Unable to read index file: {str(fh):s}")

    #             # print("fh: ", fh)
    #         return d
    #     if force or not indexfile.is_file():
    #         # cprint("c", f"Building a NEW .pkl index file for
    #         # {str(indexdir):s}")
    #         dpath = Path(indexfile.parent, indexfile.stem)
    #         runs = list(dpath.glob("*.p"))
    #         if len(runs) == 0:
    #             return None
    #         for r in runs:
    #             p = self.read_pfile_params(r)
    #             if p is None:
    #                 return None
    #             else:
    #                 pars, runinfo, indexfile = p
    #         cprint("m", f"     ... to build indexdir: {str(indexdir):s}")
    #         indexdata = self.write_indexfile(pars, runinfo, indexdir)
    #         return indexdata

    @WP.winprint_continuous
    def read_pfile_params(self, datafile) -> Union[tuple, None]:
        """
        Reads the Params and runinfo entry from the simulation data file
        """
        self.textappend(f"Reading pfile: {str(datafile.name):s}", color="white")
        try:
            with open(datafile, "rb") as fh:
                d = pickle.load(fh, fix_imports=True, encoding="latin1")
        except:
            # import vcnmodel.util.fixpicklemodule as FPM

            # try:
            #     with open(datafile, "rb") as fh:
            #         d = FPM.pickle_load(
            #             fh
            #         )  # , fix_imports=True) # , encoding="latin1")
            # (ModuleNotFoundError, IOError, pickle.UnpicklingError):
            self.textappend("SKIPPING: File is too old; re-run for new structure", color="red")
            self.textappend(f"File: {str(datafile):s}", color="red")
            return None

        if "runInfo" not in list(d.keys()):
            self.textappend(
                'SKIPPING: File is too old (missing "runinfo"); re-run for new structure',
                color="red",
            )
            # print('  Avail keys: ', list(d.keys()))
            return None
        if "Params" not in list(d.keys()):
            self.textappend(
                'SKIPPING: File is too old (missing "Params"); re-run for new structure',
                color="red",
            )
            # print('  Avail keys: ', list(d.keys())) print(d['Results'][0])
            return None
        # print(d["Params"]) print(d["runInfo"]) exit()
        return (d["Params"], d["runInfo"], str(datafile.name))  # just the dicts

    def get_sim_runtime(self, filename):
        """
        Switch the time stamp to different format Here the initial value is a
        string, which we convert to a datetime

        """
        try:
            with open(filename, "rb") as fh:
                d = FPM.pickle_load(fh, fix_imports=True, encoding="latin1")
        except:
            # import vcnmodel.util.fixpicklemodule as FPM

            # try:
            #     with open(filename, "rb") as fh:
            #         d = FPM.pickle_load(fh)
            # except (ModuleNotFoundError, IOError, pickle.UnpicklingError):
            self.textappend("SKIPPING: File is too old; re-run for new structure", color="red")
            self.textappend(f"File: {str(filename):s}", color="red")
            return None
        if d["runInfo"] is None:
            cprint("r", f"runinfo is None? file = {str(filename):s}")
            return None
        if isinstance(d["runInfo"], dict):
            ts = d["runInfo"]["runTime"]
        else:
            ts = d["runInfo"].runTime
        times = datetime.datetime.strptime(ts, "%a %b %d %H:%M:%S %Y")
        return times

    def make_indexdata(self, row):
        """
        Load up the index data class with selected information from the datasummary
        """
        if pd.isnull(row.cell_id):
            return None
        Index_data = IndexData()
        Index_data.ephys_hash = git_hashes["ephys"]  # save hash for the model code
        Index_data.project_code_hash = git_hashes["project"]  # this repository!
        Index_data.cell_id = str(row.cell_id)
        try:
            Index_data.date = str(row.date)
        except:
            Index_data.date = str(row.Date)
        Index_data.age = str(row.age)
        Index_data.cell_type = str(row.cell_type)
        if "slice_mosaic" in row.keys():
            Index_data.slice_mosiac = str(row.slice_mosaic)
        else: 
            Index_data.slice_mosiac = ""

        if "cell_mosaic" in row.keys():
            Index_data.cell_mosiac = str(row.cell_mosaic)
        else:
            Index_data.cell_mosiac = ""
        Index_data.sex = str(row.sex)
        Index_data.weight = str(row.weight)
        # Index_data.temperature = str(row.temperature)

        Index_data.Group = str(row.Group)
        Index_data.RMP = f"{row.RMP:6.2f}"
        Index_data.RMP_SD = f"{row.RMP_SD:6.2f}"
        Index_data.flag = False
        if row.RMP_SD > 2.0:
            Index_data.flag = True
        Index_data.Rin = f"{row.Rin:6.2f}"
        Index_data.taum = f"{row.taum*1e3:6.2f}"  # convert to ms
        Index_data.holding = f"{row.holding*1e12:6.2f}"  # convert to pA
        # print("row.cellid: ", row.cell_id)
        # print("row.protocols: ", row.protocols)
        prots = "; ".join([Path(prot).name for prot in row.protocols])
        Index_data.protocols = textwrap.fill(str(prots), width=40)
        # Index_data.data_complete = str(row.data_complete)
        return Index_data

    # def write_indexfile(self, params, runinfo, indexdir):
    #     Index_data = self.make_indexdata(params, runinfo, indexdir=indexdir)
    #     indexfile = self.force_suffix(indexdir)
    #     with open(indexfile, "wb") as fh:
    #         pickle.dump(Index_data, fh)
    #     return Index_data

    # def read_indexfile(self, indexfilename):
    #     """
    #     Read the index file that we will use to populate the table, and to
    #     provide "hidden" information such as the file list, for analyses.

    #     """
    #     indexfilename = self.force_suffix(indexfilename)
    #     with open(indexfilename, "rb") as fh:
    #         try:
    #             indexdata = pickle.load(fh, fix_imports=True)  #
    #         except:
    #             # cprint('y', f"Failed to read with basic pickle: {str(fh):s}")
    #             # try:
    #             #     import vcnmodel.util.fixpicklemodule as FPM

    #             #     indexdata = FPM.pickle_load(fh)  # , encoding="latin1")
    #             # except:
    #                 raise ValueError(f"Unable to read index file: {str(fh):s}")
    #     cprint('g', f"Success reading indexfile")
    #     return indexdata

    def remove_table_entry(self, indexrow):
        if len(indexrow) != 1:
            self.parent.error_message("Selection Error: Can only delete one row at a time")
            return
        indexrow = indexrow[0]
        data = self.get_table_data(indexrow)
        for f in data.files:
            fn = Path(data.simulation_path, f)
            print(f"Would delete: {str(fn):s}")
            print(fn.is_file())
            Path(fn).unlink()
        indexfilename = self.force_suffix(data.datestr)
        if Path(indexfilename).is_file():
            print(f" and index file: {str(indexfilename):s}")
        # now update the table print(indexrow)
        ind = self.get_table_data_index(indexrow)
        if ind is None:
            return
        # if ind is not None:
        # print(ind) print(self.table_data[ind])
        # print(type(self.table_data[ind]))
        self.table.removeRow(ind)
        # print(dir(self.table)) print(self.table.viewport)
        # print(dir(self.table.viewport()))
        self.table.viewport().update()

    def print_indexfile(self, indexrow):
        """
        Print the values in the index file
        """
        print("=" * 80)
        print("\nIndex file and data file params")
        cprint("c", f"Index row: {str(indexrow.row):s}")
        data = self.get_table_data(indexrow)
        print("Data: ", data)
        for f in data.files:
            with open(f, "rb") as fh:
                fdata = pickle.load(
                    fh, fix_imports=True
                )  # FPM.pickle_load(fh) # , encoding="latin1")
                print("*" * 80)
                cprint("c", f)
                cprint("y", "File Data Keys")
                print(fdata.keys())
                # print(dir(fdata['Params']))
                cprint("y", "Parameters")
                PP.pprint(fdata["Params"].__dict__)  # , sort_dicts=True)
                print("-" * 80)
                cprint("y", "RunInfo")
                PP.pprint(fdata["runInfo"].__dict__)  # , sort_dicts=True
                print("-" * 80)
                cprint("y", "ModelPars")
                PP.pprint(fdata["modelPars"])  # , sort_dicts=True

                print("*" * 80)

    def select_dates(self, rundirs, mode="D"):
        """
        For every entry in rundirs, see if the date is later than or equal to
        our limit date if the mode is "D", we treat the directory format if the
        mode is "F", we treat it as a file (very old format) rundirs is a list
        of directories - in the case of the old format, just pass a single file
        as a list...
        """
        if self.parent.start_date == "None" and self.parent.end_date == "None":
            return rundirs

        if self.parent.start_date != "None":
            sd = int(self.parent.start_date.replace("-", ""))
        else:
            sd = 0
        if self.parent.end_date != "None":
            ed = int(self.parent.end_date.replace("-", ""))
        else:
            ed = 30000000
        sel_runs = []

        for d in rundirs:
            if mode == "D":
                dname = d.stem
                if d.is_dir():
                    fdate = dname[-10:].replace("-", "")
                else:
                    fdate = dname[-19:-9].replace("-", "")  # extract the date
            elif mode == "F":
                fdate = d.datestr[-19:-9].replace("-", "")
            else:
                raise ValueError("table manager select_dates: bad mode in call: ", mode)
            if fdate[:4] not in ["2020", "2021", "2022"]:
                continue
            idate = int(fdate)
            if idate >= sd and idate <= ed:
                sel_runs.append(d)
        return sel_runs

    @WP.winprint_continuous
    def build_table(self, dataframe, mode="scan", QtCore=None, QtGui=None):
        if mode == "scan":
            force = False
        if mode == "update":
            force = True
        self.data = []
        self.table.setData(self.data)
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
                    indxs[i].flag,
                    indxs[i].cell_type,
                    indxs[i].age,
                    indxs[i].weight,
                    indxs[i].sex,
                    indxs[i].Group,
                    indxs[i].RMP,
                    indxs[i].RMP_SD,
                    indxs[i].Rin,
                    indxs[i].taum,
                    indxs[i].holding,
                    indxs[i].slice_mosaic,
                    indxs[i].cell_mosaic,
                    indxs[i].protocols,
                    indxs[i].data_complete,
                )
                for i in range(len(indxs))
            ],
            dtype=[
                ("cell_id", object),  # 0
                ("flag", object),  # 1
                ("cell_type", object),  # 2
                ("age", object),  # 3
                ("weight", object),  # 4
                ("sex", object),  # 5
                ("Group", object),  # 6
                ("RMP", object),  # 7
                ("RMP_SD", object),  # 8
                ("Rin", object),  # 9
                ("taum", object),  # 10
                ("holding", object),  # 11
                ("slice_mosaic", object),  # 12
                ("cell_mosaic", object),  # 13
                ("protocols", object),  # 14
                ("data_complete", object),  # 15
            ],
        )
        self.update_table(self.data, QtCore=QtCore, QtGui=QtGui)
        cprint("g", "Finished updating index files")

    def update_table(self, data, QtCore=None, QtGui=None):
        cprint("g", "Updating data table")
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
        if QtGui is None:
            return
        for i in range(self.table.rowCount()):
            if self.table_data[i].flag:
                self.set_color_to_row(i, QtGui.QColor(0xff, 0xef, 0x00, 0xee))
                self.set_color_to_row_text(i, QtGui.QColor(0x00, 0x00, 0x00))
            else:
                if i % 2:
                    self.set_color_to_row(i, QtGui.QColor(0x00, 0x00, 0x00))
                else:
                    self.set_color_to_row(i, QtGui.QColor(0x33, 0x33, 0x33))
                self.set_color_to_row_text(i, QtGui.QColor(0xff, 0xff, 0xff))

        # self.parent.Dock_Table.raiseDock()

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
        coldict = {
            "flag": 1,
            "cell_type": 2,
            "age": 3,
            "sex": 5,
            "Group": 6,
            "DataTable": 18,
        }
        self.parent.doing_reload = True
        filtered_table = self.data.copy()
        matchsets = dict([(x, set()) for x in filters.keys() if x != "Use Filter"])
        for k, d in enumerate(self.data):
            for f, v in filters.items():
                if v is None or v == "None":
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

    def set_color_to_row(self, rowIndex, color):
        for j in range(self.table.columnCount()):
            if self.table.item(rowIndex, j) is not None:
                self.table.item(rowIndex, j).setBackground(color)
    
    def set_color_to_row_text(self, rowIndex, color):
        for j in range(self.table.columnCount()):
            if self.table.item(rowIndex, j) is not None:
                self.table.item(rowIndex, j).setForeground(color)


    def get_table_data_index(self, index_row, use_sibling=False):
        if use_sibling:
            value = index_row.sibling(index_row.row(), 1).data()
            for i, d in enumerate(self.data):
                if self.data[i][1] == value:
                    return i
            return None

        else:
            value = index_row.row()
            return value

    def get_table_data(self, index_row):
        """
        Regardless of the sort, read the current index row and map it back to
        the data in the table.
        """
        # print("get_table_data")
        # print("  index row: ", index_row)
        # print(dir(index_row))
        # print(index_row.data())
        ind = self.get_table_data_index(index_row)
        # print("  ind: ", ind)
        for i in range(len(self.table_data)):
            if index_row.data() == self.table_data[i].cell_id:
                # print("  found: ", i)
                return self.table_data[i]

        if ind is not None:
            return self.table_data[ind]
        else:
            return None

    def select_row_by_cell_id(self, cell_id):
        for i in range(len(self.table_data)):
            # print("i: ", i, cell_id, self.table_data[i].cell_id)
            if cell_id == self.table_data[i].cell_id:
                self.table.selectRow(i)
                return i
        return None

    def select_row_by_row(self, irow: int):
        try:
            self.table.selectRow(irow)
            return irow
        except:
            return None

    def export_brief_table(self, textbox):
        #  table for coding stuff
        FUNCS.textbox_setup(textbox)
        FUNCS.textclear()
        FUNCS.textappend("date\tStrain\tGroup\treporters\tage\tdob\tAnimal_ID\tsex\tslice_slice\tcell_cell\tcell_expression")
        for i in range(len(self.table_data)):
            if i == 0:
                print(self.table_data[i])
            cell_id = self.table_data[i].cell_id
            Group = self.table_data[i].Group
            strain = self.table_data[i].strain
            reporters = self.table_data[i].reporters
            age = self.table_data[i].age
            dob = self.table_data[i].dob
            animal_id = " "  # self.table_data[i].animal_id
            sex = self.table_data[i].sex
            slice_slice = self.table_data[i].slice_slice
            cell_cell = self.table_data[i].cell_cell
            cell_expression = self.table_data[i].cell_expression
            msg = f"{cell_id:s}\t{strain:s}\t{Group:s}\t{reporters:s}\t{age:s}\t{sex:s}\t{dob:s}\t{animal_id:s}\t{slice_slice:s}\t{cell_cell:s}\t{cell_expression:s}"
            FUNCS.textappend(msg)
        print("Table exported in Report")
