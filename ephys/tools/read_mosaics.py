# type: ignore
# ignore because this is incomplete code/stub
"""Read the mosaic files, and find the markers.
DCN Mosaics.

Then for each cell, compute the depth based on a line orthogonal to that reference line.
This "standardizes" the depth measurements.

"""
import datetime
import json
import os
from pathlib import Path
from typing import List, Union

# import ephys.mapanalysistools.define_markers as define_markers
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import pint

import seaborn as sns
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint as CP
from sympy import Line, Point

import ephys.datareaders.acq4_reader as AR
import ephys.mapanalysistools.get_markers as get_markers
import ephys.tools.configfile as CF
import ephys.tools.show_paraview_csv as show_paraview_csv
import ephys.tools.tools_plot_maps as TPM
from ephys.datareaders import acq4_reader
from ephys.ephys_analysis.analysis_common import Analysis
from ephys.gui import data_table_functions as functions
from ephys.mapanalysistools import analyze_map_data
from ephys.mini_analyses import mini_event_dataclasses as MEDC
from ephys.tools import data_summary
from ephys.tools.get_configuration import get_configuration

FUNCS = functions.Functions()
# PP = PrettyPrinter(indent=4)
DSUM = data_summary.DataSummary()
UR = pint.UnitRegistry()


def define_markers():
    print(os.getcwd())
    definedMarkers = CF.readConfigFile("config/MosaicEditor.cfg")["definedMarkers"]
    all_markernames: list = []
    for k in definedMarkers.keys():
        all_markernames.extend([tkey for tkey in definedMarkers[k].keys()])

    mark_colors = {}
    mark_symbols = {}
    mark_alpha = {}
    for k in all_markernames:
        mark_alpha[k] = 1.0
    for k in ["cell", "soma"]:
        mark_alpha[k] = 0.8
    for k in all_markernames:
        if k.startswith(("surface")):
            mark_colors[k] = "c"
            mark_symbols[k] = "o"
        elif k.startswith(("rostralborder", "caudalborder")):
            mark_colors[k] = "b"
            mark_symbols[k] = "+"
        elif k.startswith(("medial", "lateral", "dorsal", "ventral", "rostral", "caudal")):
            mark_colors[k] = "g"
            mark_symbols[k] = "o"
        elif k.startswith(("VN", "injection", "hpc")):
            mark_colors[k] = "r"
            mark_symbols[k] = "D"
        elif k.startswith(("AN_Notch")):
            mark_colors[k] = "magenta"
            mark_symbols[k] = "X"
        elif k == "AN":
            mark_colors[k] = "r"
            mark_symbols[k] = "D"
        elif k.startswith(("soma", "cell")):
            mark_colors[k] = "y"
            mark_symbols[k] = "*"
            mark_alpha[k] = 0.8
        else:
            # CP.cprint("r", f"Did not tag marker type {k:s}, using default")
            mark_colors[k] = "m"
            mark_symbols[k] = "o"

    for c in ["soma", "cell"]:  # may not be in the list above
        if c not in mark_colors.keys():
            mark_colors[c] = "y"
            mark_symbols[c] = "*"
            mark_alpha[c] = 0.8

    return definedMarkers, mark_colors, mark_symbols, mark_alpha, all_markernames


class MosaicData:
    def __init__(self, experiment_name: str):
        datasets, experiments = get_configuration("./config/experiments.cfg")
        self.experiment_name = experiment_name
        self.experiment = experiments[experiment_name]
        self.mosaic_data = {}
        # print(experiment.keys())

    def get_from_master_directory(self, experiment_name: str):
        mosaic_dir = "mosaics"
        mosaic_paths = Path(self.experiment["analyzeddatapath"], self.experiment_name, mosaic_dir)
        assert mosaic_paths.exists(), f"Path {mosaic_paths} does not exist"

    def get_from_original_data(self, experiment_name: str, level: int = 3) -> List[Path]:
        mosaic_ext = ".mosaic"
        datadir = Path(self.experiment["rawdatapath"], self.experiment["directory"])
        mosaic_files = list(self.get_subitems(datadir, extension=mosaic_ext, level=level))
        return mosaic_files

    def get_subitems(self, folder: Path, extension: str, level: int) -> List[Path]:
        """get_subitems Get all files with extension in subfolders
        up to 'level' deep.
        """
        if level == 0:
            return [folder]
        pattern = "/".join(["*"] * level) + f"/*{extension}"
        return sorted(folder.glob(pattern))

    def read_mosaic(self, mosaic_file):
        """read_mosaic Just get the markers and cells from the mosaic
        file.

        Parameters
        ----------
        mosaic_file : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print("mosaic_file: ", mosaic_file)
        with open(mosaic_file, "r") as f:
            mdata = json.load(f)
            markers = None
            cells = []
            rulers = []
            # print(mdata['items'])
            for item in mdata["items"]:
                if item["type"] == "MarkersCanvasItem":
                    markers = item
                elif item["type"] == "CellCanvasItem":
                    cells.append(item)
                elif item["type"] == "RulerCanvasItem":
                    rulers.append(item)
                else:
                    pass
                # print("item type: ", item["type"])
        return markers, cells

    def parse_transstrial(self, fullfile, ax=None):
        marker_dict = get_markers.get_markers(fullfile)
        # get the marker types and colors from a dictionary
        definedMarkers, mark_colors, mark_symbols, mark_alpha, all_markernames = define_markers()

        if ax is None:
            f, ax = mpl.subplots(1, 1)
        measures, smoothed_poly = get_markers.plot_mosaic_markers(
            marker_dict,
            axp=ax,
            mark_colors=mark_colors,
            mark_alpha=mark_alpha,
            mark_symbols=mark_symbols,
        )
        ax.set_aspect("equal")
        # if ax is None:
        #     mpl.show()
        return measures

    def parse_coronal(self, markers):
        """parse_coronal _summary_
        Note we reduce to 2D as the z values are always 0.

        Parameters
        ----------
        markers : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        marks = markers  # get the list of marks.
        for m in marks:
            if m[0] == "surface":
                surface = m[1][:2]
            if m[0] == "medialdorsal":
                medialdorsal = m[1][:2]
            if m[0] == "medialventral":
                medialventral = m[1][:2]
        S = Point(*surface)
        MD = Point(*medialdorsal)
        ML = Point(*medialventral)
        L6 = Line(MD, ML)  # line along layer 6 between md and ml points
        Refline = L6.perpendicular_segment(S)  # line perpendicular to L6line at S
        return Refline, S, L6

    def parse_horizontal(self, markers):
        """parse_coronal _summary_
        Note we reduce to 2D as the z values are always 0.

        Parameters
        ----------
        markers : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        marks = markers  # get the list of marks.
        for m in marks:
            if m[0] == "surface":
                surface = m[1][:2]
            if m[0] == "medialrostral":
                medialrostral = m[1][:2]
            if m[0] == "medialcaudal":
                medialcaudal = m[1][:2]
        S = Point(*surface)
        MR = Point(*medialrostral)
        MC = Point(*medialcaudal)
        try:
            L6 = Line(MR, MC)  # line along layer 6 between md and ml points
        except ValueError as e:
            print(e)
            print("MR: ", MR)
            print("MC: ", MC)
            return None, None, None
        Refline = L6.perpendicular_segment(S)  # line perpendicular to L6line at S
        return Refline, S, L6

    def convert_line(self, line):
        """convert_line Convert a sympy line to a pair of numpy arrays

        Parameters
        ----------
        line : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        px = np.array([line.p1.x.evalf(), line.p2.x.evalf()])
        py = np.array([line.p1.y.evalf(), line.p2.y.evalf()])
        return px, py

    def get_depths(self, markers, cells):
        """get_depths Compute the depths of the cells based on the
        markers in the mosaic file.

        The markers canvas item for 'Cortex Coronal' has:
        {'type': 'MarkersCanvasItem', 'name': 'Cortex Coronal',
        'visible': True, 'alpha': 1.0,
        'userTransform': {'pos': [0.0, 0.0], 'scale': [1.0, 1.0], 'angle': 0.0},
        'z': 140, 'scalable': False, 'rotatable': False, 'movable': False, 'filename': None,
        'markers': [['surface', [-0.0015130248371471467, -0.0039721714273873755, 0.0]],
            ['medialdorsal', [-0.0005302668855183845, -0.003967946765084235, 0.0]],
            ['lateraldorsal', [-0.0014278865820606277, -0.0037269449583405634, 0.0]],
            ['injectionsite', [-0.0009077799463142794, -0.004169225674768161, 0.0]],
            ['medialventral', [-0.0006268827791634361, -0.004479236823747612, 0.0]],
            ['lateralventral', [-0.001578929303509599, -0.004240005547702646, 0.0]]]}
        The item for 'Cortex Horizontal' has:
        {'type': 'MarkersCanvasItem', 'name': 'Cortex Horizontal',
        etc.
        Parameters
        ----------
        markers : _type_
            _description_
        cells : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        depths = {}
        refline = None
        l6c = None
        refc = None
        S = None
        if markers is None:
            return None, depths

        if markers["name"] == "Cortex Coronal":
            refline, S, L6 = self.parse_coronal(markers["markers"])
            l6c = self.convert_line(L6)
            refc = self.convert_line(refline)

        elif markers["name"] == "Cortex Horizontal":
            refline, S, L6 = self.parse_horizontal(markers["markers"])
            if refline is None:
                return "bad marker", depths
            l6c = self.convert_line(L6)
            refc = self.convert_line(refline)

        elif markers["name"] == "DCN Transstrial":
            m = self.parse_transstrial(markers["markers"])
            exit()
        if S is None:
            print("No surface marker found in markers")
            return "bad marker", depths
        depths["type"] = markers["name"]
        depths["L6"] = l6c
        depths["Ref"] = refc
        depths["depths"] = {}
        depths["normalized_depths"] = {}
        depths["positions"] = {}
        depths["lines"] = {}
        for cell in cells:
            name = cell["name"]
            pos = cell["userTransform"]["pos"]
            pos.append(0.0)  # add Z
            cell_pos = Point(*pos)[:2]
            # print("cellpos: ", cell_pos)
            cell_line_segment = refline.perpendicular_segment(cell_pos)
            cell_line = self.convert_line(cell_line_segment)
            # ax.plot(cell_pos[0], cell_pos[1], "bo")
            # ax.text(cell_pos[0], cell_pos[1], cell["name"], color="c")
            # ax.plot(cell_line[0], cell_line[1], "b")
            inters = refline.intersection(cell_line_segment)
            # print(cell['name'], 'intersection: ', inters[0].evalf())
            # print('surface: ', S.evalf())
            if len(inters) > 0:
                depth = S.distance(inters[0]).evalf() * 1e6
                print(f"{cell['name']:s} Depth = {depth:6.1f} microns")
                depths["depths"][name] = depth.evalf()
                depths["normalized_depths"][name] = (depth / (refline.length * 1e6)).evalf()
                depths["positions"][name] = cell_pos
                depths["lines"][name] = cell_line
            else:
                print(cell["name"], "No intersection")
                # depths.append(np.nan)
        return "OK", depths

    def get_depths_transstrial(self, markers, cells):
        """get_depths Compute the depths of the cells based on the
        markers in the mosaic file.

        The markers canvas item for 'Transstrial' has:
        {'type': 'MarkersCanvasItem', 'name': 'DCN Transstrial', 'visible': True,
        'alpha': 1.0, 'userTransform': {'pos': [0.0, 0.0], 'scale': [1.0, 1.0], 'angle':
        0.0}, 'z': 100, 'scalable': False, 'rotatable': False, 'movable': False,
        'filename': None, 'markers': [['surface', [-0.0003354911723025581,
        -0.0026602019266781293, 0.0]], ['rostralsurface', [-0.0004321254678409253,
        -0.003058276290985064, 0.0]], ['rostralborder', [-0.0005830210241805457,
        -0.0031487762151599114, 0.0]], ['medialborder', [-0.0007880433183075958,
        -0.002850403332073103, 0.0]], ['caudalborder', [-0.0007312299115013049,
        -0.002363702154153586, 0.0]], ['caudalsurface', [-0.0004395359122069634,
        -0.0024622095602542184, 0.0]], ['AN', [-0.0006908716889654565,
        -0.0025163362980416474, 0.0]]]}
        etc.
        Parameters ---------- markers : _type_
            _description_
        cells : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        depths = {}

        if markers is None:
            return None, depths

        if markers["name"] == "DCN Transstrial":
            self.parse_transstrial(markers["markers"])
        depths["type"] = markers["name"]
        depths["Cell"] = l6c
        depths["Ref"] = refc
        depths["depths"] = {}
        depths["normalized_depths"] = {}
        depths["positions"] = {}
        depths["lines"] = {}
        for cell in cells:
            name = cell["name"]
            pos = cell["userTransform"]["pos"]
            pos.append(0.0)  # add Z
            cell_pos = Point(*pos)[:2]
            # print("cellpos: ", cell_pos)
            cell_line_segment = refline.perpendicular_segment(cell_pos)
            cell_line = self.convert_line(cell_line_segment)
            # ax.plot(cell_pos[0], cell_pos[1], "bo")
            # ax.text(cell_pos[0], cell_pos[1], cell["name"], color="c")
            # ax.plot(cell_line[0], cell_line[1], "b")
            inters = refline.intersection(cell_line_segment)
            # print(cell['name'], 'intersection: ', inters[0].evalf())
            # print('surface: ', S.evalf())
            if len(inters) > 0:
                depth = S.distance(inters[0]).evalf() * 1e6
                print(f"{cell['name']:s} Depth = {depth:6.1f} microns")
                depths["depths"][name] = depth.evalf()
                depths["normalized_depths"][name] = (depth / (refline.length * 1e6)).evalf()
                depths["positions"][name] = cell_pos
                depths["lines"][name] = cell_line
            else:
                print(cell["name"], "No intersection")
                # depths.append(np.nan)
        return "OK", depths

    def plot_references_and_cells(self, ax, depths, date=None):
        ax.axis("equal")
        l6line = depths["L6"]
        refline = depths["Ref"]
        ax.plot(l6line[0], l6line[1], "r")
        ax.plot(refline[0], refline[1], "g")
        ax.text(l6line[0][0], l6line[1][0], "L6", color="r")
        if date is not None:
            ax.text(refline[0][0], refline[1][0], date, color="g")
        for cell in depths["depths"].keys():
            cell_line = depths["lines"][cell]
            cell_pos = depths["positions"][cell]
            ax.plot(cell_line[0], cell_line[1], "b")
            ax.plot(cell_pos[0], cell_pos[1], "bo")
            ax.text(cell_pos[0], cell_pos[1], cell, color="c")

    def clean_strains(self, row):
        if row.strain == "NTSR-Cre":
            row.strain = "NTSR1-Cre"
        if row.strain == "VGAT-cre":
            row.strain = "VGAT-Cre:CBA/J"
        return row

    def get_slice_mosaic(self, slicepath: Path, date, strain):
        """get_slice_mosaic Get the depths of cells from the mosaic file.
        If the mosaic data is not in the dictionary, then read the file,
        otherwise just return the data from the dictionary.

        Parameters
        ----------
        slicepath : Path
            slice path that may have a mosaic file
        mosaic_data : Union[dict, None], optional
            dictionary of mosaic data (keyed by mosaic_file name), by default None

        Returns
        -------
        dict
            dict of depths and meausuremnets from the mosaic, by cell.
        """

        mosaicfiles = list(slicepath.glob("*.mosaic"))
        if len(mosaicfiles) == 0:
            return False, None  # no file
        mosaic_file = Path(mosaicfiles[0])
        if self.mosaic_data is not None and mosaic_file in self.mosaic_data.keys():
            return True, mosaic_file
        markers, cells = self.read_mosaic(mosaic_file)
        result, depths = self.get_depths(markers, cells)  # calculate cell positions
        if result is None or result == "bad marker":
            return False, mosaic_file
        depths["mosaic_file"] = mosaic_file
        mosaic_file = depths["mosaic_file"]  # retrieve the key
        self.mosaic_data[mosaic_file] = {
            "markers": markers,
            "depths": depths,
            "strain": strain,
            "date": date,
        }  # store slice mosaic on first encounter
        return True, mosaic_file

    def get_datasummary(self, experiment):
        datasummary = FUNCS.get_datasummary(experiment)
        datasummary = datasummary.apply(self.clean_strains, axis=1)
        return datasummary

    def evaluate_depths(self, experiment, plot_raw: bool = True):
        # print(experiment)
        datasummary = self.get_datasummary(experiment)
        coding_columns = [
            "date",
            "aniaml identifier",
            "strain",
            "reporters",
            "slice_slice",
            "cell_cell",
            "mosaic_file",
            "cell_layer",
            "depth",
            "normalized_depth",
        ]
        # print(datasummary.columns)

        self.get_slice_mosaicmosaic_data = {}
        coding = pd.DataFrame(columns=coding_columns)
        if "cell_depth" not in datasummary.columns:
            datasummary["cell_depth"] = np.nan
        if "noramlized_depth" not in datasummary.columns:
            datasummary["normalized_depth"] = np.nan
        if "mosaic_file" not in datasummary.columns:
            datasummary["mosaic_file"] = ""
        for index in datasummary.index:
            ai = datasummary.loc[index, "animal identifier"]
            date = datasummary.loc[index, "date"]
            strain = datasummary.loc[index, "strain"]
            reporters = datasummary.loc[index, "reporters"]
            cell_layer = datasummary.loc[index, "cell_layer"]
            cell_cell = datasummary.loc[index, "cell_cell"]
            cell_type = datasummary.loc[index, "cell_type"]
            sliceno = datasummary.loc[index, "slice_slice"]
            # look for all slices in the date, regardless of whether they are in the datasummary or not
            datepath = Path(experiment["rawdatapath"], self.experiment_name, date)
            slicepath = Path(experiment["rawdatapath"], self.experiment_name, date, sliceno)
            if slicepath.is_dir():
                ok, mosaic_file = self.get_slice_mosaic(slicepath, date, strain)
                if not ok:
                    continue
                if cell_cell not in self.mosaic_data[mosaic_file]["depths"]["depths"].keys():
                    continue
                datasummary.loc[index, "cell_depth"] = float(
                    self.mosaic_data[mosaic_file]["depths"]["depths"][cell_cell]
                )
                datasummary.loc[index, "normalized_depth"] = float(
                    self.mosaic_data[mosaic_file]["depths"]["normalized_depths"][cell_cell]
                )
                datasummary.loc[index, "mosaic_file"] = Path(mosaic_file).name

                coding = coding._append(
                    {
                        "date": date,
                        "animal identifier": ai,
                        "strain": strain,
                        "reporters": reporters,
                        "slice_slice": sliceno,
                        "cell_cell": cell_cell,
                        "mosaic_file": mosaic_file,
                        "layer": cell_layer,
                        "depth": float(
                            self.mosaic_data[mosaic_file]["depths"]["depths"][cell_cell]
                        ),
                        "normalized_depth": float(
                            self.mosaic_data[mosaic_file]["depths"]["normalized_depths"][cell_cell]
                        ),
                        "coding": "",
                    },
                    ignore_index=True,
                )
                # if index > 20:
                #     break

        coding.dropna(subset=["depth"], inplace=True)
        # print(coding.head())
        return coding, datasummary
        # print(coding.head(20))
        # sheet_name = experiment["coding_sheet"]
        # excelfilename = Path(experiment["analyzeddatapath"], exptname, experiment["coding_file"])
        # CE = write_excel_sheet.ColorExcel()
        # CE.make_excel(
        #     coding,
        #     outfile=excelfilename,
        #     sheetname=sheet_name,
        #     columns=coding_columns,
        # )

        # print("Coding Sheet written to: ", excelfilename)

    def plot_raw(self, mosaic_data, rig: Union[str, None] = "Rig2"):
        f, ax = mpl.subplots(1, 1)
        for mosaic_file in mosaic_data.keys():
            if mosaic_data[mosaic_file]["strain"] not in ["VGAT-Cre:CBA/J"]:
                continue
            if rig is None or mosaic_data[mosaic_file]["date"].startswith(rig):
                markers = mosaic_data[mosaic_file]["markers"]
                depths = mosaic_data[mosaic_file]["depths"]
                self.plot_references_and_cells(ax, depths, mosaic_data[mosaic_file]["date"])
        ax.axis("equal")
        mpl.show()


def find_mosiaic_files(
    experiment_name: str, level: int = 2, mosaic_file: Path = None
) -> pd.DataFrame:
    assert level in [1, 2, 3, 4]
    if experiment_name is None:
        experiment_name = "NF107Ai32_Het"
    if mosaic_file is None:
        mosaic_file = Path(
            "/Volumes/Pegasus_002/ManisLab_Data3/NF107Ai32_Het/2012.12.21_000/slice_000/cell_000/2017.12.21.S0C0.mosaic"
        )
    MOS = MosaicData(experiment_name)
    print("MOS.experiment_name: ", MOS.experiment_name)
    levelname = {
        1: "Day Level Mosaics",
        2: "Slice Level Mosaics",
        3: "Cell Level Mosaics",
        4: "Protocol level mosaics",
    }

    print(f"{levelname[level]}")
    fn = []
    ct = []
    ts = []
    mfiles = MOS.get_from_original_data(experiment_name, level=level)
    for i, mfile in enumerate(mfiles):
        ctime = mfile.stat().st_ctime
        time_str = datetime.datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
        fn.append("/".join(mfile.parts[-4:]))
        ct.append(ctime)
        ts.append(time_str)
    df = pd.DataFrame({"filename": fn, "ctime": ct, "time_str": ts})
    df = df.sort_values(by="ctime")

    return df


def get_markers_from_file(experiment_name: str = None, levels:list = [2,3], mosaic_file: Path = None):
    if experiment_name is None:
        experiment_name = "NF107Ai32_Het"
    if mosaic_file is None:
        mosaic_file = Path(
            "/Volumes/Pegasus_002/ManisLab_Data3/NF107Ai32_Het/2012.12.21_000/slice_000/cell_000/2017.12.21.S0C0.mosaic"
        )
    MOS = MosaicData(experiment_name)
    print("MOS.experiment_name: ", MOS.experiment_name)
    for level in levels:
        df = find_mosiaic_files(experiment_name, level=level, mosaic_file=mosaic_file)
        for i, row in df.iterrows():
            print(f"#{i:3d}: {row['filename']} {row['time_str']}")


def main():

    need_update = False
    experiment_name = "CBA_Age"
    Expt = "CBA_Age"
    datasets, experiments = get_configuration("config/experiments.cfg")
    experiment = experiments[Expt]

    # experiment_name = "GlyT2_NIHL"
    MOS = MosaicData(experiment_name)
    print("MOS.experiment_name: ", MOS.experiment_name)
    mfiles = MOS.get_from_original_data(experiment_name)
    # print("mfiles: ", mfiles)
    # for f in mfiles:
    #     print(f.name)

    relevant_files = [
        "2023.11.10.S0.mosaic",
        "2023.11.10.S1.mosaic",
        "2024.01.12.s2.mosaic",
        "2024.05.07.s0.mosaic",
        "2024.05.07.s2.mosaic",
        "2024.05.16.S2.mosaic",
        "2024.07.29.S2.mosaic",
        "2024.08.21.S1.mosaic",
    ]
    # relevant_files = ["2024.12.17.s0.mosaic"]
    mfiles = set(mfiles)
    fig, ax = mpl.subplots(3, 3, figsize=(12, 12))
    ax = np.ravel(ax)
    n = 0
    measures = []
    for f in mfiles:
        if f.name in relevant_files:
            # print("file: ", f.name)
            # continue
            # markers, cells= MOS.read_mosaic(f)

            # print("Markers: ")
            # print(markers)
            # print("cells: ", cells)
            print("Parsing mosiac: ", str(f))
            meas = MOS.parse_transstrial(f, ax=ax[n])
            meas["mosaic_file"] = f.name
            measures.append(meas)
            ax[n].set_title(str(f.name), fontsize=7)
            n += 1
    mpl.tight_layout()
    ms = f"{measures!s}"
    print(ms)

    mpl.show()


if __name__ == "__main__":

    d = get_markers_from_file(
        "NF107Ai32_Het",
        levels=[2,3],
        mosaic_file=Path(
            "/Volumes/Pegasus_002/ManisLab_Data3/NF107Ai32_Het/2012.12.21_000/slice_000/cell_000/2017.12.21.S0C0.mosaic"
        ),
    )
    # main()

    # for m in markers['markers']:
    #     print(m)

    # datasummary = FUNCS.get_datasummary(MOS.experiment)

    # if 'mosaic_file' not in datasummary.columns:
    #     print("updating mosaic information to datasummary")
    #     df, datasummary = MOS.evaluate_depths(MOS.experiment)
    #     need_update = True
    # outpath = Path(MOS.experiment['analyzeddatapath'],
    #                                   experiment_name,
    #                                   MOS.experiment['datasummaryFilename'])
    # if need_update:
    #     datasummary.to_pickle(outpath)
    #     DSUM.make_excel(datasummary, outpath)
    #     exit()
    # # MOS.plot_raw(MOS.mosaic_data)
    # # exit()

    # # f2, ax2 = mpl.subplots(1, 1)
    # # sns.histplot(all_depths, ax=ax2)
    # df = datasummary
    # # sns.displot(data=df, y="cell_layer", x="strain")
    # N_NTSR1 = df[df.strain == "NTSR1-Cre"].count()
    # N_tdTomato = df[df.strain == "VGAT-Cre:Ai9"].count()
    # N_VGAT = df[df.strain == "VGAT-Cre:CBA/J"].count()
    # print("NTSR1-Cre: ", N_NTSR1)
    # print("tdTomato: ", N_tdTomato)
    # print("VGAT-Cre:CBA/J: ", N_VGAT)

    # layerorder = ["L1", "L2/3", "L4", "L5", "L5/6", "L6", " "]
    # df.dropna(subset=['normalized_depth', 'cell_layer', 'strain'], ignore_index=True, inplace=True)

    # print(df['normalized_depth'])
    # sns.displot(data=df,
    #             x="normalized_depth", hue="cell_layer", multiple="stack")

    # f, ax = mpl.subplots(1, 1)
    # # mpl.scatter(df["cell_depth"], df["normalized_depth"]) #, c=df["cell_layer"])
    # sns.swarmplot(data=df,
    #              x="cell_layer", y="normalized_depth", hue="strain", order=layerorder, ax=ax)
    # ax.set_aspect("auto")
    # ax.set_ylim([1.0, 0])
    # ax.set_ybound(lower=0.0, upper=1.00)
    # mpl.show()

    # print(cells)
