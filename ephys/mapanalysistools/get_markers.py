from pathlib import Path
from typing import Tuple, Dict
import json
import numpy as np
from pylibrary.tools import cprint as CP
import sympy.geometry
import scipy.interpolate
import numpy as np
import pint

UR = pint.UnitRegistry()


def get_markers(fullfile: Path, verbose: bool = True) -> dict:
    # dict of known markers and one calculation of distance from soma to surface
    dist = np.nan
    # this may be too restrictive....
    marker_dict = {}
    # :dict = {
    #     "soma": [],
    #     "surface": [],
    #     "medialborder": [],
    #     "lateralborder": [],
    #     "rostralborder": [],
    #     "caudalborder": [],
    #     "ventralborder": [],
    #     "dorsalborder": [],
    #     "rostralsurface": [],
    #     "caudalsurface": [],
    #     "AN": [],
    #     "dist": dist,
    # }

    mosaic_file = list(fullfile.glob("*.mosaic"))
    if len(mosaic_file) > 0:
        if verbose:
            CP.cprint("c", f"    Have mosaic_file: {mosaic_file[0].name:s}")
        state = json.load(open(mosaic_file[0], "r"))
        cellmark = None
        for item in state["items"]:
            if item["type"] == "MarkersCanvasItem":
                markers = item["markers"]
                for markitem in markers:
                    if verbose:
                        CP.cprint(
                            "c",
                            f"    {markitem[0]:>20s} x={markitem[1][0]*1e3:8.3f} y={markitem[1][1]*1e3:8.3f} z={markitem[1][2]*1e3:8.3f} mm ",
                        )
                    marker_dict[markitem[0]] = [
                        markitem[1][0],
                        markitem[1][1],
                        markitem[1][2],
                    ]
                    for j in range(len(markers)):
                        markname = markers[j][0]
                        if markname in marker_dict:
                            marker_dict[markname] = [
                                markers[j][1][0],
                                markers[j][1][1],
                            ]
            elif item["type"] == "CellCanvasItem":  # get Cell marker position also
                cellmark = item["userTransform"]
            else:
                pass
            # print("didnt parse item type: ", item["type"])
        soma_xy: list = []
        somapos = []
        if cellmark is None:
            if "soma" in marker_dict.keys():
                somapos = marker_dict["soma"]
        else:  # override soma position with cell marker position
            somapos = cellmark["pos"]
            marker_dict["soma"] = somapos

        surface_xy: list = []

        if "surface" in marker_dict.keys():
            if len(somapos) >= 2 and len(marker_dict["surface"]) >= 2:
                soma_xy = somapos
                surface_xy = marker_dict["surface"]
                dist = np.sqrt(
                    (soma_xy[0] - surface_xy[0]) ** 2 + (soma_xy[1] - surface_xy[1]) ** 2
                )
                if verbose:
                    CP.cprint("c", f"    soma-surface distance: {dist*1e6:7.1f} um")
            else:
                if verbose:
                    CP.cprint("r", "    Not enough markers to calculate soma-surface distance")
        if soma_xy == [] or surface_xy == []:
            if verbose:
                CP.cprint("r", "    No soma or surface markers found")
    else:
        if verbose:
            pass
            # CP.cprint("r", "No mosaic file found")

    return marker_dict



def plot_mosaic_markers(
    markers: dict, axp, mark_colors: dict, mark_symbols: dict, mark_alpha: dict,
) -> tuple():
    measures = {}
    smoothed_poly = None
    markers_complete = True
    print(mark_alpha.keys())
    if markers is not None and len(markers.keys()) > 0:
        for marktype in markers.keys():
            if marktype not in mark_colors.keys():
                markers_complete = False
                continue
            position = markers[marktype]
            if marktype in ["soma"]:
                markersize = 8
            elif marktype.startswith(
                ("dorsal", "rostral", "caudal", "ventral", "medial", "lateral")
            ):
                markersize = 3
            else:
                markersize = 4
            if axp is not None and position is not None and len(position) >= 2 and marktype in mark_alpha.keys():
                axp.plot(
                    [position[0], position[0]],
                    [position[1], position[1]],
                    marker=mark_symbols[marktype],
                    color=mark_colors[marktype],
                    markersize=markersize,
                    alpha=mark_alpha[marktype],
                )
        pcoors: list = []
        for marker in [
            "rostralborder",
            "medialborder",
            "caudalborder",
            "caudalsurface",
            "surface",
            "rostralsurface",
        ]:
            if marker in markers.keys():
                pcoors.append(tuple(markers[marker]))
            else:
                if marker == "soma":
                    continue
                # print("marker : ", marker, " not found in markers\n", "markers: ", markers)
                markers_complete = False
                continue
        # print("got markers...", pcoors, markers_complete)
        if len(pcoors) > 0 and markers_complete:
            # marker_poly = sympy.geometry.polygon.Polygon(*pcoors)
            # patch = descartes.PolygonPatch(self.marker_poly, fc=None, ec='red', alpha=0.5, zorder=2)
            pcoors.append(pcoors[0])
            pcoors_x = [x[0] for x in pcoors]
            pcoors_y = [x[1] for x in pcoors]

            tck, _ = scipy.interpolate.splprep([pcoors_x, pcoors_y], s=0, per=True)
            xx, yy = scipy.interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
            if axp is not None:
                axp.plot(xx, yy, "g-", lw=0.75, zorder=2)
            smoothed_poly = [(xx[i], yy[i]) for i in range(len(xx))]
            smoothed_poly = sympy.geometry.polygon.Polygon(*smoothed_poly)
            medialpt = sympy.geometry.point.Point(markers["medialborder"])
            lateralpt = sympy.geometry.point.Point(markers["surface"])
            measures["medial_lateral_distance"] = medialpt.distance(lateralpt) * UR.m
            measures["area"] = smoothed_poly.area * UR.m * UR.m
            rostralpt = sympy.geometry.point.Point(markers["rostralborder"])
            caudalpt = sympy.geometry.point.Point(markers["caudalborder"])
            measures["rostral_caudal_distance"] = rostralpt.distance(caudalpt) * UR.m
            UR.define("mm = 1e-3 * m")
            UR.define("um = 1e-6 * m")
            print(f"Smoothed Polygon Slice area: {measures['area'].to(UR.mm*UR.mm):L.4f}")
            area_txt = f"Area={measures['area'].to(UR.mm*UR.mm):P5.3f} D={measures['medial_lateral_distance'].to(UR.um):P5.1f}"
            area_txt += f"RC={measures['rostral_caudal_distance'].to(UR.um):P5.1f}"
            if axp is not None:
                axp.text(
                s=area_txt,
                x=1.0,
                y=-0.05,
                transform=axp.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="k",
            )
    return measures, smoothed_poly
