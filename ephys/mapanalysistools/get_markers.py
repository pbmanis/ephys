from pathlib import Path
from typing import Tuple, Dict
import json
import numpy as np
from pylibrary.tools import cprint as CP
import sympy.geometry
import shapely
import scipy.interpolate
import numpy as np
import pint
import re
re_slice = re.compile(r"slice_(\d+)")
re_cell = re.compile(r"cell_(\d+)")
re_day = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})_00") 
re_mosaic_day = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\.")

UR = pint.UnitRegistry()

def look_for_mosaic_file(protodir: Path) -> Tuple[Path, bool]:
    """look_for_mosaic_file find the mosaic file associated with this cell.
    Working up from the CELL directory, we look for a suitable .mosaic file
    in either the slice or day directory.
    This will be a .mosaic file that matches .SN.mosaic for the slice,
    or .SN.CM.mosiaic for the cell.
    The presumed file name is one of:
    year.mo.day.SN.mosaic or year.mo.day.SN.CM.mosaic

    Parameters
    ----------
    fullfile : Path
        Path to the Cell directory

    Returns
    -------
    Tuple[Path, bool]
        _description_
    """
    # print(protodir)
    if not protodir.name.startswith("cell_"):
        raise ValueError(f"Looking for mosaic file: need to start with cell. Got invalid cell path name: {protodir.name}")
    celldir = Path(protodir)  # generally, mosaics should NOT be in cell directory, which is above the prrotocol dir.
    slicedir = Path(celldir).parent  # may be in slice directory however
    daydir = Path(slicedir).parent
    daystr  = str(daydir.name)[:-4]
    # print("slicedir: ", slicedir)
    # print("daydir: ", daydir)
    slicen = int(slicedir.name.split("_")[-1])  # get the slice number
    # print("slicen: ", slicen, "daystr: ", daystr)
    day_mosaic = list(daydir.glob("*.mosaic"))

    # print("day_mosaic: ", day_mosaic)   
    if len(day_mosaic) > 0:
        mosaic_filename = str(daydir) + f"/{daystr:s}.S{slicen:d}.mosaic"
        # print("expected filename: ", mosaic_filename)
        for dm in day_mosaic:
            # print("day mosaics: ", dm)
            if str(dm).casefold() == mosaic_filename.casefold():
                print("get_markers: Found day mosaic: ", dm)
                return dm
    # check for slice directory next
    slice_mosaic = list(slicedir.glob("*.mosaic"))
    if len(slice_mosaic) > 0:
        mosaic_filename = str(slicedir) + f"/{daystr:s}.S{slicen:d}.mosaic"
        # print("expected filename: ", mosaic_filename)
        for sm in slice_mosaic:
            # print("slice mosaics: ", sm)
            if str(sm).casefold() == mosaic_filename.casefold():
                # print("Found slice mosaic: ", sm)
                return sm
    else:
        print("didn't find markers for: ", protodir )
        print("looked for day mosaic: ", day_mosaic)
        return None





def get_markers(fullfile: Path, verbose: bool = True) -> dict:
    # dict of known markers and one calculation of distance from soma to surface
    dist = np.nan
    # this may be too restrictive....
    marker_dict = {}
    # :dict = {
    #     "somas": [],  list of soma positions, as dict of cellname: [x, y, z]
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

    if fullfile.is_dir():
        mosaic_file = list(fullfile.glob("*.mosaic"))
    elif fullfile.is_file():
        mosaic_file = [fullfile]
    if len(mosaic_file) > 0:
        if verbose:
            CP.cprint("c", f"    Have mosaic_file: {mosaic_file[0].name:s}")
        state = json.load(open(mosaic_file[0], "r"))
        cellmarks = {}
        # pick up the individual items and parse the data from them
        for item in state["items"]:
            match item["type"]:
                case "MarkersCanvasItem":
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
                case "CellCanvasItem":  # get Cell marker position also
                    xyz = [
                        item["userTransform"]["pos"][0],
                        item["userTransform"]["pos"][1],
                        item["z"],
                    ]
                    cellmarks[item["name"]] = xyz
                    if verbose:
                        # print(item)
                        CP.cprint(
                            "c",
                            f"    {item['type']:>20s} x={item['userTransform']['pos'][0]*1e3:8.3f} y={item['userTransform']['pos'][1]*1e3:8.3f} z={item['z']*1e-3:8.3f} mm ",
                        )
                case _:
                    pass
            # CP.cprint('r', f"didnt parse item type: {item['type']!s}")

        soma_xy: list = []
        somapos = []
        # print("marker_dict.keys(): ", marker_dict.keys())
        # print("cellmark: ", cellmark)
        if len(cellmarks) > 0:
            for cellname, somapos in cellmarks.items():
                # print("cellname: ", cellname)
                if "somas" not in marker_dict.keys():
                    marker_dict["somas"] = {cellname: somapos}
                else:
                    marker_dict["somas"][cellname] = somapos # .append({cellname: somapos})
        # print(marker_dict.keys())
        # exit()
        soma_xy = []
        surface_xy: list = []
        if "surface" in marker_dict.keys():
            if len(marker_dict["surface"]) >= 2:
                surface_xy = marker_dict["surface"]

            # now compute distance from soma to surface marker for each soma
            # this may not be appropriate for all datasets.
            if 'somas' in list(marker_dict.keys()):
                for i_soma, cellname in enumerate(marker_dict["somas"]):
                    soma_xy = marker_dict["somas"][cellname]
                    print("soma_xy: ", soma_xy)

                    surface_xy = marker_dict["surface"]
                    dist = np.sqrt(
                        (soma_xy[0] - surface_xy[0]) ** 2
                        + (soma_xy[1] - surface_xy[1]) ** 2
                    )
                    if verbose:
                        CP.cprint(
                            "c", f"   {cellname:s} soma-'surface marker' distance: {dist*1e6:7.1f} um"
                        )

        # if soma_xy == [] or surface_xy == []:
        #     if verbose:
        #         CP.cprint("r", "    No soma or surface markers found")
    else:
        if verbose:
            pass
            # CP.cprint("r", "No mosaic file found")
    return marker_dict


def find_min_max_axes(boundary):
    """Find the shortest line in a boundary polygon that goes through the centroid,
    and the longest line crossing the boundary polygon that is perpendicular to that line.

    Parameters
    ----------
    boundary : shapely.geometry.Polygon
        The boundary polygon to find the axes in.

    Returns
    -------
    tuple
        A tuple containing the shortest line and the longest line.
        The shortest line is a shapely LineString that goes through the centroid of the boundary polygon.
        The longest line is a shapely LineString that crosses the boundary polygon and is perpendicular to the shortest line.

    """
    if isinstance(boundary, sympy.geometry.polygon.Polygon):
        # convert sympy polygon to shapely polygon
        boundary = shapely.geometry.Polygon(
            [(v.x, v.y) for v in boundary.vertices]
        )
    if not isinstance(boundary, shapely.geometry.Polygon):
        raise ValueError("Boundary must be a shapely Polygon.")

    min_length = float("inf")
    shortest_line = None
    longest_line = None
    centroid = boundary.centroid
    # first find the shortest line through the centroid
    lfac = 2000  # length factor to extend the line beyond the boundary
    for i in range(360):  # brute force search for the shortest line
        # by rotatiing it around the centroid
        ray = shapely.LineString(  # make a line through the centroid.
            [
                (
                    centroid.x - lfac * np.cos(np.radians(i)),
                    centroid.y - lfac * np.sin(np.radians(i)),
                ),
                (
                    centroid.x + lfac * np.cos(np.radians(i)),
                    centroid.y + lfac * np.sin(np.radians(i)),
                ),
            ]
        )
        intersection = boundary.intersection(ray)  # find the intersection with the boundary
        length = intersection.length
        if length < min_length:
            min_length = length
            shortest_line = shapely.LineString(intersection.coords)
    # now find the longest line that is perpendicular to the shortest line
    if shortest_line is None:
        raise ValueError("No intersection found with the boundary polygon.")

    # shortest_line holds the shapely LineString of the shortest line
    # now find the longest line that is perpendicular to the shortest line
    perp_length = shortest_line.length * 10.0  # length of the perpendicular line
    for t in np.linspace(0, 1, 20):
        # get a parameterized point on the shortest line - this is the point where the perpendicular line will be drawn
        pt = shortest_line.line_interpolate_point(t, normalized=True)
        # now get a perpendicular line to the shortest line at this point
        offset_dist = perp_length / 2.0
        left = shortest_line.parallel_offset(offset_dist, "left")
        right = shortest_line.parallel_offset(offset_dist, "right")

        # Find the points on the left and right offset lines corresponding to the original point
        # make a simple linestring for each of the points, and then
        # make a final linestring
        left_point = left.interpolate(shortest_line.project(pt))
        right_point = right.interpolate(shortest_line.project(pt))
        li = shapely.LineString([left_point, right_point])
        long_intersection = boundary.intersection(li)
        if isinstance(long_intersection, shapely.geometry.MultiLineString):
            # if the intersection is a MultiLineString, skip it
            # but we probably should go through each line in the MultiLineString
            # and figure out if it is the longest line of the group (e.g., 
            # if the line crossed the boundary twice in one direction))
            continue
        li = shapely.LineString(long_intersection)
        if longest_line is None or li.length > longest_line.length:
            longest_line = li

    return shortest_line, longest_line


def compute_splines(coord_pairs, npoints: int = 100, remove_ends=False):
    coordinates_x = np.array([x[0] for x in coord_pairs])
    coordinates_y = np.array([x[1] for x in coord_pairs])

    dist = np.sqrt(
        (coordinates_x[:-1] - coordinates_x[1:]) ** 2
        + (coordinates_y[:-1] - coordinates_y[1:]) ** 2
    )
    cumul_dist = np.concatenate(([0], dist.cumsum()))
    b_spline, u = scipy.interpolate.splprep([coordinates_x, coordinates_y], u=cumul_dist, s=0)
    if remove_ends:
        cumul_dist = cumul_dist[1:-1]

    xx = np.linspace(cumul_dist[0], cumul_dist[-1], npoints)
    xx, yy = scipy.interpolate.splev(xx, b_spline)
    return xx, yy

def compute_measures(poly: sympy.geometry.polygon.Polygon) -> dict:
    """compute_measures computes the area and medial-lateral distance of a polygon"""
    measures = {}
    if poly is not None:
        measures["area"] = np.abs(poly.area)
        medialpt = sympy.geometry.point.Point(poly.vertices[0], dim=2)
        lateralpt = sympy.geometry.point.Point(poly.vertices[1], dim=2)
        measures["medial_lateral_distance"] = medialpt.distance(lateralpt)
    else:
        measures["area"] = np.nan
        measures["medial_lateral_distance"] = np.nan
    return measures

def plot_mosaic_markers(
    markers: dict,
    axp,
    mark_colors: dict,
    mark_symbols: dict,
    mark_alpha: dict,
) -> tuple():
    measures = {}
    smoothed_poly = None
    markers_complete = True
    # print(mark_alpha.keys())
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
            if (
                axp is not None
                and position is not None
                and len(position) >= 2
                and marktype in mark_alpha.keys()
            ):
                axp.plot(
                    [position[0], position[0]],
                    [position[1], position[1]],
                    marker=mark_symbols[marktype],
                    color=mark_colors[marktype],
                    markersize=markersize,
                    alpha=mark_alpha[marktype],
                )
        surface_coordinates: list = []
        for marker in [
            "rostralborder",
            "rostralsurface",
            "surface",
            "caudalsurface",
            "caudalborder",
        ]:

            surface_coordinates.append(tuple(markers[marker]))
            # marker_poly = sympy.geometry.polygon.Polygon(*pcoors)
            # patch = descartes.PolygonPatch(self.marker_poly, fc=None, ec='red', alpha=0.5, zorder=2)

        xx, yy = compute_splines(surface_coordinates, npoints=20)

        if axp is not None:
            axp.plot(xx, yy, "g-", lw=0.75, zorder=2)
            axp.plot(xx[0], yy[0], "ro", markersize=6, alpha=0.5)
            axp.plot(xx[-2], yy[-2], "rX", markersize=6, alpha=0.5)
        deep_boundary_coordinates: list = []
        # we use a wrap-around here to mesh with the surface points, but then will limit the data
        # to the end points for the caudal border and rostral border.
        for marker in [
            "caudalsurface",
            "caudalborder",
            "AN_Notch1",
            "AN_Notch2",
            "medialborder",
            "rostralborder",
            "rostralsurface",
        ]:
            if marker in markers.keys():
                deep_boundary_coordinates.append(tuple(markers.get(marker, {})))
        # print("deep_boundary_coordinates: ", deep_boundary_coordinates)
        # b_spline, u_par = scipy.interpolate.make_splprep([deep_boundary_coordinates_x, deep_boundary_coordinates_y], s=0)
        deep_xx, deep_yy = compute_splines(deep_boundary_coordinates, npoints=20, remove_ends=True)

        if axp is not None:
            axp.plot(deep_xx, deep_yy, "b-", lw=0.75, zorder=2)
            axp.plot(deep_xx[0], deep_yy[0], "bo", markersize=4, alpha=0.5)
            axp.plot(deep_xx[-1], deep_yy[-1], "bX", markersize=4, alpha=0.5)

        smoothed_poly_list = [(xx[i], yy[i]) for i in range(len(xx) - 1)]
        surface_poly_list = smoothed_poly_list.copy()
        smoothed_poly_list.extend([(deep_xx[i], deep_yy[i]) for i in range(len(deep_xx) - 1)])
        # smoothed_poly.append(smoothed_poly[0])  # close the loop
        # print("smoothed_poly: ", smoothed_poly)
        smoothed_poly = sympy.geometry.polygon.Polygon(*smoothed_poly_list)
        shapely_smoothed_poly = shapely.geometry.Polygon(smoothed_poly_list)
        medialpt = sympy.geometry.point.Point(markers["medialborder"], dim=2)
        lateralpt = sympy.geometry.point.Point(markers["surface"], dim=2)
        measures["medial_lateral_distance"] = float(medialpt.distance(lateralpt))
        measures["area"] = np.abs(float(smoothed_poly.area))
        measures["cir_circle"] = float(1.0 - (measures['area'] / shapely.minimum_bounding_circle(shapely_smoothed_poly).area))
        measures['perimeter'] = float(smoothed_poly.perimeter)
        measures['shape_index'] = float(0.25*measures['perimeter'] / np.sqrt(float(measures['area'])))  # shape index, see https://en.wikipedia.org/wiki/Shape_index
        measures['fractal_dimension'] = float(2.0*np.log(float(measures['perimeter'])/4.0) / np.log(float(measures['area']))) # fractal dimension, see https://en.wikipedia.org/wiki/Fractal_dimension
        
        shortest_line, longest_line = find_min_max_axes(smoothed_poly)
        measures["short_axis"] = shortest_line.length
        measures["long_axis"] = longest_line.length if longest_line is not None else 0.0
        measures["eccentricty"] = (
            longest_line.length / shortest_line.length if shortest_line.length > 0 else 0.0
        )
        measures["shortest_line"] = np.array(shortest_line.coords)
        measures["longest_line"] = np.array(longest_line.coords)
        rostralpt = sympy.geometry.point.Point(markers["rostralborder"], dim=2)
        caudalpt = sympy.geometry.point.Point(markers["caudalborder"], dim=2)
        measures["rostral_caudal_distance"] = float(rostralpt.distance(caudalpt))
        # print(markers.keys())
        somas = markers.get('somas', {})
        soma_depth = {}
        soma_radius = {}
        # print("somas: ", somas)
        for i_soma, cellname in enumerate(somas):
            print("cellname: ", cellname)

            cellpos = somas[cellname]  # z coordinate seems to be in microns...
            axp.plot(cellpos[0], cellpos[1], "y*", markersize=4, alpha=0.33)
            axp.text(cellpos[0], cellpos[1], cellname, fontsize=6)
            cell_pt = sympy.geometry.point.Point(cellpos[:2], dim=2)  # keep 2d anyway
            cell_from_medial = medialpt.distance(cell_pt)
            cell_line = sympy.geometry.Line(medialpt, cell_pt)
            # print("cell line: ", cell_line)
            # print("surface_poly: ", surface_poly)
            # intersect = cell_line.intersection(surface_poly)
            intersect = smoothed_poly.intersection(cell_line)
            soma_depth[cellname] = []
            if len(intersect) > 0:
                for n in range(len(intersect)):
                    soma_depth[cellname].append(intersect[n].distance(cell_pt))
                    # print("intersect: ", n, intersect[n].evalf())
            else:
                soma_depth[cellname] = []
            soma_radius[cellname] = medialpt.distance(intersect[1]) # cell_from_medial
            for n in range(len(soma_depth[cellname])):
                if n == 0:
                    st = "from medial:"
                else:
                    st = "depth:"
                frac_depth = soma_depth[cellname][n].evalf() / soma_radius[cellname]

                print(f"      {st:>12s} {1e6*soma_depth[cellname][n].evalf()!s} radius: {soma_radius[cellname]*1e6:.4f},  frac_depth: {frac_depth:5.2f}")
            axp.plot([medialpt.x, intersect[1].x], [medialpt.y, intersect[1].y], "y-", lw=0.5, alpha=0.7)
            # print('medialpt: ', medialpt.evalf())
            # print('intersect with surface: ', intersect[0].evalf())

            axp.plot([cell_pt.x, intersect[0].x], [cell_pt.y, intersect[0].y], "c-", lw=1.0, alpha=0.5)
        axp.plot(measures["shortest_line"][:, 0], measures["shortest_line"][:, 1], "k-", lw=0.5, alpha=0.2)
        axp.plot(measures["longest_line"][:, 0], measures["longest_line"][:, 1], "k-", lw=0.5, alpha=0.2)
        # print("soma_depth: ", soma_depth)
        # if cell_pt is not None:
        #     cell_pt = sympy.geometry.point.Point(cell_pt)
        #     cell_line = sympy.geometry.Line(
        #         medialpt, cell_pt
        #     )  # line along layer 6 between md and ml points
        #     measures["cell_distance"] = lateralpt.distance(cell_pt)
        UR.define("mm = 1e-3 * m")
        UR.define("um = 1e-6 * m")
        print(f"Smoothed Polygon Slice area: {measures['area']*1e6:.4f}")
        print(f"Medial-Lateral distance: {measures['medial_lateral_distance']*1e6:.4f} um")
        print(f"Rostral-Caudal distance: {measures['rostral_caudal_distance']*1e6:.4f} um")
        print(f"Perimeter: {measures['perimeter']*1e6:.4f} um")
        print(f"Shape index: {measures['shape_index']:.4f}")
        print(f"Fractal dimension: {measures['fractal_dimension']:.4f}")
        print(f"Circulatory circle: {measures['cir_circle']:.4f}")
        print(f"Eccentricity: {measures['eccentricty']:.4f}")
        area_txt = f"Area={measures['area']*1e6:5.3f} D={measures['medial_lateral_distance']*1e6:5.1f}"
        area_txt += f"RC={measures['rostral_caudal_distance']*1e6:5.1f}"
        if axp is not None:
            axp.text(
                s=area_txt,
                x=1.0,
                y=0.95,
                transform=axp.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="k",
            )
            axp.set_aspect("equal")
    return measures, smoothed_poly
