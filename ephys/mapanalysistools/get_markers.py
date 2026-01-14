from pathlib import Path
from typing import Tuple, Dict, Union
import json
import numpy as np
from pylibrary.tools import cprint as CP
import ephys.tools.configfile as CF
import matplotlib.pyplot as mpl
import sympy.geometry
import shapely
import scipy.interpolate
import numpy as np
import pint
import pprint
import re
import ephys.mapanalysistools.markers as MARKS


re_slice = re.compile(r"slice_(\d+)")
re_cell = re.compile(r"cell_(\d+)")
re_day = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})_00")
re_mosaic_day = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\.")

UR = pint.UnitRegistry()


def look_for_mosaic_file(protodir: Path) -> [Union[Path, None]]:
    """look_for_mosaic_file find the mosaic file associated with this cell.
    Working up from the CELL directory, we look for a suitable .mosaic file
    in either the cell, the slice or day directory.
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
    re_mosaic = re.compile(r"^(?P<date>(\d{4})\.(\d{2})\.(\d{2}))[\._]{1}(?P<slice>S(\d+))[\.]{0,1}(?P<cell>C(\d+))\.mosaic$", re.IGNORECASE)
        # print(protodir)
    pdir = protodir
    while not pdir.name.startswith("cell_"):
        pdir = pdir.parent
    if not pdir.name.startswith("cell_"):
        raise ValueError(
            f"Looking for mosaic file: need to start with cell. Got invalid cell path name: {protodir.name}"
        )
    celldir = Path(
        protodir
    )  

    slicedir = Path(celldir).parent 
    daydir = Path(slicedir).parent
    daystr = str(daydir.name)[:-4]
    
    cell_mosaics = list(celldir.glob("*.mosaic"))

    if len(cell_mosaics) > 0:
        for cm in cell_mosaics:
            # print("cell mosaics: ", cm)
            matchm = re_mosaic.match(cm.name)
            if matchm:
                # datepart = matchm.group("date")
                # slicepart = matchm.group("slice")
                cellpart = matchm.group("cell")
                if len(cellpart) > 0: 
                    return cm   # return the first matching cell mosaic file in the directory
    slice_mosaics = list(slicedir.glob("*.mosaic"))
    if len(slice_mosaics) > 0:
        for sm in slice_mosaics:
            matchm = re_mosaic.match(sm.name)
            if matchm:
                # datepart = matchm.group("date")
                slicepart = matchm.group("slice")
                cellpart = matchm.group("cell")
                if len(slicepart) >= 0: # cell part is optional.
                    return sm   # return the first matching slice mosaic file in the directory

    day_mosaics = list(daydir.glob("*.mosaic"))
    if len(day_mosaics) > 0:
        for dm in day_mosaics:
            # print("day mosaics: ", dm)
            matchm = re_mosaic.match(dm.name)
            if matchm:
                datepart = matchm.group("date")
                slicepart = matchm.group("slice")
                # cellpart = matchm.group("cell")
                if len(datepart) >= 0 and len(slicepart) >= 0:
                    return dm   # return the first matching day mosaic file in the directory
  
    else:
        print("didn't find markers for: ", protodir)
        print("looked for day mosaic: ", day_mosaics)

    return None

def markers_to_dict(markers: MARKS.MarkerGroup) -> dict:
    marker_dict = markers.to_dict()

    return marker_dict

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
    # print("EPHYS: get_markers: Looking for mosaic file in: ", fullfile)
    mosaic_file = look_for_mosaic_file(fullfile)
    if mosaic_file is not None :
        # if verbose:
        CP.cprint("c", f"    Have mosaic_file: {mosaic_file.name:s}")
        state = json.load(open(mosaic_file, "r"))
        cellmarks = {}
        # pick up the individual items and parse the data from them
        for item in state["items"]:
            match item["type"]:
                case "MarkersCanvasItem":
                    markers = item["markers"]
                    marker_name = item["name"]
                    marker_dict["name"] = marker_name
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
                    marker_dict["somas"][cellname] = somapos  # .append({cellname: somapos})
        # print(marker_dict.keys())
        # exit()
        soma_xy = []
        surface_xy: list = []
        # we rename these old keys because they make the assignment of the splines
        # more complicated, but we don't want to have to rewrite all the old
        # mosaic files either.
        if "medialborder" in marker_dict.keys():  # rename the key to "medialboundary"
            marker_dict["medialboundary"] = marker_dict.pop("medialborder")
        if "surface" in marker_dict.keys():
            if len(marker_dict["surface"]) >= 2:
                surface_xy = marker_dict["surface"]

            # now compute distance from soma to surface marker for each soma
            # this may not be appropriate for all datasets.

            if "somas" in list(marker_dict.keys()):
                for i_soma, cellname in enumerate(marker_dict["somas"]):
                    soma_xy = marker_dict["somas"][cellname]
                    surface_xy = marker_dict["surface"]
                    dist = np.sqrt(
                        (soma_xy[0] - surface_xy[0]) ** 2 + (soma_xy[1] - surface_xy[1]) ** 2
                    )
                    if verbose:
                        CP.cprint(
                            "c",
                            f"   {cellname:s} soma-'surface marker' distance: {dist*1e6:7.1f} um",
                        )

        # if soma_xy == [] or surface_xy == []:
        #     if verbose:
        #         CP.cprint("r", "    No soma or surface markers found")
    else:
        if verbose:
            # pass
            CP.cprint("m", "*** No mosaic file found")
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
        boundary = shapely.geometry.Polygon([(v.x, v.y) for v in boundary.vertices])
    if not isinstance(boundary, shapely.geometry.Polygon):
        raise ValueError("Boundary must be a shapely Polygon.")

    min_length = float("inf")
    shortest_line = None
    longest_line = None
    centroid = boundary.centroid
    # first find the shortest line through the centroid
    lfac = 2000  # length factor to extend the line beyond the boundary
    try:
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
    except Exception as e:
        print("Error finding shortest line: ", e)
        pass
    # now find the longest line that is perpendicular to the shortest line
    # if shortest_line is None:
    #     raise ValueError("No intersection found with the boundary polygon.")

    # shortest_line holds the shapely LineString of the shortest line
    # now find the longest line that is perpendicular to the shortest line
    if shortest_line is not None:
        perp_length = shortest_line.length * 10.0  # length of the perpendicular line
        for t in np.linspace(0, 1, 20):
            # get a parameterized point on the shortest line - this is the point where the perpendicular line will be drawn
            pt = shortest_line.line_interpolate_point(t, normalized=True)
            if pt.is_empty:
                continue

            # now get a perpendicular line to the shortest line at this point
            offset_dist = perp_length / 2.0
            left = shortest_line.parallel_offset(offset_dist, "left")
            right = shortest_line.parallel_offset(offset_dist, "right")
            # print("pt: ", pt, left, right)
            # Find the points on the left and right offset lines corresponding to the original point
            # make a simple linestring for each of the points, and then
            # make a final linestring
            left_point = left.interpolate(shortest_line.project(pt))
            right_point = right.interpolate(shortest_line.project(pt))
            # print("left_point: ", left_point, " right_point: ", right_point)
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
    if len(coord_pairs) == 0:
        return None, None
    print("\n", coord_pairs)
    coordinates_x = np.array([x[0] for x in coord_pairs])
    coordinates_y = np.array([x[1] for x in coord_pairs])
    dist = np.sqrt(
        (coordinates_x[:-1] - coordinates_x[1:]) ** 2
        + (coordinates_y[:-1] - coordinates_y[1:]) ** 2
    )
    cumul_dist = np.concatenate(([0], dist.cumsum()))
    # print([coordinates_x, coordinates_y])
    # print("cumul_dist: ", cumul_dist)
    b_spline = scipy.interpolate.make_interp_spline(
        cumul_dist, np.c_[coordinates_x, coordinates_y], bc_type="clamped"
    )
    if remove_ends:
        cumul_dist = cumul_dist[1:-1]

    u_pts = np.linspace(cumul_dist[0], cumul_dist[-1], npoints)  # parameterized points
    xx, yy = b_spline(u_pts).T  # scipy.interpolate.BSpline.__call__(u_pts, b_spline)
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


def find_marker_type(markers: dict):
    """find_marker_type finds the type of mosaic from the markers present.

    Parameters
    ----------
    markers : dict
        Dictionary of markers.

    Returns
    -------
    str
        The type of mosaic.
    """

    raise ValueError("Not implemented yet.")


def get_coordinates_of_a_type(markers: list, marker_types: list, marker_info, start_pos=0) -> list:
    """get_coordinates_of_a_type finds the markers of a specific type in the markers dictionary.

    Parameters
    ----------
    markers : list
        Dictionary of markers.

    Returns
    -------
    list
        List of marker coordinates.
    """
    coordinates: list = []
    coord_names: list = []
    last_valid_pos = -1
    print("\n   ### get_coordinates_of_a_type: marker_info")
    print("         ### GCOAT markers: ", markers)
    print("         ### GCOAT marker_types: ", marker_types)
    # pprint.pprint(marker_info)
    for imark, markerkey in enumerate(marker_info):  # in order
        marker = marker_info[markerkey]
        print("\n     ### GCOAT: checking marker: ", imark, marker)
        print(f"{'='*80}\n", marker.markers)
        print(imark, len(marker.markers))
        if imark >= len(marker.markers):
            continue
        if marker.name in coord_names or marker.markers[imark].group == "point":
            continue
        if marker.name in ["soma", "somas"]:
            continue
        for mt in marker_types:
            if (marker.name).find(mt) != -1 or (marker.markers[imark].group).find(mt) != -1:
                try:
                    print(marker.markers[imark] )
                    coordinates.append(tuple(marker.markers[imark].coordinates[0]))
                except KeyError:
                    print("markers had keys: ", markers.keys())
                    print("looking for marker name: ", marker.name)
                    raise()
                coord_names.append(marker.name)
                last_valid_pos = imark
    if start_pos > 0:  # allow to wrap around
        for imark, marker in enumerate(marker_info.markers[: start_pos - 1]):  # in order
            if marker.name in coord_names:
                continue
            if marker.name in ["soma", "somas"]:
                continue
            for mt in marker_types:
                if (marker.name).find(mt) != -1 or (marker.group).find(mt) != -1:
                    coordinates.append(tuple(markers[marker.name]))
                    coord_names.append(marker.name)
                    last_valid_pos = imark
    w = dict(zip(coord_names, coordinates))  # remove any duplicates by name
    coord_names = [x for x in w]
    coordinates = [x for x in w.values()]
    print("coords in order")
    pprint.pprint(w)
    print("="*80)
    return coordinates, coord_names, last_valid_pos


def get_markers_of_a_class(markers: dict, marker_class: str) -> tuple:
    """get_markers_of_a_class find markers with certain attributes ('class')
    and return those markers as a list.

    Parameters
    ----------
    markers : dict
        _description_
    marker_class : str
        _description_

    Returns
    -------
    tuple
        _description_
    """
    print("get_markers_of_a_class: markers: ", markers)
    raise ()

def compute_spline_segmenet(markers, marker_type_list, marker_definitions, axp=None, line_style="g-") -> tuple:
    coordinates, coord_names, end_pos = get_coordinates_of_a_type(
        markers,
        marker_type_list, # ["surface", "rostralborder", "caudalborder", "dorsalborder", "ventralborder"],
        marker_definitions,
        start_pos=0,
    )
    xx, yy = compute_splines(coordinates, npoints=100)
    if axp is not None and xx is not None:
        axp.plot(xx, yy, line_style, lw=0.75, zorder=2)
        axp.plot(xx[0], yy[0], "ro", markersize=6, alpha=0.5)
        axp.plot(xx[-2], yy[-2], "rX", markersize=6, alpha=0.5)
    return xx, yy

def get_marker_definitions(markers: dict, marker_template) -> MARKS.MarkerGroup:
    """get_marker_definitions gets the marker definitions for new and old style markers.

    Parameters
    ----------
    markers : dict
        _description_
    marker_template : _type_
        _description_

    Returns
    -------
    MARKS.MarkerGroup
        _description_
    """
    # print("get_old_marker_definitions: marker_template: ", marker_template)
    # print("marker_template.defined_markers: ", marker_template.defined_markers)
    print("markers: ", markers)
    import importlib
    importlib.reload(MARKS)
    print("marker_template.defined_markers kess: ", marker_template.defined_markers.keys())
    print(marker_template.defined_markers["Markers"])
    try:
        defined = marker_template.defined_markers["Markers"]
        return defined
    except KeyError:
        print("marker_template.defined_markers keys: ", marker_template.defined_markers.keys())
        raise KeyError("Could not find old marker definitions for 'Markers'.")

def get_marker_info(marker_definitions, marktype: str, info_field: str):
    """get_marker_info gets the marker info for a specific marker type and info field.

    Parameters
    ----------
    marker_definitions : _type_
        _description_
    marktype : str
        _description_
    info_field : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    try:
        mtypes = marker_definitions.markers
    except Exception as e:
        mtypes = marker_definitions
    for m in mtypes:
        try:
            if m.name == marktype:
                return m.__getattribute__(info_field)
        except Exception as e:
            # CP.cprint("r", f"Error getting marker info for {marktype}: {e}")
            # raise ValueError(f"Error getting marker info for {marktype}: {e}")
            if m == marktype:
                # print("returning: ", m, info_field)
                if hasattr(m, info_field):  
                    return mtypes[m].__getattribute__(info_field)
                else: # provide some defaults for the old version:
                    match info_field:
                        case "markersize":
                            return 6
                        case "symbol":
                            return "o"
                        case "color":
                            return "r"
                        case "alpha":
                            return 0.8
                        case "short_name":
                            return marktype[:3]
    return None


def plot_mosaic_markers(markers: dict, axp, marker_template=None) -> tuple():
    measures = {
        "area": np.nan,
        "medial_lateral_distance": np.nan,
        "short_axis": np.nan,
        "long_axis": np.nan,
        "eccentricty": np.nan,
        "cir_circle": np.nan,
        "perimeter": np.nan,
        "shape_index": np.nan,
        "fractal_dimension": np.nan,
        "shortest_line": None,
        "longest_line": None,
        "rostral_caudal_distance": np.nan,
    }

    smoothed_poly_list = []
    surface_poly_list = []
    new_style = True
    marker_definitions = marker_template.defined_markers
    if markers is None: #  or len(markers.keys()) == 0:
        return None, None
    # print("\n***Plot Mosaic Markers")
    for imark, marker in enumerate(markers):
        # print("   ***PMM: marker: ", marker)
        marktype = marker.name
        if marktype in ["soma"] or get_marker_info(marker_definitions, marktype, "group") in ["point"]:  # draw cell position
            markersize = get_marker_info(marker_definitions, marktype, "markersize")
            marksymbol = get_marker_info(marker_definitions, marktype, "symbol")
            markcolor = get_marker_info(marker_definitions, marktype, "color")
            markalpha = get_marker_info(marker_definitions, marktype, "alpha")
    
            cellpos = marker.coordinates[0]   # it is a point marker, so only one coordinate
            if axp is not None:
                axp.plot(
                    cellpos[0],
                    cellpos[1],
                    marker=marksymbol,
                    color=markcolor,
                    markersize=markersize,
                    alpha=markalpha,
                    zorder=10000,
                )
                axp.text(
                    cellpos[0],
                    cellpos[1],
                    marktype,
                    fontsize=6,
                    color="w",
                    zorder=10001,
                )

        # if axp is not None and position is not None and len(position) == 2:

        #     axp.plot(
        #         [position[0], position[0]],
        #         [position[1], position[1]],
        #         marker=get_marker_info(marker_definitions, marktype, "symbol"),
        #         color=get_marker_info(marker_definitions, marktype, "color"),
        #         markersize=get_marker_info(marker_definitions, marktype, "markersize"),
        #         alpha=get_marker_info(marker_definitions, marktype, "alpha"),
        #     )

        #     axp.text(
        #         position[0],
        #         position[1],
        #         s=get_marker_info(marker_definitions, marktype, "short_name"),
        #         fontsize=6,
        #         color="w",
        #     )
    # note that by passing marker_template.defined_markers[marktype] (marker_definitions),
    # are providing the ORDER in which the markers should be drawn

    # surface_coordinates, surface_coord_names, end_pos = get_coordinates_of_a_type(
    #     markers,
    #     ["surface", "rostralborder", "caudalborder", "dorsalborder", "ventralborder"],
    #     marker_definitions,
    #     start_pos=0,
    # )
    surface_xx, surface_yy = compute_spline_segmenet(
        markers,
        ["surface", "rostralborder", "caudalborder", "dorsalborder", "ventralborder"],
        marker_definitions,
        axp,
        line_style="g-",
    )
    deep_xx, deep_yy = compute_spline_segmenet(
        markers,
        ["interior", "boundary", "deep", "border"],
        marker_definitions,
        axp,
        line_style="b-",
    )
    sympy_smoothed_poly = None
    shapely_smoothed_poly = None
    names = [markers[i].name for i in range(len(markers))]
    if "deep_reference" in names and "surface_reference" in names:
        dr = [m.coordinates for i, m in enumerate(markers) if m.name == "deep_reference"][0][0]
        sr = [m.coordinates for i, m in enumerate(markers) if m.name == "surface_reference"][0][0]
        print(dr, sr)
        deep_pt = sympy.geometry.point.Point(dr, dim=2)
        surface_pt = sympy.geometry.point.Point(sr, dim=2)
    else:
        deep_pt = None
        surface_pt = None


    if surface_xx is not None:
        CP.cprint("c", "   ...computing surface_xx")
        surface_poly_list = [(surface_xx[i], surface_yy[i]) for i in range(len(surface_xx) - 1)]
        smoothed_poly_list = surface_poly_list.copy()
        sympy_surface_poly = sympy.geometry.polygon.Polygon(*surface_poly_list)
        shapely_smoothed_surface_poly = shapely.geometry.Polygon(surface_poly_list)

    if deep_xx is not None and surface_xx is not None:
        CP.cprint("c", "   ...computing deep_xx")
        smoothed_poly_list.extend([(deep_xx[i], deep_yy[i]) for i in range(len(deep_xx) - 1)])
        sympy_smoothed_poly = sympy.geometry.polygon.Polygon(*smoothed_poly_list)
        shapely_smoothed_poly = shapely.geometry.Polygon(smoothed_poly_list)

        if deep_pt is not None:
            measures["medial_lateral_distance"] = float(deep_pt.distance(surface_pt))
        else:
            measures["medial_lateral_distance"] = np.nan
        measures["area"] = np.abs(float(sympy_smoothed_poly.area))
        measures["cir_circle"] = float(
            1.0 - (measures["area"] / shapely.minimum_bounding_circle(shapely_smoothed_poly).area)
        )
        measures["perimeter"] = float(sympy_smoothed_poly.perimeter)
        measures["shape_index"] = float(
            0.25 * measures["perimeter"] / np.sqrt(float(measures["area"]))
        )  # shape index, see https://en.wikipedia.org/wiki/Shape_index
        measures["fractal_dimension"] = float(
            2.0 * np.log(float(measures["perimeter"]) / 4.0) / np.log(float(measures["area"]))
        )  # fractal dimension, see https://en.wikipedia.org/wiki/Fractal_dimension

        shortest_line, longest_line = find_min_max_axes(shapely_smoothed_poly)
        if shortest_line is not None:
            measures["short_axis"] = shortest_line.length
            measures["long_axis"] = longest_line.length if longest_line is not None else 0.0
            measures["eccentricty"] = (
                longest_line.length / shortest_line.length if shortest_line.length > 0 else 0.0
            )
            measures["shortest_line"] = np.array(shortest_line.coords)
            if longest_line is not None:
                measures["longest_line"] = np.array(longest_line.coords)
    if "rostralborder" in names and "caudalborder" in names:
        rb = [m.coordinates for i, m in enumerate(markers) if m.name == "rostralborder"][0][0]
        cb = [m.coordinates for i, m in enumerate(markers) if m.name == "caudalborder"][0][0]
        rostralpt = sympy.geometry.point.Point(rb, dim=2)
        caudalpt = sympy.geometry.point.Point(cb, dim=2)
        measures["rostral_caudal_distance"] = float(rostralpt.distance(caudalpt))
    somas = []
    if 'soma' in names:
        print("names: ", names)
        for m in markers:
            somas.append(m)

    # if isinstance(somas, list):
    #     somas_dict = {
    #         'cell': somas
    #     }
    #     somas = somas_dict  
    soma_depth = {}
    soma_radius = {}
    # Locate somas and draw perpendiculars...
    for i_soma, cellname in enumerate(somas):
        print(somas, i_soma, cellname)
        CP.cprint("c", f"  Computing depth for soma: {somas[i_soma].name:s}")
        # for each soma, mark its position
        cellpos = somas[i_soma].coordinates[0]  # z coordinate seems to be in microns...
        print("  cellpos: ", cellpos)
        axp.plot(cellpos[0], cellpos[1], "y*", markersize=7, alpha=1.0, zorder=10000)
        axp.text(cellpos[0], cellpos[1], cellname, fontsize=6, color="y", zorder=10001)
        if sympy_smoothed_poly is None:
            continue
        # compute the "reference" line, from the surface_reference marker to the deep_reference marker.
        cell_pt = sympy.geometry.point.Point(cellpos[:2], dim=2)  # keep 2d anyway
        soma_depth_ref_surface_marker = None
        if deep_pt is not None and surface_pt is not None:
            reference_line = sympy.geometry.Line(deep_pt, surface_pt)
            # compute the shortest line surface to the cell.
            cell_line = reference_line.perpendicular_line(cell_pt)
            # and get the intersection with the surface polygon.
            intersect = sympy_smoothed_poly.intersection(cell_line)
            soma_depth[cellname] = []
            if len(intersect) > 0:  # save the distances
                for n in range(len(intersect)):
                    soma_depth[cellname].append(intersect[n].distance(cell_pt))
                    soma_depth_ref_surface_marker = surface_pt.distance(cell_pt)
                    # CP.cprint("c", f"  Soma depth: {cellname:s} depth pt {n:d}: {soma_depth[cellname][0]*1e6:.3f} um")
                    CP.cprint("c", f"  Soma depth {cellname:s} re surface marker: {soma_depth_ref_surface_marker*1e6:.3f} um")
        else:
            soma_depth[cellname] = []  # the logic did not work, so no depth found
        if deep_pt is not None:
            soma_radius[cellname] = deep_pt.distance(intersect[1])  # cell_from_medial
        else:
            soma_radius[cellname] = np.nan
        # for n in range(len(soma_depth[cellname])):
        #     if n == 0:
        #         st = "from deep marker:"
        #         frac_depth = soma_depth[cellname][n].evalf() / soma_radius[cellname]
        #     else:
        #         st = "from surface:    "
        #         frac_depth = 1.0 - (soma_depth[cellname][n].evalf() / soma_radius[cellname])
        #     print(
        #         f"      {st:>12s} {1e6*soma_depth[cellname][n].evalf():.3f} radius: {soma_radius[cellname]*1e6:.3f},  frac_depth: {frac_depth:5.2f}"
        #     )
        if surface_pt is not None:
            axp.plot([cell_pt.x, surface_pt.x], [cell_pt.y, surface_pt.y], "y-", lw=0.65, alpha=0.8)
        if deep_pt is not None:
            axp.plot([cell_pt.x, deep_pt.x], [cell_pt.y, deep_pt.y], "c-", lw=0.6, alpha=0.8)
    if measures["shortest_line"] is not None:
        axp.plot(
            measures["shortest_line"][:, 0],
            measures["shortest_line"][:, 1],
            "k-",
            lw=0.5,
            alpha=0.2,
        )
    if measures["longest_line"] is not None:
        axp.plot(
            measures["longest_line"][:, 0],
            measures["longest_line"][:, 1],
            "k-",
            lw=0.5,
            alpha=0.2,
        )
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
    return measures, sympy_smoothed_poly

if __name__ == "__main__":
    markers = MARKS.defined_markers
    nplots = len(markers)
    import pylibrary.plotting.plothelpers as PH
    r, c = PH.getLayoutDimensions(nplots)
    P = PH.regular_grid(r, c, figsize=(8, 11))
    axr = P.axarr.ravel()
    for imark, marktype in enumerate(markers):
        axp = axr[imark]
        print(f"\nPlotting marker type: {marktype:s}")
        # print("marker_template: ", MARKS)
        print("markers[marktype]: ", markers[marktype])
        print("markers.to_dict: ", markers[marktype].to_dict())
        print("dir MARKS:\n", dir(MARKS))
        measure, poly = plot_mosaic_markers(markers[marktype].markers, axp, marker_template=MARKS)
        axp.set_title(f"Marker type: {marktype:s}")
    mpl.show()
