# Define markers for mosaic editor in a data class. 

# Markers are in named groups with a sequence of locations that should be set by the user.
# The locations are ordered [first element of tuple] for computing the area and distances.
# The second and third elements of the tuple are the x and y coordinates of the marker, in meters.
# If the first element is None, then the marker stands alone and is not used for computing area or distances.
# CN Markers for transstrial or horizontal should be listed in order of clockwise order around the structure,
# starting from the caudal border and moving rostrally,
# then medially and back to the caudal border.

# The MosaicEditor exists in acq4, and is used to define brain regions and landmarks for ephys experiments.
# These markers are used to help align electrode tracks and map data to brain regions.
# They MUST match with the markers used in the ephys_analysis/map_analysis.py code.

# Here we use a data class to define the markers, including their names,
# coordinates, and grouping (surface, interior, point, etc). 
# The grouping is used for plotting and analysis of enclosed areas or
# measurements of distances.

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as mpl
from pylibrary.tools import cprint as CP
@dataclass
class Marker:
    name: str
    short_name: str  # Optional short name for display
    coordinates: List[Tuple[float, float]]  # List of (x, y) coordinates in meters
    group: str  # e.g., 'surface', 'interiorborder', 'point'
    connections: Tuple[Union[None, str], Union[None, str]]  # Names of markers this one connects to
    order: int = 0  # Order for computing area or distances, lower numbers first
    color: str = "k"  # Optional color for plotting
    symbol: str = "o"  # Optional symbol for plotting
    markersize: int = 3  # Optional marker size for plotting
    alpha: float = 0.75  # Optional alpha for plotting

    def get(self, argument: str):
        match str:
            case 'symbol':
                return self.symbol
            case 'color':
                return self.color
            case 'markersize':
                return self.markersize
            case 'alpha':
                return self.alpha
            case 'short_name':
                return self.short_name
            case _:
                raise ValueError(f"Marker has no support to get attribute {argument}")
    def items(self):
        return self.__dict__.items()
    def to_dict(self) -> dict:
        return asdict(self)
    
@dataclass
class MarkerGroup:
    name: str
    markers: List[ Marker] = field(default_factory=list)

    def add_marker(self, marker: Marker):
        self.markers[marker.name] = marker

    def items(self):
        return self.markers.__dict__.items()
    def to_dict(self) -> dict:
        return asdict(self)
    
defined_markers = {}
defined_markers["soma"] = MarkerGroup(name="soma", markers=[
    Marker(name="soma", short_name="Soma", coordinates=[(0, 0)], group="point", connections=(None, None)),
])
defined_markers['DCN_Transstrial'] = MarkerGroup(name="DCN_Transstrial", markers=[
    Marker(name="caudalborder", short_name="CB", coordinates=[(-120e-6, -200e-6)], group="border", order=0, connections=("AN_Notch2", "caudalborder")),
    Marker(name="caudalsurface", short_name="CS", coordinates=[(75e-6, -75e-6)], group="surface", order=1, connections=("caudalborder", "caudalsurface")),
    Marker(name="surface", short_name="S", coordinates=[(100e-6, 0)], group="surface", order=2, connections=("caudalsurface", "surface")),
    Marker(name="rostralsurface", short_name="RS", coordinates=[(75e-6, 175e-6)], group="surface", order=3, connections=("surface", "rostralsurface")),
    Marker(name="rostralborder", short_name="RB", coordinates=[(-170e-6, 250e-6)], group="border", order=4, connections=("rostralsurface", "rostralborder")),
    Marker(name="medialboundary", short_name="MB", coordinates=[(-150e-6, 0)], group="interior", order=5, connections=("rostralborder", "medialboundary")),
    Marker(name="AN_Notch1", short_name="AN1", coordinates=[(-125e-6, -50e-6)], group="interior", order=6, connections=("medialboundary", "AN_Notch1")),
    Marker(name="AN_Notch2", short_name="AN2", coordinates=[(-125e-6, -175e-6)], group="interior", order=7, connections=("AN_Notch1", "AN_Notch2")),
    Marker(name="AN", short_name="AN", coordinates=[(-100e-6, -125e-6)], group="point", connections=(None, None)),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])
defined_markers['DCN Transstrial'] = MarkerGroup(name="DCN Transstrial", markers=[
    Marker(name="caudalborder", short_name="CB", coordinates=[(-120e-6, -200e-6)], group="border", order=0, connections=("AN_Notch2", "caudalborder")),
    Marker(name="caudalsurface", short_name="CS", coordinates=[(75e-6, -75e-6)], group="surface", order=1, connections=("caudalborder", "caudalsurface")),
    Marker(name="surface", short_name="S", coordinates=[(100e-6, 0)], group="surface", order=2, connections=("caudalsurface", "surface")),
    Marker(name="rostralsurface", short_name="RS", coordinates=[(75e-6, 175e-6)], group="surface", order=3, connections=("surface", "rostralsurface")),
    Marker(name="rostralborder", short_name="RB", coordinates=[(-170e-6, 250e-6)], group="border", order=4, connections=("rostralsurface", "rostralborder")),
    Marker(name="medialboundary", short_name="MB", coordinates=[(-150e-6, 0)], group="interior", order=5, connections=("rostralborder", "medialboundary")),
    Marker(name="AN_Notch1", short_name="AN1", coordinates=[(-125e-6, -50e-6)], group="interior", order=6, connections=("medialboundary", "AN_Notch1")),
    Marker(name="AN_Notch2", short_name="AN2", coordinates=[(-125e-6, -175e-6)], group="interior", order=7, connections=("AN_Notch1", "AN_Notch2")),
    Marker(name="AN", short_name="AN", coordinates=[(-100e-6, -125e-6)], group="point", connections=(None, None)),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])

defined_markers['DCN_Transstrial_NEW'] = MarkerGroup(name="DCN Transstrial NEW", markers=[
    Marker(name="caudalborder", short_name="CB", coordinates=[(-120e-6, -200e-6)], group="border", order=0, connections=(None, "caudalborder")),
    Marker(name="caudalsurface", short_name="CS",  coordinates=[(75e-6, -75e-6)], group="surface", order=1, connections=("caudalborder", "caudalsurface")),
    Marker(name="surface", short_name="S", coordinates=[(100e-6, 0)], group="surface", order=2, connections=("caudalsurface", "surface")),
    Marker(name="rostralsurface", short_name="RS", coordinates=[(75e-6, 175e-6)], group="surface", order=3, connections=("surface", "rostralsurface")),
    Marker(name="rostralborder", short_name="RB", coordinates=[(-170e-6, 250e-6)], group="surface", order=4, connections=("rostralsurface", "rostralborder")),
    Marker(name="medialboundary", short_name="MB", coordinates=[(-150e-6, 0)], group="interior", order=5, connections=("rostralborder", "medialboundary")),
    Marker(name="AN_Notch1", short_name="AN1", coordinates=[(-125e-6, -50e-6)], group="interior", order=6, connections=("medialboundary", "AN_Notch1")),
    Marker(name="AN_Notch2", short_name="AN2", coordinates=[(-125e-6, -175e-6)], group="interior", connections=("AN_Notch1", "caudalborder")),
    Marker(name="AN", short_name="AN", coordinates=[(-100e-6, -125e-6)], group="point", connections=(None, None)),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])

defined_markers['DCN Parasagittal'] = MarkerGroup(name="DCN Parasagittal", markers=[
    Marker(name="caudalsurface", short_name="CS", coordinates=[(300e-6, -200e-6)], group="surface", connections=(None, "caudalsurface")),  
    Marker(name="dorsalsurface", short_name="DS", coordinates=[(0e-6, 300e-6)], group="surface", connections=("caudalsurface", "dorsalsurface")),
    Marker(name="rostralsurface", short_name="RS", coordinates=[(-300e-6, 100e-6)], group="surface", connections=("dorsalsurface", "rostralsurface")),
    Marker(name="rostralborder", short_name="VS", coordinates=[(0e-6, -300e-6)], group="surface", connections=("rostralsurface", "rostralborder")),
    Marker(name="medialboundary", short_name="VS", coordinates=[(0e-6, -300e-6)], group="surface", connections=("rostralborder", "medialboundary")),
    Marker(name="caudalborder", short_name="VS", coordinates=[(0e-6, -300e-6)], group="surface", connections=("medialboundary", "caudalborder")),
    Marker(name="AN", short_name="AN", coordinates=[(0e-6, -150e-6)], group="point", connections=(None, None)),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])

defined_markers['DCN Coronal'] = MarkerGroup(name="DCN Coronal", markers=[
    Marker(name="dorsal_border", short_name="D", coordinates=[(-100e-6, 500e-6)], group="border", connections=("lateral4", "dorsal_border")), 
    Marker(name="medial_dorsal", short_name="M1", coordinates=[(-100e-6, 0e-6)], group="interior", connections=("dorsal", "medial_dorsal")),
    Marker(name="medial_ventral", short_name="M2", coordinates=[(-100e-6, 0e-6)], group="interior", connections=("medial_dorsal", "medial_ventral")),
    Marker(name="ventral_border", short_name="V", coordinates=[(-100e-6, -500e-6)], group="border", connections=("medial_ventral", "ventral_border")),
    Marker(name="lateral1", short_name="L1", coordinates=[(-50e-6, -400e-6)], group="surface", connections=("ventral", "lateral1")),
    Marker(name="lateral2", short_name="L2", coordinates=[(250e-6, -200e-6)], group="surface", connections=("lateral1", "lateral2")),
    Marker(name="lateral3", short_name="L3", coordinates=[(350e-6, 50e-6)], group="surface", connections=("lateral2", "lateral3")),
    Marker(name="lateral4", short_name="L4", coordinates=[(250e-6, 350e-6)], group="surface", connections=("lateral3", "lateral4")),
    Marker(name="AN", short_name="AN", coordinates=[(0e-6, -150e-6)], group="point", connections=(None, None)),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])

defined_markers["CN Parasagittal"] = MarkerGroup(name="CN Parasagittal", markers=[
    Marker(name="rostralventralAN", short_name="RVAN", coordinates=[(-300e-6, -75e-6)], group="border", connections=(None, "rostralventralAN")),    
    Marker(name="AVCNrostral", short_name="AVCNR", coordinates=[(-300e-6, 100e-6)], group="surface", connections=("rostralventralAN", "AVCNrostral")),
    Marker(name="AVCN-DCNborder", short_name="AVCN-DCN", coordinates=[(-100e-6, 300e-6)], group="border", connections=("AVCNrostral", "AVCN-DCNborder")),
    Marker(name="DCN-V1", short_name="DC-VC1", coordinates=[(0e-6, 250e-6)], group="surface", connections=("AVCN-DCNborder", "DCN-V1")),
    Marker(name="DCN-V2", short_name="DC-VC2", coordinates=[(100e-6, 150e-6)], group="surface", connections=("DCN-V1", "DCN-V2")),
    Marker(name="PVCN-DCNborder", short_name="PV-DC", coordinates=[(200e-6, -100e-6)], group="border", connections=("DCN-V2", "PVCN-DCNborder")),
    Marker(name="PVCN-midcaudal", short_name="PV-MC", coordinates=[(150e-6, -250e-6)], group="surface", connections=("PVCN-DCNborder", "PVCN-midcaudal")),
    Marker(name="PVCNcaudal", short_name="PV-C", coordinates=[(0e-6, -300e-6)], group="surface", connections=("PVCN-midcaudal", "PVCNcaudal")),
    Marker(name="caudalventralAN", short_name="CV-AV", coordinates=[(-100e-6, -300e-6)], group="border", connections=("PVCNcaudal", "caudalventralAN")),
    Marker(name="dorsalAN", short_name="D_AN", coordinates=[(-200e-6, 200e-6)], group="point", connections=(None, None)),
    Marker(name="AN", short_name="AN", coordinates=[(0e-6, -150e-6)], group="point", connections=(None, None)),
    Marker(name="VN", short_name="VN", coordinates=[(-100e-6, -200e-6)], group="point", connections=(None, None)),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])

defined_markers["VCN Horizontal"] = MarkerGroup(name="VCN Horizontal", markers=[
    Marker(name="rostralborder", short_name="RB", coordinates=[(-150e-6, 200e-6)], group="border", connections=("medialboundary", "rostralborder")),
    Marker(name="rostralsurface", short_name="RS", coordinates=[(75e-6, 175e-6)], group="surface", connections=("rostralborder", "rostralsurface")),
    Marker(name="surface", short_name="S", coordinates=[(0, 100e-6)], group="surface", connections=("rostralsurface", "surface")),
    Marker(name="caudalsurface", short_name="CS", coordinates=[(75e-6, -175e-6)], group="surface", connections=("surface", "caudalsurface")),
    Marker(name="caudalborder", short_name="CB", coordinates=[(-150e-6, -200e-6)], group="border", connections=("caudalsurface", "caudalborder")),
    Marker(name="medialboundary", short_name="MB", coordinates=[(-150e-6, 0)], group="interior", connections=("caudalborder", "medialboundary")),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])

defined_markers["VCN Parasagittal"] = MarkerGroup(name="VCN Parasagittal", markers=[
    Marker(name="rostralventralAN", short_name="RVAN", coordinates=[(-300e-6, -75e-6)], group="border", connections=(None, "rostralventralAN")),    
    Marker(name="AVCNrostral", short_name="AVCNR", coordinates=[(-300e-6, 100e-6)], group="surface", connections=("rostralventralAN", "AVCNrostral")),
    Marker(name="AVCN-DCNborder", short_name="AVCN-DCN", coordinates=[(-100e-6, 300e-6)], group="border", connections=("AVCNrostral", "AVCN-DCNborder")),
    Marker(name="PVCN-DCNborder", short_name="PV-DC", coordinates=[(200e-6, -100e-6)], group="border", connections=("DCN-V2", "PVCN-DCNborder")),
    Marker(name="PVCN-midcaudal", short_name="PV-MC", coordinates=[(150e-6, -250e-6)], group="surface", connections=("PVCN-DCNborder", "PVCN-midcaudal")),
    Marker(name="PVCNcaudal", short_name="PV-C", coordinates=[(0e-6, -300e-6)], group="surface", connections=("PVCN-midcaudal", "PVCNcaudal")),
    Marker(name="caudalventralAN", short_name="CV-AV", coordinates=[(-100e-6, -300e-6)], group="border", connections=("PVCNcaudal", "caudalventralAN")),
    Marker(name="dorsalAN", short_name="D_AN", coordinates=[(-200e-6, 200e-6)], group="point", connections=(None, None)),
    Marker(name="AN", short_name="AN", coordinates=[(0e-6, -150e-6)], group="point", connections=(None, None)),
    Marker(name="VN", short_name="VN", coordinates=[(-100e-6, -200e-6)], group="point", connections=(None, None)),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])

defined_markers["VCN Coronal"] = MarkerGroup(name="VCN Coronal", markers=[
    Marker(name="dorsomedial", short_name="DM", coordinates=[(-100e-6, 500e-6)], group="border", connections=(None, "dorsomedial")),
    Marker(name="medial", short_name="M", coordinates=[(-100e-6, 0e-6)], group="border", connections=("dorsomedial", "medial")),
    Marker(name="ventromedial", short_name="VM", coordinates=[(-100e-6, -500e-6)], group="border", connections=("medial", "ventromedial")),
    Marker(name="lateral1", short_name="L1", coordinates=[(-50e-6, -400e-6)], group="border", connections=("ventromedial", "lateral1")),
    Marker(name="lateral2", short_name="L2", coordinates=[(250e-6, -200e-6)], group="border", connections=("lateral1", "lateral2")),
    Marker(name="lateral3", short_name="L3", coordinates=[(350e-6, 50e-6)], group="border", connections=("lateral2", "lateral3")),
    Marker(name="lateral4", short_name="L4", coordinates=[(250e-6, 350e-6)], group="border", connections=("lateral3", "lateral4")),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])


defined_markers["Cortex Horizontal"] = MarkerGroup(name="Cortex Horizontal", markers=[
    Marker(name="lateralrostral", short_name="LR", coordinates=[(1000e-6, 200e-6)], group="border", connections=(None, "lateralrostral")),
    Marker(name="surface", short_name="S", coordinates=[(1000e-6, 0)], group="surface", connections=("lateralrostral", "surface")),
    Marker(name="lateralcaudal", short_name="LC", coordinates=[(1000e-6, -200e-6)], group="border", connections=("surface", "lateralcaudal")),
    Marker(name="medialcaudal", short_name="MC", coordinates=[(0e-6, -200e-6)], group="border", connections=("lateralcaudal", "medialcaudal")),
    Marker(name="medialrostral", short_name="MR", coordinates=[(0e-6, 200e-6)], group="border", connections=("medialcaudal", "medialrostral")),
    Marker(name="injectionsite", short_name="Inj", coordinates=[(0, 0)], group="point", connections=(None, None)),
    Marker(name="hpcanteriorpole", short_name="HPC", coordinates=[(200e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])

defined_markers["Cortex Coronal"] = MarkerGroup(name="Cortex Coronal", markers=[
    Marker(name="lateraldorsal", short_name="LD", coordinates=[(1000e-6, 200e-6)], group="border", connections=(None, "lateraldorsal")),
    Marker(name="surface", short_name="S", coordinates=[(1000e-6, 0)], group="surface", connections=("lateraldorsal", "surface")),
    Marker(name="lateralventral", short_name="LV", coordinates=[(1000e-6, -200e-6)], group="border", connections=("surface", "lateralventral")),
    Marker(name="medialventral", short_name="MV", coordinates=[(-150e-6,- 200e-6)], group="border", connections=("lateralventral", "medialventral")),
    Marker(name="medialdorsal", short_name="MD", coordinates=[(-150e-6, 200e-6)], group="border", connections=("medialventral", "medialdorsal")),
    Marker(name="injectionsite", short_name="Inj", coordinates=[(0, 0)], group="point", connections=(None, None)),
    Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
    Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
])

defined_markers["soma"] = MarkerGroup(name="soma", markers=[
    Marker(name="soma", short_name="Soma", coordinates=[(0, 0)], group="point", connections=(None, None),
                   color='y', symbol='*', alpha=1.0),
])

def identify_marker(marker):

    mark_keys = set(sorted(list(marker.keys())))  # keys in markers in the current mosaic
    print("="*40)
    print("marker: ", marker)
    print("identify_marker: Mark keys: ", mark_keys)
    print("="*40)
    if marker['name'] == "soma":
        CP.cprint("g", "Identified marker set as: soma")
        return "soma", defined_markers["soma"]
    for k, def_marker in defined_markers.items():  # check the dictionary of all of the pre-defined markers
        print(f"defined marker name: {def_marker.name}\n")
        all_names = [d.name for d in def_marker.markers]
        print("   had keys: ", all_names)
        if "somas" not in all_names:
            all_names.append("somas")
        if set(mark_keys) == set(all_names):
            CP.cprint("g", f"\nIdentified marker set as: {def_marker.name}")
            return def_marker

    CP.cprint("r", "Could not identify marker set.")
    raise ValueError("Could not identify marker set.")

def marker_groups():
    mgroups = list(defined_markers.keys())
    for marker in marker_group.markers.values():
        display_name = marker.short_name
        if marker.group == 'surface':
            color = 'g'
            linestyle = '-'
            ax.plot(xs, ys, color=color, linestyle=linestyle, marker='o', markersize=5)
            ax.text(xs[0], ys[0], display_name)
        elif marker.group == 'border':
            color = 'r'
            linestyle = '--'
            ax.plot(xs, ys, color=color, linestyle=linestyle, marker='o', markersize=5)
            ax.text(xs[0], ys[0], display_name)
        elif marker.group == 'interior':
            color = 'b'
            linestyle = '-.'
            ax.plot(xs, ys, color=color, linestyle=linestyle, marker='o', markersize=5)
            ax.text(xs[0], ys[0], display_name)
        else:
            color = 'k'
            linestyle = ':'
            ax.plot(xs, ys, color=color, linestyle=linestyle, marker='o', markersize=5)
            ax.text(xs[0], ys[0], display_name)

def draw_stick_outline(marker_group, ax, use_short_names:bool=True):
    """draw_stick_outline onto selected axis

    Parameters
    ----------
    marker_group : string
        the name of the marker group (see defined_markers above)
    ax : matplotlib axis
        the axis on which to draw the outline
    """
    closure_marker = None
    print("marker_group: ", marker_group)
    print("nmarkers: ", len(marker_group))
    from_marker = None
    to_marker = None
    first_point = True
    for imark, marker in enumerate(marker_group):
        print("marker: name ", marker.name)
        xs = [coord[0]*1e6 for coord in marker.coordinates]
        ys = [coord[1]*1e6 for coord in marker.coordinates]
        if use_short_names:
            display_name = marker.short_name
        else:
            display_name = marker.name
        if first_point: 
            marker_size = 8
            first_point = False
        else:
            marker_size = 5
        if marker.group == 'surface':
            color = 'g'
            linestyle = '-'
            ax.plot(xs, ys, color=color, linestyle=linestyle, marker='o', markersize=marker_size)
            ax.text(xs[0], ys[0], display_name)
        elif marker.group == 'border':
            color = 'r'
            linestyle = '--'
            ax.plot(xs, ys, color=color, linestyle=linestyle, marker='o', markersize=marker_size)
            ax.text(xs[0], ys[0], display_name)
        elif marker.group == 'interior':
            color = 'b'
            linestyle = '-.'
            ax.plot(xs, ys, color=color, linestyle=linestyle, marker='o', markersize=marker_size)
            ax.text(xs[0], ys[0], display_name)
        else:
            color = 'k'
            linestyle = ':'
            ax.plot(xs, ys, color=color, linestyle=linestyle, marker='o', markersize=marker_size)
            ax.text(xs[0], ys[0], display_name)

        if marker.group == 'point':
            continue  # do not advance the from_marker/to_marker for point markers
        to_marker = marker_group[imark+1] if imark < len(marker_group)-1 else None
        if to_marker.group == 'point':
            continue
        if from_marker is None:
            from_marker = marker  # start from this marker
        if from_marker is not None and to_marker is not None:  # draw a connection with an arrow
            ax.plot([from_marker.coordinates[-1][0]*1e6, to_marker.coordinates[0][0]*1e6],
                    [from_marker.coordinates[-1][1]*1e6, to_marker.coordinates[0][1]*1e6],
                    color=color, linestyle=linestyle)
            ax.annotate("", xy=(to_marker.coordinates[0][0]*1e6, to_marker.coordinates[0][1]*1e6),
                        xytext=(from_marker.coordinates[-1][0]*1e6, from_marker.coordinates[-1][1]*1e6),
                        arrowprops=dict(arrowstyle="->", color=color))
            # update the from position.
            from_marker = to_marker
            from_color = color
        # do a closure from the last point to the first marker that is not a point.
    for imark, marker in enumerate(marker_group):
        if marker.group != 'point':
            ax.plot([from_marker.coordinates[-1][0]*1e6, marker.coordinates[0][0]*1e6],
                    [from_marker.coordinates[-1][1]*1e6, marker.coordinates[0][1]*1e6],
                    color=from_color, linestyle='-')
            ax.annotate("", xy=(marker.coordinates[0][0]*1e6, marker.coordinates[0][1]*1e6),
                        xytext=(from_marker.coordinates[-1][0]*1e6, from_marker.coordinates[-1][1]*1e6),
                        arrowprops=dict(arrowstyle="->", color=from_color))
            break  # only do this once
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.axis('equal')


if __name__ == "__main__":
    nplots = len(defined_markers)
    ncols = 4
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = mpl.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()
    for ax in axes[nplots:]:
        ax.axis('off') 
    print("# defined markers: ", len(defined_markers))
    for i, (group_name, marker_group) in enumerate(defined_markers.items()):
        ax = axes[i]
        ax.set_title(group_name)
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.axis('equal')
        print("i: ", i, " group_name: ", group_name)
        if marker_group.markers[0].name == "soma":
            continue
        # ax.legend()
        draw_stick_outline(marker_group.markers, ax)

        # identify_marker(marker_group.markers[i].to_dict())
    mpl.tight_layout()
    mpl.show()
