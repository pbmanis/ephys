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

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as mpl
import numpy as np
from pylibrary.tools import cprint as CP
from scipy.interpolate import CubicSpline
from ephys.mapanalysistools import get_markers


@dataclass
class Marker:
    name: str
    short_name: str  # Optional short name for display
    coordinates: List[Tuple[float, float]]  # List of (x, y) coordinates in meters
    group: str  # e.g., 'surface', 'interiorborder', 'point'
    connections: Tuple[Union[None, str], Union[None, str]]  # Names of markers this one connects to
    final: Union[None, str] = None  # Name of final marker in a series
    order: int = 0  # Order for computing area or distances, lower numbers first
    color: str = "k"  # Optional color for plotting
    symbol: str = "o"  # Optional symbol for plotting
    markersize: int = 3  # Optional marker size for plotting
    alpha: float = 0.75  # Optional alpha for plotting
    marktype: Union[None, str] = None  # "type" of marker, e.g., surface, interior. NOT border or point. Set automatically if None.

    def get(self, argument: str):
        match str:
            case "symbol":
                return self.symbol
            case "color":
                return self.color
            case "markersize":
                return self.markersize
            case "alpha":
                return self.alpha
            case "short_name":
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
    markers: List[Marker] = field(default_factory=list)

    def add_marker(self, marker: Marker):
        self.markers[marker.name] = marker

    def items(self):
        return self.markers.__dict__.items()

    def to_dict(self) -> dict:
        return asdict(self)


defined_markers = {}
defined_markers["soma"] = MarkerGroup(
    name="soma",
    markers=[
        Marker(
            name="soma",
            short_name="Soma",
            coordinates=[(0, 0)],
            group="point",
            connections=(None, None),
        ),
    ],
)
defined_markers["Markers"] = MarkerGroup(
    name="Markers",
    markers=[
        Marker(
            name="surface",
            short_name="S",
            coordinates=[(0, 0)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="medialboundary",
            short_name="D",
            coordinates=[(0, -100e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)
defined_markers["DCN Transstrial"] = MarkerGroup(
    name="DCN Transstrial",
    markers=[
        Marker(
            name="caudalborder",
            short_name="CB",
            coordinates=[(-120e-6, -200e-6)],
            group="border",
            order=0,
            connections=(None, "caudalborder"),
        ),
        Marker(
            name="caudalsurface",
            short_name="CS",
            coordinates=[(75e-6, -75e-6)],
            group="surface",
            order=1,
            connections=("caudalborder", "caudalsurface"),
        ),
        Marker(
            name="surface",
            short_name="S",
            coordinates=[(100e-6, 0)],
            group="surface",
            order=2,
            connections=("caudalsurface", "surface"),
        ),
        Marker(
            name="rostralsurface",
            short_name="RS",
            coordinates=[(75e-6, 175e-6)],
            group="surface",
            order=3,
            connections=("surface", "rostralsurface"),
        ),
        Marker(
            name="rostralborder",
            short_name="RB",
            coordinates=[(-170e-6, 250e-6)],
            group="border",
            order=4,
            connections=("rostralsurface", "rostralborder"),
            final = "rostralborder"
        ),
        Marker(
            name="medialboundary",
            short_name="MB",
            coordinates=[(-150e-6, 0)],
            group="interior",
            order=5,
            connections=("rostralborder", "medialboundary"),
        ),
        Marker(
            name="AN_Notch1",
            short_name="AN1",
            coordinates=[(-125e-6, -50e-6)],
            group="interior",
            order=6,
            connections=("medialboundary", "AN_Notch1"),
        ),
        Marker(
            name="AN_Notch2",
            short_name="AN2",
            coordinates=[(-125e-6, -175e-6)],
            group="interior",
            order=7,
            connections=("AN_Notch1", "AN_Notch2"),
            final = "caudalborder"
        ),
        Marker(
            name="AN",
            short_name="AN",
            coordinates=[(-100e-6, -125e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)
# defined_markers['DCN Transstrial'] = MarkerGroup(name="DCN Transstrial", markers=[
#     Marker(name="caudalborder", short_name="CB", coordinates=[(-120e-6, -200e-6)], group="border", order=0, connections=("AN_Notch2", "caudalborder")),
#     Marker(name="caudalsurface", short_name="CS", coordinates=[(75e-6, -75e-6)], group="surface", order=1, connections=("caudalborder", "caudalsurface")),
#     Marker(name="surface", short_name="S", coordinates=[(100e-6, 0)], group="surface", order=2, connections=("caudalsurface", "surface")),
#     Marker(name="rostralsurface", short_name="RS", coordinates=[(75e-6, 175e-6)], group="surface", order=3, connections=("surface", "rostralsurface")),
#     Marker(name="rostralborder", short_name="RB", coordinates=[(-170e-6, 250e-6)], group="border", order=4, connections=("rostralsurface", "rostralborder")),
#     Marker(name="medialboundary", short_name="MB", coordinates=[(-150e-6, 0)], group="interior", order=5, connections=("rostralborder", "medialboundary")),
#     Marker(name="AN_Notch1", short_name="AN1", coordinates=[(-125e-6, -50e-6)], group="interior", order=6, connections=("medialboundary", "AN_Notch1")),
#     Marker(name="AN_Notch2", short_name="AN2", coordinates=[(-125e-6, -175e-6)], group="interior", order=7, connections=("AN_Notch1", "AN_Notch2")),
#     Marker(name="AN", short_name="AN", coordinates=[(-100e-6, -125e-6)], group="point", connections=(None, None)),
#     Marker(name="surface_reference", short_name="SR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
#     Marker(name="deep_reference", short_name="IR", coordinates=[(0e-6, 0e-6)], group="point", connections=(None, None)),
# ])

defined_markers["DCN_Transstrial_NEW"] = MarkerGroup(
    name="DCN Transstrial NEW",
    markers=[
        Marker(
            name="caudalborder",
            short_name="CB",
            coordinates=[(-120e-6, -200e-6)],
            group="border",
            order=0,
            connections=(None, "caudalborder"),
        ),
        Marker(
            name="caudalsurface",
            short_name="CS",
            coordinates=[(75e-6, -75e-6)],
            group="surface",
            order=1,
            connections=("caudalborder", "caudalsurface"),
        ),
        Marker(
            name="surface",
            short_name="S",
            coordinates=[(100e-6, 0)],
            group="surface",
            order=2,
            connections=("caudalsurface", "surface"),
        ),
        Marker(
            name="rostralsurface",
            short_name="RS",
            coordinates=[(75e-6, 175e-6)],
            group="surface",
            order=3,
            connections=("surface", "rostralsurface"),
        ),
        Marker(
            name="rostralborder",
            short_name="RB",
            coordinates=[(-170e-6, 250e-6)],
            group="border",
            order=4,
            connections=("rostralsurface", "rostralborder"),
            final = "rostralborder"
        ),
        Marker(
            name="medialboundary",
            short_name="MB",
            coordinates=[(-150e-6, 0)],
            group="interior",
            order=5,
            connections=("rostralborder", "medialboundary"),
        ),
        Marker(
            name="AN_Notch1",
            short_name="AN1",
            coordinates=[(-125e-6, -50e-6)],
            group="interior",
            order=6,
            connections=("medialboundary", "AN_Notch1"),
        ),
        Marker(
            name="AN_Notch2",
            short_name="AN2",
            coordinates=[(-125e-6, -175e-6)],
            group="interior",
            order=7,
            connections=("AN_Notch1", "AN_Notch2"),
            final = "caudalborder"
        ),
        Marker(
            name="AN",
            short_name="AN",
            coordinates=[(-100e-6, -125e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)

defined_markers["DCN Parasagittal"] = MarkerGroup(
    name="DCN Parasagittal",
    markers=[
        Marker(
            name="caudalsurface",
            short_name="CS",
            coordinates=[(250e-6, 0e-6)],
            group="surface",
            connections=(None, "caudalsurface"),
        ),
        Marker(
            name="dorsalsurface",
            short_name="DS",
            coordinates=[(150e-6, 300e-6)],
            group="surface",
            connections=("caudalsurface", "dorsalsurface"),
        ),
        Marker(
            name="rostralsurface",
            short_name="RS",
            coordinates=[(-200e-6, 150e-6)],
            group="surface",
            connections=("dorsalsurface", "rostralsurface"),
        ),
        Marker(
            name="rostralborder",
            short_name="RB",
            coordinates=[(-250e-6, -100e-6)],
            group="surface",
            connections=("rostralsurface", "rostralborder"),
        ),
        Marker(
            name="medialboundary",
            short_name="MB",
            coordinates=[(-100e-6, -150e-6)],
            group="surface",
            connections=("rostralborder", "medialboundary"),
        ),
        Marker(
            name="caudalborder",
            short_name="CB",
            coordinates=[(120e-6, -200e-6)],
            group="surface",
            connections=("medialboundary", "caudalborder"),
            final="caudalsurface"
        ),
        Marker(
            name="AN",
            short_name="AN",
            coordinates=[(0e-6, -150e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)

defined_markers["DCN Coronal"] = MarkerGroup(
    name="DCN Coronal",
    markers=[
        Marker(
            name="dorsal_border",
            short_name="D",
            coordinates=[(-100e-6, 500e-6)],
            group="border",
            connections=(None, "dorsal_border"),
        ),
        Marker(
            name="medial_dorsal",
            short_name="MD",
            coordinates=[(-50e-6, 250e-6)],
            group="interior",
            connections=("dorsal_border", "medial_dorsal"),
        ),
        Marker(
            name="medial_ventral",
            short_name="MV",
            coordinates=[(-50e-6, -250e-6)],
            group="interior",
            connections=("medial_dorsal", "medial_ventral"),
        ),
        Marker(
            name="ventral_border",
            short_name="VB",
            coordinates=[(-100e-6, -500e-6)],
            group="border",
            connections=("medial_ventral", "ventral_border"),
            final="ventral_border",
        ),
        Marker(
            name="lateral1",
            short_name="LV1",
            coordinates=[(100e-6, -400e-6)],
            group="surface",
            connections=("ventral_border", "lateral1"),
        ),
        Marker(
            name="lateral2",
            short_name="LV2",
            coordinates=[(250e-6, -200e-6)],
            group="surface",
            connections=("lateral1", "lateral2"),
        ),
        Marker(
            name="lateral3",
            short_name="LD3",
            coordinates=[(350e-6, 50e-6)],
            group="surface",
            connections=("lateral2", "lateral3"),
        ),
        Marker(
            name="lateral4",
            short_name="LD4",
            coordinates=[(250e-6, 350e-6)],
            group="surface",
            connections=("lateral3", "lateral4"),
            final="dorsal_border",
        ),
        Marker(
            name="AN",
            short_name="AN",
            coordinates=[(0e-6, -150e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)

defined_markers["CN Parasagittal"] = MarkerGroup(
    name="CN Parasagittal",
    markers=[
        Marker(
            name="rostralventralAN",
            short_name="RVAN",
            coordinates=[(-175e-6, -300e-6)],
            group="border",
            connections=(None, "rostralventralAN"),
        ),
        Marker(
            name="AVCNrostral",
            short_name="AVCNR",
            coordinates=[(-300e-6, 0e-6)],
            group="surface",
            connections=("rostralventralAN", "AVCNrostral"),
        ),
        Marker(
            name="AVCN-DCNborder",
            short_name="AVCN-DCN",
            coordinates=[(-250e-6, 100e-6)],
            group="border",
            connections=("AVCNrostral", "AVCN-DCNborder"),
            final ="AVCN-DCNborder"
        ),
        Marker(
            name="DCN-V1",
            short_name="DC-VC1",
            coordinates=[(0e-6, 350e-6)],
            group="surface",
            connections=("AVCN-DCNborder", "DCN-V1"),
        ),
        Marker(
            name="DCN-V2",
            short_name="DC-VC2",
            coordinates=[(100e-6, 150e-6)],
            group="surface",
            connections=("DCN-V1", "DCN-V2"),
            # final="PVCN-DCNborder"
        ),
        Marker(
            name="PVCN-DCNborder",
            short_name="PV-DC",
            coordinates=[(200e-6, -100e-6)],
            group="border",
            connections=("DCN-V2", "PVCN-DCNborder"),
            final="PVCN-DCNborder"

        ),
        Marker(
            name="PVCN-midcaudal",
            short_name="PV-MC",
            coordinates=[(150e-6, -250e-6)],
            group="surface",
            connections=("PVCN-DCNborder", "PVCN-midcaudal"),
        ),
        Marker(
            name="PVCNcaudal",
            short_name="PV-C",
            coordinates=[(0e-6, -300e-6)],
            group="surface",
            connections=("PVCN-midcaudal", "PVCNcaudal"),
        ),
        Marker(
            name="caudalventralAN",
            short_name="CV-AV",
            coordinates=[(-100e-6, -300e-6)],
            group="border",
            connections=("PVCNcaudal", "caudalventralAN"),
            final="PVCNcaudal"
        ),
        Marker(
            name="dorsalAN",
            short_name="D_AN",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="AN",
            short_name="AN",
            coordinates=[(0e-6, -150e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="VN",
            short_name="VN",
            coordinates=[(-100e-6, -200e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)

defined_markers["VCN Horizontal"] = MarkerGroup(
    name="VCN Horizontal",
    markers=[
        Marker(
            name="rostralborder",
            short_name="RB",
            coordinates=[(-150e-6, 100e-6)],
            group="border",
            connections=(None, "rostralborder"),
        ),
        Marker(
            name="rostralsurface",
            short_name="RS",
            coordinates=[(-75e-6, 175e-6)],
            group="surface",
            connections=("rostralborder", "rostralsurface"),
        ),
        Marker(
            name="surface",
            short_name="S",
            coordinates=[(0, 0e-6)],
            group="surface",
            connections=("rostralsurface", "surface"),
        ),
        Marker(
            name="caudalsurface",
            short_name="CS",
            coordinates=[(0e-6, -175e-6)],
            group="surface",
            connections=("surface", "caudalsurface"),
        ),
        Marker(
            name="caudalborder",
            short_name="CB",
            coordinates=[(-150e-6, -200e-6)],
            group="border",
            connections=("caudalsurface", "caudalborder"),
            final="caudalborder"
        ),
        Marker(
            name="medialboundary",
            short_name="MB",
            coordinates=[(-150e-6, 0)],
            group="interior",
            connections=("caudalborder", "medialboundary"),
            final="rostralborder"
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)

defined_markers["VCN Parasagittal"] = MarkerGroup(
    name="VCN Parasagittal",
    markers=[
        Marker(
            name="rostralventralAN",
            short_name="RVAN",
            coordinates=[(-300e-6, -75e-6)],
            group="border",
            connections=(None, "rostralventralAN"),
        ),
        Marker(
            name="AVCNrostral",
            short_name="AVCNR",
            coordinates=[(-300e-6, 100e-6)],
            group="surface",
            connections=("rostralventralAN", "AVCNrostral"),
            final="AVCNrostral"
        ),
        Marker(
            name="AVCN-DCNborder",
            short_name="AVCN-DCN",
            coordinates=[(-100e-6, 300e-6)],
            group="border",
            connections=("AVCNrostral", "AVCN-DCNborder"),
            # final="PVCN-DCNborder"
        ),
        Marker(
            name="PVCN-DCNborder",
            short_name="PV-DC",
            coordinates=[(200e-6, -100e-6)],
            group="border",
            connections=("AVCN-DCNborder", "PVCN-DCNborder"),
            final="PVCN-DCNborder"
        ),
        Marker(
            name="PVCN-midcaudal",
            short_name="PV-MC",
            coordinates=[(150e-6, -250e-6)],
            group="surface",
            connections=("PVCN-DCNborder", "PVCN-midcaudal"),
        ),
        Marker(
            name="PVCNcaudal",
            short_name="PV-C",
            coordinates=[(0e-6, -300e-6)],
            group="surface",
            connections=("PVCN-midcaudal", "PVCNcaudal"),
        ),
        Marker(
            name="caudalventralAN",
            short_name="CV-AV",
            coordinates=[(-100e-6, -300e-6)],
            group="border",
            connections=("PVCNcaudal", "caudalventralAN"),
            final="rostralventralAN"
        ),
        Marker(
            name="dorsalAN",
            short_name="D_AN",
            coordinates=[(-200e-6, 200e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="AN",
            short_name="AN",
            coordinates=[(0e-6, -150e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="VN",
            short_name="VN",
            coordinates=[(-100e-6, -200e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)

defined_markers["VCN Coronal"] = MarkerGroup(
    name="VCN Coronal",
    markers=[
        Marker(
            name="dorsomedial",
            short_name="DM",
            coordinates=[(-100e-6, 500e-6)],
            group="border",
            connections=(None, "dorsomedial"),
        ),
        Marker(
            name="medial",
            short_name="M",
            coordinates=[(-100e-6, 0e-6)],
            group="border",
            connections=("dorsomedial", "medial"),
        ),
        Marker(
            name="ventromedial",
            short_name="VM",
            coordinates=[(-100e-6, -500e-6)],
            group="border",
            connections=("medial", "ventromedial"),
            final="ventromedial"
        ),
        Marker(
            name="lateral1",
            short_name="L1",
            coordinates=[(-50e-6, -400e-6)],
            group="surface",
            connections=("ventromedial", "lateral1"),
        ),
        Marker(
            name="lateral2",
            short_name="L2",
            coordinates=[(250e-6, -200e-6)],
            group="surface",
            connections=("lateral1", "lateral2"),
        ),
        Marker(
            name="lateral3",
            short_name="L3",
            coordinates=[(350e-6, 50e-6)],
            group="surface",
            connections=("lateral2", "lateral3"),
        ),
        Marker(
            name="lateral4",
            short_name="L4",
            coordinates=[(250e-6, 350e-6)],
            group="surface",
            connections=("lateral3", "lateral4"),
            final ="dorsomedial"
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)


defined_markers["Cortex Horizontal"] = MarkerGroup(
    name="Cortex Horizontal",
    markers=[
        Marker(
            name="lateralrostral",
            short_name="LR",
            coordinates=[(1000e-6, 200e-6)],
            group="border",
            connections=(None, "lateralrostral"),
        ),
        Marker(
            name="surface",
            short_name="S",
            coordinates=[(1000e-6, 0)],
            group="surface",
            connections=("lateralrostral", "surface"),
        ),
        Marker(
            name="lateralcaudal",
            short_name="LC",
            coordinates=[(1000e-6, -200e-6)],
            group="border",
            connections=("surface", "lateralcaudal"),
            final="lateralcaudal"
        ),
        Marker(
            name="medialcaudal",
            short_name="MC",
            coordinates=[(0e-6, -200e-6)],
            group="border",
            connections=("lateralcaudal", "medialcaudal"),
        ),
        Marker(
            name="medialrostral",
            short_name="MR",
            coordinates=[(0e-6, 200e-6)],
            group="border",
            connections=("medialcaudal", "medialrostral"),
            final="lateralrostral"
        ),
        Marker(
            name="injectionsite",
            short_name="Inj",
            coordinates=[(0, 0)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="hpcanteriorpole",
            short_name="HPC",
            coordinates=[(200e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)

defined_markers["Cortex Coronal"] = MarkerGroup(
    name="Cortex Coronal",
    markers=[
        Marker(
            name="lateraldorsal",
            short_name="LD",
            coordinates=[(1000e-6, 200e-6)],
            group="border",
            connections=(None, "lateraldorsal"),
        ),
        Marker(
            name="surface",
            short_name="S",
            coordinates=[(1000e-6, 0)],
            group="surface",
            connections=("lateraldorsal", "surface"),
        ),
        Marker(
            name="lateralventral",
            short_name="LV",
            coordinates=[(1000e-6, -200e-6)],
            group="border",
            connections=("surface", "lateralventral"),
            final="lateralventral"
        ),
        Marker(
            name="medialventral",
            short_name="MV",
            coordinates=[(-150e-6, -200e-6)],
            group="border",
            connections=("lateralventral", "medialventral"),
        ),
        Marker(
            name="medialdorsal",
            short_name="MD",
            coordinates=[(-150e-6, 200e-6)],
            group="border",
            connections=("medialventral", "medialdorsal"),
            final="lateraldorsal"
        ),
        Marker(
            name="injectionsite",
            short_name="Inj",
            coordinates=[(0, 0)],
            group="point",
            connections=(None, None),
            final="lateraldorsal"
        ),
        Marker(
            name="surface_reference",
            short_name="SR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
        Marker(
            name="deep_reference",
            short_name="IR",
            coordinates=[(0e-6, 0e-6)],
            group="point",
            connections=(None, None),
        ),
    ],
)

defined_markers["soma"] = MarkerGroup(
    name="soma",
    markers=[
        Marker(
            name="soma",
            short_name="Soma",
            coordinates=[(0, 0)],
            group="point",
            connections=(None, None),
            color="y",
            symbol="*",
            alpha=1.0,
        ),
    ],
)


def identify_marker(marker):

    mark_keys = set(sorted(list(marker.keys())))  # keys in markers in the current mosaic
    print("=" * 40)
    print("marker: ", marker)
    print("identify_marker: Mark keys: ", mark_keys)
    print("=" * 40)
    if marker["name"] == "soma":
        CP.cprint("g", "Identified marker set as: soma")
        return "soma", defined_markers["soma"]
    for (
        k,
        def_marker,
    ) in defined_markers.items():  # check the dictionary of all of the pre-defined markers
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


def build_connection_lists(marker_group):
    """Build connection lists from marker group
    Markers with connection values of None are special and designate starts or ends of connection lists
    If a marker has a "final" connection, that indicates the final connection to be made from that marker
    to the designated final marker in that list

    Parameters
    ----------
    marker_group : MarkerGroup
        the marker group from which to build connection lists

    Returns
    -------
    connection_lists : list of tuples
        list of connections between markers that should be connected
    """
    connection_lists = []
    connection_list = []
    in_group = False
    for imark, marker in enumerate(marker_group):
        if marker.group == "point":
            continue  # skip point markers
        if marker.connections[0] is not None and marker.connections[1] is not None and marker.final is None:
            connection_list.append((marker.connections[0], marker.connections[1]))
            if len(connection_list) > 1 and marker.group == 'border':
                # end of that connection list, so start a new connection list
                connection_lists.append(connection_list)
                connection_list = []
        if marker.connections[0] is None and marker.connections[1] is not None:
            if len(connection_list) > 0:  # start a new connection list
                connection_lists.append(connection_list)
                connection_list = []
        if marker.connections[0] is not None and marker.connections[1] is None:
            connection_list.append((marker.connections[0], marker.connections[1]))
            if len(connection_list) > 0:
                connection_lists.append(connection_list)
                connection_list = []
        if marker.final is not None:
            connection_list.append((marker.connections[0], marker.connections[1]))
            connection_list.append((marker.connections[1], marker.final))
            if len(connection_list) > 0:  # add the final list
                connection_lists.append(connection_list)
                connection_list = []

    return connection_lists

def find_marker_by_name(marker_group, name: str):
    """find marker in marker group by name

    Parameters
    ----------
    marker_group : MarkerGroup
        the marker group to search
    name : string
        the name of the marker to find

    Returns
    -------
    marker : Marker
        the found marker
    """
    for marker in marker_group:
        if marker.name == name:
            return marker
    return None

def plot_lines(coord_xy, coord_labels, ax, i, colors):
    ax.plot(coord_xy[0], coord_xy[1], color=colors[i % len(colors)], linestyle="-")
    ax.quiver(
        coord_xy[0, :-1],
        coord_xy[1, :-1],
        coord_xy[0, 1:] - coord_xy[0, :-1],
        coord_xy[1, 1:] - coord_xy[1, :-1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color=colors[i % len(colors)],
        width=0.005,
        headwidth=5,
        headlength=7,
    )
    for i in range(len(coord_xy[0])):
        ax.text(coord_xy[0, i], coord_xy[1, i], coord_labels[i])
        ax.plot(coord_xy[0, i], coord_xy[1, i], marker=marker_group[i].symbol, color=marker_group[i].color, markersize=marker_group[i].markersize)

def plot_splines(coord_xy, coord_labels, marker_group, ax:mpl.axes = None, i: int=0, colors:list = None):

    # cs_x = CubicSpline(
    #     np.arange(coord_xy.shape[1]), coord_xy[0, :], bc_type="natural"
    # )
    # cs_y = CubicSpline(
    #     np.arange(coord_xy.shape[1]), coord_xy[1, :], bc_type="natural"
    # )
    cs_x, cs_y = get_markers.compute_splines(coord_xy, npoints=100, remove_ends=False)
    xnew = np.linspace(0, coord_xy.shape[1] - 1, 100)
    ax.plot(cs_x, cs_y, color=colors[i % len(colors)], linestyle="-")
    for i in range(len(coord_xy[0])):
        ax.text(coord_xy[0, i], coord_xy[1, i], coord_labels[i])
        ax.plot(coord_xy[0, i], coord_xy[1, i], marker=marker_group[i].symbol, color=marker_group[i].color, markersize=marker_group[i].markersize)


def plot_points(marker_group, ax, use_short_names: bool = True):
    # now plot the reference points:
    for marker in marker_group:
        if marker.group == "point":
            xs = [coord[0] * 1e6 for coord in marker.coordinates][0]
            ys = [coord[1] * 1e6 for coord in marker.coordinates][0]
            if use_short_names:
                display_name = marker.short_name
            else:
                display_name = marker.name
            ax.plot(xs, ys, marker=marker.symbol, color=marker.color, markersize = marker.markersize, alpha=marker.alpha)
            ax.text(xs, ys, display_name)

def draw_mosaic_outline(marker_group, ax, use_short_names: bool = True, splines:bool=False):
    """draw_stick_outline onto selected axis

    Parameters
    ----------
    marker_group : string
        the name of the marker group (see defined_markers above)
    ax : matplotlib axis
        the axis on which to draw the outline
    """


    connection_lists = build_connection_lists(marker_group)
    # for i, cl in enumerate(connection_lists):
    #     print(f"connection_list {i} ", cl)

    #for each connection list, get coordinates (in the list order) and plot them
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    for i, connection_list in enumerate(connection_lists):
        coord_list = []
        coord_labels = []
        # print("connection_list: ", connection_list)
        for marker in connection_list:
            # print("connections: ", marker)
            marker1 = find_marker_by_name(marker_group, marker[0])
            marker2 = find_marker_by_name(marker_group, marker[1])
            if use_short_names:
                display_name = marker1.short_name
            else:
                display_name = marker1.name
            xs = [coord[0] * 1e6 for coord in marker1.coordinates][0]
            ys = [coord[1] * 1e6 for coord in marker1.coordinates][0]
            coord_list.append([xs, ys])
            coord_labels.append(display_name)
            if marker1.final is not None:
                marker2 = find_marker_by_name(marker_group, marker1.final)
                # print("Finalizing: Marker2: ", marker2.name)
                xs = [coord[0] * 1e6 for coord in marker2.coordinates][0]
                ys = [coord[1] * 1e6 for coord in marker2.coordinates][0]
                coord_list.append([xs, ys])
                coord_labels.append(marker2.short_name)
        
        coord_xy = np.array(coord_list).T
        if ax is not None and not splines:
            plot_lines(coord_xy, coord_labels, ax=ax, i=i, colors=colors)
        if ax is not None and splines:
            plot_splines(coord_xy, coord_labels, marker_group, ax=ax, i=i, colors=colors)
    if ax is not None:
        plot_points(marker_group, ax, use_short_names)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.axis("equal")


if __name__ == "__main__":
    nplots = len(defined_markers)
    ncols = 4
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = mpl.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()
    for ax in axes[nplots:]:
        ax.axis("off")
    # print("# defined markers: ", len(defined_markers))
    for i, (group_name, marker_group) in enumerate(defined_markers.items()):
        ax = axes[i]
        ax.set_title(group_name)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.axis("equal")
        if marker_group.markers[0].name == "soma":
            continue
        # ax.legend()
        # if group_name in ["Cortex Horizontal", "Cortex Coronal"]:
        draw_mosaic_outline(marker_group.markers, ax=ax, splines=True)

        # identify_marker(marker_group.markers[i].to_dict())
    mpl.tight_layout()
    mpl.show()
