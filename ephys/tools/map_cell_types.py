""" Map cell type names from the variations that might occur in the data
to a standard set of cell types. 

"""

import re
from pylibrary.tools.cprint import cprint as CP

re_bu = re.compile(r"^bushy[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_globular = re.compile(r"^globular[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_spherical = re.compile(r"^spherical[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_tstellate = re.compile(r"^t[-_ ]*stellate", re.IGNORECASE)
re_dstellate = re.compile(r"^d[-_ ]*stellate", re.IGNORECASE)
re_lstellate = re.compile(r"^l[-_ ]*stellate", re.IGNORECASE)
re_stellate = re.compile(r"^stellate[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_octopus = re.compile(r"^octopus[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_pyramidal = re.compile(r"^pyramidal[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_granule = re.compile(r"^granule[\s\w_\?]*(?P<word1>\w+)*\s*(?P<word2>\w+)*", re.IGNORECASE)
re_basket = re.compile(r"^basket[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_golgi = re.compile(r"^golgi[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_horizontal = re.compile(r"^horizontal[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_fusiform = re.compile(r"^fusiform[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_vertical = re.compile(r"^vertical[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_typeB = re.compile(r"^type[ _-]*B[ _]*(cell)*", re.IGNORECASE)
re_chestnut = re.compile(r"^chestnut[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)

re_cartwheel = re.compile(r"^cartwheel[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_unipolar_brush = re.compile(r"^UBC$|^unipolar[ _-]*brush[ _-]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_multipolar = re.compile(r"^multipolar[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_giant = re.compile(r"^giant[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_giant_maybe = re.compile(r"^giant[ _]{1}maybe", re.IGNORECASE)
re_tuberculoventral = re.compile(r"^(?=tv|tuberculoventral|vertical)[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_unknown = re.compile(r"^unknown[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_nomorphology = re.compile(r"^no[ _]*morphology[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)
re_glia = re.compile(r"^glia[l]*[ _]*(?P<word1>\w+)*\s*", re.IGNORECASE)

all_types = [
    [re_bu, "bushy"],
    [re_globular, "globular bushy"],
    [re_spherical, "spherical bushy"],
    [re_tstellate, "t-stellate"],
    [re_dstellate, "d-stellate"],
    [re_lstellate, "l-stellate"],
    [re_stellate, "stellate"],
    [re_octopus, "octopus"],
    [re_pyramidal, "pyramidal"],
    [re_fusiform, "pyramidal"],
    [re_granule, "granule"],
    [re_basket, "basket"],
    [re_golgi, "golgi"],
    [re_horizontal, "horizontal"],
    [re_typeB, "type-B"],
    [re_chestnut, "chestnut"],
    [re_cartwheel, "cartwheel"],
    [re_unipolar_brush, "unipolar-brush-cell"],
    [re_multipolar, "multipolar"],
    [re_giant, "giant"],
    [re_giant_maybe, "giant_maybe"],
    [re_tuberculoventral, "tuberculoventral"],
    [re_unknown, "unknown"],
    [re_nomorphology, "no morphology"],
    [re_glia, "glial"],
]


def map_cell_type(variant, retval = None):
    any = False
    matching = False
    for exp in all_types:
        m = exp[0].match(variant)
        if m is not None:
            matching = True
            return exp[1]
    if not matching:
        return retval # 


def test():
    variants = {
        "bushy": [
            "bushy",
            "Bushy",
            "bushy cell",
            "Bushy cell",
            "Bushy_Cell",
            "bushy_Cell",
            "Bushy_Cell",
            "bushy_Cell",
        ],
        "globular bushy": [
            "globular",
            "Globular",
            "globular cell",
            "Globular cell",
            "globular bushy",
            "Globular bushy",
            "globular bushy cell",
            "Globular bushy cell",
        ],
        "spherical bushy": [
            "spherical",
            "Spherical",
            "spherical cell",
            "Spherical cell",
            "spherical bushy",
            "Spherical bushy",
            "spherical bushy cell",
            "Spherical bushy cell",
        ],
        "t-stellate": [
            "T-Stellate",
            "tstellate",
            "t-stellate",
            "t stellate",
            "t stellate cell",
            "T stellate cell",
            "T stellate",
        ],
        "d-stellate": ["d-stellate", "d stellate", "dstellate", "D-stellate"],
        "l-stellate": ["l-stellate", "lstellate", "L-Stellate"],
        "octopus": [
            "Octopus",
            "octopus cell",
            "Octopus cell",
        ],
        "pyramidal": [
            "Pyramidal",
            "pyramidal cell",
            "Pyramidal cell",
            "Fusiform",
            "fusiform cell",
            "Fusiform cell",
        ],
        "fusiform": [
            "Fusiform",
            "fusiform cell",
            "Fusiform cell",
        ],
        "stellate": ["Stellate", "stellate cell", "Stellate cell"],
        "granule": [
            "Granule",
            "granule cell",
            "Granule cell",
            "granule? tiny"
        ],
        "basket": ["Basket", "basket cell", "Basket cell"],
        "golgi": ["Golgi", "golgi cell", "Golgi cell"],
        "horizontal": ["Horizontal", "horizontal cell", "Horizontal cell"],
        #   "fusiform": ["Fusiform", "fusiform cell", "Fusiform cell"],
        "tuberculoventral": [
            "Tuberculoventral",
            "tuberculoventral cell",
            "Tuberculoventral cell",
            "TV",
            "TV cell",
            "Vertical",
            "vertical cell",
            "Vertical cell",
        ],
        "cartwheel": ["Cartwheel", "cartwheel cell", "Cartwheel cell"],
        "unipolar-brush-cell": ["Unipolar brush", "unipolar brush cell", "Unipolar brush cell", "ubc", "UBC"],
        "multipolar": ["Multipolar", "multipolar cell", "Multipolar cell"],
        "giant": ["Giant", "giant cell", "Giant cell", "GIANT"],
        "giant_maybe": ["giant_maybe", "Giant_maybe", "Giant_Maybe"],
        "failed": ["Failed", "failed cell", "Failed cell"],
        "unknown": ["Unknown", "unknown cell", "Unknown cell"],
        "no morphology": ["No morphology", "no morphology cell", "No morphology cell"],
        "glia": ["Glial", "glial cell", "Glial cell", "Glia", "glial"],
        "horizontal": ["Horizontal bipolar", "horizontal bipolar cell", "Horizontal bipolar cell",  "horizontal"],
        "chestnut": ["Chestnut", "chestnut cell", "Chestnut cell"],
        "type-B": ["Type B", "type-B cell", "Type-B cell", "type_B", "type-b", "type B cell"],
        "dabney": ["Dabney", "dabney cell", "Dabney cell"],
    }
    for ct in variants.keys():
        matching = False
        for v in variants[ct]:
            result = map_cell_type(v, retval=None)
            if result is not None:
                CP("g", f"matched {v}, => { map_cell_type(v)}")
                matching = True
            else:
                print("    No match for", v)
        if not matching:
            CP("r", f"NO MATCH for cell type {ct}")
        print("\n")
    # print("fusiform: ")
    # print(map_cell_type("fusiform"))
    # print("glial cell: ", end="")
    # print(map_cell_type("glial cell"))
    # print("map cell type: glial", end=": ")
    # print(map_cell_type("glial"))
    # print("UBC: ")
    # print(map_cell_type("UBC"))
    # print(map_cell_type("unipolar brush"))
    # print(map_cell_type("granule? tiny"))

if __name__ == "__main__":
    test()
