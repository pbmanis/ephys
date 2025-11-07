""" Map cell type names from the variations that might occur in the data
to a standard set of cell types. 

"""

import re

re_bu = re.compile(r"^bushy[ _]*[cell]*", re.IGNORECASE)
re_globular = re.compile(r"^globular[ _]*[cell]*", re.IGNORECASE)
re_spherical = re.compile(r"^spherical[ _]*[cell]*", re.IGNORECASE)
re_tstellate = re.compile(r"^t[-_ ]*stellate", re.IGNORECASE)
re_dstellate = re.compile(r"^d[-_ ]*stellate", re.IGNORECASE)
re_lstellate = re.compile(r"^l[-_ ]*stellate", re.IGNORECASE)
re_stellate = re.compile(r"^stellate[ _]*[cell]*", re.IGNORECASE)
re_octopus = re.compile(r"^octopus[ _]*[cell]*", re.IGNORECASE)
re_pyramidal = re.compile(r"^pyramidal[ _]*[cell]*", re.IGNORECASE)
re_granule = re.compile(r"^granule[ _]*[cell]*", re.IGNORECASE)
re_basket = re.compile(r"^basket[ _]*[cell]*", re.IGNORECASE)
re_golgi = re.compile(r"^golgi[ _]*[cell]*", re.IGNORECASE)
re_horizontal = re.compile(r"^horizontal[ _]*[cell]*", re.IGNORECASE)
re_fusiform = re.compile(r"^fusiform[ _]*[cell]*", re.IGNORECASE)
re_vertical = re.compile(r"^vertical[ _]*[cell]*", re.IGNORECASE)
re_typeB = re.compile(r"^type[ _-]*B[ _]*[cell]*", re.IGNORECASE)
re_chestnut = re.compile(r"^chestnut[ _]*[cell]*", re.IGNORECASE)

re_cartwheel = re.compile(r"^cartwheel[ _]*[cell]*", re.IGNORECASE)
re_unipolar_brush = re.compile(r"^UBC$|^unipolar[ _]*brush[ _]*[cell]*", re.IGNORECASE)
re_multipolar = re.compile(r"^multipolar[ _]*[cell]*", re.IGNORECASE)
re_giant = re.compile(r"^giant[ _]*[cell]*", re.IGNORECASE)
re_giant_maybe = re.compile(r"^giant[ _]{1}maybe", re.IGNORECASE)
re_tuberculoventral = re.compile(r"^(?=tv|tuberculoventral|vertical)[ _]*[cell]*", re.IGNORECASE)
re_unknown = re.compile(r"^unknown[ _]*[cell]*", re.IGNORECASE)
re_nomorphology = re.compile(r"^no[ _]*morphology[ _]*[cell]*", re.IGNORECASE)
re_glia = re.compile(r"^glia[ _]*[cell]*", re.IGNORECASE)

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
    [re_typeB, "type B"],
    [re_chestnut, "chestnut"],
    [re_cartwheel, "cartwheel"],
    [re_unipolar_brush, "unipolar brush"],
    [re_multipolar, "multipolar"],
    [re_giant, "giant"],
    [re_giant_maybe, "giant_maybe"],
    [re_tuberculoventral, "tuberculoventral"],
    [re_unknown, "unknown"],
    [re_nomorphology, "no morphology"],
    [re_glia, "glial"],
]


def map_cell_type(variant):
    any = False
    for exp in all_types:
        m = exp[0].match(variant)
        # print("  m=", m)
        if m is not None:
            # r = exp[0].sub(variant, exp[1])
            # # print("     r=", r)
            # print("m is not none, returning: ", m.group(), exp[1])
            return exp[1]
    return 'None'


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
        "unipolar brush": ["Unipolar brush", "unipolar brush cell", "Unipolar brush cell", "ubc", "UBC"],
        "multipolar": ["Multipolar", "multipolar cell", "Multipolar cell"],
        "giant": ["Giant", "giant cell", "Giant cell", "GIANT"],
        "giant_maybe": ["giant_maybe", "Giant_maybe", "Giant_Maybe"],
        "failed": ["Failed", "failed cell", "Failed cell"],
        "unknown": ["Unknown", "unknown cell", "Unknown cell"],
        "no morphology": ["No morphology", "no morphology cell", "No morphology cell"],
        "glial": ["Glial", "glial cell", "Glial cell", "Glia"],
        "horizontal": ["Horizontal bipolar", "horizontal bipolar cell", "Horizontal bipolar cell",  "horizontal"],
        "chestnut": ["Chestnut", "chestnut cell", "Chestnut cell"],
        "type B": ["Type B", "type-B cell", "Type-B cell", "type_B", "type B cell"],
    }
    for ct in variants.keys():
        for v in variants[ct]:
            result = map_cell_type(v)
            if result is not None:
                print(v, "=>", map_cell_type(v))
            else:
                print("No match for", v)

        print("\n")
    print(map_cell_type("fusiform"))
    print(map_cell_type("glial cell"))
    print(map_cell_type("UBC"))
    print(map_cell_type("unipolar brush cell"))

if __name__ == "__main__":
    test()
