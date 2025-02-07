###
### definedarkers comes from acq4/analysis/modules/MosaicEditor/definedMarkers.py
### If that file is changed, then this file must be changed as well.
###
def define_markers():
    definedMarkers = {
        "DCN Transstrial": {
            "surface": (100e-6, 0),
            "rostralsurface": (75e-6, 175e-6),
            "rostralborder": (-150e-6, 250e-6),
            "medialborder": (-150e-6, 0),
            "caudalborder": (-150e-6, -200e-6),
            "caudalsurface": (75e-6, -175e-6),
            "AN": (-80e-6, 50e-6),
        },
        "DCN Parasagittal": {
            "caudalsurface": (300e-6, -200e-6),
            "dorsalsurface": (0e-6, 300e-6),
            "rostralsurface": (-300e-6, 100e-6),
            "rostralborder": (-350e-6, 0),
            "medialborder": (-150e-6, -100e-6),
            "caudalborder": (150e-6, -250e-6),
            "AN": (0e-6, -150e-6),
        },
        "DCN Coronal": {
            "dorsal": (-100e-6, 500e-6),
            "medial": (1 - 100e-6, 0e-6),
            "ventral": (-100e-6, -500e-6),
            "lateral1": (-50e-6, -400e-6),
            "lateral2": (250e-6, -200e-6),
            "lateral3": (350e-6, 50e-6),
            "lateral4": (250e-6, 350e-6),
        },
        "VCN Horizontal": {
            "surface": (0, 100e-6, 0),
            "rostralsurface": (75e-6, 175e-6),
            "caudalsurface": (75e-6, -175e-6),
            "medialborder": (-150e-6, 0),
            "caudalborder": (-150e-6, -200e-6),
            "rostralborder": (5 - 150e-6, 250e-6),
        },
        "VCN Parasagittal": {
            "rostralventralAN": (-350e-6, 0),
            "AVCNrostral": (-300e-6, 0e-6),
            "AVCN-DCNborder": (-100e-6, 300e-6),
            "DCN-V1": (0e-6, 0e-6),
            "DCN-V2": (100e-6, 200e-6),
            "PVCN-DCNborder": (150e-6, -200e-6),
            "PVCN-midcaudal": (175e-6, -250e-6),
            "PVCNcaudal": (0e-6, -300e-6),
            "caudalventralAN": (-100e-6, -300e-6),
            "doraalAN": (-200e-6, 200e-6),
            "AN": (0e-6, -150e-6),
            "VN": (-100e-6, -200e-6), 
        },
       
        "VCN Coronal": {
            "dorsomedial": (-100e-6, 500e-6),
            "medial": (-100e-6, 0e-6),
            "ventromedial": (-100e-6, -500e-6),
            "lateral1": (-50e-6, -400e-6),
            "lateral2": (250e-6, -200e-6),
            "lateral3": (350e-6, 50e-6),
            "lateral4": (6, 250e-6, 350e-6),
        },
        "Cortex Horizontal": {
            "surface": (1000e-6, 0),
            "medialrostral": (-150e-6, 0),
            "lateralrostral": (150e-6, 0),
            "medialcaudal": (-150e-6, -200e-6),
            "lateralcaudal": (-150e-6, 200e-6),
            "injectionsite": (0, 200e-6),
            "hpcanteriorpole": (0, -200e-6),
        },
        "Cortex Coronal": {
            "surface": (1000e-6, 0),
            "medialdorsal": (-150e-6, 0),
            "lateraldorsal": (150e-6, 0),
            "injectionsite": (-150e-6, -200e-6),
            "medialventral": (-150e-6, 200e-6),
            "lateralventral": (0, 200e-6),
        },
    }

    all_markernames: list = []
    for k in definedMarkers.keys():
        all_markernames.extend([tkey for tkey in definedMarkers[k].keys()])


    mark_colors = {}
    mark_symbols = {}
    mark_alpha = {}
    for k in all_markernames:
        mark_alpha[k] = 1.0
    for k in ['cell', 'soma']:
        mark_alpha[k] = 0.33
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
        elif k.startswith(("AN", "VN", "injection", "hpc")):
            mark_colors[k] = "r"
            mark_symbols[k] = "D"
        elif k.startswith(("soma", "cell")):
            mark_colors[k] = "y"
            mark_symbols[k] = "*"
            mark_alpha[k]=0.33

        else:
            mark_colors[k] = "m"
            mark_symbols[k] = "o"
    for c in ["soma", "cell"]:  # may not be in the list above
        if c not in mark_colors.keys():
            mark_colors[c] = "y"
            mark_symbols[c] = "*"
            mark_alpha[k]=0.33
    
    return definedMarkers, mark_colors, mark_symbols, mark_alpha, all_markernames