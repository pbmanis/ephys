import os
import platform
from pathlib import Path
import tomllib
import pprint
pp = pprint.PrettyPrinter(width=88, compact=False)

"""
Set up experiment paths for different computers for the map and IV analysis

This also sets up the excluded files (ones that are aborted, noisy or otherwise
not good)

"""


# Paths is a dict. Keys are computer names, values are paths to the top level
# data files


def get_computer(datapaths: dict):
    """Identify which computer we are running on
    and confirm that it is known to our "wheres_my_data.toml" file.

    Parameters
    ----------
    datapaths : dict
        the dictionary from "wheres_my_data.toml"

    Returns
    -------
    computer_name: str

    Raises
    ------
    ValueError
        if the computer is not in the list

    """    
    if os.name == "nt":
        computer_name = os.environ["COMPUTERNAME"]
    else:
        computer_name = platform.node()
        computer_name = computer_name.split(".")[0].title()
    if computer_name not in datapaths['computer'].keys():
        raise ValueError(
            "Computer name {0:s} is not in known systems to set data paths: {1:s}".format(
                computer_name, str(list(datapaths.keys()))
            )
        )
    return computer_name

def get_paths():
    """Get the paths associated with this computer. Reads the 
    "wheres_my_data.toml" file, identifies the computer, and returns
    the relevant paths.

    Parameters
    ----------
    None
    Returns
    -------
    baseDataDirectory and codeDirectory for the current computer
        , as strings
    """
    print(os.getcwd())
    datapaths = tomllib.load(open("wheres_my_data.toml", "r"))
    computer = get_computer(datapaths)
    codeDirectory = Path(datapaths['computer'][computer]['codeDirectory'])
    datasetDirectory = Path(datapaths['computer'][computer]['datasetDirectory'])
    for i, p in enumerate(datapaths['computer'][computer]['baseDataDirectory']):
        baseDataDirectory = Path(datapaths['computer'][computer]['baseDataDirectory'][i])
        if baseDataDirectory.is_dir():
            return baseDataDirectory, codeDirectory, datasetDirectory
    return None, codeDirectory, datasetDirectory

baseDataDirectory, codeDirectory, datasetDirectory = get_paths()


"""
Mean spont values for quantal events by cell type and temperature. This table is
only used if there is no spont data for a given cell (e.g., low spont rate).
These values are derived from all the mean spont amplitudes for each cell type,
by temperature.

"""

mean_spont_by_cell = {
    ("bushy", 34): -29.3,
    ("bushy", 25): -52.1,
    ("t-stellate", 34): -37.7,
    ("t-stellate", 25): -27.7,  # n = 1 for 25, 8 for 34
    ("d-stellate", 34): -23.0,
    ("d-stellate", 25): -23.0,  # no d stellates at 25; use 34 C
    ("pyramidal", 34): -17.5,
    ("pyramidal", 25): -14.7,
    ("tuberculoventral", 34): -22.9,
    ("tuberculoventral", 25): -15.6,
    ("cartwheel", 34): -21.7,
    ("cartwheel", 25): -16.0,
    ("giant", 34): -16.7,
    ("cartwheel", 25): -16.7,  # no giants at 25
}

############################################################################

"""
B : control A : NIHL, measured 14 d later, 106 or 109 dB SPL AA: NIHL, measured
14 d later, 115 dB SPL AAA: NIHL, measured ~ 3d later, 115 dB SPL
"""


coding_NF107_nihl = {
    "2018.06.19": ["Animal1", "A", "ok"],
    "2018.06.20": ["Animal2", "A", "ok"],
    "2018.06.22": ["Animal3", "A", "ok"],
    "2018.07.17": ["MS3", "D", "outlier"],  # B
    "2018.07.20": ["MS8", "C", "outlier"],  # A
    "2018.07.23": ["MS4", "C", "outlier"],  # A
    "2018.07.25": ["MS1", "D", "outlier"],  # B
    "2018.07.27": ["NI4", "B", "ok"],
    "2018.07.30": ["NI3", "B", "ok"],
    "2018.08.01": ["NI1", "A", "ok"],
    "2018.08.03": ["NI2", "A", "ok"],
    "2018.09.12": ["unclipped", "A", "ok"],
    "2018.09.19": ["clipped", "A", "ok"],
    # ---------------------------added after 11/28/2018
    "2018.10.30": ["animan1", "B", "ok"],
    "2018.10.31": ["animal2", "A", "ok"],
    "2018.11.06": ["animal3", "A", "ok"],
    "2018.11.08": ["animal4", "B", "ok"],
    "2018.12.27": ["OE1", "A", "ok"],
    "2019.01.02": ["OE3", "B", "ok"],
    "2019.01.04": ["OE2", "B", "ok"],
    "2019.01.09": ["BNE9", "AA", "ok"],
    "2019.01.11": ["BNE4", "AA", "ok"],
    "2019.01.14": ["BNE5", "B", "ok"],  # note reads: Animal #BNE5 (1-5?)
    "2019.01.16": ["BNE2", "AA", "ok"],  # Notes both read: Animal #BNE2 (1-2)
    "2019.01.18": ["BNE2", "AA", "ok"],  # Animal #BNE2 (1-2)
    "2019.01.23": ["BNE1", "AA", "ok"],  # Animal #BNE1 (1-1)
    "2019.01.24": ["BNE7", "AA", "ok"],  # Animal #BNE7 (1-7)
    "2019.01.30": [
        "BNE7",
        "A",
        "ok",
    ],  # ???? Animal #BNE17/18/20 (need verification from Tessa) No discernable marks on mouse ears/toes.
    "2019.02.01": ["BNE18", "A", "ok"],  # Clear upper? left notch
    "2019.02.15": ["BNE24", "A", "ok"],  # left upper ear notch
    "2019.02.18": ["BNE21", "A", "ok"],  # both ears notched
    "2019.02.19": ["BNE23", "B", "ok"],  # lower right ear notched
    "2019.02.20": ["BNE22", "B", "ok"],  # lower left ear notched
    "2019.02.22": ["BNE27", "AA", "ok"],  # P24 At exposure
    # '2019.02.25': ['BNE25', 'AA', 'no'], # no data from this animal  P24 at
    # exposure
    "2019.03.04": ["BNE32", "AA", "ok"],
    "2019.03.05": ["BNE31", "AA", "ok"],
    "2019.03.06": ["BNE30", "AA", "ok"],
    "2019.03.01": ["BNE2Y", "AAA", "ok"],  # noise exposed high level; 3 day wait
    "2019.03.15": ["BND3Y", "AAA", "ok"],
    "2019.03.18": ["BNE102", "AAA", "ok"],  # BNE102 - exp 3/15, 3 day wait
    "2019.04.15": ["BNE103", "AA"   "no"],  # BND103   - exp 4/15 - 33 day wait
}

"""
From TFR Notebook Noise exposures:
Book 2:
P63: NF107Ai32 @ P31, OE1, OE4
P65: NF107AI32: BNE1-4 noise exposure.  BNE5: unexposed. 
P65: 12/14/18 OA1, OA2, OA8 ABR
P66: 12/14/18 OA4, AO3, OA9 ABR
P67: 12/20/18 BNE 6, 7, 8, 9, 10 : exposed    BNE 11, 12, 13 Not exposed.
P67: 12/20/18: OE2, OE4 ABR
P68: 12/21/18: OE1, OE3 ABR
P68: 12/27/18: BNE5, BNE1 ABR
P69: 12/27/18: BNE2, BNE3, BNE4, BNE7 ABR
P70: 1/2/19: BNE8, BNE9, BNE6, BNE10
P71: 1/3/19 BNE 14-16 noise exposure
P71: 1/4/19: Note BNE 1-13, OA1-4 (VGAT) 103.4 dB SPL instead of 110.
P72: 1/9/19: BNE10 ABR
P72: 1/9/19: BNE 17, 18, 19, 20 Noise exposure. 
P72: 1/14/19: OE4 ABR
P74: 1/17/19: BNE14, 15, 16, 13 ABR
P75: 1/24/19: BNE17, BNE18, BNE19, BNE20 ABR (BNE19 euthanized)
P76: 1/25/19: BNE21-24 : Noise exposure (brown coats exposed, 2 m; black coates unexposed)
P76: 1/25/19: BNE 25-29 3m brown coat exposed; white coat unexposed.
P76: 2/11/19: BNE21, 22, 23, 24 ABR
P77: 2/12/19: BNE25, 26, 27, 28, 29 ABR
P78: 2/15/19: 4 mice exposed; no label info.
P80: 3/1/19: BNE 30, 31, 32, 100, 33, 34, 35 ABR tests. First one says P45, exposed 2/15/19.
P82: 3/11/19: BNE36, 37 ABR
P83: 3/13/19: BNE 37, 38 ABR
P84: 3/15/19: BNE101 ABR
P84: 3/15/19: NF107 Exposed 2 mice, OR1 and OR3 (BNE102, BNE103)

P85: 4/12/19: NF107 exposed : one ear punch left, one right
P86: 4/15/19: ABR on one 4/12 mice used on 4/15 - acq4 has wrong mouse designator. 
P87: 5/3/19: NF107 P40 exposed No label
P87: 5/6/19: NF107 ABR (from 5/3 mouse?)
P88: 5/7/19: NF107 noise exposre P46 male no label
P89: 5/10/19: NF107 ABR (3d exp)
P90: 5/13/19: NF107 ABR "PL"
P90: 5/17/19: NF107 ABR, noise exposed on 5/14/19
P90: 5/17/19: NF107 Noise expsoure 115
P92: 5/21/19: BNE39, BNE40 ABR (2 week
P93: 5/21/19: BNE108 noise exposure (115)
P93: 6/3/19: ABR BNE108
P93: P32 NF107 noise exposure 2 m, 115dB no label
P94: 6/17/19 : BNE ? NF107 ABR males, exposed 6/3/19.

"""
############################################################################
coding_VGAT_nihl = {
    "2018.08.31": ["NM3", "B", "ok"],
    "2018.09.04": ["NM4", "B", "ok"],
    "2018.10.11": ["NS9", "A", "ok"],
    "2018.10.16": ["NS3", "A", "ok"],
    "2018.10.19": ["NT3", "A", "ok"],
    "2018.10.23": ["NT1", "A", "ok"],
    "2018.10.24": ["NT4", "B", "ok"],
    "2018.10.26": ["NU1", "B", "ok"],
    "2018.11.26": ["NF7", "", "ok"],
    "2018.11.27": ["NM7", "", "ok"],
    "2018.12.14": ["OA2", "A", "ok"],
    "2018.12.17": ["OA1", "B", "ok"],
    "2018.12.18": ["OA8", "A", "ok"],
    "2018.12.20": ["OA4", "B", "ok"],
    "2019.03.13": ["OM7", "AA", "ok"],
    "2019.05.21": ["PF2", "", "ok"],
    "2019.05.22": ["PF12", "", "ok"],
    "2021.03.17": ["SA2"],
    "2021.03.19": ["SA1"],
    "2021.12.19": ["SY3"],
    "2021.12.30": ["SY1"],  # question about genotype. 
    "2022.01.05": ["TB1"],
    "2022.01.06": ["TB7"],

}

"""From TFR data notebook, starting:
Book 2:
Page 62: noise expose VGAT OA2, OA9, OA3, OA8
P81: 3/4/19: OL3 ABR
P82: 3/4/19: OM3, OM7, OM2 ABR
P83: 3/12/19: OR litter at P41
P84: "OR 1,3 P44 noise expose, 1 black, 3 white" ?
P85: 4/11/19: OZ10, OW11 noise expose
P88: 5/7/19 OZ4 ABR
P89: 5/9/19 OZ10, OW4 ABR
P91: 5/20/19 PF2 ABR
P91: 5/20/19 PF10, PF12 ABR
Book 3:
P50: 3/1/21  SA1, SA2 noise expoosure 13.5 dB (109?)
P52: 3/15/21 SA1, SA2 ABR
P59: 12/15/21: SY2, SY3 exposure 5 dB (115)
P59: 12/16/21: SY1 exposure 5 dB
P59: 12/22/21: TB1, 2, 4, 7 exposure (2m 2f)
P60: 12/29/21: SY1, SY3 ABR
P60: 1/5/22: TB1, TB7, TB2(-/-)ABR
"""
coding_VGAT = {}


############################################################################

# experiments is a dict. Keys are the names of the expeirments (defined also in
# the argparse -E command). The values are a dict with a disk (Path), directory
# where results and intermediate files are stored, name of the datasummary file,
# and name of the annotation file (if it exists).


experiments = {
    "None": {},
    "nf107": {
        "datadisk": Path(baseDataDirectory, "NF107Ai32_Het/"),  # list of source directories/paths for original data
        "resultdisk": Path(codeDirectory),
        "db_directory": "datasets/NF107Ai32_Het",  # database directory
        "datasummary": "NF107Ai32_Het",
        "annotation": "NF107Ai32_Het_cell_annotations.xlsx",
        "maps": "NF107Ai32_Het_maps",
        "IVs": "NF107Ai32_Het_IVs",
        "coding": {},
    },
    "nf107nihl": {
        "datadisk": Path(baseDataDirectory, "NF107Ai32_NIHL"),
        "resultdisk": Path(codeDirectory),
        "db_directory": Path(datasetDirectory, "NF107Ai32_NIHL" ),
        "datasummary": Path("NF107Ai32_NIHL"),
        "annotation": None,
        "maps": None, # "NF107Ai32_NIHL_maps",
        "IVs": "NF107Ai32_NIHL_IVs",
        "coding": coding_NF107_nihl,
    },
    "vgatnihl": {
        "datadisk": Path(baseDataDirectory, "VGAT_DCNmap/"),
        "resultdisk": Path(codeDirectory),
        "db_directory": "datasets/VGAT_NIHL",
        "datasummary": "VGAT_NIHL",
        "annotation": "VGAT_NIHL_cell_annotations.xlsx",
        "maps": "VGAT_NIHL_maps",
        "IVs": None,
        "coding": coding_VGAT_nihl,
    },
    "noisetest": {
        "datadisk": Path(baseDataDirectory, "Noise_test/"),
        "resultdisk": Path(codeDirectory),
        "db_directory": "Noise_test",
        "datasummary": None,
        "annotation": None,
        "maps": None,
        "IVs": None,
        "coding": None,
    },
    "vgat": {
        "datadisk": Path(baseDataDirectory, "VGAT"),
        "db_directory": "datasets/VGAT",
        "datasummary": "VGAT",
        "annotation": "VGAT_cell_annotations.xlsx",
        "maps": "VGAT_maps",
        "IVs": None,
        "coding": coding_VGAT,
    },
    "ank2b": {
        "datadisk": Path("/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/Maness_Ank2_PFC_stim"),
        "resultdisk": Path("~/Desktop/Python/cs_minis/ANK2"),
        "db_directory": Path("~/Desktop/Python/cs_minis/ANK2"),
        "datasummary": "Intrinsics",
        "annotation": None,
        "maps": None,
        "IVs": Path("~/Desktop/Python/cs_minis/ANK2/Intrinsics"),
        "coding": None,
    }
}



def get_experiments():
    return experiments

# coding  From Tessa Ropp, gatekeeper. The letters "A", "AA", "AAA", and "B" refers to group
# (but we do not know which group - exposed or not)
#
# code: B not exposed, A exposed (@109); AA exposed (@116) 8-16 kHz Gaussian
# Noise, ABR'd

# Exclusions are **individual protocols** (and sometimes **cells**) that are
# excluded for the reasons indicated to the right. Often these are a) noisy
# recordings, b) breakthrough spikes, c) change of protocol (e.g., held at
# positive potentials), and d) sometimes because the recording went "quiet"
# between two normal maps

exclusions = [
    # NF107 : Pyramidal
    "2017.03.24_000/slice_001/cell_001//Map_NewBlueLaser_VC_pt1mW_001" # fails in analysis
    "2017.04.12_000/slice_003/cell_001",  # noisy
    "2017.06.28_000/slice_000/cell_000/Map_NewBlueLaser_VC_10Hz_002",  # break through spikes
    "2017.06.28_000/slice_000/cell_000/Map_NewBlueLaser_VC_10Hz_002",  # cell broke in to spikng
    "2017.07.05_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_003",  # single spot
    "2017.07.05_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_002",  # really noisyt
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_IC_10Hz_003",  # spikes
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_IC_10Hz_004",  # spikes
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_IC_10Hz_005",  # single point
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_IC_10Hz_006",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_IC_10Hz_007",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_IC_10Hz_008",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_IC_10Hz_009",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_005",  # sinlge point single trial
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_006",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_007",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_008",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_009",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_010",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_011",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_012",
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_013",  # single point
    "2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_014",  # single point
    "2017.08.16_000/slice_001/cell_001",  # cell ID very uncertain; removed (possible cartwheel)
    "2017.08.28_000/slice_000/cell_000",  # all cell attached, discard
    "2017.08.28_000/slice_000/cell_000/Map_NewBlueLaser_CA_000",  # all cell-attached
    "2017.08.28_000/slice_000/cell_000/Map_NewBlueLaser_CA_001",  # all cell-attached
    "2017.08.28_000/slice_000/cell_000/Map_NewBlueLaser_CA_002",  # all cell-attached
    "2017.08.28_000/slice_000/cell_000/Map_NewBlueLaser_CA_003",  # all cell-attached
    "2017.09.11_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_004",  # something wrong with data - bad structure
    "2017.09.11_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_005",  # something wrong with data - no data
    "2017.09.11_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_006",  # something wrong with data - no data
    "2017.09.20_000/slice_000/cell_000",  # because it is REALLY noisy
    "2017.10.02_000/slice_000/cell_000",  # too many break through spikes
    "2017.11.13_000/slice_001/cell_000",  # certainly t-stellate; weird spiking behavior, so remove...
    "2017.12.05_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_000",  # last record is missing data
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_002",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_003",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_004",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_005",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_006",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_007",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_017",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_018",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_019",  # some at positive V, some small maps
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_020",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_021",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_022",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_023",
    "2018.02.21_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_024",
    "2018.02.23_000/slice_000/cell_000",  # probably CW cells, image not distinct
    "2018.02.23_000/slice_000/cell_001",  # probably CW cell; image not distinct
    "2018.02.23_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_003",  # positive V
    "2018.02.23_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_004",  # positive V
    "2018.02.23_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_005",  # positive V
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_003",  # positive V  # ALL traces after +V run are excluded
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_004",  # positive V
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_005",  # positive V
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_001",  # positive V, SHOWS POSSIBLE NMDA COMPONENT.
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_002",  # positive V
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_003",  # Seems small, but responses are slower also.
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_004",  # Seems small, but responses are slower also.
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_005",  # Seems small, but responses are slower also.
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_006",  # Seems small, but responses are slower also.
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_007",  # Seems small, but responses are slower also.
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_increase_000",  #  Looks ok, but after depolarizaiton/pairing...
    "2018.02.23_000/slice_002/cell_000/Map_NewBlueLaser_VC_increase_001",  # positive V
    "2018.02.26_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_001",  # traces with larger than normal artifacts
    "2018.02.26_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_002",  # cannot eliminate from analysis
    "2018.02.26_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_003",
    "2018.02.26_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_004",
    "2018.02.27_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_002_plus60",
    "2018.02.27_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_003",  # [positive V]
    "2018.02.27_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_004",  # [positive V]
    "2018.02.27_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_005",  # [positive V]
    # '2018.02.28_000/slice_001/cell_000/', # NO RESPONSES NO SPONT; NEEDS
    # REANALYSIS WITH CB; renalyzed; include (reanalyzed)
    # '2018.02.28_000/slice_001/cell_001/Map_NewBlueLaser_VC_Single_002', # NO
    # RESPONSES NO SPONT; NEEDS REANALYSIS WITH CB (reanalyzed)
    "2018.03.06_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_004",
    "2018.03.06_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_005",
    "2018.03.07_000/slice_000/cell_000/Map_NewBlueLaser_VC_10Hz_003_objUP",  # bad optic config
    "2018.03.07_000/slice_000/cell_000/Map_NewBlueLaser_VC_10Hz_004_objUPplus60",  # positive bad optic config
    "2018.03.07_000/slice_000/cell_000/Map_NewBlueLaser_VC_10Hz_006_plus60",  # positive
    "2018.03.07_000/slice_000/cell_000/Map_NewBlueLaser_VC_10Hz_007",  # probably positive CHECK
    "2018.03.12_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_002",
    "2018.03.12_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_003",
    "2018.03.12_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_002",  # spont rate dropped - probably bad recording
    "2018.03.14_000/slice_000/cell_002/Map_NewBlueLaser_VC_10Hz_003_plus60strychnine",  # positive + strychnine
    "2018.03.14_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_002",  # ????? 001 and 003 are ok, this one has no events.
    "2018.06.04_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_000",  # clipped events
    "2018.06.04_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_000",  # clipped events
    "2018.06.04_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_000",  # clipped events
    "2018.06.04_000/slice_001/cell_000/Map_NewBlueLaser_VC_increase_1ms_000",  # clipped events
    "2018.06.04_000/slice_003/cell_000/Map_NewBlueLaser_VC_increase_1ms_001",  # lots of random noise
    "2019.04.16_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_002"  # excess noise in part of the recording
    "2019.04.16_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_003"  # excess noise in most of the recording
    # Giant NF107
    "2017.05.17_000/slice_000/cell_001/Map_NewBlueLaser_CA_000",  # one trace
    "2017.05.17_000/slice_000/cell_001/Map_NewBlueLaser_CA_001",  # one trace
    "2017.05.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_weird_001",  # one trace
    "2017.05.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_weird_002",  # one trace
    "2017.10.11_000/slice_000/cell_001",  # mirrors were off so no scans
    # TV NF107:
    "2017.05.22_000/slice_001/cell_000",  # bad recording according to notes - no mirror action, table bumped
    "2017.05.22_000/slice_001/cell_000/Map_NewBlueLaser_VC_weird_000",  # protocols have break-through spikes
    "2017.05.22_000/slice_001/cell_000/Map_NewBlueLaser_VC_weird_003",
    "2017.09.11_000/slice_002/cell_000_signflip",  # skip the signflip
    "2017.07.31_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_002",  # extraneous noise
    "2017.07.31_000/slice_001/cell_001/Map_NewBlueLaser_VC_increase_000",  # extraneous noise
    "2017.10.11_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_002",  # noise on recording
    "2017.10.11_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_003",  # noise on recording
    "2017.11.13_000/slice_004/cell_000/Map_NewBlueLaser_VC_Single_000",  # spiking
    "2017.12.13_000/slice_001/cell_001_signflip",  # skip the signflip
    "2018.01.08_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_000",  # too few events
    "2018.01.08_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_001",  # too few events
    "2018.01.08_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_002",  # too few events
    "2018.01.16_000/slice_001/cell_000",  # too few events, hf noise
    "2018.02.27_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_002",  # Positive V
    "2018.02.27_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_003",  # Positive V
    "2018.02.27_000/slice_001/cell_000/Map_NewBlueLaser_VC_increase_001",  # Positive V
    "2018.02.27_000/slice_001/cell_000/Map_NewBlueLaser_VC_increase_002",  # Positive V
    "2018.03.14_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_002_plus60strych",  # Positive
    "2018.03.14_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_003",  # Positive
    "2018.03.14_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_007",  # ? dead cell
    "2018.04.16_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_002",  # ? dead cell
    # NF107: Cartwheel:
    "2017.05.01_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_001",  # artifacts are too large - average event is squarish.
    "2017.05.01_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_005",  # different cell than rest of the data for this cell.
    "2017.06.26_000/slice_003/cell_000/",  # noisy  60 Hz, cell phone.
    "2017.12.13_000/slice_003/cell_001/Map_NewBlueLaser_VC_10Hz_003",  # small map, needs renalaysis with CB
    "2018.02.26_000/slice_001/cell_001/MapNewBlueLaster_VC_10Hz_006",  # noisy
    "2018.02.26_000/slice_001/cell_001/MapNewBlueLaster_VC_10Hz_008",  # looks like cell lost in middle of protocol
    # T-stellate
    "2017.07.12_000/slice_002/cell_001/Map_NewBlueLaser_VC_10Hz_000",  # extraneous noise
    "2017.07.12_000/slice_002/cell_001/Map_NewBlueLaser_VC_10Hz_002",  # extraneous noise
    "2019.09.10_000/slice_000/cell_002/Map_NewBlueLaser_VC_10Hz_002",  # noise, trash
    # NF107: Octopus
    "2019.09.10_000/slice_000/cell_000/Map_NewBlueLaser_VC_10Hz_003",
    # NF107 : bushy exclusions
    "2017.03.01_000/slice_000/cell_001/Map_NewBlueLaser_VC_single_test_002",  # incomplete; data at end unanalyzable
    "2017.03.24_000/slice_001/cell_000/Map_NewBlueLaser_VC_range test_000",  # single spot
    "2017.03.24_000/slice_001/cell_000/Map_NewBlueLaser_VC_range test_001",  # single spot
    "2017.03.28_000/slice_000/cell_000/Map_NewBlueLaser_VC_1mW_017",  # partial
    "2017.03.28_000/slice_000/cell_001/Map_NewBlueLaser_VC_1mW_003",  # junk at end of a copule of traces
    # '2017.03.28_000/slice_000/cell_000/Map_NewBlueLaser_VC_1mW_004', #
    # possible file corruption - could not get protocol reps, has value of [0],
    # but file has 20 entries.
    # '2017.03.28_000/slice_000/cell_000/Map_NewBlueLaser_VC_1mW_005', #
    # possible file corruption - could notget protocol reps
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_1mW_005",  # small map
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_1mW_006",  # small map
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_1mW_007",  # small map
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_1mW_008",  # small map
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_1mW_009",  # small map
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_1mW_011",  # partial
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_1mW_012",  # small map
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_4mW_001",  # different frequency
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_4mW_002",  # single, not train
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_range test_000",  # doesn't parse; do not need data
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_range test_001",  # doesn't parse; do not need data
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_multipulse_000",  # doesn't parse; do not need data
    "2017.04.03_000/slice_000/cell_001/Map_NewBlueLaser_VC_multipulse_001",  # doesn't parse; do not need data
    "2017.04.12_000/slice_003/cell_000/Map_NewBlueLaser_VC_000",  # has breakthrough spikes
    # '2017.04.03_000/slice_000/cell_003', # responses, but just a really noisy
    # recording, not useful.
    "2017.05.01_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_005"  # data is from a different cell than the rest in this entry
    # (moved on Pegasus drive to new cell #2)
    "2017.05.15_000/slice_001/cell_000",  # possibly bushy, but not clear; spont epscs are too slow... unknown is better choice.
    "2017.05.17_000/slice_001/cell_001",  # not bushy morphology - and weird CC traces
    "2017.05.17_000/slice_001/cell_001/Map_NewBlueLaser_VC_weird_002",  # partial map
    "2017.05.17_000/slice_001/cell_001/Map_NewBlueLaser_VC_weird_003",  # noisy, and cannot get analysis
    "2017.05.17_000/slice_001/cell_001/Map_NewBlueLaser_VC_weird_004",  # partial
    #'2017.05.17_000/slice_001/cell_000/Map_NewBlueLaser_VC_weird_000', # evoked
    #events are unclamped spike (see IC_000)
    "2017.05.17_000/slice_001/cell_000/Map_NewBlueLaser_IC_000",  # evoked events are just a spike  - count as responding, but can't get measurement.
    "2017.05.22_000/slice_000/cell_000",  # bad recording, scanning mirrors off
    "2017.06.23_000/slice_003/cell_000",  # major issues including using acsf as internal - discard
    "2017.07.20_000/slice_002/cell_000/Map_NewBlueLaser_VC_weird_003",  # partials
    "2017.07.20_000/slice_002/cell_000/Map_NewBlueLaser_VC_weird_004",
    "2017.07.20_000/slice_002/cell_000/Map_NewBlueLaser_VC_weird_005",
    "2017.07.20_000/slice_002/cell_000/Map_NewBlueLaser_VC_weird_006",
    "2017.07.20_000/slice_002/cell_000/Map_NewBlueLaser_VC_weird_007",
    "2017.11.21_000/slice_000/cell_001/Map_NewBlueLaser_VC_Single_002",  # ? weak stim
    "2017.11.21_000/slice_000/cell_001/Map_NewBlueLaser_VC_Single_004",  # seems to have faded
    # NOTE: The following are ok data sets but need to be reanalyzed
    # '2018.02.16_000/slice_000/cell_001/'
    # '2018.02.28_000/slice_000/cell_000/Map_NewBlueLaser_VC_10Hz_000'
    # 2018.03.16_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_000  HAS LONG
    # LATENCY RESPONSES
    "2018.02.26_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_005",
    "2018.02.26_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_006",
    "2018.02.26_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_007",  # the rest for this cell
    "2018.02.26_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_008",
    "2018.02.26_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_009",
    "2018.02.26_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_010",
    "2018.02.26_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_011",
    "2018.02.27_000/slice_003/cell_000/Map_NewBlueLaser_VC_10Hz_002",  # noisy, bad
    "2018.02.27_000/slice_003/cell_000/Map_NewBlueLaser_VC_10Hz_003",  # loos like is going at end of recording
    "2018.03.16_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_003",  # sparse map
    "2018.03.16_000/slice_002/cell_000/Map_NewBlueLaser_VC_10Hz_004",  # ? sparse map
    # Unknown excluseions:
    "2017.04.07_000/slice_000/cell_000/Map_NewBlueLaser_VC_000",  # no discernable responses
    "2017.04.07_000/slice_000/cell_000/Map_NewBlueLaser_VC_001",  # no clear responses
    "2017.04.07_000/slice_000/cell_000/Map_NewBlueLaser_VC_002",  # average is noise; remainder of this cell is ok
    # NF107 - NIHL
    # =======================================================================================#
    # Pyramidal NF107 NIHL
    "2018.08.01_000/slice_000/cell_001/Map_NewBlueLaser_VC_Single_000",  # noisy
    "2018.10.30_000/slice_001/cell_001/Map_NewBlueLaser_VC_10Hz_002",  # spiking
    "2018.08.01_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_002",  # spike at end
    "2018.08.01_000/slice_000/cell_001/Map_NewBlueLaser_VC_Single_000",  # noise on several traces (low freq)
    "2018.08.03_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_000",  # bursts of spont activity
    "2019.01.30_000/slice_001/cell_000/Map_NewBlueLaser_VC_10Hz_003",  # noise, some unstable traces
    "2019.01.30_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_000",  # not stable
    "2019.02.01_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_000",  # wandering baseline
    "2019.01.18_000/slice_001/cell_001/Map_NewBlueLaser_VC_Single_002",  # unstable baseline, small maps
    "2019.01.18_000/slice_001/cell_001/Map_NewBlueLaser_VC_Single_003",
    "2019.01.18_000/slice_001/cell_001/Map_NewBlueLaser_VC_Single_004",
    "2019.01.18_000/slice_001/cell_001/Map_NewBlueLaser_VC_Single_002",
    "2019.01.24_000/slice_002/cell_000/Map_NewBlueLaser_VC_Single_003",  # high levels of synaptic spont; varies across map
    "2019.01.02_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_001",  # unstable baselines
    "2019.02.20_000/slice_001/cell_000/Map_NewBlueLaser_VC_Single_000",  # unstable baselines
    # TV NF107NIHL
    "2019.02.18_000/slice_000/cell_000/Map_NewBlueLaser_VC_Single_000",  # unstable baseline

    "2019.09.10_000/slice_000/cell_002/Map_NewBlueLaser_VC_10Hz_002", # failure in recording? channel switching?

    # IVs 
    "Parasagittal/2020.11.02_000/slice_000/cell_000/CCIV_long_000", # trace 2 is broken
]


def get_exclusions():
    return exclusions


def main():
    e = get_exclusions()
    pp.pprint(e)


if __name__ == "__main__":
    main()
