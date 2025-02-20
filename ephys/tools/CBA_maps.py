# new_nf107_maps.py

import datetime
import json
from pathlib import Path

import ephys.ephys_analysis as EP
import numpy as np
import pandas as pd
import src.set_expt_paths as set_expt_paths
from ephys.ephys_analysis import (
    analysis_common,
    iv_analysis,
    map_analysis,
    summarize_ivs,
)
from ephys.tools import write_excel_sheet as WES
from pylibrary.tools import cprint as CP

import toml

rawdatadisk = "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/"
analyzeddatapath = "/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets/"
experiments = {} # no mapping done in this study
# {
#     "NF107Ai32_Het": {  # name of experiment
#         "rawdatapath": rawdatadisk,
#         "databasepath": analyzeddatapath,  # location of database files (summary, coding, annotation)
#         "analyzeddatapath": analyzeddatapath,  # analyzed data set directory
#         "directory": "NF107Ai32_Het",  # directory for the raw data, under rawdatadisk
#         "pdfFilename": None,  # "NF107Ai32_Maps_12-2022.pdf",  # PDF figure output file
#         "datasummaryFilename": "NF107Ai32_Het_08Aug2023.pkl", # "NF107Ai32_Het_26Jan2023.pkl",  # name of the dataSummary output file, in resultdisk
#         "IVs": None,  # "NF107Ai32_IVs",
#         "iv_analysisFilename": None,  # "iv_analysis.h5", # name of the pkl or HDF5 file that holds all of the IV analysis results
#         "eventsummaryFilename": "NF107Ai32_Het_event_summary.pkl", # name of the event summary file, in resultdisk
#         "coding_file": None,  # "Intrinsics.xlsx",
#         "coding_sheet": None,  # "codes",
#         "cell_annotationFilename": None, # "NF107Ai32_Het_cell_annotations.xlsx",  # annotation file, in resultdisk
#         "bridgeCorrectionFilename": None,  # location of the excel table with the bridge corrections
#         "map_annotationFilename": "NF107Ai32_Het_maps_08.15.2023.xlsx",  # map annotation files
#         "extra_subdirectories": ["Parasagittal", "OLD", "NF107Ai32-TTX-4AP"],
#         "artifactPath": "/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets/artifact_templates/",
#         "artifactFilename":  None,  # This should be the name of the default artifact file: specific files are pulled from the excel sheet
#         # the epoch refers to the time frame around the recording data, and is used to select the appropriate artifact template
#             # "Map_NewBlueLaser_VC_10Hz": "template_data_map_10Hz.pkl",
#             # "Map_NewBlueLaser_VC_Single": "template_data_map_Singles.pkl",
#         # },
#         "selectedMapsTable": "ExampleMaps/SelectedMapsTable.xlsx",  # table with maps and cell numbers selected
#     }
# }


def shortpath(row, leftclip=4):
    print(row.Cell_ID)
    ipath = Path(row.Cell_ID).parts
    opath = str(Path(*ipath[leftclip:]))
    return opath


def analyze():
    # experiments = nf107.set_expt_paths.get_experiments()
    expt ="NF107Ai32_Het"
    exclusions = set_expt_paths.get_exclusions()
    args = analysis_common.cmdargs  # get from default class
    args.dry_run = False
    args.autoout = True
    args.merge_flag = True
    args.experiment = experiments[expt]
    args.iv_flag = False
    args.map_flag = True
    args.mapsZQA_plot = False
    args.zscore_threshold = 1.96 # p = 0.05 for charge relative to baseline

    args.plotmode = "document"
    args.recalculate_events = True
    args.artifact_filename = experiments[expt]['artifactFilename']
    args.artifact_path = experiments[expt]['artifactPath']

    args.artifact_suppression = True
    args.artifact_derivative = False
    args.post_analysis_artifact_rejection = False
    args.autoout = True
    args.verbose = False
    # these are all now in the excel table
    # args.LPF = 3000.0
    # args.HPF = 0.

    args.detector = "aj"
    args.spike_threshold = -0.020 # always in Volts
    # args.threshold = 7

    # args.day = "2019.09.10_000"
    # args.slicecell = "S0C0" 
    # args.after = "2019.07.25"
    args.celltype = "d-stellate"
    # args.protocol = "Map_NewBlueLaser_VC_increase_1ms_005"

    args.notchfilter = True
    odds = np.arange(1, 43, 2)*60.  # odd harmonics
    nf = np.hstack((odds, [30, 15, 120., 240., 360.]))  # default values - replaced by what is in table
    str_nf = "[]" # "[" + ", ".join(str(f) for f in nf) + "]"
    args.notchfreqs = str_nf # "[60., 120., 180., 240., 300., 360., 600., 4000]"
    args.notchQ = 90.

    if args.configfile is not None:
        config = None
        if args.configfile is not None:
            if ".json" in args.configfile:
                config = json.load(open(args.configfile))
            elif ".toml" in args.configfile:
                config = toml.load(open(args.configfile))

        vargs = vars(args)  # reach into the dict to change values in namespace
        for c in config:
            if c in args:
                # print("c: ", c)
                vargs[c] = config[c]
    CP.cprint(
        "cyan",
        f"Starting MAP analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
    )
    print("\n" * 3)
    CP.cprint("r", "=" * 80)

    MAP = map_analysis.MAP_Analysis(args)
    MAP.set_experiment(experiments[expt])
    MAP.set_exclusions(exclusions)
    MAP.AM.set_artifact_suppression(args.artifact_suppression)
    MAP.AM.set_artifact_path(experiments[expt]['artifactPath'])
    MAP.AM.set_artifact_filename(experiments[expt]['artifactFilename'])
    MAP.AM.set_post_analysis_artifact_rejection(args.post_analysis_artifact_rejection)
    MAP.AM.set_template_parameters(tmax=0.009, pre_time=0.001)
    MAP.AM.set_shutter_artifact_time(0.050)
 
    CP.cprint("b", "=" * 80)
    MAP.setup()

    CP.cprint("c", "=" * 80)
    MAP.run()

    # allp = sorted(list(set(NF.allprots)))
    # print('All protocols in this dataset:')
    # for p in allp:
    #     print('   ', path)
    # print('---')
    #
    CP.cprint(
        "cyan",
        f"Finished analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
    )


def summarize(celltype="any"):
    args = {
        "experiment": "NF107AI32_Het",
        "mode": "OU",
        "outputFilename": "test.pdf",
        "celltype": celltype,
    }
    g = summarize_ivs.GetAllIVs(args)
    g.set_protocols(["Map_", "LED_"])
    g.set_experiments(experiments)
    g.run()

    sheet_names = ["spikes", "ivs"]
    with pd.ExcelWriter(
        Path(f"NF107_Het/NF107_spike_data_{g.mode:s}_{celltype:s}.xlsx")
    ) as writer:
        bg_format = writer.book.add_format(
            {"bg_color": "#DEE2E6"}
        )  # light green cell background color
        for idf, df in enumerate([g.spike_dataframe, g.iv_dataframe]):
            df.Cell_ID = df.apply(shortpath, axis=1)
            df.to_excel(writer, sheet_name=sheet_names[idf])
            # print(dir(writer.sheets[sheet_names[idf]]))
            for i, row in enumerate(df.index):
                if i % 2 == 0:
                    continue
                writer.sheets[sheet_names[idf]].set_row(
                    i, None, cell_format=bg_format
                )  # column_dimensions[str(column.title())].width = column_width
            for i, column in enumerate(df.columns):
                column_width = max(
                    [df[column].astype(str).map(len).max(), 12]
                )  # len(column))
                writer.sheets[sheet_names[idf]].set_column(
                    first_col=i + 1, last_col=i + 1, width=column_width
                )  # column_dimensions[str(column.title())].width = column_width


if __name__ == "__main__":
    analyze()
    # summarize()
