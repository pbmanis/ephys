# analyzeIVs.py
import pyqtgraph as pg
import datetime
import json
from pathlib import Path
import ephys.ephys_analysis as EP
import pandas as pd
import pprint
from ephys.ephys_analysis import (
    analysis_common,
    iv_analysis,
    map_analysis,
    summarize_ivs,
)
from ephys.tools.get_configuration import get_configuration
from pylibrary.tools import cprint as CP

import toml
PP = pprint.PrettyPrinter(indent=4)

datasets, experiments = get_configuration()

print("analyze_ivs finds experiments: ")
PP.pprint(experiments)


def shortpath(row, leftclip=4):
    print(row.Cell_ID)
    ipath = Path(row.Cell_ID).parts
    opath = str(Path(*ipath[leftclip:]))
    return opath


def analyze(expt, slicecell=None):
    # experiments = nf107.set_expt_paths.get_experiments()
    exclusions = None # set_expt_paths.get_exclusions()
    args = analysis_common.cmdargs  # get from default class
    args.dry_run = False
    args.noparallel = True
    args.merge_flag = True
    args.experiment = experiments[expt]
    args.iv_flag = True
    args.map_flag = False
    args.autoout = True

    args.verbose = False
    args.spike_threshold = -0.020  # always in Volts
    if slicecell is not None:
        args.day = slicecell[0]
        args.slicecell = slicecell[1]
    else:
        args.day = "all"
        args.slicecell = None
    args.after = "2023.11.28"
    # args.slicecell = "S0C1"
    #args.after= "2023.09.06"
    # args.celltype = "giant"

    if args.configfile is not None:
        config = None
        if args.configfile is not None:
            if ".json" in args.configfile:
                # The escaping of "\t" in the config file is necesarry as
                # otherwise Python will try to treat is as the string escape
                # sequence for ASCII Horizontal Tab when it encounters it
                # during json.load
                config = json.load(open(args.configfile))
            elif ".toml" in args.configfile:
                config = toml.load(open(args.configfile))

        vargs = vars(args)  # reach into the dict to change values in namespace
        for c in config:
            if c in args:
                # print("c: ", c)
                vargs[c] = config[c]
    CP.cprint(
        "g",
        f"Starting IV analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
    )
    print("\n" * 3)
    CP.cprint("r", "=" * 80)
    IV = iv_analysis.IVAnalysis(args)
    IV.set_experiment(experiments[expt])
    IV.set_exclusions(experiments[expt]['excludeIVs'])
    IV.setup()
    IV.run()

    # allp = sorted(list(set(NF.allprots)))
    # print('All protocols in this dataset:')
    # for p in allp:
    #     print('   ', path)
    # print('---')
    #
    CP.cprint(
        "cyan",
        f"Finished IV analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
    )
    CP.cprint("r", "=" * 80)
    CP.cprint("r", f"Now run 'nf107/util/process_spike_analysis.py' to generate the summary data file")
    CP.cprint("r", f"Then run 'nf107/util/plot_spike_info.py -a' to assemble the results into a single file")
    CP.cprint("r", f"Then run 'nf107/util/plot_spike_info.py' to plot summaries and get statistical results")
    CP.cprint("r", "=" * 80)

def process(exptname):
    with PdfPages(pdf_filename) as pdfs:
        protos = find_protocols(datasummary=datasummary, 
            codesheet=code_list,
            result_sheet=result_sheet,
            pdf_pages = pdfs)
        
def summarize(exptname, celltype="any"):
    args = {
        "experiment": exptname,
        "mode": "OU",
        "outputFilename": "test.pdf",
        "celltype": celltype,
    }
    g = summarize_ivs.GetAllIVs(args)
    g.set_protocols(["CCIV_long_HK", "CCIV_1nA_max_1s", "CCIV_200pA", "CCIV_long"])
    g.set_group_mode("Group")
    g.set_experiments(experiments)
    g.run()

    sheet_names = ["spikes", "ivs"]
    with pd.ExcelWriter(
        Path(f"{analyzeddatapath:s}/CBA_spike_data_{g.mode:s}_{celltype:s}.xlsx")
    ) as writer:
        bg_format = writer.book.add_format(
            {"bg_color": "#DEE2E6"}
        )  # light green cell background color
        for idf, df in enumerate([g.spike_dataframe, g.iv_dataframe]):
            if df is None:
                continue
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


def make_ds_excel(exptname):
    """Simple make an excel of the pkl file driving the analysis.
    This file is also made by dataSummary, in a more sophisticated manner.
    """
    toppath = experiments[exptname]["databasepath"]
    dbpath = Path(toppath, experiments[expt]["datasummaryFilename"])
    with open(dbpath, "rb") as fh:
        df = pd.read_pickle(fh)
    epath = dbpath.with_suffix(".xlsx")
    df.to_excel(epath)


if __name__ == "__main__":
    exptname = "CBA_Age"
    analyze(exptname)
    # summarize(exptname)
    # make_ds_excel(exptname)
