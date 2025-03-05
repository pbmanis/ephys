# check spont firing VC data sets

import ephys
from typing import Union
import numpy as np
from pathlib import Path
from ephys.tools.get_configuration import get_configuration
import ephys.ephys_analysis as EP
import ephys.datareaders as DR
import pandas as pd
import pyqtgraph as pg
import ephys.tools.categorize_ages as CA
import ephys.tools.filename_tools as FT
import re
import ephys.tools.check_inclusions_exclusions as CIE
import ephys.tools.assemble_datasets as assemble


vc_re = re.compile(r"\s*VC_Spont_", re.IGNORECASE)
cc_re = re.compile(r"\s*CC_Spont_", re.IGNORECASE)
# print('VC_re: ', vc_re.match("VC_Spont_001"))
# exit()


def numeric_age(row):
    """numeric_age convert age to numeric for pandas row.apply

    Parameters
    ----------
    row : pd.row_

    Returns
    -------
    value for row entry
    """
    if isinstance(row.age, float):
        return row.age
    row.age = int("".join(filter(str.isdigit, row.age)))
    return float(row.age)


def categorize_ages(row, experiment):
    row.age = numeric_age(row)
    for k in experiment["age_categories"].keys():
        if (
            row.age >= experiment["age_categories"][k][0]
            and row.age <= experiment["age_categories"][k][1]
        ):
            row.age_category = k
    return row.age_category


AR = DR.acq4_reader.acq4_reader()
SP = EP.spike_analysis.SpikeAnalysis()

def setup(config_file_path:Union[Path, str]="./config/experiments.cfg", dataset: str="NF107Ai32_Het"):
    

    datasets, experiments = get_configuration(config_file_path)

    if dataset in datasets:
        database = Path(
            experiments[dataset]["databasepath"],
            experiments[dataset]["directory"],
            experiments[dataset]["datasummaryFilename"],
        ).with_suffix(".pkl")
    else:
        database = Path(dataset)
        print("Expt keys: ", experiments.keys(), dataset)
    databasedict = experiments[dataset]
    parentpath = database.parent
    print("Database, parentpath: ", database, parentpath)
    eventsummaryfile = Path(parentpath, str(parentpath) + "_event_summary.pkl")
    eventpath = Path(parentpath, "events")
    map_annotationFile = None
    if "map_annotationFilename" in list(experiments[dataset].keys()):
        if experiments[dataset]["map_annotationFilename"] is not None:
            map_annotationFile = Path(parentpath, experiments[dataset]["map_annotationFilename"])

        if map_annotationFile is not None:
            map_annotations = pd.read_excel(
                Path(map_annotationFile).with_suffix(".xlsx"), sheet_name="Sheet1"
            )
            print("Reading map annotation file: ", map_annotationFile)
    
    dbfile = Path(
        experiments[dataset]["databasepath"],
        experiments[dataset]["directory"],
        experiments[dataset]["datasummaryFilename"],
    )
    main_db = pd.read_pickle(str(dbfile))
    expt = experiments[dataset]
    AD = assemble.AssembleDatasets()
    AD.experiment = expt
    coding_file = expt["coding_file"]
    coding_sheet = expt["coding_sheet"]
    coding_level = expt["coding_level"]
    coding_name = expt["coding_name"]

    main_db = AD.combine_summary_and_coding(
        df_summary=main_db,
        coding_file=coding_file,
        coding_sheet=coding_sheet,
        coding_level=coding_level,
        coding_name=coding_name,
        exclude_unimportant=False,
        status_bar=None,
    )

    age_cats = expt["age_categories"]
    main_db["age_category"] = {}
    main_db["age_category"] = main_db.apply(lambda row: CA.categorize_ages(row, age_cats), axis=1)
    print("Main database: ", main_db.age_category)
    return main_db, expt


def main(db: pd.DataFrame, expt: dict=None, sortby: str = "age", categories:list=[None], celltypes:list=["pyramidal"], show_vc:bool=True, show_cc:bool=False):
    main_db = db
    AR = DR.acq4_reader.acq4_reader()
    app = pg.mkQApp("sponts")
    win = pg.GraphicsLayoutWidget(show=True, title="sponts")
    win.resize(2000, 1400)
    win.setWindowTitle(f"Spont Firing")
    symbols = ["o", "s", "t", "d", "+", "x"]
    cat_colors = {'B': pg.mkColor('k'), 'A': pg.mkColor('b'), 'AA': pg.mkColor('c'), "AAA": pg.mkColor('magenta')}
    cat_colors = {"Sham": pg.mkColor("k"), "NE106": pg.mkColor("b"), "NE115": pg.mkColor("c")}
    win.setBackground("w")
    cell_count = 0
    rowcount = 0
    colcount = 0
    px = win.addPlot(title=categories)
    ncells = 0
    nvcs = 0
    main_dbs = main_db.sort_values(by=sortby)
    for ncell, cellid in enumerate(main_dbs["cell_id"]):
        # print("Cell_id: ", cellid)
        # print(main_db.columns)
        # if ncell > 10:
        #     break
        cell_row = main_dbs[main_dbs["cell_id"].str.match(cellid)]

        if sortby == "age":
            if cell_row["age_category"].values[0] not in categories:
                continue
        elif sortby == "Group" and categories is not [None]:
            if cell_row["Group"].values[0] not in categories:
                print("Group not in cats: ", cell_row["Group"].values[0])
                continue
        if cell_row["cell_type"].values[0] not in  celltypes:
            continue
        print("sort by: ", sortby, "category: ", categories)
        # print("cell row: ", cell_row.keys())
        print("cell,  group: ", cellid, cell_row["Group"].values[0])
        # continue
        all_protocols = cell_row["data_complete"].values[0]
        protocols = all_protocols.split(",")
        allprotocols = []
        protocols, additional_ivs, additional_iv_records = CIE.include_exclude(
            cellid,
            exclusions=expt["excludeIVs"],
            inclusions=expt["includeIVs"],
            allivs=protocols,
        )
        print("Protocols: ", protocols)

        nprots = len(protocols)
        pcount = 0
        t0 = 0
        ztime = -1

        for nprot, protocol in enumerate(protocols):
            print("protocol: ", protocol)
            protocol = protocol.lstrip()
            if vc_re.match(protocol) and show_vc:
                print("matched VC Protocol: ", protocol)
                protopath = Path(expt["rawdatapath"], expt["directory"], cellid, protocol)
                AR.setProtocol(protopath)
                ok = AR.getData()
                if not ok:
                    continue
                    # px = win.addPlot(title=c)
                # print('Traces shape: ', AR.traces.shape)
                # print("time_base shape: ", AR.time_base.shape)
                for tr in range(AR.traces.shape[0]):
                    start_time = AR.trace_StartTimes[tr]
                    if ztime < 0:
                        ztime = start_time
                    startindex = np.where(AR.time_base > 0.1)[0][0]
                    ca_data = AR.traces[tr, startindex:].view(typ=np.ndarray) * 1e12
                    ca_data -= np.mean(ca_data)
                    # ca_diff = np.max(ca_data) - np.min(ca_data) 
                    # if ca_diff != 0:
                    #     ca_data = 100 * ca_data / (np.max(ca_data) - np.min(ca_data))
                    # else:
                    #     ca_data = 100*ca_data/np.max(ca_data)
                    if (start_time - ztime) < -300 or (start_time - ztime) > 240:
                        continue
                    if len(categories) == 1:
                        pcolor = pg.intColor(ncell)
                    else:
                        pcolor = cat_colors[cell_row["Group"].values[0]]
                    px.plot(
                        AR.time_base[startindex:] + (start_time - ztime),
                        ca_data + ncells * 100,
                        title="CC Spontaneous Activity",
                        pen=pcolor,
                    )
                    if not show_cc:
                        pcount += 1
                    # t0 += np.max(AR.time_base) + 1
            if cc_re.match(protocol) and show_cc:
                protopath = Path(expt["rawdatapath"], expt["directory"], cellid, protocol)
                print("matched CC Protocol: ", protocol)
                AR.setProtocol(protopath)
                ok = AR.getData()
                if not ok:
                    continue
                    # px = win.addPlot(title=c)
                # print('Traces shape: ', AR.traces.shape)
                # print("time_base shape: ", AR.time_base.shape)
                for tr in range(AR.traces.shape[0]):
                    start_time = AR.trace_StartTimes[tr]
                    if ztime < 0:
                        ztime = start_time
                    startindex = np.where(AR.time_base > 0.1)[0][0]
                    if len(categories) == 1:
                        pcolor = pg.intColor(ncell)
                    else:
                        pcolor = cat_colors[cell_row["Group"].values[0]]
                    if np.abs(start_time - ztime) > 240:
                        continue
                    px.plot(
                        AR.time_base[startindex:] + (start_time - ztime),
                        AR.traces[tr, startindex:] * 1e3 + ncells * 100,
                        title="CC Spontaneous Activity",
                        pen=pcolor,
                    )
                    # t0 += np.max(AR.time_base) + 1

                pcount += 1
        if pcount > 0:
            ncells += 1
        # if 'VC' in protocol:
        #     print('VC protocol: ', protocol)
        #     data = AR.read_dataset(database, c, protocol)
        #     SP.spont_firing(data, eventpath, eventsummaryfile, map_annotations, databasedict)
        #     print('Spontaneous firing analysis completed for cell: ', c)
        # else:
        #     print('Not a VC protocol')
        # win.nextRow()
        cell_count += 1
        if cell_count % 10 == 0:
            colcount += 1
            rowcount = 0
            # px = win.addPlot(title=c)
    pg.exec()


if __name__ == "__main__":
    print("Working dir: ", Path.cwd())
    main_db, expt = setup(config_file_path=Path(Path.cwd(), "config/experiments.cfg"), dataset="GlyT2_NIHL")

    main(db = main_db, expt=expt, sortby="Group", categories=["Sham", "NE106", "NE115"], # ["B", "A", "AA", "AAA"], 
         celltypes=["tuberculoventral"], show_vc=True, show_cc=True)
