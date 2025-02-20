# check spont firing VC data sets

import ephys
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

vc_re = re.compile(r"\s*VC_Spont_[\d]{3}")
cc_re = re.compile(r"\s*CC_spont_activity_[\d]{3}", re.IGNORECASE)
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
config_file_path = "./config/experiments.cfg"

datasets, experiments = get_configuration(config_file_path)
dataset = "CBA_Age"
if dataset in list(experiments.keys()):
    database = Path(
        experiments[dataset]["directory"], experiments[dataset]["datasummaryFilename"]
    ).with_suffix(".pkl")
else:
    database = Path(dataset)
databasedict = experiments[dataset]
parentpath = database.parent
print("Database, parentpath: ", database, parentpath)
eventsummaryfile = Path(parentpath, str(parentpath) + "_event_summary.pkl")
eventpath = Path(parentpath, "events")
map_annotationFile = None
if experiments[dataset]["maps"] is not None:
    map_annotationFile = Path(parentpath, experiments[dataset]["maps"])

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
age_cats = expt["age_categories"]
main_db["age_category"] = {}
main_db["age_category"] = main_db.apply(lambda row: CA.categorize_ages(row, age_cats), axis=1)
print("Main database: ", main_db.age_category)


def main(agecat=None):
    AR = DR.acq4_reader.acq4_reader()
    app = pg.mkQApp("sponts")
    win = pg.GraphicsLayoutWidget(show=True, title="sponts")
    win.resize(2000, 1400)
    win.setWindowTitle(f"Spont Firing")
    symbols = ["o", "s", "t", "d", "+", "x"]
    win.setBackground("w")
    cell_count = 0
    rowcount = 0
    colcount = 0
    px = win.addPlot(title=agecat)
    ncells = 0
    nvcs = 0
    for ncell, c in enumerate(main_db["cell_id"]):
        print("Cell_id: ", c)
        # print(main_db.columns)
        # if ncell > 10:
        #     break
        cr = main_db[main_db["cell_id"].str.match(c)]
        if cr["age_category"].values[0] != agecat:
            continue

        all_protocols = cr["data_complete"].values[0]
        protocols = all_protocols.split(",")
        nprots = len(protocols)
        pcount = 0
        t0 = 0
        ztime = -1

        for nprot, protocol in enumerate(protocols):
            protocol = protocol.lstrip()
            if vc_re.match(protocol):
                print("matched VC Protocol: ", protocol)
                protopath = Path(expt["rawdatapath"], expt["directory"], c, protocol)
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
                    ca_data = 100 * ca_data / (np.max(ca_data) - np.min(ca_data))
                    px.plot(
                        AR.time_base[startindex:] + (start_time - ztime),
                        ca_data + ncells * 100,
                        title="CC Spontaneous Activity",
                        pen=pg.intColor(ncell),
                    )
                    # t0 += np.max(AR.time_base) + 1
            if cc_re.match(protocol):
                protopath = Path(expt["rawdatapath"], expt["directory"], c, protocol)
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
                    px.plot(
                        AR.time_base[startindex:] + (start_time - ztime),
                        AR.traces[tr, startindex:] * 1e3 + ncells * 100,
                        title="CC Spontaneous Activity",
                        pen=pg.intColor(ncell),
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
    main("Preweaning")
