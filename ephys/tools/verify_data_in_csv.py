""" Check that all needed data are included in the CSV files.

"""
import datetime
import pprint
from pathlib import Path

import ephys
import ephys.datareaders as DR
import ephys.ephys_analysis as EP
import ephys.tools.categorize_ages as CA
import ephys.tools.check_inclusions_exclusions as CIE
import ephys.tools.filename_tools as FT
import numpy as np
import pandas as pd
import pyqtgraph as pg
from ephys.tools.get_configuration import get_configuration
from pylibrary.tools import cprint as CP

CP = CP.cprint
import re

config_file_path = "./config/experiments.cfg"
r_stats_path = "./R_statistics_summaries"

def get_experiment(dataset):
    datasets, experiments = get_configuration(config_file_path)
    if dataset in list(experiments.keys()):
        database = Path(
            experiments[dataset]["directory"], experiments[dataset]["datasummaryFilename"]
        ).with_suffix(".pkl")
    else:
        database = Path(dataset)
    expt = experiments[dataset]
    return database, expt

def get_latest_csv(measure_name):
    csv_files = list(Path(r_stats_path).glob(f"{measure_name}*.csv"))
    return csv_files

def make_long_cell_id(row):
    df_cell_id = FT.make_cellid_from_slicecell(row.cell_id)
    return df_cell_id

def get_datasummary(experiment):
    datasummaryfile = Path(
        experiment["databasepath"],
        experiment["directory"],
        experiment["datasummaryFilename"],
    )
    if not datasummaryfile.is_file():
        print(
            f"DataSummary file: {datasummaryfile!s} does not yet exist - please generate it first"
        )
        return
    msg = f"DataSummary file: {datasummaryfile!s}  exists"
    msg += f"    Last updated: {datetime.datetime.fromtimestamp(datasummaryfile.stat().st_mtime)!s}"

    datasummary = pd.read_pickle(datasummaryfile)
    return datasummary

def main(dataset):
    db, experiment = get_experiment(dataset)
    inclusion_dict = experiment["includeIVs"]
    exclusion_dict = experiment["excludeIVs"]
    dfs = get_datasummary(experiment)
    verbose = False
    for measure in ["firing_parameters", "rmtau", "spike_shapes"]:
        csv_files = get_latest_csv(measure)
        latest_path = max(csv_files, key=lambda p: p.stat().st_ctime)
        print(f"    Using latest {measure} file: {latest_path}")  
        df = pd.read_csv(latest_path)
        df["long_id"] = {}
        df["long_id"] = df.apply(make_long_cell_id, axis=1)
        # print(df.columns)
        # exit()
        cell_ids = df.long_id.unique()
        for cell_count, cell_id in enumerate(cell_ids):
            all_ivs = list(dfs[dfs.cell_id == cell_id]['data_complete'].values[0].replace(" ", "").split(','))
            print(f"    {cell_id} data complete: {all_ivs}")
            validivs, additional_ivs, additional_iv_records = CIE.include_exclude(
                cell_id=cell_id,
                exclusions=exclusion_dict,
                inclusions=inclusion_dict,
                allivs=all_ivs,
                verbose=verbose,
            )
            csv_protocols = df[df.long_id == cell_id]["protocols"].values[0]
            # print("csv protocols: ", csv_protocols)
            for f in additional_ivs:
                if f not in csv_protocols:
                    CP("r", f" {cell_id} Missing included file {f} in {latest_path} for {measure}")
                else:
                    CP("g", f" {cell_id} Found included file {f} in {latest_path} for {measure}")


def main2(dataset):
    db, experiment = get_experiment(dataset)
    print(f"Checking dataset: {dataset}")
    for measure in ["firing_parameters", "rmtau", "spike_shapes"]:
        print(f"Checking measure: ", measure)
        csv_files = get_latest_csv(measure)
        latest_path = max(csv_files, key=lambda p: p.stat().st_ctime)
        print(f"    Using latest {measure} file: {latest_path}")  
        excludes = list(experiment["excludeIVs"].keys())
        includes = list(experiment["includeIVs"].keys())
        df = pd.read_csv(latest_path)
        df["long_id"] = {}
        df["long_id"] = df.apply(make_long_cell_id, axis=1)
        cells = df.long_id.unique() # all the cells in the data frame.
        # print("cells: ", cells)
        for f in includes:
            print("    Checking includes for: ", f, end=' ')
            if f not in cells:
                CP("r", f" Missing included file {f} in {latest_path}")
                print(f" {experiment['includeIVs'][f]!s}  ")
            else:
                print(f" Found included file {f} in {latest_path}")
        # print("excludes: ", excludes)
        print("\n")

        print("Checking Excludes: ")
        for f in excludes:
            allprots = ' '.join(map(str, experiment["excludeIVs"][f]['protocols']))
            # print("\nallprots: ", allprots)
            # print(" has all? : ", len(re.findall(r"all", allprots, re.IGNORECASE)))
            print("   ", f, end=' ')

            ex_all_prots = len(re.findall(r"all{3}", allprots, re.IGNORECASE)) > 0
            if f in includes:
                print("  !! cell has protocols in Includes")
                continue
            elif f in cells and ex_all_prots:
                print(f"     All excluded for file in {latest_path}")
            elif f in cells and not ex_all_prots and f in includes:
                    # print(f"     **Excluded: {f:s}")
                    # pprint.pp(experiment['excludeIVs'][f], indent=8)
                    print(f"     **Included: {experiment['includeIVs'][f]['protocols']}")
                    # pprint.pp(experiment['includeIVs'][f], indent=8)
            elif f in excludes and f in cells:
                    print(f"     **Excluded protocols: {experiment['excludeIVs'][f]['protocols']}")
                    # pprint.pp(experiment['excludeIVs'][f], indent=8)
            elif f not in cells:
                print(f"     **Not found in csv (previously excluded ?): {experiment['excludeIVs'][f]['protocols']}")
            else:
                print("Logic error")
        # print(df.columns)
        # print(df.head())
        # print("includes: ", includes)
        # print(df.cell_id.unique())

    


if __name__ == "__main__":
    # Load the database
    dataset = "CBA_Age"
    main(dataset)
    