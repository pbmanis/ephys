import argparse
import datetime
import pickle
from pathlib import Path

import pandas as pd
import pylibrary.tools.cprint as cprint

import ephys.datareaders as DR
from ephys.tools import get_configuration

CP = cprint.cprint

get_config = get_configuration.get_configuration


""" extract recording parameters
This reads the datasummary file to find all cells, then 
reads the relevant header information to build a table with the following columns:
    srs = []  # samplling rates
    durations = []  # durations of pulses
    delays = []  # delays to first pulse
    Rs = []  # series resistance from amplifier
    CNeut = []  # capacitance neutralization from amplifier
    important = []  # importance flag.

The purpose is to allow assembly of the data analysis without having access to the raw data
files, which are large and may not be readily available.


Raises
------
ValueError
    _description_
ValueError
    _description_
"""
def main():
    parser = argparse.ArgumentParser(description="Get ancillary data for a given experiment")
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Name of the experiment to assemble (e.g., 'CBA_Age')",
    )
    parser.add_argument("--show", action="store_true", help="Show the recording parameters in the most recent file")
    parser.add_argument("--date", type=str, help="Date of the recording parameters file to show (format: YYYY.MM.DD)")
    args = parser.parse_args()
    experiment_name = args.experiment_name

    config = get_config("./config/experiments.cfg")
    if experiment_name not in config[0]:
        raise ValueError(
            f"Experiment '{experiment_name}' not found in configuration ({config[0]})."
        )

    experiment = config[1][experiment_name]

    if args.show:
        if args.date is None:
            date = datetime.date.today().strftime("%Y.%m.%d")
        else:
            date = datetime.datetime.strptime(args.date, "%Y.%m.%d").date()
        output_fn = Path(experiment["directory"], experiment["databasepath"], experiment_name)
        output_fn = Path(output_fn, f"{experiment_name}_recording_pars_{date}.pkl")
        if not output_fn.exists():
            print(f"No recording parameters file found for {date} at: {output_fn}")
            print("Please run the script without --show to extract recording parameters first.")
        else:
            df_recording_pars = pd.read_pickle(output_fn)
            print("Recording parameters for experiment: ", experiment_name)
            print("for file: ", output_fn)
            pd.set_option("display.max_columns", None)  # Show all columns
            pd.set_option("display.width", None)  # Don't wrap lines
            pd.set_option("display.max_rows", None)  # Show all rows
            n = 20  # Repeat headers every 20 rows
            max_rows = 20
            for i in range(0, max_rows, n):
                print("\n",df_recording_pars.iloc[i : i + n])
        exit()

    # go through the datasummary file and extract the relevant parameters for each cell and protocol.
    datasummary_file = Path(
        experiment["directory"],
        experiment["databasepath"],
        experiment_name,
        experiment["datasummaryFilename"],
    )
    if not datasummary_file.exists():
        raise ValueError(f"Datasummary file not found at: {datasummary_file}")

    df_summary = pd.read_pickle(datasummary_file)

    n_missing = 0

    df_recording_pars = pd.DataFrame(
        columns=[
            "cell_id",
            "protocol",
            "sample_rate",
            "duration",
            "delay",
            "Rs",
            "CNeut",
            "important",
        ]
    )

    for irow, df_cell in df_summary.iterrows():
        cellpath = Path(experiment["directory"], experiment["rawdatapath"], df_cell["cell_id"])
        # print(cellpath, cellpath.exists())
        if not cellpath.exists():
            n_missing += 1
        else:
            protocol_list = [p.lstrip() for p in df_cell["data_complete"].split(",") if p.strip()]
            print("protocol list: ", protocol_list)
            day_slice_cell = str(Path(df_cell.date, df_cell.slice_slice, df_cell.cell_cell))
            for protocol in protocol_list:
                if protocol.startswith("CC-CapBridgeTune"):
                    continue
                fullcellpath = Path(cellpath, protocol)
                with DR.acq4_reader.acq4_reader(fullcellpath, "MultiClamp1.ma") as AR:

                    try:
                        dataok = AR.getData(
                        fullcellpath, allow_partial=True, record_list=[0]
                    )  # just get the first record.
                    except:
                        CP('r', f"??? {fullcellpath}, cannot read data, skipping")
                        continue
                    if not dataok:
                        CP('r', f"??? {fullcellpath}, cannot read data, skipping")
                        continue
                    index_info = {
                        "cell_id": df_cell["cell_id"],
                        "protocol": protocol,
                        "sample_rate": AR.sample_rate[0],
                        "duration": AR.tend - AR.tstart,
                        "delay": AR.tstart,
                        "Rs": AR.CCComp["CCBridgeResistance"],
                        "CNeut": AR.CCComp["CCNeutralizationCap"],
                        "important": AR.checkProtocolImportant(fullcellpath),
                    }

                    CP("g", f"    Protocol {protocol:s} has sample rate of {index_info['sample_rate']:e}")
                    df_recording_pars = df_recording_pars._append(index_info, ignore_index=True)
    print(f"Total cells: {len(df_summary)}, Missing cells: {n_missing}")
    print(df_recording_pars.head(20))
    
    today = datetime.date.today().strftime("%Y-%m-%d")
    output_fn = Path(experiment["directory"], experiment["databasepath"], experiment_name)
    output_fn = Path(output_fn, f"{experiment_name}_recording_pars_{today}.pkl")
    df_recording_pars.to_pickle(output_fn)
    print(f"Recording parameters saved to: {output_fn}")
    # if "assembled_filename" not in experiment:
    #     raise ValueError(
    #         f"Experiment '{experiment_name}' does not have an 'assembled_filename' defined."
    #     )

    # assembled_fn = Path(experiment["directory"]) / experiment["assembled_filename"]

    # if assembled_fn.exists():
    #     print(f"Assembled file already exists at: {assembled_fn}")
    #     print("Skipping assembly to avoid overwriting existing file.")
    #     print("If you want to reassemble, please delete the existing file first.")
    # else:
    #     print(
    #         f"Assembled file does not exist. Proceeding with assembly for experiment: {experiment_name}"
    #     )
        # Call the function to assemble datasets here, passing the necessary parameters
        # For example: assemble_datasets(experiment)

        # After assembling, you would typically save the assembled data to 'assembled_fn'
        # For example: save_assembled_data(assembled_data, assembled_fn)

        # print(f"Assembly complete. Assembled data would be saved to: {assembled_fn}")

if __name__ == "__main__":
    main()
