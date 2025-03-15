import concurrent.futures
import datetime
from pathlib import Path
from typing import Optional

import dateutil.parser as DUP
import numpy as np
import pandas as pd
from pylibrary.tools import cprint

from ephys.gui import data_table_functions
from ephys.tools import filter_data
from ephys.tools import functions as FUNCS
from ephys.tools.get_computer import get_computer

CP = cprint.cprint
FUNCS = data_table_functions.Functions()

def make_datetime_date(row, colname="date"):
    if colname == "date" and "Date" in row.keys():
        colname = "Date"
    if pd.isnull(row[colname]) or row[colname] == "nan":
        row.shortdate = 0
        return row.shortdate

    date = str(Path(row[colname]).name)
    date = date.split("_", maxsplit=1)[0]
    shortdate = datetime.datetime.strptime(date, "%Y.%m.%d")
    shortdate = datetime.datetime.timestamp(shortdate)
    st = datetime.datetime.timestamp(
        datetime.datetime.strptime("1970-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
    )
    row.shortdate = shortdate - st
    if pd.isnull(row.shortdate):
        raise ValueError("row.shortdate is null ... in make_datetime_date")

    return row.shortdate


def make_cell_id(row):
    sliceno = int(row["slice_slice"][-3:])
    cellno = int(row["cell_cell"][-3:])
    cell_id = f"{row['date']:s}_S{sliceno:d}C{cellno:d}"
    row["cell_id"] = cell_id
    return row

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


class AssembleDatasets:
    def __init__(self, status_bar: object = None):
        self.status_bar = status_bar  # get the status bar so we can report progress
        self.experiment = None

    def _data_complete_to_series(self, row):
        dc = row.data_complete.split(",")
        dc = [p.strip(" ") for p in dc if p != "nan" and "CCIV".casefold() in p.casefold()]
        # print("\ndc: ", dc)
        row.protocol = pd.Series(dc)
        # print(row.date, row.data_complete.values)
        return row

    def get_assembled_filename(self, experiment):
        """get_assembled_filename Create the filename for the assembled FI data.

        Parameters
        ----------
        experiment : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        assembled_fn = Path(
            Path(
                experiment["analyzeddatapath"],
                experiment["directory"],
                experiment["assembled_filename"],
            )
        )
        return assembled_fn

    def assemble_datasets(
        self,
        df_summary: pd.DataFrame,
        fn: str = "",
        experiment: dict = None,
        exclude_unimportant: bool = False,
    ):
        """assemble_datasets : Assemble the datasets from the summary and coding files,
        then combine FI curves (selected) in IV protocols for each cell.
        We also calculate some spike rate measures


        Parameters
        ----------
        df_summary : pd.DataFrame
            _description_
        coding_file : Optional[str], optional
            _description_, by default None
        coding_sheet : Optional[str], optional
            _description_, by default None
        coding_level : Optional[str], optional
            _description_, by default None
        coding_name : Optional[str], optional
            _description_, by default "Group
        exclude_unimportant : bool, optional
            _description_, by default False
        fn : str, optional
            _description_, by default ""

        """
        if experiment is None and self.experiment is None:
            raise ValueError("Experiment not defined in AssembleData; should be done at time of the call")
        self.experiment = experiment
        coding_file = experiment["coding_file"]
        coding_sheet = experiment["coding_sheet"]
        coding_level = experiment["coding_level"]
        coding_name = experiment["coding_name"]


        print(
            f"Assembling:\n  coding file: {coding_file!s}\n    Cells: {self.experiment['celltypes']!s}"
        )
        df = self.combine_summary_and_coding(
            df_summary=df_summary,
            coding_file=coding_file,
            coding_sheet=coding_sheet,
            coding_level=coding_level,
            coding_name=coding_name,
            exclude_unimportant=exclude_unimportant,
            status_bar=self.status_bar,
        )
        if "protocol" not in df.columns:
            df["protocol"] = ""
        df = df.apply(self._data_complete_to_series, axis=1)
        print(len(df), " rows after data complete to series")

        # now make a new dataframe that has a separate row for each protocol
        df = df.explode(["protocol"], ignore_index=True)
        print("Number of protocols after explode", len(df))
        df = df.dropna(subset=["protocol"])
        print("Number of protocols after dropna", len(df))

        df_null = df[df["cell_id"].isnull()]
        print("Null columns: ", df_null)
        df = df.dropna(subset=["cell_id"])
        print("# of protocols with ID: ", len(df))
        protostrings = "|".join(list(self.experiment["protocols"]["CCIV"].keys()))
        print("protostrings: ", protostrings)
        print("Protocols: ", df["protocol"].unique())
        print(df.head())
        # return
        df = self.combine_by_cell(df)
        print("\nWriting assembled data to : ", fn)
        print(df.head())
        print("Assembled groups: dataframe Groups: ", df.Group.unique())
        df.to_pickle(fn, compression="gzip")

    def categorize_ages(self, row):
        row.age = numeric_age(row)
        for k in self.experiment["age_categories"].keys():
            if (
                row.age >= self.experiment["age_categories"][k][0]
                and row.age <= self.experiment["age_categories"][k][1]
            ):
                row.age_category = k
        return row.age_category

    def clean_sex_column(self, row):
        if row.sex not in ["F", "M"]:
            row.sex = "U"
        return row.sex

    def combine_summary_and_coding(
        self,
        # excelsheet,
        # adddata=None,
        df_summary: pd.DataFrame,
        coding_file: Optional[str] = None,
        coding_sheet: Optional[str] = "Sheet1",
        coding_level: Optional[str] = "date",
        coding_name: Optional[str] = "Group",
        exclude_unimportant=False,
        status_bar=None,
    ):
        """combine_summary_and_coding: combine the summary data with the coding file

        Parameters
        ----------
        excelsheet : string or Path
            excel filename to read to get the data to plot. This is the sheet generated by process_spike_info
        adddata : _type_, optional
            analysis result data to merge with main excel sheet, by default None
        coding_file: string or Path, optional
            excel file with coding information (full date, plus Group and sex columns as needed),
            by default None.
            Do not specify this if the Group column in the excelsheet already has valid groups,
            and the sex is already specified as well.
        analysis_cell_types : list, optional
            a list of cell type names, to specify which cell types will be analyzed, by default []

        Returns
        -------
        _type_
            _description_
        """
        CP("r", "\nReading intermediate result files")

        # if Path(excelsheet).suffix == ".pkl":  # need to respecify as excel sheet
        #     excelsheet = Path(excelsheet).with_suffix(".xlsx")
        # print(f"    Excelsheet (from process_spike_info): {excelsheet!s}")
        # print("     coding_level: ", coding_level)
        # df = pd.read_excel(excelsheet)
        print("    # entries in summary sheet: ", len(df_summary))
        if "cell_id" not in list(df_summary.columns):
            df_summary.apply(make_cell_id, axis=1)

        # print(f"    Adddata in read_intermediate_result_files: {adddata!s}")
        if coding_file is not None:  # add coding from the coding file
            coding_filename = Path(
            Path(
                self.experiment["analyzeddatapath"],
                self.experiment["directory"],
                self.experiment["coding_file"],
            )
        )
            df = self.read_coding_file(df_summary, coding_filename, coding_sheet, coding_level)
            print(
                "coding file: ",
                coding_filename,
                " sheet: ",
                coding_sheet,
                " level: ",
                coding_level,
                "coding_name: ",
                coding_name,
                "Group: ",
                df.Group.unique(),
            )
            print("coding data: ", df.columns)
            print("Groups from coding file: ", df[coding_name].unique())

        else:
            df = df_summary
            df["Group"] = "Control"

        # raise ValueError("Need to fix the coding file reading")
        FD = filter_data.FilterDataset(df, self.experiment["junction_potential"])
        if "remove_expression" not in self.experiment.keys():
            self.experiment["remove_expression"] = []
        df = FD.filter_data_entries(
            df,
            remove_groups=self.experiment["remove_groups"],
            remove_expression=self.experiment["remove_expression"],
            excludeIVs=self.experiment["excludeIVs"],
            exclude_internals=["cesium", "Cesium"],
            exclude_temperatures=["25C", "room temp"],
            exclude_unimportant=exclude_unimportant,
            verbose=True,
        )

        CP("m", "Finished reading files\n")
        if status_bar is not None:
            status_bar("Assembling Datasets: Finished reading files")
        print("df.Groups: ", df.Group)
        return df

    def combine_by_cell(self, df, valid_protocols=None):
        """
        Rules for combining cells and pulling the data from the original analysis:
        1. Combine data from cells with the same ID
        2. Check the cell name and whether it fits the S00C00 or S1C1 format.
        3. When getting spike parameters, use a logical set of restrictions:
            a. Use only the first spike at the lowest current level that evoke spikes
                for AP HW, AP_thr_V, AP15Rate, AdaptRatio, AHP_trough_V, AHP_depth_V, AHP_trough_T
                This is in ['spikes']["LowestCurrentSpike"]

            b. Do not use traces that are above the spike firing rate turnover point (non-monotonic)
            c. compute the Adaptation Index (Manis et al., 2019, PLoS One) for a selected firing rate
                range. (e.g., 20-40 Hz)
            c. compute Adaptation Index by eFEL method (all across train; same limited firing range)
               Try using Allen Institute version to catch values for adaptation_index instead of eFEL version.


        """
        CP("y", "Combine by cell")

        df = df.apply(make_cell_id, axis=1)
        df.dropna(subset=["cell_id"], inplace=True)
        df.rename(columns={"sex_x": "sex"}, inplace=True)
        if self.experiment["celltypes"] != ["all"]:
            df = df[df.cell_type.isin(self.experiment["celltypes"])]
        df["shortdate"] = df.apply(
            make_datetime_date, colname="date", axis=1
        )  # make a short date as a datetime for sorting
        after = "2000.01.01"
        after_parsed = datetime.datetime.timestamp(DUP.parse(after))
        after_parsedts = after_parsed
        df = df[df["shortdate"] >= after_parsedts]
        cell_list = list(set(df.cell_id))
        cell_list = sorted(cell_list)
        dfdict = {}  # {col: [] for col in cols}
        df_new = pd.DataFrame.from_dict(dfdict)
        computer_name = get_computer()
        nworkers = self.experiment["NWORKERS"][computer_name]
        cells_to_do = [cell for cell in cell_list if cell is not None]

        # here we should check to see if cell has been done in the current file,
        # and remove it from the list.
        combined_file = Path(
            self.experiment["analyzeddatapath"],
            self.experiment["directory"],
            self.experiment["assembled_filename"],
        )
        # first be sure that we even have a combined file!
        if combined_file.is_file():
            print("Combined File exists: ", combined_file)
            try:
                already_done = pd.read_pickle(
                    Path(
                        self.experiment["analyzeddatapath"],
                        self.experiment["directory"],
                        self.experiment["assembled_filename"],
                    ),
                    compression="gzip",
                )
            except:
                already_done = pd.read_pickle(
                    Path(
                        self.experiment["analyzeddatapath"],
                        self.experiment["directory"],
                        self.experiment["assembled_filename"],
                    )
                )  # try without compression
            already_done = already_done.cell_id.unique()
        else:
            already_done = []
        # cells_to_do = [cell for cell in cells_to_do if cell not in already_done]
        # instrument up to to a limited set of the data for testing
        # The limit numbers refer to the IV data table.
        ilimit = None  # list(range(67, 78))
        tasks = 1
        if ilimit is None:
            limit = len(cells_to_do)
            tasks = range(limit)
        elif isinstance(ilimit, int):
            limit = min(ilimit, len(cells_to_do))
            tasks = range(limit)
        elif isinstance(ilimit, list):
            limit = ilimit
            tasks = limit

        result = [None] * len(tasks)
        results = dfdict
        parallel = True
        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
                results = executor.map(
                    FUNCS.compute_FI_Fits,
                    [self.experiment] * len(tasks),
                    [df] * len(tasks),
                    [cell_list[i] for i in tasks],
                    [self.experiment["FI_protocols"]] * len(tasks),
                )
                for i, result in enumerate(results):
                    if result is not None:
                        df_new = pd.concat(
                            [df_new, pd.Series(result).to_frame().T], ignore_index=True
                        )
            # with MP.Parallelize(enumerate(tasks), results=results, workers=nworkers) as tasker:
            #     for i, x in tasker:
            #         result = FUNCS.compute_FI_Fits(
            #             self.experiment, df, cell_list[i], protodurs=self.experiment["FI_protocols"]
            #         )
            #         tasker.results[cell_list[i]] = result

            # for r in results:
            #     df_new = pd.concat([df_new, pd.Series(results[r]).to_frame().T], ignore_index=True)
        else:

            # do each cell in the database
            for icell, cell in enumerate(cell_list):
                print("icell: ", icell)
                if not isinstance(ilimit, list):
                    if ilimit is not None and icell > ilimit:
                        break
                elif isinstance(ilimit, list):
                    if icell not in ilimit:
                        continue
                if cell is None:
                    CP("r", f"    Cell # {icell:d} in the database is None")
                    continue
                CP(
                    "c", f"    Computing FI_Fits for cell: {cell:s}"
                )  # df[df.cell_id==cell].cell_type)
                datadict = FUNCS.compute_FI_Fits(
                    self.experiment, df, cell, protodurs=self.experiment["FI_protocols"]
                )
                if datadict is None:
                    print("    datadict is none for cell: ", cell)
                    continue
                df_new = pd.concat([df_new, pd.Series(datadict).to_frame().T], ignore_index=True)
        return df_new

    def check_coding_file(self, df_coding):
        """check_coding_file Verify that the coding sheet has appropriate column names.


        Parameters
        ----------
        df_coding: Pandas DataFrame
            pd dataframe representation of the coding file as read by pd.read_excel.

        """
        cols = df_coding.columns
        if "date" not in cols:
            raise ValueError("Coding file must have a 'date' column")
        if "Group" not in cols:
            raise ValueError("Coding file must have a 'Group' column")
        if "Subject" not in cols:
            raise ValueError("Coding file must have a 'Subject' column")
        return
    

    def read_coding_file(self, df, coding_file, coding_sheet, level="date"):
        df_coding = pd.read_excel(coding_file, sheet_name=coding_sheet)
        self.check_coding_file(df_coding)
        print("Coding file head: \n", df_coding.head())
        for index in df.index:
            row = df.loc[index]
            if pd.isnull(row.date):
                continue
            if "coding_name" in self.experiment.keys():
                coding_name = self.experiment["coding_name"]
            else:
                coding_name = "Group"
            # print(row.date, df_coding.date.values)
            print("date in the date values: ", row.date, row.date in df_coding.date.values)
            # Here we apply what is in the CODING file to the combined file.
            if row.date in df_coding.date.values:
                if "sex" in df_coding.columns:  # update sex? Should be in main table.
                    df.loc[index, "sex"] = (
                        df_coding[df_coding.date == row.date].sex.astype(str).values[0]
                    )
                if "cell_expression" in df_coding.columns:
                    df.loc[index, "cell_expression"] = (
                        df_coding[df_coding.date == row.date].cell_expression.astype(str).values[0]
                    )

                # how to assign groups: by date or subject?
                print("Level: ", level.lower())
                if level.casefold() == "date".casefold():
                    print("row.date: ", row.date)
                    df.loc[index, "Group"] = (
                        df_coding[df_coding.date == row.date][coding_name].astype(str).values[0]
                    )
                    if df.loc[index, "Group"] == np.nan:
                        print(
                            "     df.loc[index, 'Group']: ",
                            df.loc[index, "Group"],
                            "is Nan, but wanted: ",
                            row[coding_name],
                            "from coding file column: ",
                            coding_name,
                        )
                elif level.casefold() == "subject".casefold():
                    mask = df_coding.subject == row.subject
                    df.loc[index, "Group"] = df_coding[mask][coding_name].astype(str).values[0]
                elif level.casefold() == "slice".casefold:
                    mask = (df_coding.date == row.date) & (df_coding.slice_slice == row.slice_slice)
                    df.loc[index, "Group"] = df_coding[mask][coding_name].astype(str).values[0]
                elif level.casefold() == "cell".casefold:
                    mask = (
                        (df_coding.date == row.date)
                        & (df_coding.slice_slice == row.slice_slice)
                        & (df_coding.cell_cell == row.cell_cell)
                    )
                    print("mask: ", mask)
                    print("df_coding.date: ", row.date)
                    print("df_coding.slice_slice: ", row.slice_slice)
                    print("df_coding.cell_cell: ", row.cell_cell)
                    print("coding name: ", coding_name)
                    print("Mask: ", df_coding[mask][coding_name].astype(str))
                    df.loc[index, "Group"] = df_coding[mask][coding_name].astype(str).values[0]
            else:
                # print("Assigning nan to : ", df.loc[index].cell_id)
                df.loc[index, "Group"] = np.nan
        return df
