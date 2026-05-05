# check inclusions/exclusions in the configuration file
# Logic:
# The exclusion dictionary consists of entries as follows:
#        2024.03.28_000/slice_000/cell_000:
#            protocols: ["CCIV_1nA_max_1s_pulse_000"]
#            reason: "Broken protocol"
#
#     This means that the CCIV_1nA_max_1s_pulse_000 protocol for the cell 2024.03.28_000/slice_000/cell_000
#     is excluded from analysis because it was a broken protocol.
#     There are also entries such as:
#       2024.03.13_000/slice_000/cell_000:
#           protocols: ["all"]
#           reason: "Bad Cell"
#     Setting protocols to "all" (case insensitive) means that all protocols for this cell are excluded from analysis.
#
#     Setting protocols to "except" (case insensitive) means that all protocols for this cell are excluded from analysis
#     with the exception of those listed in the "exceptions" dictionary.
#
# The inclusion dictionary consists of entries as follows::
#       2023.01.26_000/slice_000/cell_001:
#            protocols: ["CCIV_1nA_max_1s_pulse_000"]
#            records:  [[x for x in range(0, 20)]]
#            reason: "These sweeps are ok to include"

#       Here, the protocols in the 'protocols' list are included in the analysis.
#       The records list is a list of lists, where each list is a list of sweep numbers to include in the analysis.
#       The reason is a string that is included in the output to explain why these sweeps are included.
#       The inclusion is used to include data where the protocol was stopped early, or where there
#       are 'bad' records that do not influence the overall analysis. An example is doing a current clamp
#       protocol with large negative current steps, but starting with the positive steps. The negative steps
#       may lead to cell damage or loss, but the positive steps are still useful for spike counting.
#
# The return from include_exclude is a list of valid protocols for the cell, exclusive of those that are excluded.

import datetime
from pathlib import Path
from typing import Union
import pandas as pd
from pylibrary.tools import cprint as CP


def include_exclude(
    cell_id: str, exclusions: dict, inclusions: dict, allivs: list, verbose: bool = False,
    inclusions_flag: bool = True, exclusions_flag: bool = True,
):
    """
    Get the included and remove the excluded data from the list of IVs.
    """
    validivs = check_exclusions(cell_id, exclusions,
                                allivs, verbose=verbose, exclusions_flag=exclusions_flag)
    additional_ivs, additional_iv_records = check_inclusions(
        cell_id, inclusions, verbose=verbose, inclusions_flag=inclusions_flag)
    if verbose or (exclusions_flag and cell_id in exclusions) or (inclusions_flag and cell_id in inclusions):
        print(f"     Valid IVs after exclusions: {validivs}")
        print(f"     Additional IVs from inclusions: {additional_ivs}")
        print(f"     Additional IV records from inclusions: {additional_iv_records}")
    return validivs, additional_ivs, additional_iv_records


def check_exclusions(cell_id: str, exclusions: dict, allivs: list, 
                     verbose:bool=False, exclusions_flag:bool=True) -> list:
    validivs = []
    if exclusions is not None:
        # handle shortcut cases first:
        day_id = Path(cell_id).parts[0]
        slice_id = str(Path(*Path(cell_id).parts[:2]))  # must make a string again.
        if slice_id in list(exclusions.keys()):
            if verbose or exclusions:
                CP.cprint("y", f"\nCell(sliceid level): {cell_id}: {slice_id} is completely excluded")
            return []  # exclude everyting for this slice.
        if day_id in list(exclusions.keys()):
            if verbose or exclusions_flag:
                CP.cprint("y", f"\nCell(day level): {cell_id}: {day_id} is completely excluded")
            return []  # exclude everyting for this day.
        # The rest of the exclusions are are on cell by cell and protocol basis
        if cell_id in list(exclusions.keys()):
            if (verbose or exclusions_flag):
                CP.cprint("y", f"\nCell (cell_level): {cell_id} has excluded protocols: {exclusions[cell_id]['protocols']}")
            if exclusions[cell_id]["protocols"] in ["all", "All", ["all"], ["All"]]:
                msg = f"       All protocols for {cell_id} are excluded from analysis in the configuration file."
                CP.cprint("r", msg)
                return []  # nothing to do - entire cell is excluded.
            # exclude all protocols except those in the "exceptions" key
            elif exclusions[cell_id]["protocols"] in ["except", ["except"]]:
                if (
                    "exceptions" not in exclusions[cell_id].keys()
                ):  # confirm that there is an exceptions key
                    raise ValueError(
                        f"Configuration file error: No 'exceptions' key in the 'except' exclusion (cell_id = {cell_id})"
                    )
                # print("   ... Appending excepted protocols from 'except': ")
                for protocol in  exclusions[cell_id]["exceptions"]:
                    if Path(protocol).name in exclusions[cell_id]["exceptions"]:
                        if verbose or exclusions_flag:
                            CP.cprint("y", f"     adding excepted protocol: {protocol}")
                        if protocol not in validivs:
                            validivs.append(protocol)
            else:
                # print("   ... Appending valid protocols: ")
                # print("   all ivs is : ", allivs)
                for protocol in allivs:
                    # print("protocol: ", protocol)
                    # print("    protocol name: ", Path(protocol).name)
                    # print("    exclusions: ", exclusions[cell_id]["protocols"])
                    if Path(protocol).name not in exclusions[cell_id]["protocols"]:
                        # print("    adding valid protocol: ", protocol)
                        if protocol not in validivs:
                            validivs.append(protocol)
                # print("     After appending, validivs now is: ", validivs)
                # validivs.append([protocol for protocol in allivs if Path(protocol).name not in exclusions[cell_id]['protocols']])
        else:
            if verbose or exclusions_flag:
                CP.cprint("g",f"\nCell: {cell_id}: No exclusions")
            validivs = allivs
        # print("after Exclusions, validivs is: ", validivs)
    else:
        if verbose or exclusions_flag:
            CP.cprint("g",f"\nCell: {cell_id}: No exclusions")
        validivs = allivs
    return validivs


def check_inclusions(cell_id: str, inclusions: dict, verbose: bool = False, inclusions_flag: bool = True,
                     ):
    additional_ivs = []
    additional_iv_records = None
    if inclusions is not None:
        # Add ivs that are in the inclusions list. These may include data that is from
        # protocols that did not run to completion, but which might still have useful data
        # We also add the records for these ivs, so that we can select the appropriate sweeps
        if cell_id in inclusions:
            if verbose or inclusions_flag:
                CP.cprint("g",f"\nCell: {cell_id}")

            for iprot, protocol in enumerate(inclusions[cell_id]["protocols"]):
                # protopath = str(Path(cell_id, protocol))
                additional_ivs.append(protocol) # protopath)
                if additional_iv_records is None:
                    additional_iv_records = {
                        protocol: [cell_id, inclusions[cell_id]["records"][iprot]]
                    }
                else:
                    additional_iv_records[protocol] = [
                        cell_id,
                        inclusions[cell_id]["records"][iprot],
                    ]
                if verbose or inclusions_flag:
                    CP.cprint("m", f"       has included protocols: {additional_ivs}")
                    CP.cprint("m", f"       with records: {additional_iv_records}")
            # for iaiv, aiv in enumerate(additional_ivs):
            #     if additional_ivs[iaiv] not in validivs:
            #         validivs.append(additional_ivs[iaiv])
    return additional_ivs, additional_iv_records


def list_inclusions_exclusions(df, exclusion_dict, inclusion_dict, all_ivs:list, verbose:bool=False,
                               inclusions_flag:bool=True, exclusions_flag:bool=True, find_cell:Union[str, None]=None):
    """
    List the inclusions and exclusions for each cell in the DataFrame
    """
    cell_ids = df.cell_id.unique()
    if find_cell is not None and find_cell not in cell_ids:
        print(f"Cell {find_cell} not found in the DataFrame.")
        return
    for cell_count, cell_id in enumerate(cell_ids):
        if find_cell is not None and cell_id != find_cell:
            continue
        all_ivs = list(df[df.cell_id == cell_id]['data_complete'].values[0].replace(" ", "").split(','))
        validivs, additional_ivs, additional_iv_records = include_exclude(
            cell_id=cell_id,
            exclusions=exclusion_dict,
            inclusions=inclusion_dict,
            allivs=all_ivs,
            verbose=verbose,
            inclusions_flag=inclusions_flag,
            exclusions_flag=exclusions_flag,
        )
        # if cell_id == "2024.04.23_000/slice_001/cell_000":
        # if verbose:
        #     print("*" * 80)
        #     print(f"Cell: {cell_id}")
        #     print(f"     Valid IVs: {validivs}")
        #     print(f"     Additional IVs: {additional_ivs}")
        #     print(f"           Additional IV Records: {additional_iv_records}")
        #     print("*" * 80)
        #     print("\n\n")

    if verbose:
        print(exclusion_dict.keys())
        print(inclusion_dict.keys())
    if not verbose and exclusions_flag:
        if find_cell is not None and find_cell not in exclusion_dict.keys():
            print(f"\nCell {find_cell} not found in the exclusion dictionary.")
        print(f"\nExclusions: {len(exclusion_dict.keys())} cells with exclusions")
        print(f"     {len(exclusion_dict.keys())} cells with some protocol exclusions")
    if not verbose and inclusions_flag:
        if find_cell is not None and find_cell not in inclusion_dict.keys():
            print(f"\nCell {find_cell} not found in the inclusion dictionary.")
        print(f"\nInclusions: {len(inclusion_dict.keys())} cells with inclusions")
        print(f"     {len(inclusion_dict.keys())} cells with some protocol inclusions")


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

def main():
    import argparse
    from ephys.tools.get_configuration import get_configuration
    import datetime

    parser = argparse.ArgumentParser(description="Check inclusions and exclusions in the configuration file")
    parser.add_argument("-c", "--configfile", type=str, default="config/experiments.cfg", help="Path to the configuration file")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="Dataset to check (default: all datasets in the configuration file)")
    parser.add_argument("-e", "--exclude", action="store_true", help="Check exclusions")
    parser.add_argument("-i", "--include", action="store_true", help="Check inclusions")
    parser.add_argument("-b",  action="store_true", help="check inclusions and exclusions")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("-f", "--find", type=str, default=None, help="Find the cell and report its inclusions and exclusions")

    args = parser.parse_args()
    configfile = args.configfile
    verbose = args.verbose
    inclusions_flag = args.include
    exclusions_flag = args.exclude
    check_both = args.b
    if check_both:
        inclusions_flag = True
        exclusions_flag = True
    find_cell = args.find
    datasets, experiments = get_configuration(configfile)
    if args.dataset is not None:
        if args.dataset in datasets:
            datasets = [args.dataset]
        else:
            print(f"Dataset {args.dataset} not found in configuration file. Checking all datasets in experiment.cfg.")
    for dataset in datasets:
        experiment = experiments[dataset]
        inclusion_dict = experiment["includeIVs"]
        exclusion_dict = experiment["excludeIVs"]
        datasummary = get_datasummary(experiment)
        all_ivs = []
        print(f"\n{'='*80}\nDataset: {dataset}\n{'='*80}")
        list_inclusions_exclusions(datasummary, exclusion_dict, inclusion_dict, all_ivs, verbose=verbose,
                                   inclusions_flag=inclusions_flag, exclusions_flag=exclusions_flag, find_cell=find_cell)

if __name__ == "__main__":
    main()
