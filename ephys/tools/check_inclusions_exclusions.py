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

from pathlib import Path
from pylibrary.tools import cprint as CP


def include_exclude(
    cell_id: str, exclusions: dict, inclusions: dict, allivs: list, verbose: bool = False
):
    """
    Get the included and remove the excluded data from the list of IVs.
    """
    validivs = check_exclusions(cell_id, exclusions, allivs, verbose=verbose)
    additional_ivs, additional_iv_records = check_inclusions(
        cell_id, inclusions, verbose=verbose)
    
    return validivs, additional_ivs, additional_iv_records


def check_exclusions(cell_id: str, exclusions: dict, allivs: list, 
                     verbose:bool=False) -> list:
    validivs = []
    if exclusions is not None:
        # handle shortcut cases first:
        day_id = Path(cell_id).parts[0]
        slice_id = str(Path(*Path(cell_id).parts[:2]))  # must make a string again.
        if slice_id in list(exclusions.keys()):
            if verbose:
                print(f"\nCell: {cell_id}")
                print(f"    Slice {slice_id} is completely excluded")
            return []  # exclude everyting for this slice.
        if day_id in exclusions.keys():
            if verbose:
                print(f"\nCell: {cell_id}")
                print(f"    Day {day_id} is completely excluded")
            return []  # exclude everyting for this day.
        # The rest of the exclusions are are on cell by cell and protocol basis
        if cell_id in exclusions.keys():
            if verbose:
                print(f"\nCell: {cell_id}")
                print("     has excluded protocols: ", exclusions[cell_id]["protocols"])
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
                        if verbose:
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
            if verbose:
                print(f"\nCell: {cell_id}")
                print("    no exclusions")
            validivs = allivs
        # print("after Exclusions, validivs is: ", validivs)
    else:
        if verbose:
            print("cell_inclusion_exclusion: check_exclusio:: No exclusions for this cell: ", cell_id)
        validivs = allivs
    return validivs


def check_inclusions(cell_id: str, inclusions: dict, verbose: bool = False):
    additional_ivs = []
    additional_iv_records = None
    if inclusions is not None:
        # Add ivs that are in the inclusions list. These may include data that is from
        # protocols that did not run to completion, but which might still have useful data
        # We also add the records for these ivs, so that we can select the appropriate sweeps
        if cell_id in inclusions:
            if verbose:
                print(f"\nCell: {cell_id}")

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
                if verbose:
                    print("       has included protocols: ", additional_ivs)
                    print(
                    "       with records: ",
                    additional_iv_records,
                )
            # for iaiv, aiv in enumerate(additional_ivs):
            #     if additional_ivs[iaiv] not in validivs:
            #         validivs.append(additional_ivs[iaiv])
    return additional_ivs, additional_iv_records


def list_inclusions_exclusions(df, exclusion_dict, inclusion_dict, all_ivs:list, verbose:bool=False):
    """
    List the inclusions and exclusions for each cell in the DataFrame
    """
    cell_ids = df.cell_id.unique()
    for cell_count, cell_id in enumerate(cell_ids):
        all_ivs = list(df[df.cell_id == cell_id]['data_complete'].values[0].replace(" ", "").split(','))
        validivs, additional_ivs, additional_iv_records = include_exclude(
            cell_id=cell_id,
            exclusions=exclusion_dict,
            inclusions=inclusion_dict,
            allivs=all_ivs,
            verbose=verbose,
        )
        # if cell_id == "2024.04.23_000/slice_001/cell_000":
        if verbose:
            print("*" * 80)
            print(f"Cell: {cell_id}")
            print(f"     Valid IVs: {validivs}")
            print(f"     Additional IVs: {additional_ivs}")
            print(f"           Additional IV Records: {additional_iv_records}")
            print("*" * 80)
            print("\n\n")

    if verbose:
        print(exclusion_dict.keys())


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


if __name__ == "__main__":
    from ephys.tools.get_configuration import get_configuration
    import datetime
    import pandas as pd

    configfile = "/Users/pbmanis/Desktop/Python/mrk-nf107/config/experiments.cfg"
    datasets, experiments = get_configuration(configfile)
    experiment = experiments["NF107Ai32_NIHL"]
    inclusion_dict = experiment["includeIVs"]
    exclusion_dict = experiment["excludeIVs"]
    datasummary = get_datasummary(experiment)
    all_ivs = []
    list_inclusions_exclusions(datasummary, exclusion_dict, inclusion_dict, all_ivs, verbose=True)
