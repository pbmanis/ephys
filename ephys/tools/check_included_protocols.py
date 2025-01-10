import pandas as pd
import analysis_common
import get_config
from ephys.tools.get_configuration import get_configuration
config_file_path = "./config/experiments.cfg"


def check_inclusions(experimentname: str, config_file_path: str):
    """
    """

    datasets, experiments = get_configuration(config_file_path)
    if datasets is None:
        print("Unable to get configuration file from: ", config_file_path)
    if experimentname not in datasets:
        raise ValueError(f"Experiment {experimentname} not found in configuration file")

    experiment = experiments[experimentname]
    return experiment


def gather_protocols(
    experiment: dict,
    protocols: list,
    prots: dict,
    allprots: dict = None,
    day: str = None,
):
    """
    Gather all the protocols and sort by functions/types
    First call will likely have allprots = None, to initialize
    after that will update allprots with new lists of protocols.

    The variable "allprots" is a dictionary that accumulates
    the specific protocols from this cell according to type.
    The type is then used to determine what analysis to perform.

    Parameters
    ----------
    protocols: list
        a list of all protocols that are found for this day/slice/cell
    prots: dict
        data, slice, cell information
    allprots: dict
        dict of all protocols by type in this day/slice/cell
    day : str
        str indicating the top level day for this slice/cell

    Returns
    -------
    allprots : dict
        updated copy of allprots.
    """
    if allprots is None:  # Start with the protocol groups in the configuration file
        protogroups = list(self.experiment["protocols"].keys())
        allprots = {k: [] for k in protogroups}
        # {"maps": [], "stdIVs": [], "CCIV_long": [], "CCIV_posonly": [], "VCIVs": []}
    else:
        protogroups = list(self.experiment["protocols"].keys())
    prox = sorted(list(set(protocols)))  # remove duplicates and sort alphabetically

    for i, protocol in enumerate(prox):  # construct filenames and sort by analysis types
        if len(protocol) == 0:
            continue
        # if a single protocol name has been selected, then this is the filter
        if (
            (self.protocol is not None)
            and (len(self.protocol) > 1)
            and (self.protocol != protocol)
        ):
            continue
        # clean up protocols that have a path ahead of the protocol (can happen when combining datasets in datasummary)
        protocol = Path(protocol).name

        # construct a path to the protocol, starting with the day
        if day is None:
            c = Path(prots["date"], prots["slice_slice"], prots["cell_cell"], protocol)
        else:
            c = Path(day, prots.iloc[i]["slice_slice"], prots.iloc[i]["cell_cell"], protocol)
        c_str = str(c)  # make sure it is serializable for later on with JSON.
        # maps
        this_protocol = protocol[:-4]
        for pg in protogroups:
            pg_prots = self.experiment["protocols"][pg]
            if pg_prots is None:
                continue
            if this_protocol in pg_prots:
                allprots[pg].append(c_str)


    print("Gather_protocols: Found these protocols: ", allprots)
    return allprots



