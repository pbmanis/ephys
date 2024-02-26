from pathlib import Path
import pyqtgraph as pg
import pprint
import inspect # to get path to standard_configurations files
import ephys
from pylibrary.tools import cprint
CP = cprint.cprint

PP = pprint.PrettyPrinter(indent=4)


def get_configuration(configfile: str = "experiments.cfg"):
    """get_configuration : retrieve the configuration file from the current
    working directory, and return the datasets and experiments
    Note: if some keys are missing, we will attempt to get them from the
    default configuration in the ephys.config.standard_configurations file,
    unless the main configuration has a key to pull from another "standard" configuration
    in the ephys.config directory.

    Parameters
    ----------
    configfile : str, optional
        _description_, by default "experiments.cfg"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """
    try:
        abspath = Path().absolute()
        if abspath.name == 'nb':
            abspath = Path(*abspath.parts[:-1])
        print("Getting Configuration file from: ", Path().absolute())
        cpath = Path(Path().absolute(), "config", configfile)
        config = pg.configfile.readConfigFile(cpath)
        experiments = config["experiments"]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No config file found, expected in the top-level config directory, named '{cpath!s}'"
        ) from exc

    datasets = list(experiments.keys())
    retrieve_standard_values(experiments, datasets)

    validate_configuration(experiments, datasets)
    # print("Datasets: ", datasets)  # pretty print this later
    return datasets, experiments

def retrieve_standard_values(experiments, datasets):
    """retrieve_standard_values : retrieve standard values for the configuration file.
    In particular, this captures missing configuration keys and sets them to default values
    If the standard key IS in the current configuration, file, is is NOT changed.

    The standard keys are:
        data_inclusion_criteria
        ylims
    the format is just a dict standard["ylims"] and standard["data_inclusion_criteria"]

        Almost everything else in the configuration file is required, and will raise an error
        in the validation process if it is missing.

    Parameters
    ----------
    experiments:
        The top experiments dictionary
    datasets :
        the datasets in the experiments dictionary/configuration
    """
    try:
        ephys_path = Path(str(inspect.getfile(ephys)))
        ephys_posix = ephys_path.parent.as_posix()
        cfg_path = Path(str(ephys_posix), "config", "standard_configurations.cfg")
        standard = pg.configfile.readConfigFile(cfg_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No config file found, expected in the top-level config directory, named '{cfg_path!s}'"
        ) from exc
    
    for dataset in datasets:
        if "ylims" not in experiments[dataset]:
            CP("c", f"Reading 'ylims' from standard configuration for dataset: {dataset:s}")
            experiments[dataset]["ylims"] = standard["ylims"]
        if "data_inclusion_criteria" not in experiments[dataset]:
            CP("c", f"Reading 'data_inclusion_criteria' from standard configuration for dataset: {dataset:s}")
            experiments[dataset]["data_inclusion_criteria"] = standard["data_inclusion_criteria"]
    return experiments

def validate_configuration(experiments, datasets):
    """validate_configuration : validate the configuration file

    Parameters
    ----------
    experiments :
        The top experiments dictionary
    datasets :
        the datasets in the experiments dictionary/configuration

    Raises
    ------
    ValueError
        If the configuration is missing required entries
    """

    required_keys = [
        "region",
        "celltypes",
        "rawdatapath",
        "extra_subdirectories",
        "analyzeddatapath",
        "directory",
        "datasummaryFilename",
        "coding_file",
        "coding_name",
        "coding_sheet",
        "coding_level",
        "NWORKERS",
        "excludeIVs",

        "stats_filename",
        "statistical_comparisons",
        
        "plot_order",
        "plot_colors",
        "ylims",
        
        "spike_measures",
        "rmtau_measures",
        "FI_measures",
        
        "group_by",
        "group_map",
        "group_legend_map",
        "secondary_group_by",
        
        "data_inclusion_criteria",
        "protocol_durations", # this might be optional
        "protocols",

    ]
    for dataset in datasets:
        if dataset not in experiments:
            raise ValueError(
                f"Dataset '{dataset}' not found in the experiments section of the configuration file"
            )
        missing_keys = []
        for keyvalue in required_keys:
            if keyvalue not in experiments[dataset]:
                missing_keys.append(keyvalue)
        


        if len(missing_keys) > 0:
            PP.pprint(experiments[dataset])
            print(f"\n{'='*80:s}\nConfiguration file for dataset '{dataset}' is missing the following entries ")
            for keyvalue in missing_keys:
                print(f"    {keyvalue}")
            raise ValueError(
                    f"Dataset '{dataset}' has missing entries - please fix!"
                )
            
    

