from pathlib import Path
import pyqtgraph as pg
import pprint
PP = pprint.PrettyPrinter(indent=4)


def get_configuration(configfile: str = "experiments.cfg"):
    """get_configuration : retrieve the configuration file from the current
    working directory, and return the datasets and experiments

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
    validate_configuration(experiments, datasets)
    # print("Datasets: ", datasets)  # pretty print this later
    return datasets, experiments

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
            
    

