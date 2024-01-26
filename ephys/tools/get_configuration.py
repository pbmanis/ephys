from pathlib import Path
import pyqtgraph as pg


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
    for dataset in datasets:
        if dataset not in experiments:
            raise ValueError(
                f"Dataset '{dataset}' not found in the experiments section of the configuration file"
            )
        if "region" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have a 'region' entry in the configuration file"
            )
        if "celltypes" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have a 'celltypes' entry in the configuration file"
            )
        if "rawdatapath" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have a 'rawdatapath' entry in the configuration file"
            )
        if Path(experiments[dataset]["rawdatapath"]).is_dir() is False:
            raise ValueError(
                f"The 'rawdatapath' entry for Dataset '{dataset}' was not found"
            )
        if "extra_subdirectories" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have an 'extra_subdirectories' entry in the configuration file"
            )
        if "analyzeddatapath" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have a 'analyzeddatapath' entry in the configuration file"
            )
        if "directory" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have an 'directory' entry in the configuration file"
            )
        if "datasummaryFilename" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have an 'datasummaryFilename' entry in the configuration file"
            )
        if "coding_file" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have an 'coding_file' entry in the configuration file"
            )
        if "NWORKERS" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have an 'NWORKERS' entry in the configuration file to specify # of cores to use"
            )
        if "excludeIVs" not in experiments[dataset]:
            raise ValueError(
                f"Dataset '{dataset}' does not have an 'excludeIVs' entry in the configuration file to specify which IVs to exclude"
            )
 

