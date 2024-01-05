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
        print("Getting Configuration file from: ", Path().absolute())
        cpath = Path(Path().absolute(), "config", configfile)
        config = pg.configfile.readConfigFile(cpath)
        experiments = config["experiments"]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No config file found, expected in the top-level config directory, named '{cpath!s}'"
        ) from exc

    datasets = list(experiments.keys())
    # print("Datasets: ", datasets)  # pretty print this later
    return datasets, experiments
