from pathlib import Path

""" Configuration Manager
These functions provides some tools to manage the configuration information for datatables.

Specific functions include:
1. Verification of the existence of all specified Paths.

2. Get all used prototols listed in the datasummary file, and generate
some text for the configuration file.

3. 

"""

def verify_paths(config):
    """ verify_paths
    Verify that all paths specified in the configuration file exist.
    """
    paths_to_check = [ 
        Path(config["rawdatapath"], config["directory"]),
        Path(config["databasepath"], config["directory"]),
        Path(config["analyzeddatapath"], config["directory"]),
    ]

    for dpath in paths_to_check:
        if not dpath.is_dir():  # specifically check for the directory
            print(f"Path {dpath} does not exist")
            raise FileNotFoundError(f"Path {dpath} does not exist")
    return True

def gather_protocols(config):
    """ get_protocols
    Get a list of all protocols used in the datasummary file.
    """
    pass