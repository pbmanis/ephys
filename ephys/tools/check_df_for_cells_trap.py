""" Check_trap:
Check whether the cell_ids in the experiment configuration under "trap"
are actuall in the dataframe or not.
If not, we raise an error, otherwise success passes silently.

This might be useful for debugging missing data.

In the configuration file, trap should appear as:
trap:
    <cell_id>: "reason for trap" (actually, any argument is okay, but this is a good format for documentation)
    ... (more cells as needed)

"""

def check_df_for_cells_trap(df, experiment, message:str=None):
    """check_trap Check for the trap in the configuration

    Parameters
    ----------
    df : pandas dataframe
        dataframe with cell_id column
    experiment : dict
        experiment configuration dictionary

    Raises
    ------
    ValueError
        if ANY of the cell_ids in the trap are not found in the dataframe

    """
    trap = experiment.get("trap", None)
    if trap is not None:
        for cell in trap.keys():
            if cell not in df.cell_id.values:
                raise ValueError(f"Cell {cell} in trap not found in data: {message if message is not None else ''}")
            else:
                print(f"************* Cell {cell} in trap found in data: {message if message is not None else ''}")