""" Test VC analysis routine on a bit of data
"""

# run from HK_collab to get the parameters.

from pathlib import Path
import pandas as pd
from ephys.ephys_analysis import vc_analysis
from ephys.tools import get_configuration
from ephys.datareaders import acq4_reader
import pytest

@pytest.mark.skip(reason="Not testable in current environment - must be run from an experiment diretory")
def test_vc():
    experiment_name = "DCN_IC_inj"

    datasets, experiments = get_configuration.get_configuration("./config/experiments.cfg")

    experiment = experiments[experiment_name]
    cell = "2025.07.24_000/slice_000/cell_002"
    protocol = "VC_IH_measure_000"
    ds = pd.read_pickle(Path(experiment['rawdatapath'], experiment['directory'], experiment['databasepath'],
                            experiment['directory'],
                            experiment['datasummaryFilename']))

    this_cell = ds[ds['cell_id'] == cell]
    print(this_cell)
    protocol_path = Path(experiment['rawdatapath'], experiment['directory'],
                                    cell, protocol)
    print(protocol_path)
    print(protocol_path.is_dir())
    AR = acq4_reader.acq4_reader(pathtoprotocol = protocol_path)
    VC = vc_analysis.VCAnalysis()
    VC.configure(
        datapath=protocol_path,
        altstruct=AR,
        file=None,
        experiment=experiment,
        reader=AR,
        plot=True
    )
    VC.analyze_vcs()

if __name__ == "__main__":
    test_vc()
