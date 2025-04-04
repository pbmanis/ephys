"""Simple CC map analysis.

This module provides a simple analysis of LSPS maps done in current clamp.
1. detect spikes, save timing
2.
"""

from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH

import ephys.datareaders.acq4_reader as acq4_reader
from ephys.tools import get_configuration

AR = acq4_reader.acq4_reader()


class CC_Map_Analyzer:
    def __init__(self, dataset):
        self.dataset = dataset

        datasets, experiment = get_configuration.get_configuration("config/experiments.cfg")
        if dataset not in datasets:
            raise ValueError(f"Dataset {dataset} not found in experiments.cfg")
        self.experiment = experiment[dataset]

        self.datasummary = self.get_datasummary()
        self.find_CC_maps()
        self.read_CC_maps()

    def get_datasummary(self):
        dsfile = Path(
            self.experiment["analyzeddatapath"],
            self.experiment["directory"],
            self.experiment["datasummaryFilename"],
        )
        if not dsfile.is_file():
            raise FileNotFoundError(f"Data summary file not found: {dsfile}")
        df = pd.read_pickle(dsfile)
        return df

    def find_CC_maps(self):
        # print(self.datasummary.columns)
        all_prots = []
        self.datasummary["IC_Protocols"] = {}
        for cellid in self.datasummary["cell_id"]:
            all_prots = self._find_CC_map(cellid, all_prots)
        # print(all_prots)
        # all_prots_ic = sorted(set([p.replace(" ", "")[:-4] for p in all_prots if p.lower().startswith("ic_")]))
        # print(all_prots_ic)

    def _find_CC_map(self, cell_id: str = None, all_prots: list = None):
        ds_cell = self.datasummary[self.datasummary["cell_id"] == cell_id]
        if len(ds_cell["data_complete"].values) == 0:
            return all_prots
        protos = ds_cell["data_complete"].values[0]
        protos = protos.split(",")
        protocols = [p.replace(" ", "") for p in protos]
        all_prots.extend(protocols)
        all_prots_ic = sorted(
            set([p.replace(" ", "") for p in protocols if p.lower().startswith("ic_")])
        )

        ci = self.datasummary.loc[self.datasummary["cell_id"] == cell_id].index
        if len(all_prots_ic) > 0:
            self.datasummary.loc[ci, ["IC_Protocols"]] = str(all_prots_ic)
        return all_prots

    def read_CC_map(self, cell_id, protodir):
        AR.setProtocol(protodir)
        AR.getData()

    def read_CC_maps(self):
        nplots = 0
        maxrows = 9
        maxcols = 9
        maxplots = maxrows * maxcols
        f, ax = mpl.subplots(maxrows, maxcols, figsize=(10, 10))
        for icell, cell_id in enumerate(self.datasummary["cell_id"]):
            ds_cell = self.datasummary[self.datasummary["cell_id"] == cell_id]
            protos = ds_cell["IC_Protocols"].values[0]
            if not isinstance(protos, str):
                continue
            protos = protos.replace("[", "").replace("]", "").replace("'", "")
            for iproto, proto in enumerate(protos.split(",")):
                datapath = Path(
                    self.experiment["rawdatapath"], self.experiment["directory"], cell_id
                )
                protos = list(datapath.glob(f"{proto:s}*"))
                if len(protos) == 0:
                    continue
                print(protos)
                self.read_CC_map(cell_id, protos[0])

                for tr in AR.traces:
                    te = np.array(AR.time_base).squeeze()
                    if len(te) == 0:
                        continue
                    if np.max(te) > 1.25:
                        tend = np.argwhere(te >= 1.25)[0][0]
                    else:
                        tend = np.argwhere(te >= 1.0)[0][0]
                    tb =  np.array(AR.time_base[:tend]).squeeze()
                    ax[nplots % maxcols, nplots // maxrows].plot(
                        tb,
                        1e3 * np.array(tr).squeeze()[:tend],
                        linewidth=0.35,
                    )
                    PH.referenceline(reference=-60.,
                                     limits = [0., np.max(tb)],
                                     linewidth = 0.2,
                        axl = ax[nplots % maxcols, nplots // maxrows])
                if iproto == 0:
                    ax[nplots % maxcols, nplots // maxrows].set_title(
                        f"{cell_id:s}",
                        fontsize=3,
                    )
                ax[nplots % maxcols, nplots // maxrows].text(
                    0.02,
                    1.0 - 0.02 * iproto,
                    s=f"{proto:s}",
                    fontsize=4,
                    transform=ax[nplots % maxcols, nplots // maxrows].transAxes,
                    ha="left",
                    va="top",
                )

            # if nplots // maxrows == maxrows:
            #     ax[nplots % maxcols, nplots // maxrows].set_xlabel("Time (s)", fontsize=6)
            # if nplots // maxcols == 0:
            #     ax[0, nplots // maxrows].set_ylabel("Vm (mV)", fontsize=6)
            # ax[icell % maxcols, maxplots % maxrows].set_xlim(0, 1.25)
            ax[nplots % maxcols, nplots // maxrows].set_ylim(-90, 40)
            PH.noaxes(ax[nplots % maxcols, nplots // maxrows])
            PH.calbar(
                ax[nplots % maxcols, nplots // maxrows],
                calbar=[0.8, -10, 0.2, 20],
                unitNames={"x": "s", "y": "mV"},
                fontsize=6,
                linewidth=0.5,
            )

            nplots += 1
            if nplots >= maxplots:
                break
            if nplots < maxplots:
                for i in range(nplots, maxplots):
                    ax[i % maxcols, i // maxrows].axis("off")
        mpl.show()


if __name__ == "__main__":
    CC = CC_Map_Analyzer("Thalamocortical")
