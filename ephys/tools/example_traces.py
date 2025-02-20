"""
Plot some traces
"""
from pathlib import Path
import pickle
import pandas as pd
import datetime
import numpy as np
import matplotlib

# matplotlib.use("Qt4Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as mpl
import pylibrary.plotting.plothelpers as PH
import src.nf107_ivs as NF
import set_expt_paths
import ephys.ephys_analysis as EP

set_expt_paths.get_computer()

experiments = set_expt_paths.get_experiments()
cells = [
    "2017.05.17.s1c0:CCIV_1nA_max_000",  # BU
    "2018.01.26.s0c0:CCIV_1nA_max_000",  # TS
    "2018.01.26.s0c1:CCIV_1nA_max_000",  # DS
    "2018.06.11.s0c0:CCIV_1nA_max_000",  # Pyr
    "2017.11.28.s0c1:CCIV_long_001",  # CW
    #         '2017.12.13.s1c0:CCIV_1nA_max_000', # TV
    "2017.05.22.s1c0:CCIV_1nA_max_000",  # TV
]

NFC = NF.iv_analysis()


class ExTraces(object):
    def __init__(self):
        experiment = "nf107"
        self.exptdir = Path(experiments[experiment]["directory"])
        self.inputFilename = Path(
            self.exptdir, experiments[experiment]["datasummary"]
        ).with_suffix(".pkl")
        self.basedir = Path(experiments[experiment]["disk"])
        self.df = pd.read_pickle(str(self.inputFilename))

        rows = 3
        cols = 2
        self.P = PH.regular_grid(
            rows,
            cols,
            order="rowsfirst",
            figsize=(8.0, 10),
            showgrid=False,
            verticalspacing=0.03,
            horizontalspacing=0.03,
            margins={
                "leftmargin": 0.07,
                "rightmargin": 0.05,
                "topmargin": 0.03,
                "bottommargin": 0.1,
            },
            labelposition=(0.0, 0.0),
            parent_figure=None,
            panel_labels=None,
        )

        for n, cell in enumerate(cells):
            cn, protocolstr = cell.split(":")
            day = cn[:10] + "_000"
            sc = cn[11:]
            print(day, sc)
            slicestr = "slice_00%d" % int(sc[1])
            cellstr = "cell_00%d" % int(sc[3])
            print(day, slicestr, cellstr)
            cf = NFC.find_cell(self.df, day, slicestr, cellstr, protocolstr=None)
            protname = Path(day, slicestr, cellstr, protocolstr)
            try:
                iday = cf.index.tolist()[0]
            except:
                raise ValueError("? no protocol matching found")
            print(protname)
            print(Path(self.basedir, protname).is_dir())
            for i in cf["IV"]:
                for prot in i.keys():
                    if prot == protname and Path(self.basedir, protname).is_dir():
                        print("Yeah")
                        self.doprot(protname, iday, n, cn)
                # print(cf['IV'].values.keys())
        mpl.show()

    def doprot(self, f, iday, n, cn):
        print('doprot:\n   iday: ', int(iday))
        print("   Analyzing %s" % Path(self.basedir, f))
        self.EPIV = EP.IVSummary.IVSummary(str(Path(self.basedir, f)), plot=False,)
        print(dir(self.EPIV))
        br_offset = 0.0
        # if (not pd.isnull(self.df.at[iday, 'IV']) and
        #      f in self.df.at[iday, 'IV'] and
        #      '--Adjust' in self.df.at[iday, 'IV'][f].keys()):
        #     br_offset = self.df.at[iday, 'IV'][f]['BridgeAdjust']
        #     print('Bridge: {0:.2f} Mohm'.format(self.df.at[iday, 'IV'][f]['BridgeAdjust']/1e6))
        result = {}
        self.spike_threshold = -0.035
        self.pubmode = True
        ctype = self.df.at[iday, "cell_type"]
        tgap = 0.0015
        tinit = True
        if ctype in ["bushy", "Bushy", "d-stellate", "D-stellate", "Dstellate"]:
            tgap = 0.0005
            tinit = False
        self.EPIV.AR.traces = np.array(self.EPIV.AR.traces)
        self.plot_traces(self.P.axarr.ravel()[n], self.pubmode, ctype, cn)
        return
        plotted = self.EPIV.compute_iv(
            threshold=self.spike_threshold,
            bridge_offset=br_offset,
            tgap=tgap,
            pubmode=self.pubmode,
            plotiv=False,
        )
        iv_result = self.EPIV.RM.analysis_summary
        sp_result = self.EPIV.SP.analysis_summary
        result["IV"] = iv_result
        result["Spikes"] = sp_result
        self.plot_traces(self.P.axarr.ravel()[n], self.pubmode, ctype, cn)
        ctype = self.df.at[iday, "cell_type"]
        annot = self.df.at[iday, "annotated"]
        if annot:
            ctwhen = "[revisited]"
        else:
            ctwhen = "[original]"
        # if self.EPIV.IVFigure is not None:
        #     self.EPIV.IVFigure.suptitle('{0:s}\nType: {1:s} {2:s}'.format(
        #         str(Path(self.basedir, f)), # .replace('_', '\_'),
        #         self.df.at[iday, 'cell_type'], ctwhen),
        #         fontsize=8)
        nfiles = 0
        # if plotted:
        #     t_path = Path(self.tempdir, 'temppdf_{0:04d}.pdf'.format(nfiles))
        #     # mpl.savefig(t_path, dpi=300) # use the map filename, as we will sort by this later
        #     nfiles += 1
        #     mpl.close(EPIV.IVFigure)
        return result, nfiles

    def plot_traces(self, ax, pubmode, ctype, filename):
        if ctype in ["t-stellate", "d-stellate", "tuberculoventral"]:
            dv = 65.0
        elif ctype in ["bushy"]:
            dv = 35.0
        elif ctype in ["cartwheel", "pyramidal"]:
            dv = 85.0
        jsp = 0
        max_spk_tr = 3
        print(self.EPIV.AR.traces.shape)
        for i in range(self.EPIV.AR.traces.shape[0]):
            if i in list(self.EPIV.SP.spikeShape.keys()):
                idv = float(jsp) * dv
                jsp += 1
            else:
                idv = 0.0
            if jsp > max_spk_tr:
                continue
            ax.plot(
                self.EPIV.AR.time_base * 1e3,
                idv + self.EPIV.AR.traces[i, :].view(np.ndarray) * 1e3,
                "-",
                linewidth=0.55,
            )
            ptps = np.array([])
            paps = np.array([])
            if i in list(self.EPIV.SP.spikeShape.keys()):
                for j in list(self.EPIV.SP.spikeShape[i].keys()):
                    paps = np.append(
                        paps, self.EPIV.SP.spikeShape[i][j]["peak_V"] * 1e3
                    )
                    ptps = np.append(
                        ptps, self.EPIV.SP.spikeShape[i][j]["peak_T"] * 1e3
                    )
                ax.plot(ptps, idv + paps, "ro", markersize=0.5)

            # mark spikes outside the stimlulus window
            ptps = np.array([])
            paps = np.array([])
            # for window in ['baseline', 'poststimulus']:
            #     dt = self.EPIV.AR.sample_interval
            #     ptps = np.array(self.EPIV.SP.analysis_summary[window+'_spikes'][i])
            #     ptpsn = []
            #     if len(ptps) > 0:
            #         print('p: ', ptps, dt, [int(x/dt) for x in ptps])
            #         for x in ptps:
            #             ix = int(x/dt)
            #             r = range(ix-10, ix+10)
            #             ivmax = np.argmax(self.EPIV.AR.traces[i][r])+r[0]
            #             ptpsn.append(ivmax*dt)
            #             ax.plot(self.EPIV.AR.time_base[r]*1e3, idv + 1e3*self.EPIV.AR.traces[i][r], 'k-', linewidth=0.2)
            #         print('s ', ptps)
            #         print('n' ,ptpsn)
            #     ptps = np.array(ptpsn)
            #     if len(ptps) > 0:
            #         print('p2: ', ptps)
            #     uindx = [int(u/dt) for u in ptps]
            #     paps = np.array(self.EPIV.AR.traces[i, uindx])
            #     ax.plot(ptps*1e3, idv+paps*1e3, 'bo', markersize=0.5)

        for k in self.EPIV.RM.taum_fitted.keys():
            ax.plot(
                self.EPIV.RM.taum_fitted[k][0] * 1e3,
                self.EPIV.RM.taum_fitted[k][1] * 1e3,
                "--k",
                linewidth=0.30,
            )
        # for k in self.EPIV.RM.tauh_fitted.keys():
        #     ax.plot(
        #         self.EPIV.RM.tauh_fitted[k][0] * 1e3,
        #         self.EPIV.RM.tauh_fitted[k][1] * 1e3,
        #         "--r",
        #         linewidth=0.50,
        #     )
        ax.set_title(ctype, fontsize=14)
        ax.text(
            0.95,
            0.05,
            str(filename).replace("_", "\_"),
            transform=ax.transAxes,
            horizontalalignment="right",
            fontsize=6,
        )
        if pubmode:
            PH.calbar(
                ax,
                calbar=[0.0, -95.0, 25.0, 25.0],
                axesoff=True,
                orient="left",
                unitNames={"x": "ms", "y": "mV"},
                fontsize=10,
                weight="normal",
                font="Arial",
            )
        # P.axdict['B'].plot(self.EPIV.SP.analysis_summary['FI_Curve'][0]*1e9, self.EPIV.SP.analysis_summary['FI_Curve'][1]/(self.EPIV.AR.tend-self.EPIV.AR.tstart), 'ko-', markersize=4, linewidth=0.5)


if __name__ == "__main__":
    ExTraces()
