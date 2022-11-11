import matplotlib.pyplot as mpl
from pathlib import Path
import pyabf

basepath = Path("/Volumes/PBM_007/Example_D-stellate_neuron")
abf_filename = Path(basepath, "20210112a", "2021_01_12_0008.abf")
abf = pyabf.ABF(abf_filename)
# print(abf.headerText)

def plot_one_run(subfig, abf, fn):

    nSweeps = abf.sweepCount
    ax = subfig.subplots(2, 1, sharex=True,  gridspec_kw={
                           'width_ratios': [1],
                           'height_ratios': [0.8, 0.2],
                       'wspace': 0.1,
                       'hspace': 0.0}, squeeze=True)
    mpl.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.85)
    ax[0].set_title(str(fn.name), fontsize=9)

    for i in range(nSweeps):
        print(f"Sweep = {i:d}")
        abf.setSweep(i)
        ax[0].plot(abf.sweepX, abf.sweepY, linewidth=0.5)
        ax[1].plot(abf.sweepX, abf.sweepC, linewidth=0.3)
    ax[1].set_ylim((-100, 20))
    for i in range(2):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

ephys_path = Path(basepath, "20210112a")
files = list(ephys_path.glob("*.abf"))

nf = len(files)
fig= mpl.figure(figsize=(8, 6))
subfigs = fig.subfigures(3,3) # , wspace=0.2, hspace=0.15)
subfigs = subfigs.ravel()
for i, fn in enumerate(files):
    abf = pyabf.ABF(fn)
    plot_one_run(subfigs[i], abf, fn)
mpl.show()