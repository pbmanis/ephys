""" Get ages and weights of mice form a set of experiments, and plot them. 
data are keyed by sex as well.
"""
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import pyqtgraph as pg
import seaborn as sns
from pyqtgraph import QtGui

import ephys.datareaders.acq4_reader as acq4_reader

dirs = {"VGAT:": [
    "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/VGAT_DCNmap",
    "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/VGAT",
    "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/VGAT_NIHL",
    "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/VGAT_NIHL/Blinded_2wk_NIHL",
    "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/VGAT_NIHL/Parasaggital",
],
"Ank2": [
    "/Volumes/Pegasus_004/ManisLab_Data3/Kasten_Michael/Maness_Ank2_PFC_Stim/Rig2(MRK)/L23_intrinsic",
    "/Volumes/Pegasus_004/ManisLab_Data3/Kasten_Michael/Maness_Ank2_PFC_Stim/Rig2(PBM)/L23_intrinsic",
    "/Volumes/Pegasus_004/ManisLab_Data3/Kasten_Michael/Maness_Ank2_PFC_Stim/Rig4(MRK)/L23_intrinsic",
],
"CBA_Age": [
    "/Volumes/Pegasus_004/ManisLab_Data3/Edwards_Reginald/CBA_Age",
]
}
weights = []
ages = []
sexes = []
dates = []
genotype = []
experiment = "CBA_Age"

cm = pg.colormap.get('CET-L9') # prepare a linear color map

def get_datetime(timestamp):
    utc_time = datetime.fromtimestamp(timestamp, timezone.utc)
    local_time = utc_time.astimezone()
    return local_time.strftime("%Y-%m-%d %H:%M:%S.%f%z (%Z)")

for directory in dirs[experiment]:
    files = list(Path(directory).glob("*"))
    print("Dir: ", dir)
    for file in files:
        if file.is_file():
            continue
        AR = acq4_reader.acq4_reader()
        index = AR.readDirIndex(file)["."]
        date = get_datetime(index['__timestamp__'])
        if "weight" in index.keys() and "age" in index.keys():
            wt = index["weight"].replace("g", "")
            wt = wt.replace("G", "")
            if wt in ["", " "]:
                print("No weight, date=", date)
                exit()
                continue
            sex = index["sex"]
            age = index["age"].split(" ")[0].replace("p", "").replace("D", "")
            age = age.replace("P", "")
            age = age.replace("d", "")
            age = age.replace("ish", "")
            if age in ["", " "]:
                print("no age, date=", date)
                exit()
                continue
            weights.append(int(wt))
            ages.append(int(age))
            sexes.append(sex.upper())
            dates.append(date)
            genotype.append(index["genotype"].upper())
            print(f"{dates[-1]:s}, P{ages[-1]:3d}D {weights[-1]:2d}g {sexes[-1]:1s}  {genotype[-1]:s}")

print("sex: ", sexes)
print("Males:   ", len([x for x in sexes if x == "M"]))
print("Females: ", len([x for x in sexes if x == "F"]))

redish = pg.CIELabColor(48, 72, 72)
bluish = pg.CIELabColor(77, 42, -127)
# pens = {"-/-": pg.mkPen((255, 255, 255, 255), width=2), "+/+": pg.mkPen((0, 255, 0, 255), width=2), "+/-": pg.mkPen((255, 0, 0, 255), width=2)
#         }
pens = {"M": bluish, "F": redish}
brushes = {"M": bluish, "F": redish}
symbols = {"-/-": "o", "+/+": "+", "+/-": "s", "WT": "+", "": "+"}
# scatter = pg.PlotDataItem(pxMode = False )
# spi = pg.ScatterPlotItem(name=f"Age vs Weight by genotypex")
spots= {"M": [], "F": []}
spi = {}
for sex in spots.keys():
    spi[sex] = pg.ScatterPlotItem()
for i, s in enumerate(sexes):   
    spots[sexes[i]].append(   {
                        "pos": (ages[i], weights[i]),
                        "pen": pens[sexes[i]],
                        "brush": brushes[sexes[i]],
                        "symbol": symbols[genotype[i]],
                        "size": 8,
                    }
    )
for sex in spots.keys():
     spi[sex].addPoints(spots[sex])
# app = pg.mkQApp("Summary Plot")
# app.setStyle("fusion")
# win = pg.GraphicsLayoutWidget(show=True, title=f"Age vs Weight {experiment:s} Mice Manis Lab")
P1=pg.plot()
r = 1.61803
P1.resize(int(600*r), 600)
P1.setWindowTitle("Manis Lab")

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

# P1 = win.plot(title="Age vs Weight by sex and genotype")
legend = pg.LegendItem(offset=(50,25), labelTextSize='12pt',
                       horSpacing=20, verSpacing=-5, 
                       pen=(255,255,255,255),
                       brush=(255, 255, 255, 32),
                       frame=True)
# legend.addItem(spi["M"], "Age vs Weight by genotype")
lab1 = pg.PlotDataItem(y=[0], pen=None, symbol=symbols["WT"], symbolBrush=brushes["M"], symbolSize=8,)
lab2 = pg.PlotDataItem(y=[0], pen=None, symbol=symbols["WT"], symbolBrush=brushes["F"], symbolSize=8,)

legend.addItem(lab1, "Males")
legend.addItem(lab2, "Females")
legend.setParentItem(P1.graphicsItem())

legend = P1.addLegend()

P1.addItem(spi["M"])
P1.addItem(spi["F"])

# legends = []
# legends.append(pg.LegendItem(pen=pg.mkPen(pens["M"]), brush=pg.mkBrush(brushes["M"], text="M")))
# legends.append(pg.LegendItem(pen=pg.mkPen(pens["F"]), brush=pg.mkBrush(brushes["F"], text="F")))
# legends.append(pg.LegendItem(symbol=symbols['-/-'], text="-/-"))
# legends.append(pg.LegendItem(symbol=symbols['+/+'], text="+/+"))
# for i, legend in enumerate(legends):
#     legend.setParentItem(spi)
    # legend.setOffset((0, 100+i*50))
    

# # ages, weights, pen=None, symbol="o", symbolSize=5, symbolPen=None, symbolBrush=(255, 0, 0, 255))

P1.setLabel("left", "Weight (g)")
P1.setLabel("bottom", "Age (days)")

P1.showGrid(x=True, y=True)
P1.setXRange(0, int(np.max(ages)/10)*10)
P1.setYRange(0, 50)
pg.exec()
# df = pd.DataFrame({"age": ages, "weight": weights, "sex": sexes})
# sns.scatterplot(x=ages, y=weights, hue='sex', data=df)
# # mpl.show()