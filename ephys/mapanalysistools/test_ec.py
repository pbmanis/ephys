import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.colors
import matplotlib
import matplotlib.collections as collections
from  matplotlib import colors as mcolors
import matplotlib.cm

import pylibrary.PlotHelpers as PH
#import seaborn
cm_sns = mpl.cm.get_cmap('terrain')  # terrain is not bad
#cm_sns = mpl.cm.get_cmap('parula')  # terrain is not bad
#cm_sns = mpl.cm.get_cmap('jet')  # jet is terrible
color_sequence = ['k', 'r', 'b']
colormap = 'snshelix'

spotsize = 0.5
pos = np.random.random((50,2))
measure = np.random.randn(pos.size)
npts = pos.size
radw = np.ones(npts)*spotsize/2.
radh = np.ones(npts)*spotsize/2.
colors = []
maxv = np.max(measure)
cmx = matplotlib.cm.ScalarMappable(norm=None, cmap=cm_sns)
colors = cmx.to_rgba(measure/maxv)
    #colors.append(mcolors.to_rgba(cm_sns(measure/vmax), alpha=0.6))
f, axe = mpl.subplots(1,1)
ec = collections.EllipseCollection(radw, radh, np.zeros_like(radw), offsets=pos, units='xy', transOffset=axe.transData,
            facecolor=colors, edgecolor='k')
print(ec)
axe.add_collection(ec)
axe.set_xlim(-10, 10)
axe.set_ylim(-10, 10)
axe.set_aspect('equal')
mpl.show()
