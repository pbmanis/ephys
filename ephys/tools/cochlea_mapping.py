import datetime
from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np
import pyqtgraph as pg
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

import cochlear_frequency_maps as CF2D
import read_imaris

    """ Use the cochlear frequency maps to plot Imaris filaments in 3D, with annotation
    of specific frequency locations. The filaments are assumed to be traced along the IHC
    or pillar cell apical surface, and to follow the entire length of the cochlea. 



    Returns
    -------
    Nothing is returned, but a plot is generated.
        
    """

datapath = "/Volumes/Pegasus_004/LightSheetData/Filament Test/"
fn = "16-19-37_M22-2R 6-3x R3 dyn copy_flip_filament MAC TEST .ims"
# fn = "63xoil CBA p266_LuciferYellow_GLYOXAL1hr_SOFT MOUNT_S2_C0_C1_7_30_24_647_LY_AB_Stitch_COMPLETE.ims"
filename = Path(datapath, fn)

def distance_along_path(filament):
    """
    Calculate the distance along a path defined by a set of points
    The filament is assumed to be a numpy array of shape (n, 3) where n is the number of points in the filament
    """
    d = np.zeros(len(filament))
    for i in range(1, len(filament)):
        d[i] = d[i - 1] + np.linalg.norm(filament[i] - filament[i - 1])
    return d

class Annotation3D(Annotation):
    """Annotate the point xyz with text s"""

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, s, xy=(0, 0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.xy = (xs, ys)
        Annotation.draw(self, renderer)


def annotate3D(ax, s, *args, **kwargs):
    """add anotation text s to to Axes3d ax
    all other arguments are passed to the matplotlib annotation function
    """

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)


def plot_filament_mpl(filament, radii, maptype: str = "Ou", ax=None):

    filament_distance = distance_along_path(filament)

    colors = mpl.cm.rainbow(np.linspace(0, 1, filament.shape[0]))
    # ax.view_init(elev=5.0, azim=-14.0, roll=0.0)
    ax.view_init(elev=48.0, azim=0.0, roll=0.0)
    ax.view_init(elev=2.0, azim=3.0, roll=0.0)
    ax.scatter(filament[:, 0], filament[:, 1], filament[:, 2], c=colors, s=radii)
    freq_array = (
        np.array([2.000, 4.000, 8.000, 16.000, 24.000, 32.000, 64.000]) * 1000.0
    )  # convert to Hz
    maxd = np.max(filament_distance)
    dfs, dpct = CF2D.cochlear_frequency_to_distance(maptype, f=freq_array, maxd=maxd)

    idxs = []
    for i in range(len(dpct)):
        filament_distance_norm = filament_distance / maxd
        idxs.append(int(np.argmin(np.abs(filament_distance - dfs[i]))))

    xyzn = filament[idxs]
    ax.scatter(filament[idxs, 0], filament[idxs, 1], filament[idxs, 2], c=colors[idxs], s=50)
    for j, xyz_ in enumerate(xyzn):
        read_imaris.annotate3D(
            ax,
            s=f"{int(1e-3*freq_array[j]):d} kHz",
            xyz=xyz_,
            fontsize=8,
            xytext=(-3, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )


def plot_filaments_grid(filament, radii):
    fig = mpl.figure(figsize=(12, 12))
    a = mpl.text(
        x=0.98,
        y=0.01,
        s=f"pbm {datetime.datetime.now().strftime('%Y.%m.%d::%H:%M')}",
        ha="right",
        fontsize=8,
        transform=fig.transFigure,
    )

    mpl.gca().spines.top.set(visible=False)
    mpl.gca().spines.right.set(visible=False)
    mpl.gca().spines.left.set(visible=False)
    mpl.gca().spines.bottom.set(visible=False)
    mpl.gca().set_xticks([])
    mpl.gca().set_yticks([])

    mapnames = CF2D.get_map_names()
    posmap = {1: 221, 2: 222, 3: 223, 4: 224}
    mapnames = ["Ou", "Muller", "Ehret", "Muniak"]
    for im, mapname in enumerate(mapnames):
        ax = fig.add_subplot(posmap[im + 1], projection="3d")
        plot_filament_mpl(filament, radii, mapname, ax)
        ax.set_title(f"{mapname:s} map")
    fig.suptitle("Filaments: Comparing Cochlear Frequency Maps")
    mpl.tight_layout()
    mpl.show()


if __name__ == "__main__":
    # read the filament from one cochlea and then plot them on a grid
    filaments, radii = read_imaris.read_filaments(filename)
    plot_filaments_grid(filaments[0], radii[0])

