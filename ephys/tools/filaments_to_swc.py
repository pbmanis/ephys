# import neuronvis as NV
import datetime
from pathlib import Path

import matplotlib.pyplot as mpl
import ngauge
import numpy as np
import pyqtgraph as pg
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

import src.read_imaris as read_imaris

""" Convert the filament tracing from an Imaris file (cell) to an 
SWC file for use in neuron reconstruction software.

Returns
-------
_type_
    _description_

Raises
------
ValueError
    _description_
"""
datapath = "/Users/pbmanis/Desktop/Reggie-Reconstructions/"
# fn = "63xoil CBA p85 LuciferYellow_GLYOXAL2days_SOFT MOUNT_S0_C0_PYR_12_1_23_Stitch.ims"
fn = "63xoil CBA p266_LuciferYellow_GLYOXAL1hr_SOFT MOUNT_S2_C0_C1_7_30_24_647_LY_AB_Stitch_COMPLETE.ims"


def distance_along_path(filament):
    """
    Calculate the distance along a path defined by a set of points
    """
    d = np.zeros(len(filament))
    for i in range(1, len(filament)):
        d[i] = d[i - 1] + np.linalg.norm(filament[i] - filament[i - 1])
    return d


def convert_types(filament_type_list):
    """
    Convert the types to a string: These would be the defaults, but..
    they don't match our mapping, so use different dict.
    """
    # type_dict = {
    #     0: "Undefined",
    #     1: "Soma",
    #     2: "Axon",
    #     3: "Dendrite",
    #     4: "Apical dendrite",
    #     5: "Fork point",
    #     6: "End point",
    #     7: "Custom"
    # }
    type_dict = {
        "Filaments 1": 0,  # use 0 for the straight filament
        "Filaments 1 Basal": 3,  # standard SWC mapping for basal dendrite
        "Filaments 1 Apical": 4,  # standard SWC mapping for apical dendrite
    }
    swc_types = []
    for filament_type in filament_type_list:
        if filament_type not in type_dict.keys():
            print(f"Unknown type: {filament_type}")
            raise ValueError(f"Unknown type: {filament_type}")
        else:
            swc_types.append(type_dict[filament_type])
    return swc_types


def filament_to_swc(filaments_xyz, radii, edges, filament_types, savename="test"):
    # This is modified from the swc-plugins_for-Imaris-10/ExportSEC_new.py
    # github: Elsword016/swc-plugins_for-Imaris-10
    # here, we take the filaments that ahve been extracted from the imaris file
    # by read_imaris.py and convert them to an SWC file for use in neuron reconstruction software

    # filament type conversion:
    print(filament_types)
    # main conversion
    swcs = np.zeros((0, 7))
    swc_list = []
    vCount = len(filaments_xyz)
    print(len(filament_types))

    for i in range(vCount):
        vFilamentsXYZ = filaments_xyz[i]
        vFilamentsRadius = radii[i]  # vFilaments.GetRadii(i)
        vFilamentsEdges = edges[i]  # vFilaments.GetEdges(i)
        vFilamentsTypes = [ft for ft in convert_types(filament_types[i])]  # vFilaments.GetTypes(i)

        N = len(vFilamentsXYZ)
        G = np.zeros((N, N), dtype=bool)
        visited = np.zeros(N, dtype=bool)
        for p1, p2 in vFilamentsEdges:
            G[p1, p2] = True
            G[p2, p1] = True

        head = 0
        swc = np.zeros((N, 7))
        visited[0] = True
        pixel_scale = 1.0  # microns
        pixel_offset = 0.0  # microns
        print(vFilamentsTypes)

        queue = [0]
        prevs = [-1]
        while queue:
            cur = queue.pop()
            prev = prevs.pop()
            print("cur: ", cur, prev, head, vFilamentsTypes[cur])
            swc[head] = [head + 1, vFilamentsTypes[cur], 0, 0, 0, vFilamentsRadius[cur], prev]
            pos = vFilamentsXYZ[cur] - pixel_offset
            swc[head, 2:5] = pos * pixel_scale
            for idx in np.where(G[cur])[0]:
                if not visited[idx]:
                    visited[idx] = True
                    queue.append(idx)
                    prevs.append(head + 1)
            head += 1
        filename_filament = f"{savename}_filament_{i}.swc"
        print(
            "Exported " + str(i + 1) + "/" + str(vCount) + " filaments", end="\r"
        )  ## Export individual filaments as separate swcs
        swcs = np.vstack((swcs, swc))
        np.savetxt(filename_filament, swc, fmt="%d %d %f %f %f %f %d")
    np.savetxt(savename, swcs, fmt="%d %d %f %f %f %f %d")  # Combined swcs
    print("All filaments exported!")


def plot_filament_continuous(
    filament: np.array, radii: np.array, edges: np.array, filament_type: str, ax=None, swcgen=False
):
    filament_distance = distance_along_path(filament)
    fd_diff = np.diff(filament_distance)
    # define break pointss along the path - usually corresponds to branch points,
    # but also to unconnected ends of the filament
    break_points = np.where(fd_diff > 2.0)[0]
    # print("break points: ", break_points)
    # exit()
    # colors = mpl.cm.rainbow(np.linspace(0, 1, filament.shape[0]))
    colors = mpl.cm.rainbow(np.linspace(0, 1, len(break_points)))
    # ax.view_init(elev=5.0, azim=-14.0, roll=0.0)
    # ax.view_init(elev=48.0, azim=0.0, roll=0.0)
    # ax.view_init(elev=2.0, azim=3.0, roll=0.0)
    if swcgen:
        swc_text = ""
    id = 1
    bp_id = [None] * len(break_points)
    for i in range(len(break_points) - 1):
        # print(break_points[i], break_points[i+1])
        # print(filament.shape)
        fb = filament[break_points[i] : break_points[i + 1], :]
        radius = radii[break_points[i] : break_points[i + 1]]
        # print(fb.shape, radius)
        ax.scatter(fb[:, 0], fb[:, 1], fb[:, 2], c=colors[i], s=radius * radius * 9, alpha=0.5)
        ax.scatter(fb[0, 0], fb[0, 1], fb[0, 2], c="k", s=50)
        ax.text(fb[0, 0], fb[0, 1], fb[0, 2], f"{i:d}", fontsize=8)


def plot_filaments_swc(
    filament, radii, edges, filament_types, i_fil: int = 0, ax=None, swcgen: bool = False
):
    # if ax is None:
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

    mapname = "swc"
    if ax is None:
        ax = fig.add_subplot(projection="3d")
    plot_filament_continuous(filament, radii, edges, filament_types, ax, swcgen=swcgen)
    ax.view_init(elev=90.0, azim=0.0, roll=0.0)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 650)
    ax.set_zlim(0, 250)
    ax.set_title(f"{mapname:s} filament: {filament_types}")


if __name__ == "__main__":
    filename = Path(datapath, fn)
    filaments, radii, edges, filament_types = read_imaris.read_filaments(filename)
    i_fil = 2  # select which traced filament to plot
    # figure, ax = mpl.subplots(2, 2, figsize=(10,10))
    # for i_fil in range(0, 3):
    #     plot_filaments_swc(
    #         filament=filaments[i_fil],
    #         radii=radii[i_fil],
    #         edges=edges[i_fil],
    #         filament_types=filament_types[i_fil],
    #         # ax = ax[i_fil//2, i_fil%2],
    #         i_fil=i_fil,
    #         swcgen=True,
    #     )
    # figure.tight_layout()
    # mpl.show()

    filament_to_swc(
        filaments_xyz=filaments,
        radii=radii,
        edges=edges,
        filament_types=filament_types,
        savename="test",
    )
