from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np
import pyqtgraph as pg
import tables
from mpl_toolkits.mplot3d.proj3d import proj_transform

""" read_imaris: Read and provide a basic plot of filament tracing data from Imaris files.

There are several functions in this source that do other bits with the imaris files.


Returns
-------
multiple arrays:
    filaments_xyz: list of numpy arrays of shape (n, 3) where n is the number of points in the filament
    radii: list of numpy arrays of shape (n,) where n is the number of points in the filament
    edges: list of numpy arrays of shape (n, 2) where n is the number of edges in the filament
    filament_types: list of strings, one for each filament, indicating the type of filament

    
Raises
------
FileNotFoundError
    if the requested file is not found.

"""


def distance_along_path(filament):
    """
    Calculate the distance along a path defined by a set of points
    """
    d = np.zeros(len(filament))
    for i in range(1, len(filament)):
        d[i] = d[i - 1] + np.linalg.norm(filament[i] - filament[i - 1])
    return d


def read_filaments(filename, verbose: bool = False):

    if not filename.is_file():
        raise FileNotFoundError(f"File {filename} not found")
    filaments_xyz = []
    radii = []
    edges = []
    filament_types = []
    with tables.open_file(filename, "r") as hf:
        # print(filename)
        contents = dir(hf.root.Scene8.Content)
        # print(dir(hf.root))
        # dataset = hf.root.DataSet
        # print("Dataset: ", dataset)
        # print(dir(dataset))
        # print("Dataset attrs: ", hf.root.__getattr__("DataSet").read())
        # exit()

        if verbose:
            for content in contents:
                if content.startswith("_"):
                    continue

                fdir = dir(hf.root.Scene8.Content.__getattr__(content))
                # print("    content: ", fdir, "\n")

                for s in fdir:
                    if s.startswith("_"):
                        continue
                    print("\n    Name: ", s)
                    try:
                        print("     ", hf.root.Scene8.Content.Filaments0.__getattr__(s).read())
                    except:
                        print("     ", hf.root.Scene8.Content.Filaments0.__getattr__(s))

        # print(hf.root.Scene8.Content.Filaments0.Edge.read())
        # print(hf.root.Scene8.Content.Filaments0.Vertex.read())
        # print(hf.root.Scene8.Content.Filaments0.LabelGroupNames.read())
        # print(hf.root.Scene8.Content.Filaments0.LabelSets.read())

        for c in contents:
            if c.startswith("Filaments"):
                fildir = hf.root.Scene8.Content.__getattr__(c)
                # print("Filament dir: ", c, "\n", dir(fildir))
                # print("   attrs: ", list(fildir._f_getattr('Name')))
                position_x = fildir.Vertex.col("PositionX")
                position_y = fildir.Vertex.col("PositionY")
                position_z = fildir.Vertex.col("PositionZ")
                radius = fildir.Vertex.col("Radius")
                edge1 = fildir.Edge.col("VertexA")
                edge2 = fildir.Edge.col("VertexB")
                edge = np.array([edge1, edge2]).T
                filtype = fildir._f_getattr("Name")[0].decode()
                filament_types.append(
                    [filtype] * len(position_x)
                )  # must be a type for every position.
                filament_pos = np.array([position_x, position_y, position_z]).T
                filaments_xyz.append(filament_pos)
                edges.append(edge)
                radii.append(radius)

    return filaments_xyz, radii, edges, filament_types


def plot_filament_pg(filament, radii):
    app = pg.QtWidgets.QApplication([])
    # window = GL.GLViewWidget()
    window = pg.GraphicsLayoutWidget(show=True, title="Filament")
    window.setGeometry(100, 100, 800, 800)
    window.show()
    filament_distance = np.linalg.norm(filament - filament[0], axis=1)
    # print(filament_distance.max())
    # print(filament_distance.min())
    filament_distance = (filament_distance - filament_distance.min()) / filament_distance.max()
    min = filament_distance.min()
    max = filament_distance.max()
    # print("min max distance: ", min, max)
    nPts = len(filament_distance)
    colormap_name = "viridis"
    colormap = pg.colormap.get(colormap_name)
    valueRange = np.linspace(0, 1, num=nPts, endpoint=True)
    colors = colormap.getLookupTable(0, 1.0, nPts=nPts)
    pens = colors[np.searchsorted(valueRange, filament_distance)]
    dvar = GL.GLLinePlotItem(pos=filament, color=pens, width=15, antialias=True)
    window.addWidget(dvar)
    app.exec()


def plot_scene(data):
    app = pg.QtWidgets.QApplication([])
    pg.show(data)
    app.exec()


def read_attribute(hf, location, attrib):
    """
    Location should be specified as a path:  for example
    'DataSet/ResolutionLevel 0/TimePoint 0/Channel 1'

    attrib is a string that defines the attribute to extract: for example
    'ImageSizeX'
    """
    print(location, attrib)
    print(list(hf[location].attrs.values()))
    print("items: ", list(hf[location].attrs.items()))
    return str(hf[location].attrs[attrib], encoding="ascii")


def show_thumbnail(hf):
    app = pg.QtWidgets.QApplication([])
    data = hf.root.Thumbnail.Data.read()
    pg.show(data)
    app.exec()


def plot_data(dax, day: None):
    app = pg.QtWidgets.QApplication([])
    if day is None:
        pg.plot(dax)
    else:
        idsort = np.argsort(dax)
        pg.plot(dax[idsort], day[idsort])
    app.exec()


def show_raw_image(hf):
    app = pg.QtWidgets.QApplication([])
    print("Reading image data: wait ...")
    image_data = (
        hf.root.DataSet._f_get_child("ResolutionLevel 0")
        ._f_get_child("TimePoint 0")
        ._f_get_child("Channel 0")
        ._f_get_child("Data")
        .read()
    )
    print("data read")
    pg.show(image_data.T)
    app.exec()


def show_swc_cell_using ngauge(filename, ax=None):
    cell = ngauge.Neuron.from_swc(f"{filename}")
    if ax is None:
        f, ax = mpl.subplots(1, 1)
    ax.axis("equal")
    u = cell.plot(ax=ax, color="k")
    ax.axis("equal")
    mpl.show()


if __name__ == "__main__":
    filname ="make a filename"
    filaments, radii, edges, filament_types = read_filaments(filename)
