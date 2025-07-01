import math
from pathlib import Path
from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import scipy.interpolate
import shapely
import sklearn.cluster
import sympy
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from pylibrary.plotting import plothelpers as PH
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    SpectralClustering,
)
from sklearn.datasets import make_classification

color_map = "gist_rainbow"


def cochlear_percent_distance_to_frequency(functype, d, verbose=False):
    # in this function, all distances are relative to the distance from the apex of the cochlea
    # so we need to invert the d values in some cases where the base is the equation reference
    # point.
    # d is the percent distance from the apex.
    # returns frequency, given % distance
    if functype == "Ou":
        f = 1460 * np.power(10.0, (0.0177 * d))  # percentage distance
        minf = 1460 * np.power(10.0, (0.0177 * 0))
        maxf = 1460 * np.power(10.0, (0.0177 * 100))

    elif functype == "Ou_Normal":
        f = 2553 * np.power(10.0, (0.0140 * d))  # percentage distance
        minf = 2553 * np.power(10.0, (0.0140 * 0))
        maxf = 2553 * np.power(10.0, (0.0140 * 100))

    elif functype == "Muller":
        d = 100 - d
        f = 10.0 ** ((156.5 - d) / 82.5)
        f = f * 1000.0  # Muller et al put f in kHz, not Hz
        minf = 1e3 * 10.0 ** ((156.5 - 100) / 82.5)
        maxf = 1e3 * 10.0 ** ((156.5 - 0) / 82.5)

    elif functype == "Ehret":  # Ehret,
        minf = 3350 * (np.power(10.0, 0.21 * 0) - 1)
        maxf = 3350 * (np.power(10.0, 0.21 * 7.0) - 1)
        dmm = (d / 100) * 7.0  # distance in mm from apex based on percentage
        f = 3350 * (np.power(10.0, 0.21 * dmm) - 1)

    elif functype == "Muniak":
        # relative to apex, normalization is to 1, not 100.
        dn = d / 100.0
        f = 5404 * (10.0 ** (1.16 * dn) - 0.27)
        minf = 5404 * (10.0 ** (1.16 * 0) - 0.27)
        maxf = 5404 * (10.0 ** (1.16 * 1) - 0.27)

    elif functype == "Wang":
        f = 2.109 * (np.power(10, (100.0 - d) * 0.0142) - 0.7719)
        f = f * 1000.0  # f is in kHz here.
        minf = 1e3 * 2.109 * (np.power(10, (100.0 - 100) * 0.0142) - 0.7719)
        maxf = 1e3 * 2.109 * (np.power(10, (100.0 - 0) * 0.0142) - 0.7719)

    else:
        raise ValueError(f"Unknown cochlear distance function: {functype}")
    if verbose:
        print(f" {functype:>12s} {f:8.1f} [{minf:8.1f} -- {maxf:8.1f}], d= {d:5.1f} %")
    return f, minf, maxf


def dist_from_freq_Muniak(f):
    """Convert frequency to cochlear distance."""
    # d is the percent distance from the apex.
    # returns frequency, given % distance
    dn = np.log10((f / 5404.0) + 0.27) / 1.16
    return dn * 100.0  # return as percentage distance


from itertools import combinations

import numpy as np

# import open3d as o3d
# import polar
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def cal_boundary(points, k=30, save_filename=None, visualize=False):
    neighbors = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = neighbors.kneighbors(points)

    max_b = 0
    min_b = 1000000000

    boundary = []
    for i in tqdm(range(points.shape[0]), "cal_boundary"):
        is_boundary = False
        p_neighbor, p_distance = indices[i], np.round(distances[i], 5)
        neighbor_points = points[p_neighbor[1:]]

        p_mean = np.mean(p_distance)
        p_std = np.std(p_distance)
        local_resol = round(p_mean + 2 * p_std, 5)

        if local_resol > max_b:
            max_b = local_resol
        if min_b > local_resol:
            min_b = local_resol

        pairs = list(combinations(p_neighbor[1:], 2))
        for j in range(len(pairs)):
            count = 0
            p1 = points[i]
            p2 = points[pairs[j][0]]
            p3 = points[pairs[j][1]]
            c = Circle(p1, p2, p3)
            if c.radius == None:
                continue

            if c.radius >= local_resol:
                cn_distance = np.linalg.norm((neighbor_points - c.center), axis=1)
                cn_distance = np.round(cn_distance, 5)

                for k in range(len(cn_distance)):
                    if cn_distance[k] <= c.radius:
                        count += 1

                        if count > 3:
                            break

            if count == 3:
                boundary.append(points[i])
                is_boundary = True
                break

        # if not is_boundary :
        #     pol = polar.Polar(np.array(points[i]), neighbor_points, normalize=True)

    # print(f"len : {len(boundary)}")
    # print(f"{min_b}")
    # print(f"{max_b}")

    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(np.pad(np.array(boundary), (0, 1), 'constant', constant_values=0))
    # points = np.array(pc.points)
    points = np.array(boundary)
    # if visualize:
    #     o3d.visualization.draw_geometries([pc])

    # if save_filename != None :
    #     np.savetxt(save_filename, points[:, :2])

    return points


# def order_points_by_nearest(points: List[Point]) -> List[Point]:
#     """Orders a list of points into a line based on nearest neighbor."""
#     if not points:
#         return []
#     ordered_points = [points.pop(0)]
#     while points:
#         current_point = ordered_points[-1]
#         closest_point = None
#         min_distance = float("inf")
#         for point in points:
#             distance = current_point.distance(point)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_point = point
#         ordered_points.append(closest_point)
#         points.remove(closest_point)
#     return ordered_points


def compute_splines(coord_pairs, npoints: int = 100, remove_ends=False):
    coordinates_x = np.array([x[0] for x in coord_pairs])
    coordinates_y = np.array([x[1] for x in coord_pairs])

    dist = np.sqrt(
        (coordinates_x[:-1] - coordinates_x[1:]) ** 2
        + (coordinates_y[:-1] - coordinates_y[1:]) ** 2
    )
    cumul_dist = np.concatenate(([0], dist.cumsum()))
    b_spline, u = scipy.interpolate.splprep([coordinates_x, coordinates_y], u=cumul_dist, s=0)
    if remove_ends:
        cumul_dist = cumul_dist[1:-1]

    xx = np.linspace(cumul_dist[0], cumul_dist[-1], npoints)
    xx, yy = scipy.interpolate.splev(xx, b_spline)
    return xx, yy


def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = mpl.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    cmap = mpl.colormaps["Spectral"]
    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    # print("  Info from clustering; ", title)
    mpl.tight_layout()


class Arrow3D(FancyArrowPatch):
    """A 3D arrow class for matplotlib that can be used to draw arrows in 3D space.
    from : https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    """

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)


def plot_cluster_points(X, yhat, ax, show_colorbar: bool = False):
    clusters = np.unique(yhat)
    # create scatter plot for samples from each cluster
    # print("Number of clusters to plot: ", len(clusters))

    cmap = mpl.colormaps[color_map]
    colors = [cmap(each) for each in np.linspace(0, 1, 100)]
    fr = np.zeros(100)
    for j in range(100):
        fr[j], fmin, fmax = cochlear_percent_distance_to_frequency(
            "Muniak", float(j), verbose=False
        )
    # print(fr)
    for i, cluster in enumerate(clusters):
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)[0]
        # print(X[row_ix, 0])
        frqs = [
            int(fr[int(f)] / 1000.0) for f in X[row_ix, 0]
        ]  # use the first column for cochlear distance
        fr_cols = [colors[int(f)] for f in frqs]  # map to colors
        ax.scatter(
            X[row_ix, 1],
            X[row_ix, 2],
            s=9,
            marker="o",
            facecolor=fr_cols,
            edgecolor=None,
            label=f"Cluster {cluster}",
            alpha=0.5,
        )
    if show_colorbar:
        return
        from matplotlib import cm

        norm = cm.colors.Normalize(vmin=fmin, vmax=fmax)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = mpl.gcf().colorbar(
            cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="vertical"
        )
        cbar.set_label("Cochlear Distance (%)", rotation=270, labelpad=15)
        frs = np.array([4.0, 8.0, 16.0, 32.0, 64.0]) * 1000.0
        fr_labels = [f"{f/1000.:.1f} kHz" for f in frs]
        print(frs, fr_labels)
        fr_ticks = [int(dist_from_freq_Muniak(f)) for f in frs]
        print(fr_ticks)
        print("fr_ticks: ", fr_ticks)
        cbar.set_ticks(fr_ticks, labels=fr_labels)


def cluster_points(
    X: List,
    n_clusters: int = 3,
    algorithm: str = "KMeans",
    centroids: Union[list, np.ndarray] = None,
    epsilon: float = 20.0,
    n_neighbors: int = 10,
    mode: str = "Place-CN",
    ax=None,
    show_colorbar: bool = False,
) -> List:
    scale = 1.0
    params = {
        "min_cluster_size": 10,
        "cluster_selection_epsilon": epsilon,
        "n_jobs": 12,
        "allow_single_cluster": True,
        "n_neighbors": n_neighbors,
    }
    # print(f"Clustering with parameters: {params}")
    # print("Incoming data shape: ", np.shape(X))
    Xo = X.copy()
    # X = np.array(X) - np.mean(X, axis=0)  # center the data
    # X = X / np.std(X, axis=0)  # scale the data
    # convert X to a numpy array.
    # X = np.array([[point[0].evalf(), point[1].evalf(), point[2].evalf()] for point in X])

    # connectivity matrix for structured Ward
    match algorithm:
        case "HDBSCAN":
            model = sklearn.cluster.HDBSCAN(
                min_cluster_size=60,
                cluster_selection_epsilon=epsilon,
                n_jobs=12,
                allow_single_cluster=True,
            )

        case "ward":
            from sklearn.neighbors import kneighbors_graph

            connectivity = kneighbors_graph(
                X, n_neighbors=params["n_neighbors"], include_self=False
            )
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            model = sklearn.cluster.AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward", connectivity=connectivity
            )

        case "Birch":
            model = sklearn.cluster.Birch(
                threshold=params["cluster_selection_epsilon"], n_clusters=n_clusters
            )

        case "KMeans":
            if centroids is None:
                print("Using KMeans with random initialization")
                model = sklearn.cluster.KMeans(
                    n_clusters=n_clusters,
                    n_init=20,
                    max_iter=1000,  # algorithm="elkan",
                    tol=1e-5,
                    init="k-means++",
                    random_state=42,
                )
            else:
                print("Using KMeans with provided centroids")
                model = sklearn.cluster.KMeans(
                    n_clusters=n_clusters,
                    n_init=20,
                    max_iter=5000,
                    algorithm="elkan",
                    tol=1e-6,
                    init=centroids,
                    random_state=39,
                )

        case "AffinityProp":
            # this tends to give small clusters related to narrow cochlear distances.
            model = AffinityPropagation(
                damping=0.9, preference=-10, affinity="euclidean", convergence_iter=50
            )
        case "Spectral":
            model = SpectralClustering(
                n_clusters=n_clusters,
                n_init=100,
                n_jobs=12,
                affinity="nearest_neighbors",  # note rbf may be very slow
                n_neighbors=params["n_neighbors"],
                assign_labels="kmeans",
                random_state=42,
            )

        case _:
            raise ValueError(
                f"Unknown clustering algorithm: {alg}. Use 'HDBSCAN', 'ward', 'Birch', 'KMeans', 'AffinityProp', or 'Spectral'."
            )

    if mode == "Place-CN":  # use only some data - here the cochlear distance
        # print(X.shape)
        # Xm = X[:, 0].reshape(-1, 1)  # use only the first column (cochlear distance) for clustering
        Xm = X[:, (0, 1, 2)]  # print("Xm: ", Xm.shape, Xm)
        # mpl.hist(X[:,0], bins=100, color="grey", alpha=0.5)
        # mpl.show()
        # exit()
    else:
        Xm = X  # use all the columns
    # print(Xm.shape, mode)
    model.fit(Xm)
    # assign a cluster to each example
    if hasattr(model, "labels_"):
        yhat = model.labels_.astype(int)
    else:
        yhat = model.predict(Xm)
    # retrieve unique clusters

    if ax is None:
        return yhat

    plot_cluster_points(Xo, yhat, ax, show_colorbar=show_colorbar)

    return yhat


def convert_sympy_points_to_coords(xypoints):
    """Convert a list of sympy points to a shapely LineString.
    if xypoints is a Segment2D, unpack it to a list of points."""
    if not isinstance(xypoints, (list, tuple)):
        raise ValueError("Input must be a list or tuple of sympy points.")
    if isinstance(xypoints[0], sympy.Segment2D):
        # unpack the Segment2D to a list of points
        xypoints = [xypoints[0].p1, xypoints[0].p2]
    coords = []
    x = xypoints[0][0].evalf()
    y = xypoints[0][1].evalf()
    coords.append((x, y))
    if len(xypoints) > 1:
        x = xypoints[1][0].evalf()
        y = xypoints[1][1].evalf()
        coords.append((x, y))
    return coords


def find_min_max_axes(boundary):
    """Find the shortest line in a boundary polygon that goes through the centroid,
    and the longest line crossing the boundary polygon that is perpendicular to that line.

    Parameters
    ----------
    boundary : shapely.geometry.Polygon
        The boundary polygon to find the axes in.

    Returns
    -------
    tuple
        A tuple containing the shortest line and the longest line.
        The shortest line is a shapely LineString that goes through the centroid of the boundary polygon.
        The longest line is a shapely LineString that crosses the boundary polygon and is perpendicular to the shortest line.

    """
    if not isinstance(boundary, shapely.geometry.Polygon):
        raise ValueError("Boundary must be a shapely Polygon.")

    min_length = float("inf")
    shortest_line = None
    longest_line = None
    centroid = boundary.centroid
    # first find the shortest line through the centroid
    lfac = 2000  # length factor to extend the line beyond the boundary
    for i in range(360):  # brute force search for the shortest line
        # by rotatiing it around the centroid
        ray = shapely.LineString(  # make a line through the centroid.
            [
                (
                    centroid.x - lfac * np.cos(np.radians(i)),
                    centroid.y - lfac * np.sin(np.radians(i)),
                ),
                (
                    centroid.x + lfac * np.cos(np.radians(i)),
                    centroid.y + lfac * np.sin(np.radians(i)),
                ),
            ]
        )
        intersection = boundary.intersection(ray)  # find the intersection with the boundary
        length = intersection.length
        if length < min_length:
            min_length = length
            shortest_line = shapely.LineString(intersection.coords)
    # now find the longest line that is perpendicular to the shortest line
    if shortest_line is None:
        raise ValueError("No intersection found with the boundary polygon.")

    # shortest_line holds the shapely LineString of the shortest line
    # now find the longest line that is perpendicular to the shortest line
    perp_length = shortest_line.length * 10.0  # length of the perpendicular line
    for t in np.linspace(0, 1, 20):
        # get a parameterized point on the shortest line - this is the point where the perpendicular line will be drawn
        pt = shortest_line.line_interpolate_point(t, normalized=True)
        # now get a perpendicular line to the shortest line at this point
        offset_dist = perp_length / 2.0
        left = shortest_line.parallel_offset(offset_dist, "left")
        right = shortest_line.parallel_offset(offset_dist, "right")

        # Find the points on the left and right offset lines corresponding to the original point
        # make a simple linestring for each of the points, and then
        # make a final linestring
        left_point = left.interpolate(shortest_line.project(pt))
        right_point = right.interpolate(shortest_line.project(pt))
        li = shapely.LineString([left_point, right_point])
        long_intersection = boundary.intersection(li)
        if isinstance(long_intersection, shapely.geometry.MultiLineString):
            # if the intersection is a MultiLineString, skip it
            # but we probably should go through each line in the MultiLineString
            # and figure out if it is the longest line of the group (e.g.,
            # if the line crossed the boundary twice in one direction))
            continue
        li = shapely.LineString(long_intersection)
        if longest_line is None or li.length > longest_line.length:
            longest_line = li
    return shortest_line, longest_line


def test_find_min_max_axes():
    """test_find_min_max_axes:
    Create an elliptical polygon, rotate it, and test the find_min_max_axes function.
    Plot each of the ellipses with the shortest and longest axes.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    a = 1
    b = 4
    astep = 5
    ngrid = len(np.arange(0, 360, astep))
    nr, nc = PH.getLayoutDimensions(ngrid)
    fig, axf = mpl.subplots(nr, nc, figsize=(12, 12))
    axr = axf.ravel()
    noise = 0.1  # if you make this too large, the ellipse will not be recognized as an ellipse
    # and the geometry will fail
    for i, angle in enumerate(np.arange(0, 360, astep)):
        ellipse = shapely.geometry.Polygon(
            [
                (
                    (a * (1 + noise * np.random.random())) * np.cos(theta),
                    (b * (1 + noise * np.random.random())) * np.sin(theta),
                )
                for j, theta in enumerate(np.linspace(0, 2 * np.pi, 50))
            ]
        )
        ellipse = shapely.affinity.rotate(ellipse, angle)  # rotate the ellipse for testing
        shortest_line, longest_line = find_min_max_axes(ellipse)
        ax = axr[i]
        ax.plot(*ellipse.exterior.xy, color="blue", label="Boundary")
        ax.plot(*shortest_line.xy, color="red", label="Shortest Line")
        if longest_line is not None:
            ax.plot(*longest_line.xy, color="green", label="Longest Line")
        ax.set_aspect("equal", adjustable="box")
    mpl.show()


def compute_geometry_measures(boundary: shapely.geometry.Polygon, scale:bool=True) -> dict:
    """compute_geometry_measures : Calculate various geometric measures from
    a boundary polygon.
    Computed measures in the returned dictionary include:
    - area
    - centroid
    - cir_circle (circumference circle ratio)
    - perimeter
    - shape_index
    - fractal_dimension
    - short_axis (length of the shortest axis)
    - long_axis (length of the longest axis)
    - eccentricity (ratio of the longest to shortest axis)
    - shortest_line (the shortest line through the centroid)
    - longest_line (the longest line crossing the boundary polygon that is perpendicular to the shortest line)


    Parameters
    ----------
    boundary : shapely.geometry.Polygon
        The enclosing polygon for which to compute geometric measures.

    Returns
    -------
    dict

    """
    measures = {}
    measures["area"] = np.abs(boundary.area)
    measures["centroid"] = np.array(boundary.centroid.coords[0])
    measures["cir_circle"] = 1.0 - (
        measures["area"] / shapely.minimum_bounding_circle(boundary).area
    )
    measures["perimeter"] = boundary.length
    measures["shape_index"] = (
        0.25 * measures["perimeter"] / np.sqrt(float(measures["area"]))
    )  # shape index, see https://en.wikipedia.org/wiki/Shape_index
    measures["fractal_dimension"] = (
        2.0 * np.log(float(measures["perimeter"]) / 4.0) / np.log(float(measures["area"]))
    )  # fractal dimension, see https://en.wikipedia.org/wiki/Fractal_dimension
    shortest_line, longest_line = find_min_max_axes(boundary)
    measures["short_axis"] = shortest_line.length
    measures["long_axis"] = longest_line.length if longest_line is not None else 0.0
    measures["eccentricty"] = (
        longest_line.length / shortest_line.length if shortest_line.length > 0 else 0.0
    )
    measures["shortest_line"] = shortest_line
    measures["longest_line"] = longest_line if longest_line is not None else None
    return measures


def show_paraview_csv(
    basepath: Path,
    csv_file: str,
    parameters={
        "n_clusters": 3,
        "epsilon": 20,
        "hulltype": ["convex"],
        "clustering_algorithm": "KMeans",
    },
    mode="Place-CN",  # or "fxyz" for place and location
    ax=None,
    plotting: bool = True,
    show_colorbar: bool = False,
):  # noqa: E501
    """
    Reads a CSV file generated by ParaView and plots the data.

    Parameters
    ----------
    basepath: Path
        Base path where the CSV file is located.
    csv_file (Path)
        Path to the CSV file to be read.

    parameters dict): Dictionary containing clustering parameters.

    mode (str): Mode of operation, either "Place-CN" or "fxyz".

    ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes are created.

    show_colorbar (bool): Whether to show a colorbar for the frequency map.

    """
    # Read the CSV file
    datafile = Path(basepath, csv_file)
    df = pd.read_csv(datafile, header=0, skiprows=0, skipinitialspace=True)
    print("Reading data from ", datafile.name)
    # Plot the data
    ifr = 0
    axc = None
    if ax is None and plotting:
        f, ax = mpl.subplots(1, 1, figsize=(6, 10))
        ax = [ax]
    elif plotting:
        axc = ax
  
    ordered_points = [
        list(p) for p in zip(df["Points:0"].values, df["Points:1"].values, df["Points:2"].values)
    ]
    # outlier = np.where((np.array(ordered_points[0]) > 650) & (np.array(ordered_points[1]) > 650))[0]
    # # print("outliers: ", outlier)
    # for ol in outlier:
    #     print(ordered_points[0][ol], ordered_points[1][ol])
    # ax[ifr].scatter(
    #     np.array(ordered_points[0]),
    #     np.array(ordered_points[1]),
    #     s=10,
    #     marker="X",
    #     color="grey",
    #     label=f"Place-CN: {slice_location:.1f}, Freq (kHz): {slice_freq:.1f}",
    # )

    clust_points = np.array(
        [df["Place-CN"].values, df["Points:0"].values, df["Points:1"].values, df["Points:2"].values]
    ).T
    if parameters["n_clusters"] > 1:
        # print(
        #     f"clustering... ncl: {parameters['n_clusters']:d} with eps={parameters['epsilon']}  ",
        #     end="",
        # )

        yhat = cluster_points(
            clust_points,
            n_clusters=parameters["n_clusters"],
            algorithm=parameters.get("clustering_algorithm", "KMeans"),
            epsilon=parameters["epsilon"],
            centroids=parameters.get("centroids", None),
            n_neighbors=parameters.get("n_neighbors", 10),
            mode=mode,
            ax=axc,
            show_colorbar=show_colorbar,
        )
    else:
        yhat = [0] * len(clust_points)  # if only one cluster, assign all to cluster 0
        if plotting:
            plot_cluster_points(clust_points, yhat, axc, show_colorbar=show_colorbar)
    # print("   done")
    cmap = mpl.colormaps["Spectral"]
    unique_clusters = np.unique(yhat)
    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_clusters))]
    # print("Clusters: ", unique_clusters)
    measures = [None] * len(unique_clusters)
    locs = np.zeros(len(unique_clusters))
    freqs = np.zeros(len(unique_clusters))
    for i, yhat_i in enumerate(unique_clusters):
        bound = parameters["hulltype"][i]
        row_ix = np.where(yhat == yhat_i)[0]
        slice_location = np.mean(df["Place-CN"][row_ix])
        slice_freq, _, _ = cochlear_percent_distance_to_frequency("Muniak", slice_location)
        locs[i] = slice_location
        freqs[i] = slice_freq
        match bound:
            case "BPD":  # the bpd code is not working...
                c_points = np.array(
                    [p for p in zip(df["Points:0"][row_ix], df["Points:1"][row_ix])]
                )
                boundary = cal_boundary(c_points, k=100)
                if plotting:
                    axc.plot(
                        boundary[:, 0],
                        boundary[:, 1],
                        linestyle="-",
                        marker=None,
                        color=colors[i],
                        alpha=1,
                        linewidth=0.5,
                        # label="Concave Hull",
                    )
                bpd_points = shapely.Polygon([(p[0], p[1]) for p in boundary])
                measures[i] = compute_geometry_measures(bpd_points)
            case "convex":
                c_points = [
                    shapely.Point(p) for p in zip(df["Points:0"][row_ix], df["Points:1"][row_ix])
                ]
                poly_points = shapely.Polygon([(p.x, p.y) for p in c_points])
                cvh = shapely.convex_hull(poly_points)  # , ratio=0.1)
                boundary = np.asarray(cvh.boundary.coords)
                if plotting:
                    axc.plot(
                        boundary[:, 0], boundary[:, 1], linestyle="-", color=colors[i], alpha=0.5
                    )
                measures[i] = compute_geometry_measures(cvh)

            case "concave":
                c_points = [
                    shapely.Point(p) for p in zip(df["Points:0"][row_ix], df["Points:1"][row_ix])
                ]
                poly_points = shapely.Polygon([(p.x, p.y) for p in c_points])
                ccv = shapely.concave_hull(poly_points, ratio=0.1)

                boundary = np.asarray(ccv.boundary.coords)
                measures[i] = compute_geometry_measures(ccv)
                if plotting:
                    axc.plot(
                        boundary[:, 0],
                        boundary[:, 1],
                        linestyle="-",
                        marker=None,
                        color=colors[i],
                        alpha=1,
                        linewidth=0.5,
                        # label="Concave Hull",
                    )
        sl = np.array(measures[0]["shortest_line"].coords)
        ll = np.array(measures[0]["longest_line"].coords)
        if i == 0 and plotting:
            axc.scatter(
                measures[0]["centroid"][0],
                measures[0]["centroid"][1],
                s=7,
                marker="o",
                facecolor="blue",
                edgecolor="black",
                label="Centroid",
            )
        
            axc.plot(
                sl[:, 0], sl[:, 1], color="k", linestyle="-", linewidth=0.75, label="Short Axis"
            )
            if measures[0]["longest_line"] is not None:
                axc.plot(
                    ll[:, 0],
                    ll[:, 1],
                    color="w",
                    linestyle="-",
                    linewidth=0.75,
                    label="Long Axis",
                )
    if plotting:
        axc.set_xlabel("X (um)")
        axc.set_ylabel("Y (um)")
        axc.legend()
        axc.grid()
        mpl.title("Data from " + datafile.name)
        area_str = " ".join(f"{a['area']:.0f}" for a in measures)
        area_str = "[" + area_str + "]"
        loc_str = " ".join(f"{l:.1f}" for l in locs)
        loc_str = "[" + loc_str + "]"
        freqs_str = " ".join(f"{f/1000.:.1f}" for f in freqs)
        freqs_str = "[" + freqs_str + "]"
        # print(measures.keys())
        axc.set_title(
            f"{datafile.name:s} loc: {loc_str!s} fr: {freqs_str!s} kHz\nArea: {area_str!s} um2",
            fontsize=8,
        )

    # print a report on the measurements.
    print("\nData file: ", datafile.name)
    for i, m in enumerate(measures):
        if i > 0:
            continue
        if m is not None:
            m["location"] = locs[i]
            m["frequency"] = freqs[i]
            print(
                f"Cluster {i:2d} - Loc: {locs[i]:4.1f}%  Freq: {freqs[i]/1000.:4.1f} kHz  Area: {m['area']:8.0f} um2, Perimeter: {m['perimeter']:8.0f} um, "
                f"Shape Index: {m['shape_index']:6.3f}, Fractal Dimension: {m['fractal_dimension']:6.3f}, "
                f"Circumference Circle Ratio: {m['cir_circle']:6.3f} Eccentricty: {m['eccentricty']:6.3f}"
            )

    return measures


def exemplar_cells():
    """Returns a list of example cells with their parameters."""
    cells = [
        {
            "medial_lateral_distance": 0.0004908971577306055,
            "area": np.float64(2.970099437467582e-07),
            "cir_circle": 0.40372642986393903,
            "perimeter": 0.002041676530665728,
            "shape_index": 0.9365726050996317,
            "fractal_dimension": 1.0087199482372742,
            "short_axis": 0.00048039683540399626,
            "long_axis": 0.000793182540515296,
            "eccentricty": 1.651098596118474,
            "shortest_line": np.array([[-0.0008, -0.0028], [-0.0003, -0.0027]]),
            "longest_line": np.array([[-0.0007, -0.0024], [-0.0006, -0.0031]]),
            "rostral_caudal_distance": 0.0007989412716500584,
            "mosaic_file": "2023.11.10.S1.mosaic",
        },
        {
            "medial_lateral_distance": 0.0004563885141439321,
            "area": np.float64(3.166764984037725e-07),
            "cir_circle": 0.46120730339385274,
            "perimeter": 0.0021765649723480595,
            "shape_index": 0.9669493534476478,
            "fractal_dimension": 1.0044915863660608,
            "short_axis": 0.0004534792928760983,
            "long_axis": 0.0008533643134455893,
            "eccentricty": 1.881815392348575,
            "shortest_line": np.array([[-0.0017, -0.0036], [-0.0013, -0.0036]]),
            "longest_line": np.array([[-0.0017, -0.0032], [-0.0016, -0.004]]),
            "rostral_caudal_distance": 0.0008474862397959731,
            "mosaic_file": "2024.05.16.S2.mosaic",
        },
        {
            "medial_lateral_distance": 0.0005219100593689208,
            "area": np.float64(4.4189519759133885e-07),
            "cir_circle": 0.5577543879436503,
            "perimeter": 0.0026867138995547743,
            "shape_index": 1.0104196648708927,
            "fractal_dimension": 0.9985831577999942,
            "short_axis": 0.000521475759010142,
            "long_axis": 0.0011289865658754541,
            "eccentricty": 2.164983791420105,
            "shortest_line": np.array([[0.0004, -0.0043], [-0.0001, -0.0042]]),
            "longest_line": np.array([[-0.0001, -0.0049], [0.0002, -0.0038]]),
            "rostral_caudal_distance": 0.0011244681260747672,
            "mosaic_file": "2023.11.10.S0.mosaic",
        },
        {
            "medial_lateral_distance": 0.0004473552541781912,
            "area": np.float64(3.3449881384705525e-07),
            "cir_circle": 0.48930828139788574,
            "perimeter": 0.0022614633245248246,
            "shape_index": 0.9775348887884351,
            "fractal_dimension": 1.0030476635916215,
            "short_axis": 0.00044902611675918635,
            "long_axis": 0.0009117726084721051,
            "eccentricty": 2.0305558506323824,
            "shortest_line": np.array([[-0.0009, -0.0019], [-0.0005, -0.0019]]),
            "longest_line": np.array([[-0.0008, -0.0014], [-0.0007, -0.0023]]),
            "rostral_caudal_distance": 0.0009148949951662046,
            "mosaic_file": "2024.07.29.S2.mosaic",
        },
        {
            "medial_lateral_distance": 0.0004246182700614864,
            "area": np.float64(3.046602754584032e-07),
            "cir_circle": 0.5414089222737148,
            "perimeter": 0.0021892676891974317,
            "shape_index": 0.9915873117429808,
            "fractal_dimension": 1.0011261311725077,
            "short_axis": 0.00041736378885534245,
            "long_axis": 0.0009035609972605338,
            "eccentricty": 2.1649242732308682,
            "shortest_line": np.array([[-0.0005, -0.0051], [-0.0008, -0.0049]]),
            "longest_line": np.array([[-0.0009, -0.0053], [-0.0004, -0.0045]]),
            "rostral_caudal_distance": 0.0009226718191378156,
            "mosaic_file": "2024.08.21.S1.mosaic",
        },
        {
            "medial_lateral_distance": 0.00047987386396314166,
            "area": np.float64(3.7508127512480465e-07),
            "cir_circle": 0.544914964254201,
            "perimeter": 0.002415872308737684,
            "shape_index": 0.9861688778128276,
            "fractal_dimension": 1.0018826098266942,
            "short_axis": 0.0004844836367731988,
            "long_axis": 0.0010092335663737118,
            "eccentricty": 2.0831117704934257,
            "shortest_line": np.array([[-0.0019, -0.0051], [-0.0014, -0.005]]),
            "longest_line": np.array([[-0.0018, -0.0045], [-0.0016, -0.0055]]),
            "rostral_caudal_distance": 0.0010277052044550617,
            "mosaic_file": "2024.05.07.s0.mosaic",
        },
        {
            "medial_lateral_distance": 0.00039247376724662003,
            "area": np.float64(2.3826775423825783e-07),
            "cir_circle": 0.5684092162228827,
            "perimeter": 0.0019871567949352378,
            "shape_index": 1.0177462457829132,
            "fractal_dimension": 0.9976930139397893,
            "short_axis": 0.0003890378416673056,
            "long_axis": 0.0008293281127954013,
            "eccentricty": 2.131741501652222,
            "shortest_line": np.array([[-0.0013, -0.0039], [-0.0017, -0.0037]]),
            "longest_line": np.array([[-0.0018, -0.0041], [-0.0013, -0.0034]]),
            "rostral_caudal_distance": 0.0008411020591520479,
            "mosaic_file": "2024.05.07.s2.mosaic",
        },
        {
            "medial_lateral_distance": 0.00041858726818487914,
            "area": np.float64(2.2687237721154363e-07),
            "cir_circle": 0.4734478064514275,
            "perimeter": 0.0018287591820429518,
            "shape_index": 0.9598551900694902,
            "fractal_dimension": 1.0053563207786076,
            "short_axis": 0.000416411878809748,
            "long_axis": 0.0007377030090931931,
            "eccentricty": 1.771570520999085,
            "shortest_line": np.array([[0.001, -0.0041], [0.0014, -0.0041]]),
            "longest_line": np.array([[0.0011, -0.0037], [0.0012, -0.0045]]),
            "rostral_caudal_distance": 0.000743057187190497,
            "mosaic_file": "2024.01.12.s2.mosaic",
        },
    ]
    return cells


def datasets():
    """Returns a dictionary of datasets of slices extracted from the
    Muniak atlas,  with parameters to guide clustering of different
    regions of the CN.
    """
    rds = {
        "dataset_88.csv": {
            "parameters": {
                "n_clusters": 1,
                "epsilon": 10,
                "hulltype": ["convex"],  # 'concave' or 'convex'
            }
        },
        "dataset_83.csv": {
            "parameters": {
                "n_clusters": 1,
                "epsilon": 10,
                "hulltype": ["convex"],  # 'concave' or 'convex'
            }
        },
        "dataset_76.csv": {
            "parameters": {
                "n_clusters": 1,
                "epsilon": 10,
                "hulltype": ["concave"],  # 'concave' or 'convex'
            }
        },
        "dataset_68.csv": {
            "parameters": {
                "n_clusters": 1,
                "epsilon": 30,
                "hulltype": ["concave"],  # 'concave' or 'convex'
            }
        },
        "dataset_59.csv": {
            "parameters": {
                "n_clusters": 1,
                "epsilon": 30,
                "hulltype": ["concave"],  # 'concave' or 'convex'
            }
        },
        "dataset_54.csv": {
            "parameters": {
                "n_clusters": 1,
                "epsilon": 30,
                "hulltype": ["concave"],  # 'concave' or 'convex'
            }
        },
        "dataset_40.csv": {
            "parameters": {
                "n_clusters": 1,
                "epsilon": 30,
                "hulltype": ["concave"],  # 'concave' or 'convex'
            }
        },
        "dataset_32.csv": {
            "parameters": {
                "n_clusters": 2,
                "epsilon": 10,  # minimum for HDBSCAN here is about 5, but it is not perfect
                "n_neighbors": 20,  # number of neighbors for spectral clustering
                "hulltype": [
                    "concave",
                    "convex",
                ],  # 'concave' or 'convex' or 'BPD' (boundary point detection)
                "centroids": np.array([[30.0, 550.0, 400.0], [70.0, 400.0, 625.0]]),  # dcn  # pvcn
                "clustering_algorithm": "Spectral",  # 'HDBSCAN', 'KMeans', 'AffinityProp', 'Spectral', 'Birch', 'ward'
            }
        },
        "dataset_26.csv": {
            "parameters": {
                "n_clusters": 3,
                "epsilon": 10,
                "n_neighbors": 20,  # number of neighbors for spectral and ward clustering
                "hulltype": ["concave", "convex", "convex"],  # 'concave' or 'convex'
                "centroids": np.array(
                    [
                        [15.0, 600.0, 400.0],  # dcn
                        [70.0, 400.0, 700.0],  # pvcn
                        [90.0, 375.0, 300.0],  # avcn
                    ]
                ),
                "clustering_algorithm": "Spectral",  # 'HDBSCAN', 'KMeans', 'AffinityProp', 'Spectral', 'Birch', 'ward'
            }
        },
        "dataset_02.csv": {
            "parameters": {
                "n_clusters": 2,
                "epsilon": 5,
                "n_neighbors": 100,  # number of neighbors for spectral and ward clustering
                "hulltype": ["concave", "concave", "convex"],  # 'concave' or 'convex'
                "centroids": np.array(
                    [
                        [6.0, 750.0, 500.0],  # dcn
                        [100.0, 480.0, 900.0],  # pvcn
                        # [100.0, 550.0, 450.0],  # avcn
                    ]
                ),
                "clustering_algorithm": "Spectral",  # 'HDBSCAN', 'KMeans', 'AffinityProp', 'Spectral', 'Birch', 'ward'
            }
        },
        # "dataset_01.csv": {
        #     "parameters": {
        #         "n_clusters": 3,
        #         "epsilon": 20,
        #         "hulltype": ["convex"],  # 'concave' or 'convex'
        #     }
        # },
    }
    return rds


from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# def color_line(x, y, z, c, ax, **lc_kwargs):
#     """
#     Plot a line with a color specified along the line by a third value.

#     It does this by creating a collection of line segments. Each line segment is
#     made up of two straight lines each connecting the current (x, y) point to the
#     midpoints of the lines connecting the current point with its two neighbors.
#     This creates a smooth line with no gaps between the line segments.

#     Parameters
#     ----------
#     x, y : array-like
#         The horizontal and vertical coordinates of the data points.
#     c : array-like
#         The color values, which should be the same size as x and y.
#     ax : Axes
#         Axis object on which to plot the colored line.
#     **lc_kwargs
#         Any additional arguments to pass to matplotlib.collections.LineCollection
#         constructor. This should not include the array keyword argument because
#         that is set to the color argument. If provided, it will be overridden.

#     Returns
#     -------
#     matplotlib.collections.LineCollection
#         The generated line collection representing the colored line.
#     """
#     if "array" in lc_kwargs:
#         warnings.warn('The provided "array" keyword argument will be overridden')

#     default_kwargs = {"capstyle": "butt"}
#     default_kwargs.update(lc_kwargs)

#     # Compute the midpoints of the line segments. Include the first and last points
#     # twice so we don't need any special syntax later to handle them.
#     x = np.asarray(x)
#     y = np.asarray(y)
#     z = np.asarray(z)
#     x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
#     y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
#     z_midpts = np.hstack((z[0], 0.5 * (z[1:] + z[:-1]), z[-1]))
#     # Determine the start, middle, and end coordinate pair of each line segment.
#     # Use the reshape to add an extra dimension so each pair of points is in its
#     # own list. Then concatenate them to create:
#     # [
#     #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
#     #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
#     #   ...
#     # ]
#     coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
#     coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
#     coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
#     segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

#     print("1")
#     lc = LineCollection(segments, **default_kwargs)
#     print("2")
#     lc.set_array(c)  # set the colors of each segment
#     print("3")
#     coll = ax.add_collection(lc, zs=z, zdir="z")
#     print("4")
#     return coll


def main(basepath: Path, dataset: dict):
    """
    Main function to run the script.

    Parameters
    ----------
    basepath : Path
        Base path where the CSV files are located.
    dataset : dict
        Dictionary containing dataset file names and their parameters.

    """
    plotting = True
    dataset = datasets()
    nd = len(dataset.keys())
    nr, nc = PH.getLayoutDimensions(nd)
    if plotting:
        fig, ax = mpl.subplots(nr, nc, figsize=(12, 12))
        axr = np.ravel(ax) if isinstance(ax, np.ndarray) else [ax]
    all_meas = {}
    
    # define the measures to assign to each axis
    selections = {"x": "area", "y": "shape_index", "z": "eccentricty"}
    # define the axes used to make the assignments to frequency
    dist_axes = "xyz"  # or "xy", or "xyz" for 3D plotting or "y" or "z" or yz
    for i, csv_file in enumerate(dataset.keys()):
        # if i < 8 or i > 8:
        #     continue
        if plotting:
            ax = axr[i]
        else:
            ax = None
        if i == nd - 1:
            show_colorbar = True
        else:
            show_colorbar = False
        if csv_file not in dataset:
            raise ValueError(f"Dataset {csv_file} not found in predefined datasets.")
        parameters = dataset[csv_file]["parameters"]
        print("Processing dataset:", csv_file, "with parameters:", parameters)
        meas = show_paraview_csv(
            basepath,
            csv_file=csv_file,
            parameters=parameters,
            ax=ax,
            show_colorbar=show_colorbar,
            plotting=plotting,
        )
        all_meas[csv_file] = meas
    # match all the axes
    if plotting:
        for i in range(nd):
            if i == 0:
                xlims = np.array(axr[i].get_xlim())
                ylims = np.array(axr[i].get_ylim())
            else:
                xlims = np.vstack((xlims, axr[i].get_xlim()))
                ylims = np.vstack((ylims, axr[i].get_ylim()))
        xmin = np.min(xlims[:, 0]) - 100.0
        xmax = np.max(xlims[:, 1]) + 100.0
        ymin = np.min(ylims[:, 0]) - 100
        ymax = np.max(ylims[:, 1]) + 100.0
        if plotting:
            for i in range(nd):
                axr[i].set_xlim(xmin, xmax)
                axr[i].set_ylim(ymin, ymax)
            for i in range(nd):
                axr[i].set_aspect("equal", adjustable="box")
            mpl.tight_layout()
    freqs = np.zeros(nd)
    for i in range(nd):
        csv_file = list(dataset.keys())[i]
        meas = all_meas[csv_file]

        freqs[i] = meas[0]["frequency"]
    fr, minf, maxf = cochlear_percent_distance_to_frequency("Muniak", 100.0, verbose=False)
    cmap = mpl.colormaps[color_map]
    frarray = [(f - minf) / (maxf - minf) for f in freqs]
    colors = [cmap(ifreq) for ifreq in frarray]
    for i, c in enumerate(colors):
        c = tuple(float(x) for x in c)
        colors[i] = c  # remove alpha channel for 3D plotting
    # print(colors)
    ax3d = None
    if plotting:
        fig3d = mpl.figure(figsize=(12, 12))
        ax3d = fig3d.add_subplot(111, projection="3d")

    factor_values = np.arange(0.1, 1.0, 0.02)
    nvals = len(factor_values)
    dist = np.zeros(nvals)
    sdist = np.zeros(nvals)
    factor = np.zeros(nvals)
    plotting = False
    for i, test_factor in enumerate(factor_values):
        distances = compute_3d_mapping(
            dataset,
            all_meas,
            selections,
            dist_axes,
            factor = test_factor,
            plotting=plotting,
            cmap=cmap,
            colors=colors,
            ax3d=ax3d
            )
        dist[i] = np.mean(distances)
        sdist[i] = np.std(distances)
        factor[i] = test_factor
    print(sdist)
    
    best_factor = factor[np.argmin(sdist)]

    plotting = True
    distances = compute_3d_mapping(
        dataset,
        all_meas,
        selections,
        dist_axes,
        factor = best_factor,
        plotting=plotting,
        cmap=cmap,
        colors=colors,
        ax3d=ax3d
        )
    
    # if not plotting:  # I know, inverted logic, but this is to test the best fit over a range of area/perimeter factors
    if len(factor_values) > 1:
        fe, axe = mpl.subplots(figsize=(8, 6))
        axe.errorbar(factor, dist, yerr=sdist, marker="o", markersize=3, linestyle="-", color="black")
        axe.set_xlabel(f"{selections['x']:s} Factor")
        axe.set_ylabel("Mean Distance (um)")
        axe.set_title(f"Mean Distance vs {selections['x']} Factor with {selections['y']} and {selections['z']}")
    mpl.show()

import numpy as np
    # fractal dimension and shape_index are highly correlated, so we can use either
    
def compute_3d_mapping(dataset, all_meas, selections, dist_axes, factor: float=0.5, plotting=False,
                       colors=None, cmap=None, ax3d=None):
    nd = len(dataset.keys())
    x_select = selections["x"]
    y_select = selections["y"]
    z_select = selections["z"]
    freq_vals = np.zeros(nd)
    x_vals = np.zeros(nd)
    y_vals = np.zeros(nd)
    z_vals = np.zeros(nd)
    areafactor = (factor**2) * 1.0e12
    perimeterfactor = factor * 1.0e6
    for i in range(nd):
        csv_file = list(dataset.keys())[i]
        parameters = dataset[csv_file]["parameters"]
        meas = all_meas[csv_file]
        freqs = [m["frequency"] for j, m in enumerate(meas) if m is not None and j in [0]]

        # areas = [m["area"] for j, m in enumerate(meas) if m is not None and j in [0]]
        # fractal = [m["fractal_dimension"] for j, m in enumerate(meas) if m is not None and j in [0]]
        # shape_index = [m["shape_index"] for j, m in enumerate(meas) if m is not None and j in [0]]
        # perimeter = [m["perimeter"] for j, m in enumerate(meas) if m is not None and j in [0]]
        # cir_circle = [m["cir_circle"] for j, m in enumerate(meas) if m is not None and j in [0]]
        # eccentricity = [m["eccentricty"] for j, m in enumerate(meas) if m is not None and j in [0]]
        x_val = [m[x_select] for j, m in enumerate(meas) if m is not None and j in [0]]
        y_val = [m[y_select] for j, m in enumerate(meas) if m is not None and j in [0]]
        z_val = [m[z_select] for j, m in enumerate(meas) if m is not None and j in [0]]
        freq_vals[i] = freqs[0]
        x_vals[i] = x_val[0]
        y_vals[i] = y_val[0]
        z_vals[i] = z_val[0]
    if plotting and ax3d is not None:
        ax3d.scatter(x_vals, y_vals, z_vals, s=100, c=colors, alpha=0.8)
    ifr = np.argsort(freq_vals)
    freqs = np.array(freq_vals)[ifr]
    x_vals = x_vals[ifr]
    y_vals = y_vals[ifr]
    z_vals = z_vals[ifr]

    u = scipy.interpolate.make_interp_spline(
        freqs, np.array([x_vals, y_vals, z_vals]).T, k=4
    )  # create a spline for the line representing the chosen variables.
    x_fr_s = np.linspace(np.min(freqs), np.max(freqs), 1000)
    x_vals_s = u(x_fr_s)[:, 0]
    y_vals_s = u(x_fr_s)[:, 1]
    z_vals_s = u(x_fr_s)[:, 2]

    # plot the frequency mapping line in this space
    if plotting and ax3d is not None:
        ax3d.plot(
            x_vals_s,
            y_vals_s,
            z_vals_s,
            label="Smooth Line",
            c="b",
            linewidth=0.25,
            alpha=0.5,
        )
    # add a color map to the line based on the frequency values.
        frarray = [(f - np.min(freqs)) / (np.max(freqs) - np.min(freqs)) for f in x_fr_s]
        b_colors = [cmap(ifreq) for ifreq in frarray]
        ax3d.scatter(x_vals_s, y_vals_s, z_vals_s, s=4, c=b_colors, alpha=0.8)

    # compute the cell locations in this 3d space
    cells = exemplar_cells()
    nc = len(cells)
    x_frc = np.zeros(nc)
    x_vals_i = np.zeros(nc)
    y_vals_i = np.zeros(nc)
    z_vals_i = np.zeros(nc)
    x_best_fr = np.zeros(nc)
    y_best_fr = np.zeros(nc)
    z_best_fr = np.zeros(nc)
    fr_interp = scipy.interpolate.LinearNDInterpolator(
        list(zip(x_vals_s, y_vals_s, z_vals_s)), x_fr_s, rescale=True
    )
    # assemble arrays of points from interpolation
    cm = ["maroon"] * nc
    mk = ["o"] * nc
    mosaic_files = [cell["mosaic_file"] for cell in cells]
    # find shortest distance in the x/y/z space between the
    # interpolated line and the exemplar cells.
    # brute force distance calculation.
    distances = np.zeros(nc)
    for j, cell in enumerate(cells):
        x = cell[x_select]
        y = cell[y_select]
        z = cell[z_select]
        if x_select in "area":
            x = np.array(x) * areafactor
        if y_select == "area":
            y = np.array(y) * areafactor
        if z_select == "area":
            z = np.array(z) * areafactor
        if x_select in "perimeter":
            x = np.array(x) * perimeterfactor
        if y_select == "perimeter":
            y = np.array(y) * perimeterfactor
        if z_select == "perimeter":
            z = np.array(z) * perimeterfactor
        for i, frx in enumerate(x_fr_s):  # check all frequencies along the line
            # print(frx, y, z)
            if np.isnan(frx):
                continue
            # compute the distance in the selected space (one axis, two axes, or three axes)
            match dist_axes:
                case "x":
                    dist = np.sqrt((x - x_vals_s[i]) ** 2)
                case "y":
                    dist = np.sqrt((y - y_vals_s[i]) ** 2)
                case "z":
                    dist = np.sqrt((z - z_vals_s[i]) ** 2)
                case "xy":
                    dist = np.sqrt((x - x_vals_s[i]) ** 2 + (y - y_vals_s[i]) ** 2)
                case "yz":
                    dist = np.sqrt((y - y_vals_s[i]) ** 2 + (z - z_vals_s[i]) ** 2)
                case "xz":
                    dist = np.sqrt((x - x_vals_s[i]) ** 2 + (z - z_vals_s[i]) ** 2)
                case "xyz":
                    dist = np.sqrt(
                        (x - x_vals_s[i]) ** 2 + (y - y_vals_s[i]) ** 2 + (z - z_vals_s[i]) ** 2
                    )
                case _:
                    raise ValueError(f"Invalid dist_axes value: {dist_axes}")
            if i == 0:
                min_dist = dist
                x_best = x_vals_s[i]
                y_best = y_vals_s[i]
                z_best = z_vals_s[i]
                fr_x = frx
            else:
                if dist < min_dist:
                    min_dist = dist
                    x_best = x_vals_s[i]
                    y_best = y_vals_s[i]
                    z_best = z_vals_s[i]
                    fr_x = frx
            # print("y/z select: ", y_select, z_select)
        x_frc[j] = fr_x
        x_vals_i[j] = x
        y_vals_i[j] = y
        z_vals_i[j] = z
        x_best_fr[j] = x_best
        y_best_fr[j] = y_best
        z_best_fr[j] = z_best
        distances[j] = min_dist
   
    if plotting and ax3d is not None:
        # markers where the slice measures are in 3d space
        ax3d.scatter(
            x_vals_i,  # x_frc,
            y_vals_i,
            z_vals_i,
            label="Exemplar Cells",
            c=cm,
            marker="o",
            # markersize=4,
            # linestyle=None,
            # linewidth=1,
            alpha=1.0,
        )
        # markers where the slice measures map to the line
        ax3d.scatter(
            x_best_fr,
            y_best_fr,
            z_best_fr,
            label="on lne",
            c="k",
            marker="x",
            s=24,
            alpha=1.0,
        )
        # text and line connecting the measured lines to the parametric frequency line.
        ordered_cells = np.argsort(x_frc)
        for iun, cell in enumerate(cells):
            i = ordered_cells[iun]
            if i % 2:
                ha = "left"
            else:
                ha = "right"
            ax3d.text(
                x_vals_i[i],
                y_vals_i[i],
                z_vals_i[i],
                f" {Path(mosaic_files[i]).stem:s} {x_frc[i]/1000.:.1f}kHz ",
                color="k",
                fontsize=8,
                ha=ha,
                va="center",
            )
            ax3d.arrow3D(
                x=x_vals_i[i],
                y=y_vals_i[i],
                z=z_vals_i[i],
                dx=-(x_vals_i[i] - x_best_fr[i]),
                dy=-(y_vals_i[i] - y_best_fr[i]),
                dz=-(z_vals_i[i] - z_best_fr[i]),
                mutation_scale=1,
                arrowstyle="-|>",
                linestyle="solid",
                color="k",
                linewidth=0.5,
                alpha=0.5,
                #    label=f"{Path(mosaic_files[i]).stem:s} {x_frc[i]/1000.:.1f}kHz",
            )

        # labe the axes.
        ax3d.set_xlabel(x_select)
        ax3d.set_ylabel(y_select)
        ax3d.set_zlabel(z_select)
        mpl.show()
    return distances

if __name__ == "__main__":
    # test_find_min_max_axes()
    # exit()

    basepath = Path("/Users/pbmanis/Desktop/CN-extracts")
    dataset = datasets()
    main(basepath, dataset)
    # ds = "dataset_01.csv"
    # show_paraview_csv(basepath, csv_file=ds, parameters=dataset[ds]["parameters"])
