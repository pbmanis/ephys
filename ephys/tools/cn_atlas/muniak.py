"""Muniak et al. (2013) CBA-mouse cochlear-nucleus atlas reader.

The atlas ships as ParaView files. Each (coordinate frame x scalar field) is
also provided as an AVS-UCD ascii file (``.inp``) -- a tetrahedral *volume*
mesh. We parse it with numpy, extract the outer boundary surface once, and
cache the result (vertices, faces, and both scalar fields) to a compressed
``.npz`` so the source drive is only needed on the first load of each frame.

All four frames share the stereotaxic convention +X = lateral, +Z = dorsal,
+Y = rostral/caudal (verified by aligning Original to Paxinos-coronal).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import numpy as np

from .util import cprint

# Friendly coordinate-frame name -> (subdirectory, filename infix).
FRAMES: dict[str, tuple[str, str]] = {
    "Original": ("Original", "original"),
    "Paxinos coronal": ("PaxinosCoronal", "paxinos-coronal"),
    "Paxinos horizontal": ("PaxinosHorizontal", "paxinos-horizontal"),
    "Paxinos sagittal": ("PaxinosSagittal", "paxinos-sagittal"),
}


def parse_avs_ucd(path: Union[str, Path]) -> dict:
    """Parse an AVS-UCD ascii mesh (as exported by Amira / ParaView).

    Layout::

        # comment lines ...
        nnodes ncells n_node_data n_cell_data n_model_data
        <nnodes>  node_id  x y z
        <ncells>  cell_id  mat_id  cell_type  v0 v1 ...
        n_node_comp  size0 size1 ...          (present iff n_node_data > 0)
        label0, unit0                          (one per component)
        ...
        <nnodes>  node_id  value0 value1 ...

    Nodes and node-data are listed in the same order, so we index everything by
    *row order* and ignore the id column -- this sidesteps 0- vs 1-based id
    quirks. Cell vertex indices are 0-based references into the node list.

    Returns a dict with ``points`` (N,3 float64), ``cells`` (M,4 int64, tets),
    ``scalar`` (N, float64) and ``scalar_label`` (str).
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("#"):
        i += 1
    header = lines[i].split()
    nnodes, ncells, n_node_data = (int(header[0]), int(header[1]), int(header[2]))
    i += 1

    node_block = lines[i : i + nnodes]
    i += nnodes
    cell_block = lines[i : i + ncells]
    i += ncells

    # Node coordinates (ignore the leading id column).
    points = np.array([ln.split()[1:4] for ln in node_block], dtype=np.float64)

    # Cells: id, mat, cell_type, v0 v1 v2 v3. Keep tetrahedra (the CN model is
    # entirely tets); other element types would need their own handling.
    tets = []
    for ln in cell_block:
        tok = ln.split()
        if tok[2] == "tet":
            tets.append(tok[3:7])
    cells = np.array(tets, dtype=np.int64)

    scalar = np.zeros(nnodes, dtype=np.float64)
    scalar_label = ""
    if n_node_data > 0:
        ctrl = lines[i].split()
        i += 1
        ncomp = int(ctrl[0])
        sizes = [int(x) for x in ctrl[1 : 1 + ncomp]]
        labels = []
        for _ in range(ncomp):
            labels.append(lines[i].strip())
            i += 1
        total = sum(sizes)
        data_block = lines[i : i + nnodes]
        i += nnodes
        vals = np.array([ln.split()[1 : 1 + total] for ln in data_block], dtype=np.float64)
        scalar = vals[:, 0]
        scalar_label = labels[0] if labels else ""

    return {
        "points": points,
        "cells": cells,
        "scalar": scalar,
        "scalar_label": scalar_label,
    }


def extract_boundary_surface(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return outward-oriented boundary triangles of a tetrahedral mesh.

    A triangular face lies on the boundary iff it belongs to exactly one tet.
    Faces are oriented so their normal points away from the tet's 4th vertex,
    giving consistent outward normals for shading.
    """
    v = tets
    faces = np.concatenate(
        [v[:, [0, 1, 2]], v[:, [0, 1, 3]], v[:, [0, 2, 3]], v[:, [1, 2, 3]]], axis=0
    )
    # The vertex opposite each face (used to orient the normal outward).
    opposite = np.concatenate([v[:, 3], v[:, 2], v[:, 1], v[:, 0]], axis=0)

    keys = np.sort(faces, axis=1)
    _, idx, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    bidx = idx[counts == 1]
    bfaces = faces[bidx].copy()
    bopp = opposite[bidx]

    p0, p1, p2 = points[bfaces[:, 0]], points[bfaces[:, 1]], points[bfaces[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    centroid = (p0 + p1 + p2) / 3.0
    toward = points[bopp] - centroid
    flip = np.einsum("ij,ij->i", normal, toward) > 0  # normal points inward -> flip
    bfaces[flip] = bfaces[flip][:, [0, 2, 1]]
    return bfaces.astype(np.int64)


def frame_inp_paths(atlas_root: Path, frame: str) -> tuple[Path, Path]:
    sub, infix = FRAMES[frame]
    root = Path(atlas_root)
    freq = root / sub / "Frequency" / f"CNmodel data {infix} frequency.inp"
    coch = root / sub / "Cochleotopic" / f"CNmodel data {infix} cochleotopic.inp"
    return freq, coch


def _align_scalar(ref_points: np.ndarray, other: dict) -> np.ndarray:
    """Return the ``other`` scalar reordered to match ``ref_points``.

    The Frequency and Cochleotopic files are the same mesh exported twice, so
    the node order normally matches exactly. If it does not, fall back to a
    coordinate-keyed lookup.
    """
    opts = other["points"]
    if opts.shape == ref_points.shape and np.allclose(opts, ref_points, atol=1e-3):
        return other["scalar"]
    cprint("y", "cn_alignment: cochleotopic node order differs; remapping by coordinate")
    key = {tuple(np.round(p, 2)): k for k, p in enumerate(opts)}
    out = np.full(ref_points.shape[0], np.nan)
    for k, p in enumerate(ref_points):
        j = key.get(tuple(np.round(p, 2)))
        if j is not None:
            out[k] = other["scalar"][j]
    return out


def load_frame(
    frame: str, atlas_root: Path, cache_dir: Union[str, Path], rebuild: bool = False
) -> dict:
    """Load (or build + cache) the surface + scalar fields for one frame."""
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"muniak_{FRAMES[frame][1]}.npz"
    freq_inp, coch_inp = frame_inp_paths(atlas_root, frame)

    if cache.exists() and not rebuild:
        fresh = (not freq_inp.exists()) or cache.stat().st_mtime >= freq_inp.stat().st_mtime
        if fresh:
            try:
                d = np.load(cache, allow_pickle=True)
                cprint("g", f"cn_alignment: loaded cache {cache.name}")
                return {
                    "vertices": d["vertices"],
                    "faces": d["faces"],
                    "freq": d["freq"],
                    "place": d["place"],
                    "freq_label": str(d["freq_label"]),
                    "place_label": str(d["place_label"]),
                    "frame": frame,
                }
            except Exception as exc:  # noqa: BLE001 - rebuild on bad cache
                cprint("y", f"cn_alignment: bad cache {cache.name}: {exc}; rebuilding")

    if not freq_inp.exists():
        raise FileNotFoundError(f"Atlas file not found (is the drive mounted?): {freq_inp}")

    t0 = time.time()
    fq = parse_avs_ucd(freq_inp)
    vertices = fq["points"].astype(np.float32)
    faces = extract_boundary_surface(fq["points"], fq["cells"]).astype(np.int32)
    freq = fq["scalar"].astype(np.float32)

    place = np.full(vertices.shape[0], np.nan, dtype=np.float32)
    place_label = ""
    if coch_inp.exists():
        cochd = parse_avs_ucd(coch_inp)
        place = _align_scalar(fq["points"], cochd).astype(np.float32)
        place_label = cochd["scalar_label"]
    else:
        cprint("y", f"cn_alignment: no cochleotopic file for {frame}")

    np.savez_compressed(
        cache,
        vertices=vertices,
        faces=faces,
        freq=freq,
        place=place,
        freq_label=fq["scalar_label"],
        place_label=place_label,
    )
    cprint(
        "g",
        f"cn_alignment: built {frame}: {vertices.shape[0]} verts, "
        f"{faces.shape[0]} faces in {time.time() - t0:.1f}s -> {cache.name}",
    )
    return {
        "vertices": vertices,
        "faces": faces,
        "freq": freq,
        "place": place,
        "freq_label": fq["scalar_label"],
        "place_label": place_label,
        "frame": frame,
    }
