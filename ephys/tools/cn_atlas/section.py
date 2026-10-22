"""Section-plane geometry for slicing the atlas.

The plane is defined in the Kim display world (X = lateral/ML, Y = anterior-
posterior/AP, Z = dorsal-ventral, dorsal up). A section starts from a cardinal
orientation and may be tilted by two angles (degrees, positive = that in-plane
axis tilts upward / toward +Z). A position offset (um) slides it along its
normal; a thickness (um) defines the slab.

For each orientation:
    Coronal    normal = AP (Y);  in-plane u = ML (X), v = DV (Z)
    Horizontal normal = DV (Z);  in-plane u = ML (X), v = AP (Y)
    Sagittal   normal = ML (X);  in-plane u = AP (Y), v = DV (Z)

tilt_u rotates the plane about its u axis; tilt_v rotates it about its v axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# orientation -> (normal, u, v) as world unit axes, plus friendly tilt labels.
# tilt_u rotates about the u axis and tilt_v about the v axis, but the labels
# name the anatomical direction in which the plane tips (i.e. where its normal
# swings), not the axis of rotation -- rotating about u = ML tips the plane in
# the A-P sense, etc. This reads more naturally than naming tilts by their axis.
ORIENTATIONS: dict[str, dict] = {
    "Coronal": {
        "normal": (0, 1, 0), "u": (1, 0, 0), "v": (0, 0, 1),
        "tilt_u": "D-V tilt", "tilt_v": "M-L tilt", "pos": "A-P position",
    },
    "Horizontal": {
        "normal": (0, 0, 1), "u": (1, 0, 0), "v": (0, 1, 0),
        "tilt_u": "A-P tilt", "tilt_v": "M-L tilt", "pos": "D-V position",
    },
    "Sagittal": {
        "normal": (1, 0, 0), "u": (0, 1, 0), "v": (0, 0, 1),
        "tilt_u": "D-V tilt", "tilt_v": "A-P tilt", "pos": "M-L position",
    },
}


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for ``deg`` degrees about unit ``axis``."""
    a = np.deg2rad(deg)
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    x, y, z = ax
    c, s = np.cos(a), np.sin(a)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


# Claude added 2026-10-21: orient the plane normal so the position slider/arrows
# move the plane in the intuitive "up = dorsal" sense. The sign is chosen so the
# normal's dominant anatomical component points Dorsal (+Z), or Anterior (+Y)
# then Right (+X) when there is no D/V component. Cardinal normals are already
# canonical, so this only changes tilted / oblique planes (whose line-derived
# normal sign is otherwise arbitrary). The in-plane image depends only on u/v, so
# flipping the normal reverses only the scroll direction, not the displayed slice.
def _canonical_normal(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, float)
    # dominant axis wins; exact ties favor D/V (2) > A/P (1) > M/L (0)
    ax = max((2, 1, 0), key=lambda i: abs(n[i]))
    return -n if n[ax] < 0 else n


@dataclass
class Section:
    """A (possibly tilted) section plane + slab in world coordinates."""

    orientation: str = "Coronal"
    tilt_u: float = 0.0          # degrees about the u axis
    tilt_v: float = 0.0          # degrees about the v axis
    position: float = 0.0        # um offset from ``center`` along the normal
    thickness: float = 40.0      # um (slab)
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world center
    # Optional explicit orthonormal (normal, u, v). When set (e.g. an oblique
    # plane defined by "reslice perpendicular to a drawn line"), it overrides
    # the orientation/tilt construction below.
    basis_override: "tuple | None" = field(default=None, compare=False, repr=False)

    def set_basis(self, normal, u, v) -> "Section":
        """Pin an explicit orthonormal basis (used for oblique planes)."""
        self.basis_override = (np.asarray(normal, float), np.asarray(u, float),
                               np.asarray(v, float))
        return self

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (normal, u, v) unit vectors after applying the tilts."""
        # Claude fixed 2026-10-21: canonicalize the normal sign so position
        # scrolling is "up = dorsal" for every orientation (see _canonical_normal).
        if self.basis_override is not None:
            n, u, v = self.basis_override
            return _canonical_normal(n), u, v
        o = ORIENTATIONS[self.orientation]
        n0 = np.asarray(o["normal"], float)
        u0 = np.asarray(o["u"], float)
        v0 = np.asarray(o["v"], float)
        R = _rot(u0, self.tilt_u) @ _rot(v0, self.tilt_v)
        # return R @ n0, R @ u0, R @ v0
        return _canonical_normal(R @ n0), R @ u0, R @ v0

    def plane_center(self) -> np.ndarray:
        n, _, _ = self.basis()
        return self.center + self.position * n

    def corners(self, half_u: float, half_v: float) -> np.ndarray:
        """Four world corners of the plane quad (for a 3D indicator)."""
        _, u, v = self.basis()
        c = self.plane_center()
        return np.array([
            c - half_u * u - half_v * v,
            c + half_u * u - half_v * v,
            c + half_u * u + half_v * v,
            c - half_u * u + half_v * v,
        ])


def plane_to_world(section: "Section", uv: np.ndarray) -> np.ndarray:
    """Map in-plane (u, v) coordinates (um) to 3D world points."""
    _, u, v = section.basis()
    c = section.plane_center()
    uv = np.atleast_2d(np.asarray(uv, float))
    return c + uv[:, 0:1] * u + uv[:, 1:2] * v


def world_to_plane(section: "Section", pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points onto the section: returns (uv [N,2], signed dist [N])."""
    n, u, v = section.basis()
    rel = np.atleast_2d(np.asarray(pts, float)) - section.plane_center()
    return np.column_stack([rel @ u, rel @ v]), rel @ n


def section_descriptor(section: "Section") -> dict:
    """Serializable tag identifying the plate a contour was drawn on."""
    return {"orientation": section.orientation, "tilt_u": round(section.tilt_u, 2),
            "tilt_v": round(section.tilt_v, 2), "position": round(section.position, 1)}


def descriptor_matches(section: "Section", desc: dict, pos_tol: float = 25.0) -> bool:
    """True if ``desc`` is (approximately) the current section's plate."""
    return (desc.get("orientation") == section.orientation
            and abs(desc.get("tilt_u", 0.0) - section.tilt_u) < 1.0
            and abs(desc.get("tilt_v", 0.0) - section.tilt_v) < 1.0
            and abs(desc.get("position", 0.0) - section.position) <= pos_tol)


def clip_faces(vertices: np.ndarray, faces: np.ndarray, center: np.ndarray,
               normal: np.ndarray, thickness: float) -> np.ndarray:
    """Keep faces whose extent along ``normal`` overlaps the slab."""
    if len(faces) == 0:
        return faces
    d = (vertices - center) @ normal
    df = d[faces]
    lo = df.min(axis=1)
    hi = df.max(axis=1)
    half = thickness / 2.0
    sel = (hi >= -half) & (lo <= half)
    return faces[sel]


def label_outlines(label_img: np.ndarray) -> np.ndarray:
    """Boolean edge map: True where the label changes (region boundaries)."""
    edges = np.zeros(label_img.shape, bool)
    edges[:, :-1] |= label_img[:, :-1] != label_img[:, 1:]
    edges[:-1, :] |= label_img[:-1, :] != label_img[1:, :]
    return edges


def region_contours(label_img: np.ndarray, a: np.ndarray, b: np.ndarray,
                    smooth_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Vector boundary contours for every region in a label image.

    Returns (x, y) world coordinates (um) along the ``a`` (u) and ``b`` (v)
    axes, with NaN separators between contours so a single PlotCurveItem with
    ``connect='finite'`` draws them all as thin, antialiased, smoothed lines.
    Each region is traced inside its own bounding box so the cost is ~O(image).
    """
    from scipy.ndimage import find_objects, gaussian_filter1d
    from skimage.measure import find_contours

    du = (a[-1] - a[0]) / (len(a) - 1) if len(a) > 1 else 1.0
    dv = (b[-1] - b[0]) / (len(b) - 1) if len(b) > 1 else 1.0
    slices = find_objects(label_img)
    xs: list = []
    ys: list = []
    for lid in np.unique(label_img):
        if lid == 0 or lid > len(slices):
            continue
        sl = slices[lid - 1]
        if sl is None:
            continue
        sub = np.pad(label_img[sl] == lid, 1).astype(float)
        r0, c0 = sl[0].start - 1, sl[1].start - 1
        for cont in find_contours(sub, 0.5):
            row, col = cont[:, 0], cont[:, 1]
            if smooth_sigma > 0 and len(row) > 5:
                row = gaussian_filter1d(row, smooth_sigma, mode="wrap")
                col = gaussian_filter1d(col, smooth_sigma, mode="wrap")
            xs.append(a[0] + (col + c0) * du)
            xs.append([np.nan])
            ys.append(b[0] + (row + r0) * dv)
            ys.append([np.nan])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)
