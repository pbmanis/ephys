"""Co-register the Muniak CN surface onto the Kim atlas cochlear nucleus.

Both atlases carry the cochlear nucleus, and both are expressed in an
anatomically aligned world (+X lateral, +Y A-P, +Z dorsal). We therefore fit a
*similarity* transform (rotation + uniform scale + translation) -- a conformal
map, appropriate for the CBA (Muniak) vs C57BL/6 (Kim) size/shape difference --
from the Muniak CN surface to one hemisphere's Kim CN point cloud, using ICP.

The Muniak CN is a single (one-sided) structure; the Kim CN is bilateral. We
register to each hemisphere and keep the better fit, which also resolves the
left/right handedness automatically (a rotation-only fit cannot match the
mirror-image hemisphere, so its residual stays high).
"""

from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform mapping src -> dst (corresponded).

    Returns (scale, R 3x3, t 3) minimizing ||scale*R@src + t - dst||.
    """
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xd.T @ xs) / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_corr = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_corr[2, 2] = -1.0
    rot = u @ s_corr @ vt
    if with_scale:
        var_s = (xs ** 2).sum() / len(src)
        scale = float((d * np.diag(s_corr)).sum() / var_s)
    else:
        scale = 1.0
    t = mu_d - scale * rot @ mu_s
    return scale, rot, t


def _matrix(scale: float, rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = scale * rot
    m[:3, 3] = t
    return m


def apply_transform(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (matrix[:3, :3] @ pts.T).T + matrix[:3, 3]


def _subsample(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), n, replace=False)]


def _rms_radius(v: np.ndarray) -> float:
    return float(np.sqrt(((v - v.mean(0)) ** 2).sum(1).mean()))


def _icp(src: np.ndarray, dst: np.ndarray, init: np.ndarray, with_scale: bool,
         iters: int = 80, tol: float = 1e-5) -> tuple[np.ndarray, float]:
    """ICP refining a rigid (with_scale=False) or similarity transform."""
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    matrix = init
    prev = np.inf
    for _ in range(iters):
        cur = apply_transform(matrix, src)
        dist, idx = tree.query(cur)
        scale, rot, t = umeyama(src, dst[idx], with_scale=with_scale)
        matrix = _matrix(scale, rot, t)
        rmse = float(np.sqrt((dist ** 2).mean()))
        if abs(prev - rmse) < tol * max(prev, 1.0):
            break
        prev = rmse
    return matrix, rmse


def mirror_x(pts: np.ndarray, mid_x: float) -> np.ndarray:
    """Reflect points across the sagittal plane x = mid_x (left<->right)."""
    out = np.asarray(pts, float).copy()
    out[:, 0] = 2 * mid_x - out[:, 0]
    return out


def _frame_rotation() -> np.ndarray:
    """Rotation aligning the Muniak 'Paxinos coronal' frame to the Kim (Allen)
    world: -90 deg about the ML (X) axis. The Paxinos and Allen conventions
    differ by this quarter turn, so without it the Muniak CN is canted ~90 deg
    caudally. Verified by overlap against the Kim CN surface."""
    a = np.deg2rad(-90.0)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def anatomical_placement(muniak_verts: np.ndarray, kim_sides: dict, side: str = "right",
                         mirror: bool = False, base_rotation: np.ndarray = None) -> dict:
    """Place the Muniak CN into Kim space using the shared anatomical frame.

    Geometric ICP is unreliable here -- the blobby, cross-species CN surfaces
    fit a flipped orientation slightly better, so automatic fits come out
    upside-down. We instead apply a fixed frame rotation (Paxinos->Allen),
    centroid alignment, a uniform scale from the bounding-box-diagonal ratio
    (so extents match), and an optional left/right reflection. Residual tilt is
    left to the manual nudges.
    """
    src = np.asarray(muniak_verts, float)
    dst = np.asarray(kim_sides[side], float)
    rot = _frame_rotation() if base_rotation is None else np.asarray(base_rotation, float)
    c = src.mean(0)
    rotated = (src - c) @ rot.T
    d_src = float(np.linalg.norm(rotated.max(0) - rotated.min(0)))
    d_dst = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    s = d_dst / max(d_src, 1e-9)
    reflect = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
    linear = s * reflect @ rot
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = dst.mean(0) - linear @ c      # map src centroid -> dst centroid
    return {"matrix": matrix, "scale": float(s), "side": side, "mirror": bool(mirror)}


def cached_registration(kim_atlas, muniak_root, cache_dir, frame: str = "Paxinos coronal",
                        side: str = "right", mirror: bool = False, rebuild: bool = False) -> dict:
    """Anatomical base placement of the Muniak CN into Kim space.

    The Kim target is the outer surface of the *entire* cochlear nucleus (Kim
    id 607 + descendants, a single marching-cubes surface), split into
    hemispheres. Fast (meshes are cached), so computed on demand. Returns
    dict(matrix, scale, side, mirror, frame, mid_x).
    """
    from . import muniak as muniak_mod

    md = muniak_mod.load_frame(frame, muniak_root, cache_dir)
    mid_x = float(kim_atlas.world_center()[0])
    # Target = CN excluding granule domains, matching the Muniak model's extent.
    cn = kim_atlas.cn_core_mesh(rebuild=rebuild)["vertices"]
    kim_sides = {"right": cn[cn[:, 0] > mid_x], "left": cn[cn[:, 0] < mid_x]}
    reg = anatomical_placement(md["vertices"].astype(float), kim_sides, side, mirror)
    reg.update({"frame": frame, "mid_x": mid_x})
    return reg


def register_muniak_to_kim(muniak_verts: np.ndarray, kim_sides: dict,
                           max_pts: int = 6000) -> dict:
    """Register the Muniak CN surface to the better-fitting Kim hemisphere.

    ``kim_sides`` maps a side label ("right"/"left") to that hemisphere's Kim CN
    point cloud. Returns dict(matrix, rmse, side, scale).
    """
    src = _subsample(np.asarray(muniak_verts, float), max_pts)
    c_src = src.mean(0)
    best = None
    for side, dst_full in kim_sides.items():
        if dst_full is None or len(dst_full) < 10:
            continue
        dst = _subsample(np.asarray(dst_full, float), max_pts)
        # Fixed scale from the structures' sizes (both are physical um), so ICP
        # cannot collapse the source to nest inside the target.
        s0 = _rms_radius(dst) / max(_rms_radius(src), 1e-9)
        src_s = c_src + s0 * (src - c_src)          # pre-scale about its centroid
        init = np.eye(4)
        init[:3, 3] = dst.mean(0) - c_src           # centroid-align, identity rotation
        # Stage 1: fixed-scale rigid ICP -> correct orientation + position.
        m_rigid, rmse = _icp(src_s, dst, init, with_scale=False)
        m_scale = np.eye(4)
        m_scale[:3, :3] *= s0
        m_scale[:3, 3] = c_src - s0 * c_src
        matrix = m_rigid @ m_scale                  # src -> src_s -> dst
        scale = s0
        # Stage 2: refine scale with a free-scale ICP from this good init, but
        # only accept it if the scale stays sane (guards against collapse).
        m_sim, rmse_sim = _icp(src, dst, matrix, with_scale=True)
        s_ref = float(np.cbrt(abs(np.linalg.det(m_sim[:3, :3]))))
        if 0.6 < s_ref / s0 < 1.6 and rmse_sim <= rmse:
            matrix, rmse, scale = m_sim, rmse_sim, s_ref
        if best is None or rmse < best["rmse"]:
            best = {"matrix": matrix, "rmse": rmse, "side": side, "scale": float(scale)}
    return best
