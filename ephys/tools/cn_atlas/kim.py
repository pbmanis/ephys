"""Kim / Chon (2019) Unified Mouse Brain Atlas reader.

Data (figshare 25750983, v2 2024):
    UnifiedAtlas_Label_v2_20um-isotropic.nii   uint16 label volume, 20um iso
    UnifiedAtlas_template_coronal_20um-...nii   anatomical template
    UnifiedAtlas_Label_ontology_v2.csv          structure tree

The label volume is sub-sampled from Allen CCFv3 and carries no spatial affine
(qform/sform = 0). Its index axes are (i = ML [570], j = DV [400], k = AP
[660]), 20um per voxel. We map voxels to a display world where +X = lateral,
+Y = anterior/posterior, +Z = dorsal (up) via (i, j, k) -> (i, k, -j), a pure
rotation (det +1), so no face-winding flip is needed.

Region surfaces are built on demand with marching cubes over the (cropped)
binary mask of a label set, then cached per region to a compressed .npz.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from .nifti import read_nifti
from .util import cprint

# --- Nomenclature: Kim/Paxinos acronym -> (historical/Muniak acronym, name) --
# From the in-press curation note. CN-relevant entries drive the display; the
# rest are kept for Step 3 (overlay / boundary naming).
REMAP: dict[str, tuple[str, str]] = {
    "VCA": ("AVCN", "Anteroventral cochlear nucleus"),
    "VCP": ("PVCN", "Posteroventral cochlear nucleus"),
    "VCPO": ("PVCN (octopus area)", "Posteroventral cochlear nucleus, octopus cell area"),
    "DC": ("DCN", "Dorsal cochlear nucleus"),
    "DCMo": ("DCN molecular", "Dorsal cochlear nucleus, molecular layer"),
    "DCFu": ("DCN fusiform", "Dorsal cochlear nucleus, fusiform layer"),
    "DCDp": ("DCN deep", "Dorsal cochlear nucleus, deep layer"),
    "GrC (VCCap in 81)": ("VCAGr", "Ventral cochlear nucleus, granule cell layer"),
    "Tz": ("Tz/MNTB", "Medial nucleus of the trapezoid body"),
}

# Functional CN display groups (Muniak nomenclature) -> Kim ontology root ids
# whose descendants make up the group, and a default display color (RGB 0-255).
CN_GROUPS: dict[str, dict] = {
    "DCN": {"roots": [96], "color": (214, 39, 40)},
    "AVCN": {"roots": [2238], "color": (44, 160, 44)},
    "PVCN": {"roots": [2237, 2243], "color": (31, 119, 180)},
    "VCAGr (granule)": {"roots": [112, 560], "color": (255, 160, 20)},
    "Tz/MNTB": {"roots": [642], "color": (148, 103, 189)},
}

CN_ROOT_ID = 607  # "Cochlear nuclei"


class KimAtlas:
    """Lazy reader + mesh builder for the Kim Unified atlas."""

    def __init__(self, root: Path, cfg: dict, cache_dir: Union[str, Path]):
        self.root = Path(root)
        self.cfg = cfg
        self.voxel_size = float(cfg.get("KimVoxelSize", 20.0))
        self.label_path = self.root / cfg.get("KimLabelFile", "UnifiedAtlas_Label_v2_20um-isotropic.nii")
        self.template_path = self.root / cfg.get(
            "KimTemplateFile", "UnifiedAtlas_template_coronal_20um-isotropic.nii")
        self.ontology_path = self.root / cfg.get("KimOntologyFile", "UnifiedAtlas_Label_ontology_v2.csv")
        self.cache_dir = Path(cache_dir).expanduser() / "kim"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.structures: dict[int, dict] = {}
        self.children: dict[int, list] = {}
        self._vol = None          # lazy memmap (labels)
        self._template = None     # lazy memmap (anatomical template)
        self._present: Optional[set] = None
        self._load_ontology()

    # -- ontology -----------------------------------------------------------
    def _load_ontology(self):
        with open(self.ontology_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if not r.get("id", "").strip():
                    continue
                sid = int(r["id"])
                parent = int(r["structure_id_path"]) if r.get("structure_id_path", "").strip() else None
                if parent == sid:
                    parent = None  # root points to itself in this file
                acro = r["acronym"]
                self.structures[sid] = {
                    "id": sid,
                    "name": r["name"],
                    "acronym": acro,
                    "rgb": (int(r["RGB_1"]), int(r["RGB_2"]), int(r["RGB_3"])),
                    "parent": parent,
                    "order": int(r["graph_order"]) if r.get("graph_order", "").strip() else 0,
                    # historical/Muniak display name, when remapped
                    "muniak": REMAP.get(acro, (None, None))[0],
                }
        for sid, s in self.structures.items():
            self.children.setdefault(s["parent"], []).append(sid)

    def descendants(self, root_id: int) -> set:
        """All ids in the subtree rooted at ``root_id`` (inclusive)."""
        out, stack = set(), [root_id]
        while stack:
            cid = stack.pop()
            if cid in out:
                continue
            out.add(cid)
            stack.extend(self.children.get(cid, []))
        return out

    def display_name(self, sid: int) -> str:
        s = self.structures.get(sid, {})
        m = s.get("muniak")
        return f"{m} [{s.get('acronym','?')}]" if m else s.get("acronym", str(sid))

    # -- volume -------------------------------------------------------------
    @property
    def volume(self) -> np.ndarray:
        if self._vol is None:
            if not self.label_path.exists():
                raise FileNotFoundError(
                    f"Kim label volume not found (drive mounted?): {self.label_path}"
                )
            self._vol = read_nifti(self.label_path, memmap=True)["data"]
        return self._vol

    @property
    def template(self) -> np.ndarray:
        if self._template is None:
            if not self.template_path.exists():
                raise FileNotFoundError(f"Kim template not found: {self.template_path}")
            self._template = read_nifti(self.template_path, memmap=True)["data"]
        return self._template

    def world_to_voxel(self, pts_world: np.ndarray) -> np.ndarray:
        """Inverse of _voxel_to_world: world (x,y,z) um -> voxel idx (i,j,k)."""
        p = np.asarray(pts_world, float)
        vs = self.voxel_size
        return np.column_stack([p[:, 0] / vs, -p[:, 2] / vs, p[:, 1] / vs])

    def world_bbox(self) -> tuple[np.ndarray, np.ndarray]:
        ni, nj, nk = self.volume.shape
        vs = self.voxel_size
        corners = np.array([[x, y, z]
                            for x in (0, ni * vs)
                            for y in (0, nk * vs)
                            for z in (-nj * vs, 0)])
        return corners.min(0), corners.max(0)

    def world_center(self) -> np.ndarray:
        mn, mx = self.world_bbox()
        return (mn + mx) / 2.0

    def _plane_halfspan(self, u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
        mn, mx = self.world_bbox()
        center = (mn + mx) / 2.0
        corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                            for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
        rel = corners - center
        return float(np.abs(rel @ u).max()), float(np.abs(rel @ v).max())

    def _plane_bounds(self, u: np.ndarray, v: np.ndarray,
                      center: np.ndarray) -> tuple[float, float, float, float]:
        """In-plane (u, v) extent (umin, umax, vmin, vmax) of the volume bbox
        measured from ``center``. Asymmetric, so an off-center (oblique) plane
        still spans the whole brain instead of being clipped on the far side
        (_plane_halfspan is symmetric about the brain center)."""
        mn, mx = self.world_bbox()
        corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                            for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
        rel = corners - np.asarray(center, float)
        pu = rel @ np.asarray(u, float)
        pv = rel @ np.asarray(v, float)
        return float(pu.min()), float(pu.max()), float(pv.min()), float(pv.max())

    def sample_section(self, section, step: Optional[float] = None, slab: bool = True) -> dict:
        """Resample template + labels on a section plane.

        Returns template (mean over slab), label (mid-plane, nearest), the world
        extent [u0,u1,v0,v1] (um) and the in-plane grid axes (a=u, b=v).
        """
        from scipy.ndimage import map_coordinates

        step = step or self.voxel_size
        n, u, v = section.basis()
        center = section.plane_center()
        # Claude fixed 2026-10-21: bounds measured from the actual plane center so
        # an off-center (oblique) plane is not clipped on the far (anterior) side.
        # half_u, half_v = self._plane_halfspan(u, v)
        # a = np.arange(-half_u, half_u + step, step)
        # b = np.arange(-half_v, half_v + step, step)
        u0, u1, v0, v1 = self._plane_bounds(u, v, center)
        a = np.arange(u0, u1 + step, step)
        b = np.arange(v0, v1 + step, step)
        grid_a, grid_b = np.meshgrid(a, b)  # (nb, na)
        base = center[None, None, :] + grid_a[..., None] * u + grid_b[..., None] * v

        vol = np.asarray(self.volume).reshape(self.volume.shape[:3])
        tvol = np.asarray(self.template).reshape(self.template.shape[:3])
        label = map_coordinates(
            vol, self.world_to_voxel(base.reshape(-1, 3)).T, order=0, mode="constant", cval=0
        ).reshape(grid_a.shape).astype(np.int32)

        if slab and section.thickness > self.voxel_size:
            offs = np.arange(-section.thickness / 2, section.thickness / 2 + step, step)
        else:
            offs = np.array([0.0])
        tacc = np.zeros(grid_a.shape, float)
        for o in offs:
            pts = (base + o * n).reshape(-1, 3)
            tacc += map_coordinates(
                tvol, self.world_to_voxel(pts).T, order=1, mode="constant", cval=0
            ).reshape(grid_a.shape)
        template = tacc / len(offs)

        return {
            "template": template,
            "label": label,
            "extent": [u0, u1, v0, v1],
            "a": a,
            "b": b,
        }

    def present_labels(self) -> set:
        """Set of label ids actually present in the volume (cached)."""
        if self._present is not None:
            return self._present
        cache = self.cache_dir / "present_labels.npy"
        if cache.exists():
            self._present = set(int(x) for x in np.load(cache))
            return self._present
        cprint("c", "cn_alignment: scanning Kim volume for present labels ...")
        ids = np.unique(np.asarray(self.volume))
        np.save(cache, ids)
        self._present = set(int(x) for x in ids)
        return self._present

    # -- meshes -------------------------------------------------------------
    def _voxel_to_world(self, verts_ijk_um: np.ndarray) -> np.ndarray:
        # (i=ML, j=DV, k=AP) um  ->  (x=ML, y=AP, z=-DV) : dorsal up, rotation.
        x = verts_ijk_um[:, 0]
        y = verts_ijk_um[:, 2]
        z = -verts_ijk_um[:, 1]
        return np.column_stack([x, y, z]).astype(np.float32)

    def _mask_mesh(self, mask: np.ndarray, smooth_sigma: float = 0.7) -> dict:
        """Marching cubes over a boolean mask, cropped to its bounding box.

        The mask is lightly Gaussian-smoothed first so the surface is not a
        voxel staircase; ``level=0.5`` keeps the smoothed isosurface near the
        original boundary.
        """
        from skimage.measure import marching_cubes
        from scipy.ndimage import gaussian_filter

        idx = np.argwhere(mask)
        if idx.size == 0:
            return {"vertices": np.zeros((0, 3), np.float32), "faces": np.zeros((0, 3), np.int32)}
        mn = idx.min(0)
        mx = idx.max(0) + 1
        sub = mask[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
        pad = max(2, int(round(3 * smooth_sigma)))
        sub = np.pad(sub.astype(np.float32), pad)  # close surface + room to smooth
        if smooth_sigma > 0:
            sub = gaussian_filter(sub, smooth_sigma)
        verts, faces, _, _ = marching_cubes(sub, level=0.5, spacing=(self.voxel_size,) * 3)
        # undo pad and add crop offset, both in micrometres
        verts = verts + (np.asarray(mn) - pad) * self.voxel_size
        world = self._voxel_to_world(verts)
        return {"vertices": world, "faces": faces.astype(np.int32)}

    def region_mesh(self, key: str, label_ids: Iterable[int], rebuild: bool = False) -> dict:
        """Build (or load) the surface for the union of ``label_ids``."""
        cache = self.cache_dir / f"region_{key}.npz"
        if cache.exists() and not rebuild:
            try:
                d = np.load(cache)
                return {"vertices": d["vertices"], "faces": d["faces"]}
            except Exception:  # noqa: BLE001
                pass
        ids = list(label_ids)
        mask = np.isin(np.asarray(self.volume), ids)
        mesh = self._mask_mesh(mask)
        np.savez_compressed(cache, vertices=mesh["vertices"], faces=mesh["faces"])
        cprint("g", f"cn_alignment: Kim region {key}: {len(mesh['vertices'])} verts -> {cache.name}")
        return mesh

    def group_mesh(self, group: str, rebuild: bool = False) -> dict:
        roots = CN_GROUPS[group]["roots"]
        ids = set()
        for r in roots:
            ids |= self.descendants(r)
        return self.region_mesh(group.replace("/", "-").replace(" ", "_"), ids, rebuild)

    def structure_mesh(self, sid: int, rebuild: bool = False) -> dict:
        """Surface for one ontology node (its whole subtree)."""
        ids = self.descendants(sid)
        return self.region_mesh(f"id{sid}", ids, rebuild)

    def cn_mesh(self, rebuild: bool = False) -> dict:
        """Outer surface of the entire cochlear nucleus (id 607 + descendants).

        A single marching-cubes surface over the union mask, so it has no
        internal inter-region walls.
        """
        return self.region_mesh("CN_all", self.descendants(CN_ROOT_ID), rebuild)

    def cn_core_mesh(self, rebuild: bool = False) -> dict:
        """CN surface excluding the granule-cell domains (ids 112 GrC, 560 VCAGr).

        The Muniak tonotopic model does not include the granule cell cap, so for
        registration the Kim target should likewise be the magnocellular CN.
        """
        ids = self.descendants(CN_ROOT_ID) - self.descendants(112) - self.descendants(560)
        return self.region_mesh("CN_core", ids, rebuild)

    def brain_outline(self, downsample: int = 3, rebuild: bool = False) -> dict:
        """Translucent whole-brain outline (downsampled for speed)."""
        cache = self.cache_dir / f"brain_outline_ds{downsample}.npz"
        if cache.exists() and not rebuild:
            try:
                d = np.load(cache)
                return {"vertices": d["vertices"], "faces": d["faces"]}
            except Exception:  # noqa: BLE001
                pass
        from skimage.measure import marching_cubes

        vol = np.asarray(self.volume)[::downsample, ::downsample, ::downsample]
        mask = np.pad((vol > 0).astype(np.float32), 1)
        step = self.voxel_size * downsample
        verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=(step,) * 3)
        verts = verts - step  # undo the 1-voxel pad
        world = self._voxel_to_world(verts)
        np.savez_compressed(cache, vertices=world.astype(np.float32), faces=faces.astype(np.int32))
        cprint("g", f"cn_alignment: Kim brain outline: {len(world)} verts -> {cache.name}")
        return {"vertices": world.astype(np.float32), "faces": faces.astype(np.int32)}
