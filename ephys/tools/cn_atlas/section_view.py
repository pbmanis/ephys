"""2D section-plate widget: template + Kim outlines + ground-truth editing.

Shows a resampled atlas plate (grayscale template with the Kim region contours).
Hovering reports the region under the cursor. Also hosts the adjunct
ground-truth layer: user-drawn closed contours (editable PolyLineROIs) and point
landmarks, drawn on the current plate and stored in 3D world coordinates.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from . import ground_truth as gt_mod
from . import section as section_mod


def _point_in_poly(p: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (poly is [N,2])."""
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _dist_to_poly(p: np.ndarray, poly: np.ndarray) -> float:
    """Minimum distance from point ``p`` to the closed polyline ``poly`` [N,2]."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ap = p - a
    denom = np.sum(ab * ab, axis=1)
    t = np.clip(np.sum(ap * ab, axis=1) / np.where(denom == 0, 1.0, denom), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.sqrt(np.min(np.sum((p - proj) ** 2, axis=1))))


class SectionPlate(QtWidgets.QWidget):
    """A 2D atlas plate linked to a Section, with ground-truth annotation."""

    LANDMARK_TOL = 150.0  # um: show landmarks within this distance of the plane
    # Claude added 2026-10-22: show saved alignment picks within this distance.
    PICK_TOL = 150.0  # um
    # Claude added 2026-10-21: off-plane GT contours whose nearest point is within
    # this distance of the slice are drawn as faded, dashed "ghost" references
    # (orthographically projected). On-plane contours stay editable (see refresh_gt).
    CONTOUR_GHOST_TOL = 300.0  # um

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.title = QtWidgets.QLabel("<b>Section</b>")
        layout.addWidget(self.title)

        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox(lockAspect=True)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem()
        self.outline_curve = pg.PlotCurveItem(
            pen=pg.mkPen((255, 80, 80), width=1), antialias=True, connect="finite")
        self.gt_landmarks = pg.ScatterPlotItem(
            size=11, symbol="x", pen=pg.mkPen((255, 255, 0), width=2), brush=None)
        # Claude added 2026-10-22: saved alignment picks near the plane (green +).
        self.picked_markers = pg.ScatterPlotItem(
            size=13, symbol="+", pen=pg.mkPen((0, 255, 140), width=2), brush=None)
        self.picked_markers.setZValue(97)
        self._draw_item = pg.PlotCurveItem(pen=pg.mkPen((0, 255, 255), width=2), antialias=True)
        self._sel_marker = pg.ScatterPlotItem(
            size=16, symbol="o", pen=pg.mkPen((255, 255, 255), width=2),
            brush=pg.mkBrush(255, 255, 0, 160))
        self._sel_marker.setZValue(100)
        self._sel_marker.setVisible(False)
        self._line_curve = pg.PlotCurveItem(
            pen=pg.mkPen((80, 200, 255), width=2, style=QtCore.Qt.PenStyle.DashLine),
            antialias=True)
        self._line_curve.setZValue(89)
        # Claude added 2026-10-20: fiducial-line pierce marker. When the plate is
        # resliced perpendicular to the drawn line, the line is seen end-on; this
        # 50 um (data-unit) bright translucent blue disk marks where it crosses.
        self._line_spot = pg.ScatterPlotItem(
            size=50.0, pxMode=False, symbol="o",
            pen=pg.mkPen((150, 200, 255), width=1.5),
            brush=pg.mkBrush(70, 150, 255, 200))
        self._line_spot.setZValue(96)
        self._line_spot.setVisible(False)
        # Claude added 2026-10-20: anatomical edge labels so the slice orientation
        # is unambiguous. World axes: +X=Right, +Y=Anterior, +Z=Dorsal; each edge
        # is filled/placed from the section basis in _update_edge_labels().
        self._edge_labels = {
            "right": pg.TextItem(color=(180, 210, 255), anchor=(1.0, 0.5)),
            "left": pg.TextItem(color=(180, 210, 255), anchor=(0.0, 0.5)),
            "top": pg.TextItem(color=(180, 210, 255), anchor=(0.5, 0.0)),
            "bottom": pg.TextItem(color=(180, 210, 255), anchor=(0.5, 1.0)),
        }
        for _lbl in self._edge_labels.values():
            _lbl.setZValue(98)
            _lbl.setVisible(False)
        self.vb.addItem(self.img)
        self.vb.addItem(self.outline_curve)
        self.vb.addItem(self.gt_landmarks)
        self.vb.addItem(self.picked_markers)
        self.vb.addItem(self._draw_item)
        self.vb.addItem(self._sel_marker)
        self.vb.addItem(self._line_curve)
        self.vb.addItem(self._line_spot)
        for _lbl in self._edge_labels.values():
            self.vb.addItem(_lbl)
        layout.addWidget(self.glw, stretch=1)

        self.readout = QtWidgets.QLabel("hover a region ...")
        self.readout.setStyleSheet("QLabel{padding:2px;}")
        layout.addWidget(self.readout)

        self._lab: Optional[np.ndarray] = None
        self._a: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None
        self._names: Optional[Callable[[int], str]] = None
        self._section = None

        # ground-truth state
        self.gt: Optional[gt_mod.GroundTruth] = None
        self.gt_visible = False
        # Claude added 2026-10-22: saved alignment picks (set by the viewer).
        self.picks = None
        self._gt_rois: list = []        # [(roi, region, cid)]
        self._gt_ghosts: list = []      # faded curves for off-plane GT contours
        self._mode = None               # None | "contour" | "landmark"
        self._draw_region: Optional[str] = None
        self._draw_pts: list = []
        self._sel = None                # (roi, region, cid, idx) selected vertex
        self._active = None             # (roi, region, cid) highlighted, movable contour
        self._moving = None             # drag-move state while translating a contour

        # section-line tool: two points that define an oblique reslice (plane
        # perpendicular to the line). Each point carries its own position along
        # the current normal, so the line may span several sections. Its in-plane
        # (u, v) is invariant across parallel sections, so the markers stay put as
        # the position slider scrolls; only the depth readout changes.
        self._line_mode = None          # None | "p1" | "p2" (point being placed)
        self._line_section = None       # Section fixing the orientation/tilts/center
        self._line_pts: list = [None, None]      # each: {"uv": ndarray, "pos": float}
        self._line_targets: list = [None, None]  # draggable pg.TargetItem per point
        # self._extent = None           # last (a0,a1,b0,b1): preserve zoom on scroll
        # Claude fixed 2026-10-21: the in-plane bounds move with the plane center,
        # so an extent-change test refit (autoRange) on every position step. The
        # caller now says explicitly when to refit (fit=True); we only track
        # whether the plate has ever been shown, to fit once on first display.
        self._shown = False

        # callbacks (set by the viewer)
        self.on_contour_drawn = None    # (region, world_points)
        self.on_contour_edited = None   # (region, cid, world_points)
        self.on_landmark_placed = None  # (world_point)
        self.on_section_line_changed = None  # (world_p0, world_p1) or (None, None)
        self.on_point_picked = None     # (world_point): plain left-click alignment pick

        self.glw.scene().sigMouseMoved.connect(self._on_hover)
        self.glw.scene().sigMouseClicked.connect(self._on_click)
        # shift-click to select a vertex; Delete / right-click to remove it
        self.glw.scene().installEventFilter(self)
        self.glw.installEventFilter(self)
        self.glw.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    # -- plate rendering ----------------------------------------------------
    def set_section(self, result: dict, name_lookup: Callable[[int], str], title: str = "",
                    show_outlines: bool = True, smooth: bool = True, section=None,
                    fit: bool = False):
        self._lab, self._a, self._b, self._names = result["label"], result["a"], result["b"], name_lookup
        self._section = section
        # Claude fixed 2026-10-20: a display-only ML mirror (invertX) was tried to
        # match the 3D view, but it reversed the draw->3D correspondence (drawing
        # on the slice's left landed on the 3D right). Reverted: plate-right = +X.
        # Orientation is instead made explicit with the R/L/A/P/D/V edge labels.
        self.vb.invertX(False)
        templ = result["template"]
        a, b = result["a"], result["b"]
        if title:
            self.title.setText(f"<b>{title}</b>")

        self.img.setImage(templ.T, autoLevels=False,
                          levels=(float(templ.min()), float(max(templ.max(), 1))))
        gray = np.repeat(np.arange(256, dtype=np.ubyte)[:, None], 3, axis=1)
        self.img.setLookupTable(gray)
        self.img.setRect(QtCore.QRectF(float(a[0]), float(b[0]),
                                       float(a[-1] - a[0]), float(b[-1] - b[0])))
        self._update_edge_labels(section, a, b)

        if show_outlines:
            x, y = section_mod.region_contours(result["label"], a, b,
                                               smooth_sigma=1.0 if smooth else 0.0)
            self.outline_curve.setData(x=x, y=y)
            self.outline_curve.setVisible(True)
        else:
            self.outline_curve.setVisible(False)
        self.refresh_gt()
        self.refresh_picks()
        self._refresh_line_display()
        # Claude fixed 2026-10-21: keep the current magnification while scrolling
        # position/thickness; refit only when the caller signals a basis change
        # (orientation / tilt / oblique reslice) or on first display. The old
        # extent-change test refit every position step because the in-plane
        # bounds are measured from the moving plane center.
        # extent = (float(a[0]), float(a[-1]), float(b[0]), float(b[-1]))
        # if extent != self._extent:
        #     self.vb.autoRange()
        # self._extent = extent
        if fit or not self._shown:
            self.vb.autoRange()
            self._shown = True

    def set_outlines_visible(self, on: bool):
        self.outline_curve.setVisible(on)

    # -- saved alignment picks ---------------------------------------------
    # Claude added 2026-10-22: the viewer owns a PickedPoints store; the plate
    # just mirrors it as on-plate markers and reports new picks via
    # ``on_point_picked`` (see _on_click).
    def set_picks(self, picks):
        self.picks = picks
        self.refresh_picks()

    def refresh_picks(self):
        """Show saved picks whose world point lies within PICK_TOL of the plane."""
        self.picked_markers.setData([])
        if self.picks is None or self._section is None:
            return
        spots = []
        for p in self.picks.points:
            uv, dist = section_mod.world_to_plane(self._section, np.array([p["point"]]))
            if abs(float(dist[0])) <= self.PICK_TOL:
                spots.append({"pos": (float(uv[0, 0]), float(uv[0, 1]))})
        self.picked_markers.setData(spots)

    @staticmethod
    def _anat_label(vec) -> str:
        """Anatomical letter(s) for a world direction. World +X=Right, +Y=Anterior,
        +Z=Dorsal. Off-cardinal directions get a composite (e.g. 'R-D') so the
        edge labels show the plane is no longer in cardinal coordinates."""
        arr = np.asarray(vec, float)
        names = {0: ("R", "L"), 1: ("A", "P"), 2: ("D", "V")}
        order = np.argsort(np.abs(arr))[::-1]

        def letter(ax: int) -> str:
            return names[ax][0] if arr[ax] >= 0 else names[ax][1]

        primary = int(order[0])
        label = letter(primary)
        secondary = int(order[1])
        if abs(arr[primary]) > 1e-9 and abs(arr[secondary]) > 0.35 * abs(arr[primary]):
            label += "-" + letter(secondary)
        return label

    def _update_edge_labels(self, section, a, b):
        """Place R/L/A/P/D/V labels at the plate edges from the section basis.
        Right edge = +u, left = -u, top = +v, bottom = -v (plate-right = +X)."""
        if section is None or a is None or b is None or len(a) < 2 or len(b) < 2:
            for lbl in self._edge_labels.values():
                lbl.setVisible(False)
            return
        _, u, v = section.basis()
        mid_a = 0.5 * (float(a[0]) + float(a[-1]))
        mid_b = 0.5 * (float(b[0]) + float(b[-1]))
        placements = {
            "right": (self._anat_label(u), float(a[-1]), mid_b),
            "left": (self._anat_label(-u), float(a[0]), mid_b),
            "top": (self._anat_label(v), mid_a, float(b[-1])),
            "bottom": (self._anat_label(-v), mid_a, float(b[0])),
        }
        for key, (text, x, y) in placements.items():
            lbl = self._edge_labels[key]
            lbl.setText(text)
            lbl.setPos(x, y)
            lbl.setVisible(True)

    # -- ground truth -------------------------------------------------------
    def set_gt(self, gt):
        self.gt = gt
        self.refresh_gt()

    def set_gt_visible(self, on: bool):
        self.gt_visible = on
        self.refresh_gt()

    def _clear_gt_rois(self):
        self._active = None          # ROIs are being destroyed; drop references
        self._moving = None
        for roi, _, _ in self._gt_rois:
            self.vb.removeItem(roi)
        self._gt_rois = []
        # Claude added 2026-10-21: also drop the off-plane ghost curves
        for ghost in self._gt_ghosts:
            self.vb.removeItem(ghost)
        self._gt_ghosts = []
        self._clear_selection()

    def refresh_gt(self):
        """Rebuild editable ROIs + landmark markers for the current plate."""
        self._clear_gt_rois()
        self.gt_landmarks.setData([])
        if not (self.gt_visible and self.gt and self._section is not None):
            return
        for region, info in self.gt.regions.items():
            col = tuple(info.get("color", [255, 230, 0]))
            for c in info["contours"]:
                pts = np.array(c["points"])
                if section_mod.descriptor_matches(self._section, c.get("section", {})):
                    uv, _ = section_mod.world_to_plane(self._section, pts)
                    roi = pg.PolyLineROI(uv.tolist(), closed=True,
                                         pen=pg.mkPen(col, width=2))
                    roi.sigRegionChangeFinished.connect(
                        lambda r=roi, rg=region, cid=c["id"]: self._on_roi_changed(r, rg, cid))
                    self.vb.addItem(roi)
                    self._gt_rois.append((roi, region, c["id"]))
                    continue
                # Claude added 2026-10-21: off-plane contour -> faded, dashed
                # "ghost" (orthographic projection) when it passes within
                # CONTOUR_GHOST_TOL of the slice. Read-only reference; a single
                # off-plane curve has no true in-plane outline, so this is a
                # visual aid until GT regions become volumes (see design notes).
                if pts.size == 0:
                    continue
                uv, dist = section_mod.world_to_plane(self._section, pts)
                if float(np.abs(dist).min()) > self.CONTOUR_GHOST_TOL:
                    continue
                closed = np.vstack([uv, uv[:1]])  # repeat first point to close
                ghost = pg.PlotCurveItem(
                    x=closed[:, 0], y=closed[:, 1], antialias=True,
                    pen=pg.mkPen(col + (110,), width=1,
                                 style=QtCore.Qt.PenStyle.DashLine))
                ghost.setZValue(5)  # above the image/outlines, below interactive markers
                self.vb.addItem(ghost)
                self._gt_ghosts.append(ghost)
        # landmarks near the plane
        spots = []
        for lm in self.gt.landmarks:
            uv, dist = section_mod.world_to_plane(self._section, np.array([lm["point"]]))
            if abs(float(dist[0])) <= self.LANDMARK_TOL:
                spots.append({"pos": (uv[0, 0], uv[0, 1]),
                              "pen": pg.mkPen(tuple(lm.get("color", [255, 255, 0])), width=2)})
        self.gt_landmarks.setData(spots)

    def _roi_world(self, roi) -> np.ndarray:
        uv = np.array([[p.x(), p.y()] for p in
                       (roi.mapToView(pt) for pt in roi.getState()["points"])])
        return section_mod.plane_to_world(self._section, uv)

    def _on_roi_changed(self, roi, region, cid):
        if self.on_contour_edited is not None and self._section is not None:
            self.on_contour_edited(region, cid, self._roi_world(roi))

    # -- vertex selection + deletion, whole-contour move --------------------
    def eventFilter(self, obj, ev):  # noqa: N802 - Qt signature
        et = ev.type()
        ET = QtCore.QEvent.Type
        if et == ET.GraphicsSceneMousePress and self._mode is None:
            mods = ev.modifiers()
            if mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
                if self._select_vertex_at(ev.scenePos()):
                    ev.accept()
                    return True
            elif (ev.button() == QtCore.Qt.MouseButton.RightButton
                  and self._sel is not None):
                self._delete_selected_vertex()
                ev.accept()
                return True
            elif (ev.button() == QtCore.Qt.MouseButton.LeftButton and not mods
                  and self._begin_move(ev.scenePos())):
                ev.accept()
                return True
        elif et == ET.GraphicsSceneMouseMove and self._moving is not None:
            self._drag_move(ev.scenePos())
            ev.accept()
            return True
        elif et == ET.GraphicsSceneMouseRelease and self._moving is not None:
            self._end_move()
            ev.accept()
            return True
        elif et == ET.KeyPress and self._mode is None:
            if ev.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
                if self._delete_selected_vertex():
                    return True
            elif self._active is not None and self._arrow_move(ev):
                return True
        return super().eventFilter(obj, ev)

    def _vertex_tol(self) -> float:
        """Selection radius in data units (~16 px)."""
        try:
            px = self.vb.viewPixelSize()
            return 16.0 * max(float(px[0]), float(px[1]))
        except Exception:  # noqa: BLE001
            return 25.0

    def _select_vertex_at(self, scene_pos) -> bool:
        if self._section is None or not self._gt_rois:
            return False
        vpt = self.vb.mapSceneToView(scene_pos)
        tol = self._vertex_tol()
        best = None
        for roi, region, cid in self._gt_rois:
            for i, p in enumerate(roi.getState()["points"]):
                w = roi.mapToView(p)
                d = (w.x() - vpt.x()) ** 2 + (w.y() - vpt.y()) ** 2
                if d <= tol * tol and (best is None or d < best[0]):
                    best = (d, roi, region, cid, i)
        if best is None:
            self._clear_selection()
            return False
        self._set_selection(*best[1:])
        return True

    def _set_selection(self, roi, region, cid, idx):
        self._sel = (roi, region, cid, idx)
        w = roi.mapToView(roi.getState()["points"][idx])
        self._sel_marker.setData([{"pos": (w.x(), w.y())}])
        self._sel_marker.setVisible(True)
        n = len(roi.getState()["points"])
        self.readout.setText(
            f"vertex {idx + 1}/{n} of '{region}' selected — "
            f"press Delete or right-click to remove")

    def _clear_selection(self):
        self._sel = None
        self._sel_marker.setData([])
        self._sel_marker.setVisible(False)

    def _delete_selected_vertex(self) -> bool:
        if self._sel is None or self._mode is not None or self._section is None:
            return False
        roi, region, cid, idx = self._sel
        world = self._roi_world(roi)
        if len(world) <= 3:
            self.readout.setText("cannot delete: a contour needs at least 3 points")
            return False
        world = np.delete(world, idx, axis=0)
        self._clear_selection()
        if self.on_contour_edited is not None:
            self.on_contour_edited(region, cid, world)
        self.refresh_gt()
        self.readout.setText(f"deleted a vertex from '{region}'")
        return True

    # -- highlight + whole-contour move -------------------------------------
    def _normal_pen(self, region: str):
        col = tuple(self.gt.regions.get(region, {}).get("color", [255, 230, 0])) \
            if self.gt else (255, 230, 0)
        return pg.mkPen(col, width=2)

    def _set_active(self, roi, region, cid):
        """Highlight ``roi`` to show it will move as a unit (dashed, thicker)."""
        if self._active is not None and self._active[0] is not roi:
            self._active[0].setPen(self._normal_pen(self._active[1]))
        self._active = (roi, region, cid)
        col = tuple(self.gt.regions.get(region, {}).get("color", [255, 230, 0])) \
            if self.gt else (255, 230, 0)
        roi.setPen(pg.mkPen(col, width=3, style=QtCore.Qt.PenStyle.DashLine))

    def _poly_view(self, roi) -> np.ndarray:
        pts = roi.getState()["points"]
        return np.array([[(w := roi.mapToView(p)).x(), w.y()] for p in pts])

    def _begin_move(self, scene_pos) -> bool:
        """Start a whole-contour drag if the press is on a contour body (not a handle)."""
        if self._section is None or not self._gt_rois:
            return False
        vpt = self.vb.mapSceneToView(scene_pos)
        p = np.array([vpt.x(), vpt.y()])
        tol = self._vertex_tol()
        # if the press is on a handle, let the ROI move that single vertex instead
        for roi, _, _ in self._gt_rois:
            poly = self._poly_view(roi)
            if len(poly) and np.min(np.sum((poly - p) ** 2, axis=1)) <= tol * tol:
                return False
        # prefer the already-active contour, then any whose body is hit
        ordered = self._gt_rois
        if self._active is not None:
            ordered = ([r for r in self._gt_rois if r[0] is self._active[0]]
                       + [r for r in self._gt_rois if r[0] is not self._active[0]])
        for roi, region, cid in ordered:
            poly = self._poly_view(roi)
            if len(poly) >= 3 and (_point_in_poly(p, poly)
                                   or _dist_to_poly(p, poly) <= tol):
                self._clear_selection()
                self._set_active(roi, region, cid)
                self._moving = {"roi": roi, "region": region, "cid": cid, "last": p}
                return True
        return False

    def _drag_move(self, scene_pos):
        vpt = self.vb.mapSceneToView(scene_pos)
        last = self._moving["last"]
        self._moving["roi"].translate(pg.Point(float(vpt.x() - last[0]),
                                               float(vpt.y() - last[1])), finish=False)
        self._moving["last"] = np.array([vpt.x(), vpt.y()])

    def _end_move(self):
        mv, self._moving = self._moving, None
        if self.on_contour_edited is not None and self._section is not None:
            self.on_contour_edited(mv["region"], mv["cid"], self._roi_world(mv["roi"]))

    def _arrow_move(self, ev) -> bool:
        keymap = {QtCore.Qt.Key.Key_Left: (-1, 0), QtCore.Qt.Key.Key_Right: (1, 0),
                  QtCore.Qt.Key.Key_Up: (0, 1), QtCore.Qt.Key.Key_Down: (0, -1)}
        d = keymap.get(ev.key())
        if d is None:
            return False
        step = 50.0 if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier) else 10.0
        roi, region, cid = self._active
        roi.translate(pg.Point(d[0] * step, d[1] * step), finish=False)
        if self.on_contour_edited is not None and self._section is not None:
            self.on_contour_edited(region, cid, self._roi_world(roi))
        return True

    # -- copy a nearby contour onto the current plate as a template ---------
    def copy_contour_here(self, region: str) -> bool:
        if self._section is None or not self.gt:
            return False
        info = self.gt.regions.get(region)
        if not info or not info.get("contours"):
            self.readout.setText(f"'{region}' has no contour to copy")
            return False
        cur = section_mod.section_descriptor(self._section)
        best = None
        for c in info["contours"]:
            desc = c.get("section", {})
            if desc.get("orientation") != cur.get("orientation"):
                continue
            if section_mod.descriptor_matches(self._section, desc):
                continue  # already a contour on this plate
            dp = abs(desc.get("position", 0.0) - cur.get("position", 0.0))
            if best is None or dp < best[0]:
                best = (dp, c)
        if best is None:
            self.readout.setText(
                f"no nearby '{region}' contour in this orientation to copy")
            return False
        src = np.array(best[1]["points"], float)
        uv, _ = section_mod.world_to_plane(self._section, src)   # flatten onto plane
        world = section_mod.plane_to_world(self._section, uv)
        if self.on_contour_drawn is not None:
            self.on_contour_drawn(region, world)
        self.refresh_gt()
        # highlight the just-added copy so it moves as a unit (drag body / arrows)
        new_cs = self.gt.regions.get(region, {}).get("contours", [])
        if new_cs:
            new_cid = new_cs[-1]["id"]
            for roi, rg, cid in self._gt_rois:
                if rg == region and cid == new_cid:
                    self._set_active(roi, rg, cid)
                    self.glw.setFocus()
                    break
        self.readout.setText(
            f"copied '{region}' from {best[0]:.0f} um away — drag it (off a handle) "
            "or use arrow keys to position; drag handles / shift-click+Delete to refine")
        return True

    # -- section line (two points across sections -> oblique reslice) -------
    def start_section_line(self) -> bool:
        """Begin defining a 2-point section line.

        Point 1 is placed on the current plate; the user then scrolls the
        position slider to a different section and places point 2, so the line
        may run obliquely through the volume. Endpoints are reported in world
        coordinates via ``on_section_line_changed`` (either may be None until it
        is placed) so the viewer can draw them in 3D and reslice perpendicular.
        """
        if self._section is None:
            return False
        self._cancel_draw()
        self._clear_line_items()
        self._line_pts = [None, None]
        self._line_section = self._section
        self._line_mode = "p1"
        self.readout.setText(
            "Click to place the FIRST point (drag to adjust); double-click to "
            "lock it, then scroll to another section for the second point.")
        self._emit_line()
        return True

    def clear_section_line(self):
        self._clear_line_items()
        self._line_pts = [None, None]
        self._line_section = None
        self._line_mode = None
        if self.on_section_line_changed is not None:
            self.on_section_line_changed(None, None)

    def _clear_line_items(self):
        for i, t in enumerate(self._line_targets):
            if t is not None:
                self.vb.removeItem(t)
            self._line_targets[i] = None
        self._line_curve.setData(x=[], y=[])
        self._line_curve.setVisible(False)
        self._line_spot.setData([])
        self._line_spot.setVisible(False)

    def _line_matches_current(self) -> bool:
        """The line is tied to one orientation/tilt; the position may differ
        (that is how the two endpoints land on different sections)."""
        cur, base = self._section, self._line_section
        if cur is None or base is None:
            return False
        return (cur.orientation == base.orientation
                and abs(cur.tilt_u - base.tilt_u) < 1e-6
                and abs(cur.tilt_v - base.tilt_v) < 1e-6)

    def _ensure_target(self, idx: int):
        if self._line_targets[idx] is not None:
            return
        col = (80, 200, 255) if idx == 0 else (120, 255, 160)
        t = pg.TargetItem(pos=(0, 0), size=12, movable=True,
                          pen=pg.mkPen(col, width=2), label=f"P{idx + 1}",
                          labelOpts={"offset": (10, -10)})
        t.setZValue(95)
        t.sigPositionChanged.connect(lambda _=None, i=idx: self._on_target_moved(i))
        self.vb.addItem(t)
        self._line_targets[idx] = t

    def _place_point(self, idx: int, uv):
        uv = np.asarray(uv, float)
        self._line_pts[idx] = {"uv": uv, "pos": float(self._section.position)}
        self._ensure_target(idx)
        t = self._line_targets[idx]
        t.blockSignals(True)
        t.setPos(pg.Point(float(uv[0]), float(uv[1])))
        t.blockSignals(False)

    def _on_target_moved(self, idx: int):
        t, pt = self._line_targets[idx], self._line_pts[idx]
        if t is None or pt is None:
            return
        p = t.pos()
        pt["uv"] = np.array([p.x(), p.y()], float)
        self._refresh_line_display()
        self._emit_line()

    def _line_point_world(self, pt) -> np.ndarray:
        """World point for an endpoint, using its own position along the normal
        and the orientation/tilts captured when the line was started."""
        base = self._line_section
        sec = section_mod.Section(orientation=base.orientation, tilt_u=base.tilt_u,
                                  tilt_v=base.tilt_v, position=pt["pos"],
                                  center=base.center)
        return section_mod.plane_to_world(sec, pt["uv"])[0]

    def section_line_world(self):
        """World endpoints (w0, w1) once both are placed, else None."""
        if None in self._line_pts or self._line_section is None:
            return None
        return (self._line_point_world(self._line_pts[0]),
                self._line_point_world(self._line_pts[1]))

    def _emit_line(self):
        if self.on_section_line_changed is None or self._line_section is None:
            return
        w0 = self._line_point_world(self._line_pts[0]) if self._line_pts[0] else None
        w1 = self._line_point_world(self._line_pts[1]) if self._line_pts[1] else None
        self.on_section_line_changed(w0, w1)

    def _on_line_click(self, ev, pt):
        idx = 0 if self._line_mode == "p1" else 1
        if ev.double():
            if self._line_pts[idx] is None:
                self._place_point(idx, [pt.x(), pt.y()])
            if self._line_mode == "p1":
                self._line_mode = "p2"
                self.readout.setText(
                    "First point locked. Scroll the Position slider to another "
                    "section, then click to place the SECOND point.")
            else:
                self._line_mode = None
                self.readout.setText("Section line defined — press 'Reslice ⊥ line'.")
        else:
            self._place_point(idx, [pt.x(), pt.y()])
        self._refresh_line_display()
        self._emit_line()
        ev.accept()

    def _refresh_line_display(self):
        """Update marker/curve visibility + the coordinate/angle readout. The
        markers persist (same in-plane u,v) as the position slider scrolls."""
        show = self._line_matches_current() or self._line_mode is not None
        for i, t in enumerate(self._line_targets):
            if t is not None:
                t.setVisible(show and self._line_pts[i] is not None)
        if show and self._line_pts[0] is not None and self._line_pts[1] is not None:
            p0, p1 = self._line_pts[0]["uv"], self._line_pts[1]["uv"]
            self._line_curve.setData(x=[p0[0], p1[0]], y=[p0[1], p1[1]])
            self._line_curve.setVisible(True)
        else:
            self._line_curve.setVisible(False)
        self._refresh_line_spot()
        self._update_line_readout()

    def _refresh_line_spot(self):
        """Blue disk on the plate where the fiducial line pierces this slice.

        After 'Reslice perpendicular to line' the line is normal to the plane,
        so it is seen end-on as a single point; the disk marks that crossing.
        Hidden when the line lies in-plane (the dashed curve shows it there) or
        when the plane is beyond the line's span.
        """
        both = self._line_pts[0] is not None and self._line_pts[1] is not None
        if self._section is None or self._line_section is None or not both:
            self._line_spot.setVisible(False)
            return
        w0 = self._line_point_world(self._line_pts[0])
        w1 = self._line_point_world(self._line_pts[1])
        n, _, _ = self._section.basis()
        d = w1 - w0
        denom = float(d @ n)
        if abs(denom) < 1e-6:            # line lies in-plane; curve already shows it
            self._line_spot.setVisible(False)
            return
        t = float((self._section.plane_center() - w0) @ n) / denom
        if not (-1e-6 <= t <= 1.0 + 1e-6):   # plane is outside the line segment
            self._line_spot.setVisible(False)
            return
        pierce = w0 + t * d
        uv, _ = section_mod.world_to_plane(self._section, pierce[None, :])
        self._line_spot.setData([{"pos": (float(uv[0, 0]), float(uv[0, 1]))}])
        self._line_spot.setVisible(True)

    def _update_line_readout(self):
        if self._line_section is None or self._section is None:
            return
        if self._line_pts[0] is None and self._line_pts[1] is None:
            return
        parts = []
        for i, pt in enumerate(self._line_pts):
            if pt is None:
                continue
            w = self._line_point_world(pt)
            depth = pt["pos"] - self._section.position
            parts.append(f"P{i + 1}: ML={w[0]:.0f} AP={w[1]:.0f} DV={w[2]:.0f} µm "
                         f"(Δsection {depth:+.0f})")
        if self._line_pts[0] is not None and self._line_pts[1] is not None:
            parts.append(self._line_angle_text())
        if parts:
            self.readout.setText("   |   ".join(parts))

    def _line_angle_text(self) -> str:
        """Direction of the line relative to the current section axes."""
        n, u, v = self._section.basis()
        d = self._line_point_world(self._line_pts[1]) - self._line_point_world(self._line_pts[0])
        du, dv, dn = float(d @ u), float(d @ v), float(d @ n)
        in_plane = np.degrees(np.arctan2(dv, du))
        out_plane = np.degrees(np.arctan2(dn, float(np.hypot(du, dv))))
        return (f"line {float(np.linalg.norm(d)):.0f} µm: in-plane {in_plane:+.1f}°, "
                f"out-of-plane {out_plane:+.1f}°")

    def _hover_line_readout(self, pt):
        """While placing a point, show the coordinate under the cursor live."""
        if self._line_section is None:
            return
        idx = 1 if self._line_mode == "p2" else 0
        probe = {"uv": np.array([pt.x(), pt.y()], float),
                 "pos": float(self._section.position)}
        w = self._line_point_world(probe)
        self.readout.setText(
            f"placing P{idx + 1} @ ML={w[0]:.0f} AP={w[1]:.0f} DV={w[2]:.0f} µm "
            f"(section {self._section.position:+.0f}) — click to set")

    # -- drawing ------------------------------------------------------------
    def start_draw_contour(self, region: str):
        self._cancel_draw()
        self._mode = "contour"
        self._draw_region = region

    def start_place_landmark(self):
        self._cancel_draw()
        self._mode = "landmark"

    def _cancel_draw(self):
        self._mode = None
        self._draw_pts = []
        self._draw_item.setData([], [])
        self._line_mode = None   # drawing a contour/landmark ends line placement

    def finish_contour(self):
        if self._mode == "contour" and len(self._draw_pts) >= 3 and self._section is not None:
            uv = np.array(self._draw_pts)
            world = section_mod.plane_to_world(self._section, uv)
            if self.on_contour_drawn is not None:
                self.on_contour_drawn(self._draw_region, world)
        self._cancel_draw()

    def _on_click(self, ev):
        if self._section is None:
            return
        pt = self.vb.mapSceneToView(ev.scenePos())
        if self._line_mode is not None:
            self._on_line_click(ev, pt)
            return
        if self._mode is None:
            # Claude added 2026-10-22: a plain left-click (no modifiers, not a
            # double-click, and not mid vertex-select / contour-move — those are
            # consumed by the scene event filter) records the cursor's world
            # position as a saved alignment pick.
            if (ev.button() == QtCore.Qt.MouseButton.LeftButton and not ev.double()
                    and not ev.modifiers() and self._moving is None
                    and self.on_point_picked is not None):
                world = section_mod.plane_to_world(
                    self._section, np.array([[pt.x(), pt.y()]]))[0]
                self.on_point_picked(world)
                ev.accept()
            return
        if self._mode == "landmark":
            world = section_mod.plane_to_world(self._section, np.array([[pt.x(), pt.y()]]))[0]
            if self.on_landmark_placed is not None:
                self.on_landmark_placed(world)
            self._cancel_draw()
            ev.accept()
            return
        # contour mode
        if ev.double():
            self.finish_contour()
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.finish_contour()
        else:
            self._draw_pts.append([pt.x(), pt.y()])
            arr = np.array(self._draw_pts)
            self._draw_item.setData(arr[:, 0], arr[:, 1])
        ev.accept()

    # -- hover --------------------------------------------------------------
    def _on_hover(self, pos):
        if self._lab is None or self._a is None:
            return
        pt = self.vb.mapSceneToView(pos)
        if self._line_mode is not None:   # placing a line point: show its coords
            self._hover_line_readout(pt)
            return
        # Claude fixed 2026-10-20: previously returned here whenever a section
        # line had endpoints, which suppressed region labels after a slice/
        # reslice. Fall through so hover reports the region again; the line's
        # angle summary still shows in the right-panel line_help readout.
        # if self._line_pts[0] is not None or self._line_pts[1] is not None:
        #     return   # keep the line coordinate/angle readout visible
        if len(self._a) < 2 or len(self._b) < 2:
            return
        # Claude added 2026-10-22: report the cursor's M-L / A-P / D-V world
        # position (mm, 3 dp) alongside the structure name. plane_to_world gives
        # the true 3D point for any orientation (cardinal, tilted, or oblique).
        coord = self._coord_text(pt)
        ai = int(round((pt.x() - self._a[0]) / (self._a[1] - self._a[0])))
        bi = int(round((pt.y() - self._b[0]) / (self._b[1] - self._b[0])))
        if 0 <= bi < self._lab.shape[0] and 0 <= ai < self._lab.shape[1]:
            sid = int(self._lab[bi, ai])
            name = self._names(sid) if (sid and self._names) else ""
        else:
            name = ""
        self.readout.setText(f"{name or '—'}{coord}")

    def _coord_text(self, pt) -> str:
        """'   |   ML=.. AP=.. DV=.. mm' for the cursor, or '' if no section."""
        if self._section is None:
            return ""
        w = section_mod.plane_to_world(self._section, np.array([[pt.x(), pt.y()]]))[0]
        return (f"   |   ML={w[0] / 1000.0:.3f} AP={w[1] / 1000.0:.3f} "
                f"DV={w[2] / 1000.0:.3f} mm")
