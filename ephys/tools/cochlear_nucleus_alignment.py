#!/usr/bin/env python3
"""
cochlear_nucleus_alignment.py

Tool for aligning acq4 microscopy images (and, later, video stacks) of mouse
cochlear-nucleus sections against reference atlases, and for localizing cells
within those atlases. Pure Python: pyqtgraph (OpenGL) for 3D display, no
ParaView.

Build stages:

    Step 1  Muniak et al. (2013) CBA-mouse CN atlas: translucent 3D surface in
            four coordinate frames with tonotopic / cochleotopic overlays.
    Step 2  Kim / Chon (2019) Unified Mouse Brain Atlas: translucent brain
            outline + cochlear-nucleus subregions + a checkable region tree.
    Step 3  Superimpose Muniak on Kim (region boundaries).
    Step 4  Read acq4 tiffs / video stacks and register them.
    Step 5+ Find and localize cells.

Layout
------
This file is the GUI shell + file management + entry point. The atlas readers,
mesh builders, rendering helpers, and per-atlas view controllers live in the
``cn_atlas`` subpackage::

    cn_atlas/config.py    config + path resolution
    cn_atlas/muniak.py    Muniak AVS-UCD reader + surface/cache
    cn_atlas/nifti.py     minimal NIfTI-1 reader
    cn_atlas/kim.py       Kim ontology + region mesh builder/cache
    cn_atlas/render.py    shader + colormaps
    cn_atlas/viewers.py   MuniakViewer / KimViewer controllers + ColorBar

Author: Paul Manis (with Claude Code)
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets

from ephys.tools.cn_atlas import config, kim, muniak, render
from ephys.tools.cn_atlas.kim_viewer import KimViewer
from ephys.tools.cn_atlas.section_view import SectionPlate
from ephys.tools.cn_atlas.util import cprint
from ephys.tools.cn_atlas.viewers import ColorBar, MuniakViewer

ATLASES = ("Muniak CN", "Kim Unified Brain")


class CNAlignmentWindow(QtWidgets.QMainWindow):
    """Window shell: atlas selector + swappable control panels + shared 3D view."""

    def __init__(self, cfg: dict, rebuild: bool = False):
        super().__init__()
        self.cfg = cfg
        self.rebuild = rebuild
        render.register_shaders()

        self.setWindowTitle("Cochlear Nucleus Alignment")
        self.resize(1280, 860)

        cache_dir = cfg.get("CacheDir", "~/.cache/cn_alignment")
        muniak_root = config.resolve_root(cfg, "MuniakAtlasRoots")
        kim_root = config.resolve_root(cfg, "KimAtlasRoots")

        # shared 3D view + colorbar + 2D section plate
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(pg.mkColor(18, 20, 26))
        self.colorbar = ColorBar()
        self.section_plate = SectionPlate()
        self.section_plate.setVisible(False)

        # per-atlas viewers (built lazily on first activation)
        self.viewers: dict[str, object] = {}
        self.panels = QtWidgets.QStackedWidget()
        if muniak_root is not None:
            self.viewers["Muniak CN"] = MuniakViewer(
                self.view, self.colorbar, cfg, cache_dir, muniak_root, rebuild)
        if kim_root is not None:
            kv = KimViewer(self.view, self.colorbar, cfg, cache_dir, kim_root, rebuild,
                           section_plate=self.section_plate, muniak_root=muniak_root)
            kv.on_section_visibility = self._set_section_visible
            self.viewers["Kim Unified Brain"] = kv

        self._build_ui(muniak_root, kim_root)
        self.active: Optional[str] = None
        start = cfg.get("DefaultAtlas", "Muniak CN")
        start = start if start in self.viewers else next(iter(self.viewers), None)
        if start is not None:
            self.atlas_combo.setCurrentText(start)
            self._switch_atlas(start)

    def _build_ui(self, muniak_root, kim_root):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QWidget()
        left.setFixedWidth(300)
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(QtWidgets.QLabel("<b>Atlas</b>"))
        self.atlas_combo = QtWidgets.QComboBox()
        self.atlas_combo.addItems([a for a in ATLASES if a in self.viewers])
        self.atlas_combo.currentTextChanged.connect(self._switch_atlas)
        lv.addWidget(self.atlas_combo)

        for name in ATLASES:
            if name in self.viewers:
                self.panels.addWidget(self.viewers[name].panel)
        lv.addWidget(self.panels, stretch=1)

        # Claude added 2026-10-22: launch the acq4 mosaic importer (separate
        # window). Lazy import so the main tool starts without it.
        self.mosaic_btn = QtWidgets.QPushButton("Mosaic importer…")
        self.mosaic_btn.clicked.connect(self._open_mosaic_importer)
        lv.addWidget(self.mosaic_btn)
        self._mosaic_win = None

        if not self.viewers:
            lv.addWidget(QtWidgets.QLabel(
                "<span style='color:#f66'>No atlas found. Mount the drive or "
                "edit the atlas roots in the config.</span>"))

        self.right_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.right_split.addWidget(self.view)
        self.right_split.addWidget(self.section_plate)
        self.right_split.setStretchFactor(0, 3)
        self.right_split.setStretchFactor(1, 2)

        # right-side per-atlas controls (e.g. Muniak placement/fine-tune for Kim)
        self.right_stack = QtWidgets.QStackedWidget()
        self._empty_right = QtWidgets.QWidget()
        self.right_stack.addWidget(self._empty_right)
        self._right_pages = {}
        for name in ATLASES:
            if name in self.viewers:
                rp = getattr(self.viewers[name], "right_panel", None)
                if rp is not None:
                    self.right_stack.addWidget(rp)
                    self._right_pages[name] = rp

        hbox.addWidget(left)
        hbox.addWidget(self.colorbar)
        hbox.addWidget(self.right_split, stretch=1)
        hbox.addWidget(self.right_stack)

        self._missing = []
        if muniak_root is None:
            self._missing.append("Muniak")
        if kim_root is None:
            self._missing.append("Kim")

    def _open_mosaic_importer(self):
        # Claude added 2026-10-22: open (or re-show) the mosaic importer window.
        from ephys.tools.cn_atlas.mosaic.browser import MosaicBrowserWindow
        if self._mosaic_win is None:
            self._mosaic_win = MosaicBrowserWindow(parent=self)
        self._mosaic_win.show()
        self._mosaic_win.raise_()
        self._mosaic_win.activateWindow()

    def _set_section_visible(self, on: bool):
        self.section_plate.setVisible(on)

    def _switch_atlas(self, name: str):
        if name not in self.viewers or name == self.active:
            return
        if self.active is not None:
            self.viewers[self.active].deactivate()
        self.active = name
        self.panels.setCurrentWidget(self.viewers[name].panel)
        rp = self._right_pages.get(name)
        if rp is not None:
            self.right_stack.setCurrentWidget(rp)
            self.right_stack.setVisible(True)
        else:
            self.right_stack.setCurrentWidget(self._empty_right)
            self.right_stack.setVisible(False)
        self.viewers[name].activate()

    def closeEvent(self, event):  # noqa: N802 - Qt signature
        prefs = {
            "DefaultAtlas": self.active or self.cfg.get("DefaultAtlas", "Muniak CN"),
            "CacheDir": self.cfg.get("CacheDir", "~/.cache/cn_alignment"),
        }
        for v in self.viewers.values():
            v.remember(prefs)
        config.save_user_config(prefs)
        super().closeEvent(event)


def _build_all(cfg: dict, rebuild: bool) -> int:
    """Parse + cache every atlas product without opening a GUI."""
    cache_dir = cfg["CacheDir"]
    status = 0
    mroot = config.resolve_root(cfg, "MuniakAtlasRoots")
    if mroot is not None:
        for frame in muniak.FRAMES:
            muniak.load_frame(frame, mroot, cache_dir, rebuild=rebuild)
    else:
        cprint("y", "cn_alignment: no Muniak atlas root found; skipping")
        status = 2
    kroot = config.resolve_root(cfg, "KimAtlasRoots")
    if kroot is not None:
        atlas = kim.KimAtlas(kroot, cfg, cache_dir)
        atlas.present_labels()
        atlas.brain_outline(rebuild=rebuild)
        for g in kim.CN_GROUPS:
            atlas.group_mesh(g, rebuild=rebuild)
    else:
        cprint("y", "cn_alignment: no Kim atlas root found; skipping")
        status = 2
    return status


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Cochlear nucleus atlas alignment")
    parser.add_argument("--config", default=None, help="extra config file to merge")
    parser.add_argument("--atlas", default=None, choices=list(ATLASES))
    parser.add_argument("--frame", default=None, choices=list(muniak.FRAMES.keys()))
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="re-parse source data, ignoring the cache")
    parser.add_argument("--build-only", action="store_true",
                        help="parse + cache all atlas products and exit (no GUI)")
    args = parser.parse_args(argv)

    cfg = config.get_configuration(args.config)
    if args.atlas:
        cfg["DefaultAtlas"] = args.atlas
    if args.frame:
        cfg["DefaultFrame"] = args.frame

    if args.build_only:
        return _build_all(cfg, args.rebuild_cache)

    app = pg.mkQApp("Cochlear Nucleus Alignment")
    win = CNAlignmentWindow(cfg, rebuild=args.rebuild_cache)
    win.show()
    app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
