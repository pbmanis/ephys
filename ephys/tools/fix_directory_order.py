"""
fix_directory_order.py

Advisory tool for checking and reorganizing an acq4 "day" directory tree.

This tool NEVER moves, renames, or deletes anything on disk. acq4 data must be
reorganized with acq4's own Data Manager so that the .index files travel with
the data. The purpose here is to read the .index files, diagnose structural and
temporal-order problems in a day's directory tree, and show the user exactly
what to change by hand.

Expected hierarchy (see dir_check.py):

    Day  (YYYY.MM.DD_NNN)
        slice_NNN
            [images / videos: image_*.tif, video_*.ma, 2pStack_*, 2pImage_*]
            [label dirs: FITC, TRITC, ... (holding images/videos)]
            cell_MMM
                protocol dirs (CCIV_*, VC_*, EPSP_*, Map_*, user-named, ...)

Problems flagged / corrected in the proposed tree (later stages):
  - slices / cells / protocols out of temporal order (by .index __timestamp__)
  - a label dir (FITC/TRITC) nested inside another label dir -> hoist to be a
    parallel sibling under the slice
  - a cell_* dir not directly under a slice -> relocate under the last slice
    that precedes it in time (name collision in that slice flagged in red)

Layout:
    Top    : recent-directory history (persisted across restarts) + Browse /
             Previous day / Next day / Reload current structure buttons.
    Left   : current structure, with per-directory timestamps; problems colored.
    Right  : suggested reordering (temporally ordered, hierarchy corrected).
    Center : arrows from left items to their proposed position on the right.

pbmanis 2026
"""

import datetime
import html
import inspect
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import ephys
import pylibrary.tools.cprint as CP
import pyqtgraph as pg
import pyqtgraph.dockarea as PGD
from pylibrary.tools import fileselector as FS
from pyqtgraph import configfile
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# acq4 reader: readDirIndex(dir) -> dict; dir's own metadata is under the "."
# key, which carries __timestamp__ (epoch seconds).
from ephys.datareaders import acq4_reader

# acq4 "day" directory names look like 2025.01.10_000.
_DATEDIR_RE = re.compile(r"\d{4}\.\d{2}\.\d{2}_\d{3}")
_SLICE_RE = re.compile(r"slice_\d+$")
_CELL_RE = re.compile(r"cell_\d+$")
# "pair_NNN" dirs are not normally used here; when present they were usually
# inserted by accident (wrong button in acq4). Treat them exactly like cells.
_PAIR_RE = re.compile(r"pair_\d+$")
# acq4 protocol-run subdirectories: "000", "000_001", "000_001_002", ... (3-digit
# groups joined by underscores). These hold data files, not hierarchy, so they
# are not opened, listed, or recursed into.
_RUNDIR_RE = re.compile(r"\d{3}(_\d{3})*$")
# A label (fluorophore) image directory that may be nested and need hoisting.
_LABEL_DIRS = ("FITC", "TRITC", "DAPI", "GFP", "CY3", "CY5", "ALEXA")

# Files that are image/video media (may sit at the slice level or in a label dir).
_MEDIA_RE = re.compile(
    r"^(image|video|2pStack|2pImage)_\d{3,4}\.(tif|tiff|ma)$", re.IGNORECASE
)

# Housekeeping names that are not part of the data hierarchy.
_SKIP_NAMES = {".index", ".DS_Store", "log.txt", "Thumbs.db"}

# Maximum number of recent directories to remember.
_MAX_RECENT = 15

# Problem tag -> colour used to mark a node in place (left tree) and to annotate
# the proposed tree (right tree, later stages).
_PROBLEM_COLORS = {
    "reorder": "#1f6feb",   # blue:   out of temporal order (position changes)
    "relocate": "#e36209",  # orange: cell not under a slice -> move it
    "hoist": "#e36209",     # orange: label dir nested in another -> move up
    "conflict": "#d1242f",  # red:    name collision at the proposed destination
}


def _ephys_config_dir() -> Path:
    """Return the packaged ephys/config directory (independent of cwd)."""
    return Path(inspect.getfile(ephys)).parent / "config"


def _config_path() -> Path:
    """Location of this tool's persisted settings (recent-directory history)."""
    return _ephys_config_dir() / "fix_directory_order.cfg"


def load_config() -> dict:
    """Read the persisted settings. Missing/unreadable -> defaults."""
    path = _config_path()
    try:
        if path.is_file():
            cfg = configfile.readConfigFile(str(path))
            if isinstance(cfg, dict):
                return cfg
    except Exception as exc:  # noqa: BLE001 - corrupt file -> fall back to defaults
        CP.cprint("y", f"fix_directory_order: could not read {path!s}: {exc}")
    return {"RecentDirs": []}


@dataclass
class Node:
    """One directory in the acq4 day tree (media files are summarized on their
    containing directory rather than given their own nodes)."""

    path: Path
    name: str
    kind: str  # "day" | "slice" | "cell" | "label" | "protocol"
    timestamp: Optional[float]  # epoch seconds, or None if unknown
    ts_source: str  # "index" | "parent" | "mtime" | "none"
    dir_type: Optional[str]  # acq4 .index dirType ("Slice"/"Cell"/...), if present
    depth: int
    media: list = field(default_factory=list)  # media filenames directly in this dir
    children: list = field(default_factory=list)  # child Node objects (dirs only)
    parent: Optional["Node"] = None
    # structural/temporal problems, attached by the diagnostics pass (later stage)
    problems: list = field(default_factory=list)

    @property
    def is_media_dir(self) -> bool:
        """A directory that holds image/video media (slice level or label dir)."""
        return bool(self.media)


@dataclass
class PNode:
    """One directory in the *proposed* (corrected) tree. It references the
    original model node it came from (``source``) so the arrow overlay can link
    the two trees by path, and carries the change that put it here."""

    source: Node
    change: Optional[str]  # None | "reorder" | "relocate" | "hoist" | "conflict"
    children: list = field(default_factory=list)  # child PNode objects
    note: str = ""  # extra explanation (e.g. a name-collision description)


class _ArrowCanvas(QtWidgets.QWidget):
    """Thin column between the two trees that paints connector arrows from each
    moved item on the left to its proposed position on the right. All the
    geometry lives in the owner's _paint_arrows()."""

    def __init__(self, owner: "FixDirectoryOrder"):
        super().__init__()
        self.owner = owner
        self.setMinimumWidth(64)
        # don't eat mouse events meant for the splitter handles
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, a0):  # noqa: N802 - Qt override name
        self.owner._paint_arrows(self)


class FixDirectoryOrder(pg.QtWidgets.QWidget):
    def __init__(self, config=None, experiments=None, app=None):
        super().__init__()
        self.config = config if config else {"RecentDirs": []}
        self.experiments = experiments if experiments else {}
        self.app = app

        # current acq4 "day" directory under examination
        self.day_dir: Optional[Path] = None
        # in-memory structure model, filled by scan_structure()
        self.model = None
        # proposed (corrected) tree, filled by build_proposed()
        self.proposed = None

        self.AR = acq4_reader.acq4_reader()

        # widgets created in set_window()
        self.win = None
        self.recent_list: Optional[QtWidgets.QListWidget] = None
        self.left_tree: Optional[QtWidgets.QTreeWidget] = None
        self.right_tree: Optional[QtWidgets.QTreeWidget] = None
        self.arrow_canvas: Optional["_ArrowCanvas"] = None
        self.flag_order_cb: Optional[QtWidgets.QCheckBox] = None
        self.banner: Optional[QtWidgets.QLabel] = None
        self.status_path: Optional[QtWidgets.QLabel] = None
        # path-string -> QTreeWidgetItem, so the arrow overlay can map a model
        # node to its item in each tree.
        self._left_items: dict = {}
        self._right_items: dict = {}
        # [(path, colour)] for each moved item that gets a left->right arrow
        self._arrow_pairs: list = []

        self._normalize_recent()
        self.datadir = self._resolve_datadir()

    # ------------------------------------------------------------------
    # recent-directory history (persisted to fix_directory_order.cfg)
    # ------------------------------------------------------------------
    def _normalize_recent(self):
        """Coerce RecentDirs into a list of at most _MAX_RECENT entries."""
        rd = self.config.get("RecentDirs", [])
        if isinstance(rd, str):
            rd = [rd] if rd else []
        elif rd is None:
            rd = []
        else:
            rd = list(rd)
        self.config["RecentDirs"] = rd[:_MAX_RECENT]

    def _resolve_datadir(self) -> str:
        """Pick a sensible starting directory for the Browse dialog: the newest
        existing recent entry, else the user's home directory."""
        for r in self.config.get("RecentDirs", []):
            rp = Path(r)
            if rp.exists():
                return str(rp if rp.is_dir() else rp.parent)
        return str(Path.home())

    def _save_config(self):
        """Persist settings. This file is owned solely by this tool, so the
        whole dict is rewritten (no comments to preserve)."""
        path = _config_path()
        try:
            configfile.writeConfigFile(self.config, str(path))
        except Exception as exc:  # noqa: BLE001
            CP.cprint("r", f"fix_directory_order: failed to save {path!s}: {exc}")

    def _push_recent(self, path: Union[str, Path]):
        """Push a day directory onto the recent stack (newest first, de-duped,
        capped) and persist it."""
        path = str(path)
        lst = [p for p in self.config.get("RecentDirs", []) if p != path]
        lst.insert(0, path)
        self.config["RecentDirs"] = lst[:_MAX_RECENT]
        self._refresh_recent_list()
        self._save_config()

    def _recent_label(self, path: str) -> str:
        """Shorten a path to the tail starting at the day directory."""
        parts = Path(path).parts
        for i, part in enumerate(parts):
            if _DATEDIR_RE.fullmatch(part):
                return str(Path(*parts[i:]))
        return str(Path(*parts[-3:])) if len(parts) >= 3 else str(path)

    def _refresh_recent_list(self):
        """Repopulate the recent-directories list widget from self.config."""
        if self.recent_list is None:
            return
        self.recent_list.blockSignals(True)
        self.recent_list.clear()
        for p in self.config.get("RecentDirs", []):
            item = QtWidgets.QListWidgetItem(self._recent_label(p))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self.recent_list.addItem(item)
        self.recent_list.blockSignals(False)

    # ------------------------------------------------------------------
    # day-directory discovery and navigation
    # ------------------------------------------------------------------
    def _coerce_to_day_dir(self, path: Union[str, Path]) -> Optional[Path]:
        """Resolve a user-chosen path to an acq4 day directory:
        - path itself is a day dir -> use it
        - a parent of path is a day dir -> use that parent
        - path contains day-dir children -> use the earliest one
        """
        p = Path(path)
        if _DATEDIR_RE.fullmatch(p.name):
            return p
        for anc in p.parents:
            if _DATEDIR_RE.fullmatch(anc.name):
                return anc
        if p.is_dir():
            days = sorted(c for c in p.iterdir() if c.is_dir() and _DATEDIR_RE.fullmatch(c.name))
            if days:
                return days[0]
        return None

    def _sibling_days(self) -> list:
        """Sorted list of sibling day directories in the current day's parent."""
        if self.day_dir is None:
            return []
        parent = self.day_dir.parent
        try:
            return sorted(
                c for c in parent.iterdir() if c.is_dir() and _DATEDIR_RE.fullmatch(c.name)
            )
        except OSError:
            return []

    def go_next_day(self):
        self._step_day(+1)

    def go_prev_day(self):
        self._step_day(-1)

    def _step_day(self, delta: int):
        sibs = self._sibling_days()
        if not sibs or self.day_dir is None:
            return
        try:
            i = sibs.index(self.day_dir)
        except ValueError:
            return
        j = i + delta
        if 0 <= j < len(sibs):
            self.load_day(sibs[j], push_recent=True)
        else:
            self._set_status(
                f"Already at the {'last' if delta > 0 else 'first'} day in "
                f"{self.day_dir.parent}"
            )

    def reload_structure(self):
        """Re-scan the current day (used after the user fixes things in acq4's
        Data Manager, to confirm the structure is now correct)."""
        if self.day_dir is not None:
            self.load_day(self.day_dir, push_recent=False)

    def _strict_order(self) -> bool:
        """True when the user has opted (checkbox) to flag a cell's protocol dirs
        whose name order differs from their time order. Default False: such
        order-only differences are not considered a problem to fix."""
        return bool(self.flag_order_cb is not None and self.flag_order_cb.isChecked())

    def _on_toggle_order_mode(self, _checked: bool):
        """Re-process the current day when the protocol-order mode changes."""
        if self.day_dir is not None:
            self.load_day(self.day_dir, push_recent=False)

    # ------------------------------------------------------------------
    # load / scan / render
    # ------------------------------------------------------------------
    def browse_for_day(self):
        sel = FS.FileSelector(dialogtype="dir", startingdir=str(self.datadir))
        if sel.fileName is None:
            return
        self.load_day(sel.fileName, push_recent=True)

    def load_recent(self, item: QtWidgets.QListWidgetItem):
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if path:
            self.load_day(path, push_recent=True)

    def load_day(self, path: Union[str, Path], push_recent: bool = True):
        """Set the current day directory, scan it, and render both trees."""
        day = self._coerce_to_day_dir(path)
        if day is None or not day.is_dir():
            self._set_status(f"Not an acq4 day directory: {path}")
            CP.cprint("r", f"fix_directory_order: not a day directory: {path}")
            return
        self.day_dir = day
        self.datadir = str(day.parent)
        if push_recent:
            self._push_recent(day)
        self._set_status(str(day))
        if self.win is not None:
            self.win.setWindowTitle(f"Fix Directory Order - {day.name}")

        self.model = self.scan_structure(day)
        self.diagnose(self.model)
        self.render_left()
        self.render_right()
        self.update_banner()

    def scan_structure(self, day_dir: Path):
        """Walk the day tree and build the in-memory structure model (a tree of
        Node objects rooted at the day directory). Purely descriptive: it
        reflects what is on disk, reading each .index once for timestamps and
        dirType. Diagnostics and the proposed reordering come in later stages."""
        return self._build_node(day_dir, parent_index=None, depth=0, parent=None)

    # ------------------------------------------------------------------
    # scan helpers
    # ------------------------------------------------------------------
    def _read_dir_index(self, dirpath: Path) -> Optional[dict]:
        """Read a directory's .index file (quietly). Returns the full index dict
        (keyed by "." and by child names), or None if missing/unreadable."""
        try:
            # configfile.readConfigFile returns None if the file does not exist.
            return configfile.readConfigFile(str(Path(dirpath, ".index")))
        except Exception as exc:  # noqa: BLE001 - corrupt/locked .index -> treat as absent
            CP.cprint("y", f"fix_directory_order: could not read .index in {dirpath}: {exc}")
            return None

    @staticmethod
    def _classify_dir(name: str) -> str:
        """Classify a directory by NAME (per the acq4 convention the spec
        describes). dirType from the .index is recorded separately for
        cross-checking by the diagnostics pass."""
        if _DATEDIR_RE.fullmatch(name):
            return "day"
        if _SLICE_RE.fullmatch(name):
            return "slice"
        # pair_* dirs are accidental here and handled identically to cells
        if _CELL_RE.fullmatch(name) or _PAIR_RE.fullmatch(name):
            return "cell"
        if name.upper() in _LABEL_DIRS:
            return "label"
        return "protocol"  # any other directory (named protocol or user-defined)

    @staticmethod
    def _timestamp_from(index: Optional[dict], key: str = ".") -> Optional[float]:
        """Pull __timestamp__ (epoch seconds) for ``key`` out of an index dict."""
        if not isinstance(index, dict):
            return None
        entry = index.get(key)
        if isinstance(entry, dict) and "__timestamp__" in entry:
            try:
                return float(entry["__timestamp__"])
            except (TypeError, ValueError):
                return None
        return None

    def _build_node(
        self,
        path: Path,
        parent_index: Optional[dict],
        depth: int,
        parent: Optional[Node],
    ) -> Node:
        """Recursively build a Node for ``path``.

        Timestamp precedence: this dir's own .index "." entry, then the parent's
        .index listing for this name, then the filesystem mtime."""
        name = path.name
        own_index = self._read_dir_index(path)
        dir_type = None
        if isinstance(own_index, dict) and isinstance(own_index.get("."), dict):
            dir_type = own_index["."].get("dirType")

        ts = self._timestamp_from(own_index, ".")
        ts_source = "index"
        if ts is None:
            ts = self._timestamp_from(parent_index, name)
            ts_source = "parent"
        if ts is None:
            try:
                ts = path.stat().st_mtime
                ts_source = "mtime"
            except OSError:
                ts, ts_source = None, "none"

        node = Node(
            path=path,
            name=name,
            kind=self._classify_dir(name),
            timestamp=ts,
            ts_source=ts_source,
            dir_type=dir_type,
            depth=depth,
            parent=parent,
        )

        # Separate child directories (recursed) from media files (summarized).
        if depth < 8:
            try:
                entries = sorted(path.iterdir(), key=lambda p: p.name)
            except OSError:
                entries = []
            for e in entries:
                if e.name in _SKIP_NAMES or e.name.startswith("."):
                    continue
                if e.is_dir():
                    # skip acq4 protocol-run subdirectories (000, 000_001, ...)
                    if _RUNDIR_RE.fullmatch(e.name):
                        continue
                    node.children.append(
                        self._build_node(e, own_index, depth + 1, node)
                    )
                elif _MEDIA_RE.match(e.name):
                    node.media.append(e.name)
        return node

    # ------------------------------------------------------------------
    # diagnostics: attach problem tags to nodes in the model
    # ------------------------------------------------------------------
    def diagnose(self, node: Optional[Node]):
        """Walk the model and append ``(type, message)`` problem tags to nodes.

        Three kinds of problem are detected here:
          - "hoist":    a label dir (FITC/TRITC/...) nested inside another label
                        dir, which should become a parallel sibling under the slice.
          - "relocate": a cell_* dir that is not directly under a slice.
          - "reorder":  a slice/cell/protocol out of temporal order within its
                        proper container (name order != timestamp order).
        Name-collision ("conflict") detection depends on the proposed
        destination and is handled with the proposed tree (later stage)."""
        if node is None:
            return
        self._check_sibling_order(node)
        for c in node.children:
            if c.kind == "label" and node.kind == "label":
                c.problems.append(
                    (
                        "hoist",
                        f"'{c.name}' is nested inside label dir '{node.name}'; "
                        f"move it up to be a sibling under the slice",
                    )
                )
            if c.kind == "cell" and node.kind != "slice":
                c.problems.append(
                    (
                        "relocate",
                        f"cell '{c.name}' is not directly under a slice "
                        f"(it is under '{node.name}'); move it under the correct slice",
                    )
                )
            self.diagnose(c)

    # A parent of a given kind contains an ordered run of children of the next
    # kind down; those are the ones whose temporal order we check.
    _ORDERED_CHILD = {"day": "slice", "slice": "cell", "cell": "protocol"}

    def _check_sibling_order(self, node: Node):
        """Flag children that sit in a different slot by name than they would by
        timestamp (i.e. the directory ordering does not match the recording
        order)."""
        kind = self._ORDERED_CHILD.get(node.kind)
        if kind is None:
            return
        # A pure reorder never changes an item's immediate parent, so it is not
        # suggested -- with one opt-in exception: a cell's protocol order can be
        # surfaced by ticking the "Flag protocol order" checkbox.
        if kind != "protocol" or not self._strict_order():
            return
        sibs = [c for c in node.children if c.kind == kind]
        if len(sibs) < 2:
            return
        # sibs is already in on-disk/name order; sort a copy by timestamp.
        time_order = sorted(
            sibs, key=lambda n: (n.timestamp is None, n.timestamp if n.timestamp else 0.0)
        )
        pos_name = {id(n): i for i, n in enumerate(sibs)}
        pos_time = {id(n): i for i, n in enumerate(time_order)}
        for n in sibs:
            if pos_name[id(n)] != pos_time[id(n)]:
                n.problems.append(
                    ("reorder", f"{kind} '{n.name}' is out of temporal order")
                )

    @staticmethod
    def _count_problems(node: Optional[Node]) -> int:
        if node is None:
            return 0
        n = len(node.problems)
        for c in node.children:
            n += FixDirectoryOrder._count_problems(c)
        return n

    # ------------------------------------------------------------------
    # left tree rendering (current structure, problems coloured in place)
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_ts(node: Node) -> str:
        if node.timestamp is None:
            return ""
        s = datetime.datetime.fromtimestamp(node.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        if node.ts_source != "index":  # mark fallback timestamps
            s += f"  ({node.ts_source})"
        return s

    @staticmethod
    def _media_summary(node: Node) -> str:
        if not node.media:
            return ""
        vids = sum(1 for f in node.media if f.lower().startswith("video"))
        imgs = len(node.media) - vids
        parts = []
        if imgs:
            parts.append(f"{imgs} image" + ("s" if imgs != 1 else ""))
        if vids:
            parts.append(f"{vids} video" + ("s" if vids != 1 else ""))
        return ", ".join(parts)

    def _make_item(self, node: Node, items_map: dict) -> QtWidgets.QTreeWidgetItem:
        label = node.name
        media = self._media_summary(node)
        if media:
            label += f"   — {media}"
        item = QtWidgets.QTreeWidgetItem([label, self._fmt_ts(node)])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(node.path))
        if node.problems:
            # first problem drives the colour; tooltip lists them all
            color = _PROBLEM_COLORS.get(node.problems[0][0], "#d1242f")
            brush = QtGui.QBrush(QtGui.QColor(color))
            for col in (0, 1):
                item.setForeground(col, brush)
            f = item.font(0)
            f.setBold(True)
            item.setFont(0, f)
            item.setToolTip(0, "\n".join(p[1] for p in node.problems))
        for c in node.children:
            item.addChild(self._make_item(c, items_map))
        items_map[str(node.path)] = item
        return item

    def render_left(self):
        """Render the current (on-disk) structure, with timestamps and problem
        nodes coloured in place."""
        tree = self.left_tree
        if tree is None:
            return
        tree.clear()
        self._left_items = {}
        if self.model is None:
            return
        tree.addTopLevelItem(self._make_item(self.model, self._left_items))
        tree.expandAll()
        tree.resizeColumnToContents(0)

    # ------------------------------------------------------------------
    # proposed (corrected) tree
    # ------------------------------------------------------------------
    @staticmethod
    def _ts_key(node: Node):
        """Sort key placing timestamped nodes in time order, unknowns last."""
        return (node.timestamp is None, node.timestamp if node.timestamp else 0.0)

    @staticmethod
    def _change_for(source: Node, conflict: bool) -> Optional[str]:
        """Derive a proposed-node change tag from the source node's problems."""
        if conflict:
            return "conflict"
        types = {p[0] for p in source.problems}
        for t in ("relocate", "hoist", "reorder"):
            if t in types:
                return t
        return None

    def _gather_misplaced_cells(self, node: Node, out: list):
        """Collect every cell_* dir whose parent is not a slice (recursing
        through non-cell nodes only)."""
        for c in node.children:
            if c.kind == "cell":
                if node.kind != "slice":
                    out.append(c)
            else:
                self._gather_misplaced_cells(c, out)

    def _gather_labels(self, slice_node: Node, node: Node, out: list):
        """Collect label dirs in a slice subtree as ``(label, is_hoisted)``; a
        label nested inside another label is hoisted to be a slice-level sibling."""
        for c in node.children:
            if c.kind == "label":
                out.append((c, node is not slice_node))
                self._gather_labels(slice_node, c, out)

    def _dest_slice(self, cell: Node, slices: list) -> Optional[Node]:
        """Destination slice for a misplaced cell: the last slice that precedes
        it in time (latest slice with timestamp <= the cell's). If none precedes
        it, fall back to the earliest slice."""
        if not slices:
            return None
        ct = cell.timestamp
        if ct is not None:
            preceding = [s for s in slices if s.timestamp is not None and s.timestamp <= ct]
            if preceding:
                return max(preceding, key=lambda s: s.timestamp)
        return slices[0]

    def _keep_subtree(self, node: Node) -> PNode:
        """Pass a subtree through unchanged (used for directories outside the
        slice/cell/protocol reorganization), preserving any existing flags."""
        p = PNode(source=node, change=self._change_for(node, False))
        for c in node.children:
            p.children.append(self._keep_subtree(c))
        return p

    def _build_cell_pnode(self, cell: Node, is_reloc: bool, conflict: bool) -> PNode:
        if conflict:
            change, note = "conflict", (
                f"name '{cell.name}' already exists in the destination slice — "
                f"rename or merge in acq4's Data Manager"
            )
        elif is_reloc:
            change, note = "relocate", ""
        else:
            change, note = self._change_for(cell, False), ""
        cp = PNode(source=cell, change=change, note=note)
        protos = [c for c in cell.children if c.kind == "protocol"]
        if self._strict_order():  # else keep current on-disk (name) order
            protos = sorted(protos, key=self._ts_key)
        for p in protos:
            cp.children.append(PNode(source=p, change=self._change_for(p, False)))
        return cp

    @staticmethod
    def _merge_by_time(retained: list, inserted: list) -> list:
        """Keep ``retained`` (node, pnode) pairs in their current order and drop
        each ``inserted`` pair in at its temporal slot: before the first retained
        item recorded later than it. Items that stay put never move, so they are
        never flagged as reordered."""
        result = list(retained)
        for node, pnode in sorted(
            inserted, key=lambda e: (e[0].timestamp is None, e[0].timestamp or 0.0)
        ):
            tx = node.timestamp
            idx = len(result)
            if tx is not None:
                for i, (rn, _) in enumerate(result):
                    if rn.timestamp is not None and rn.timestamp > tx:
                        idx = i
                        break
            result.insert(idx, (node, pnode))
        return result

    def _build_slice_pnode(self, s: Node, reloc_cells: list) -> PNode:
        s_p = PNode(source=s, change=self._change_for(s, False))
        used = {c.name for c in s.children if c.kind == "cell"}

        labels = []
        self._gather_labels(s, s, labels)
        hoisted_labels = [lab for lab, hoisted in labels if hoisted]

        # retained: the slice's current direct children, kept in current order.
        retained = []
        for c in s.children:
            if c.kind == "cell":  # proper cell, stays put
                retained.append((c, self._build_cell_pnode(c, False, False)))
            elif c.kind == "label":  # direct label, stays (nested ones hoist out)
                retained.append((c, PNode(source=c, change=self._change_for(c, False))))
            else:  # any other directory under the slice, unchanged
                retained.append((c, self._keep_subtree(c)))

        # inserted: items whose immediate parent changes to this slice.
        inserted = []
        for c in reloc_cells:  # cells relocated from elsewhere
            conflict = c.name in used
            used.add(c.name)
            inserted.append((c, self._build_cell_pnode(c, True, conflict)))
        for lab in hoisted_labels:  # labels hoisted up from inside another label
            inserted.append((lab, PNode(source=lab, change="hoist")))

        s_p.children = [pn for _, pn in self._merge_by_time(retained, inserted)]
        return s_p

    def build_proposed(self, root: Optional[Node]) -> Optional[PNode]:
        """Build the corrected tree: slices/cells/protocols in temporal order,
        nested label dirs hoisted to slice-level siblings, and misplaced cells
        relocated under the last slice that precedes them in time (name
        collisions flagged, but still shown at the proposed destination)."""
        if root is None or root.kind != "day":
            return None
        slices = sorted((c for c in root.children if c.kind == "slice"), key=self._ts_key)
        misplaced = []
        self._gather_misplaced_cells(root, misplaced)
        assigned = {id(s): [] for s in slices}
        unplaceable = []
        for cell in misplaced:
            dest = self._dest_slice(cell, slices)
            if dest is None:
                unplaceable.append(cell)
            else:
                assigned[id(dest)].append(cell)

        day_p = PNode(source=root, change=None)
        # retained day children, kept in current order: slices and any other
        # (non-cell) directories. Misplaced cells are relocated into slices.
        retained = []
        for c in root.children:
            if c.kind == "slice":
                retained.append((c, self._build_slice_pnode(c, assigned.get(id(c), []))))
            elif c.kind == "cell":
                continue  # relocated under a slice (handled above)
            else:
                retained.append((c, self._keep_subtree(c)))
        # cells with no slice at all to hold them: keep at day level, flagged
        inserted = []
        for c in unplaceable:
            pn = self._build_cell_pnode(c, is_reloc=True, conflict=False)
            pn.change = "conflict"
            pn.note = "no slice exists to hold this cell; create/choose a slice in Data Manager"
            inserted.append((c, pn))

        day_p.children = [pn for _, pn in self._merge_by_time(retained, inserted)]
        return day_p

    _CHANGE_TOOLTIP = {
        "reorder": "moved into temporal (recording-time) order",
        "relocate": "relocated here — under the last slice that precedes it in time",
        "hoist": "hoisted up to be a parallel sibling under the slice",
    }

    def _make_pitem(self, p: PNode) -> QtWidgets.QTreeWidgetItem:
        node = p.source
        label = node.name
        media = self._media_summary(node)
        if media:
            label += f"   — {media}"
        item = QtWidgets.QTreeWidgetItem([label, self._fmt_ts(node)])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(node.path))
        if p.change:
            color = _PROBLEM_COLORS.get(p.change, "#d1242f")
            brush = QtGui.QBrush(QtGui.QColor(color))
            for col in (0, 1):
                item.setForeground(col, brush)
            f = item.font(0)
            f.setBold(True)
            item.setFont(0, f)
            item.setToolTip(0, p.note or self._CHANGE_TOOLTIP.get(p.change, p.change))
        for c in p.children:
            item.addChild(self._make_pitem(c))
        self._right_items[str(node.path)] = item
        return item

    def render_right(self):
        """Render the proposed (corrected) structure in the right tree, then
        work out which items get a left->right arrow."""
        tree = self.right_tree
        if tree is None:
            return
        tree.clear()
        self._right_items = {}
        self._arrow_pairs = []
        self.proposed = self.build_proposed(self.model)
        if self.proposed is None:
            if self.arrow_canvas is not None:
                self.arrow_canvas.update()
            return
        tree.addTopLevelItem(self._make_pitem(self.proposed))
        tree.expandAll()
        tree.resizeColumnToContents(0)
        self._collect_arrows(self.proposed, self._arrow_pairs)
        if self.arrow_canvas is not None:
            self.arrow_canvas.update()
            # positions are only final once layout settles; repaint next tick
            QtCore.QTimer.singleShot(0, self.arrow_canvas.update)

    # ------------------------------------------------------------------
    # arrow overlay between the two trees
    # ------------------------------------------------------------------
    @staticmethod
    def _titled(title: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Wrap a widget with a bold title label above it (so each tree is
        labelled inside the shared splitter)."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        lab = QtWidgets.QLabel(title)
        lab.setStyleSheet("font-weight:bold; padding:2px;")
        v.addWidget(lab)
        v.addWidget(widget)
        w.setLayout(v)
        return w

    def _collect_arrows(self, p: PNode, out: list):
        """Collect (path, colour) for every moved proposed node."""
        if p.change in ("reorder", "relocate", "hoist", "conflict"):
            out.append((str(p.source.path), _PROBLEM_COLORS.get(p.change, "#d1242f")))
        for c in p.children:
            self._collect_arrows(c, out)

    def _paint_arrows(self, canvas: "_ArrowCanvas"):
        """Draw a connector from each moved item in the left tree to its
        proposed position in the right tree. Items that are scrolled out of
        view or hidden under a collapsed parent are skipped."""
        if not self._arrow_pairs or self.left_tree is None or self.right_tree is None:
            return
        lv = self.left_tree.viewport()
        rv = self.right_tree.viewport()
        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        try:
            for path, color in self._arrow_pairs:
                li = self._left_items.get(path)
                ri = self._right_items.get(path)
                if li is None or ri is None:
                    continue
                lr = self.left_tree.visualItemRect(li)
                rr = self.right_tree.visualItemRect(ri)
                if lr.isEmpty() or rr.isEmpty():  # under a collapsed parent
                    continue
                ly, ryv = lr.center().y(), rr.center().y()
                if not (0 <= ly <= lv.height()) or not (0 <= ryv <= rv.height()):
                    continue  # scrolled out of view
                # map the viewport y to canvas-local y via global coords
                y0 = canvas.mapFromGlobal(lv.mapToGlobal(QtCore.QPoint(0, ly))).y()
                y1 = canvas.mapFromGlobal(rv.mapToGlobal(QtCore.QPoint(0, ryv))).y()
                x0, x1 = 0, canvas.width()
                pen = QtGui.QPen(QtGui.QColor(color))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(x0, y0, x1, y1)
                self._draw_arrowhead(painter, x1, y1, x1 - x0, y1 - y0, color)
        finally:
            painter.end()

    @staticmethod
    def _draw_arrowhead(painter, x, y, dx, dy, color, size: int = 7):
        """Filled triangular arrowhead at (x, y) pointing along (dx, dy)."""
        ang = math.atan2(dy, dx)
        p0 = QtCore.QPointF(x, y)
        p1 = QtCore.QPointF(
            x - size * math.cos(ang - math.pi / 7),
            y - size * math.sin(ang - math.pi / 7),
        )
        p2 = QtCore.QPointF(
            x - size * math.cos(ang + math.pi / 7),
            y - size * math.sin(ang + math.pi / 7),
        )
        painter.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        painter.drawPolygon(QtGui.QPolygonF([p0, p1, p2]))

    def update_banner(self):
        """Header banner: green 'no reordering' or amber 'suggested reordering',
        with the current day directory shown in bold."""
        if self.banner is None:
            return
        if self.model is None:
            self.banner.setText("Select a day directory to begin.")
            self.banner.setStyleSheet("")
            return
        day = html.escape(self.day_dir.name) if self.day_dir is not None else ""
        n = self._count_problems(self.model)
        if n == 0:
            self.banner.setText(f"✓   <b>{day}</b>   —   No reordering required")
            self.banner.setStyleSheet(
                "background-color:#1a7f37; color:white; padding:4px;"
            )
        else:
            self.banner.setText(
                f"⚠   <b>{day}</b>   —   Suggested reordering   "
                f"({n} item(s) flagged)"
            )
            self.banner.setStyleSheet(
                "background-color:#9a6700; color:white; padding:4px;"
            )

    # ------------------------------------------------------------------
    # window construction
    # ------------------------------------------------------------------
    def set_window(self, parent=None):
        self.win = pg.QtWidgets.QMainWindow()
        self.win.setWindowTitle("Fix Directory Order")
        self.DockArea = PGD.DockArea()
        self.win.setCentralWidget(self.DockArea)
        self.win.resize(1500, 1000)

        # --- top dock: recent history + navigation controls ---------------
        self.Dock_Top = PGD.Dock("Day selection", size=(1500, 180))
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(6, 6, 6, 6)
        top.setLayout(top_layout)

        # navigation buttons (left, per lab convention)
        btn_box = QtWidgets.QVBoxLayout()
        btn_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        self.btn_browse.clicked.connect(self.browse_for_day)
        self.btn_prev = QtWidgets.QPushButton("Previous day")
        self.btn_prev.clicked.connect(self.go_prev_day)
        self.btn_next = QtWidgets.QPushButton("Next day")
        self.btn_next.clicked.connect(self.go_next_day)
        self.btn_reload = QtWidgets.QPushButton("Reload current structure")
        self.btn_reload.clicked.connect(self.reload_structure)
        for b in (self.btn_browse, self.btn_prev, self.btn_next, self.btn_reload):
            btn_box.addWidget(b)
        # Default (unticked): a cell's protocol dirs differing only in name order
        # vs time order is NOT treated as needing a change. Tick to flag those.
        self.flag_order_cb = QtWidgets.QCheckBox("Flag protocol order\n(name vs time)")
        self.flag_order_cb.toggled.connect(self._on_toggle_order_mode)
        btn_box.addWidget(self.flag_order_cb)
        btn_box.addStretch(1)
        top_layout.addLayout(btn_box, stretch=0)

        # recent-directories list (persisted across restarts)
        recent_box = QtWidgets.QVBoxLayout()
        recent_box.addWidget(QtWidgets.QLabel("Recent day directories:"))
        self.recent_list = QtWidgets.QListWidget()
        self.recent_list.setMinimumHeight(120)
        self.recent_list.itemClicked.connect(self.load_recent)
        recent_box.addWidget(self.recent_list)
        top_layout.addLayout(recent_box, stretch=1)

        self.Dock_Top.addWidget(top)

        # --- banner (no-change vs suggested-reordering message) -----------
        self.Dock_Banner = PGD.Dock("Status", size=(1500, 40))
        self.Dock_Banner.hideTitleBar()
        self.banner = QtWidgets.QLabel("Select a day directory to begin.")
        self.banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        f = self.banner.font()
        f.setPointSize(f.pointSize() + 2)
        self.banner.setFont(f)
        self.Dock_Banner.addWidget(self.banner)

        # --- trees + arrow canvas (one dock, three-way splitter) ----------
        # Keeping both trees and the arrow canvas in a single splitter (rather
        # than separate docks) keeps the arrow geometry valid: the canvas edges
        # always abut the two tree viewports.
        self.Dock_Trees = PGD.Dock("Current  ->  Suggested", size=(1500, 760))

        def _make_tree():
            t = QtWidgets.QTreeWidget()
            t.setHeaderLabels(["Directory", "Timestamp"])
            t.setColumnWidth(0, 360)
            t.setUniformRowHeights(True)
            t.setRootIsDecorated(True)
            return t

        self.left_tree = _make_tree()
        self.right_tree = _make_tree()
        self.arrow_canvas = _ArrowCanvas(self)

        trees_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        trees_split.addWidget(self._titled("Current structure", self.left_tree))
        trees_split.addWidget(self.arrow_canvas)
        trees_split.addWidget(self._titled("Suggested reordering", self.right_tree))
        trees_split.setStretchFactor(0, 1)
        trees_split.setStretchFactor(1, 0)
        trees_split.setStretchFactor(2, 1)
        trees_split.setSizes([680, 70, 680])
        self.Dock_Trees.addWidget(trees_split)

        # repaint the arrows whenever either tree scrolls or expands/collapses
        for t in (self.left_tree, self.right_tree):
            t.verticalScrollBar().valueChanged.connect(self.arrow_canvas.update)
            t.horizontalScrollBar().valueChanged.connect(self.arrow_canvas.update)
            t.itemExpanded.connect(lambda *_: self.arrow_canvas.update())
            t.itemCollapsed.connect(lambda *_: self.arrow_canvas.update())

        # keep the two trees scrolled together. Qt does not re-emit valueChanged
        # when the value is unchanged, so this bidirectional link cannot loop.
        lsb = self.left_tree.verticalScrollBar()
        rsb = self.right_tree.verticalScrollBar()
        lsb.valueChanged.connect(rsb.setValue)
        rsb.valueChanged.connect(lsb.setValue)

        self.DockArea.addDock(self.Dock_Top, "top")
        self.DockArea.addDock(self.Dock_Banner, "bottom", self.Dock_Top)
        self.DockArea.addDock(self.Dock_Trees, "bottom", self.Dock_Banner)

        # status bar shows the full current day path
        self.status_path = QtWidgets.QLabel("No day directory selected")
        try:
            self.status_path.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
        except Exception:  # noqa: BLE001 - enum path differs across Qt bindings
            pass
        self.win.statusBar().addWidget(self.status_path, 1)

        self._refresh_recent_list()
        self.win.show()

    def _set_status(self, msg: str):
        if self.status_path is not None:
            self.status_path.setText(str(msg))

    def quit(self):
        sys.exit(0)


def main():
    app = pg.mkQApp()
    app.setStyle("fusion")

    config = load_config()

    # Optional: the experiment configuration can provide the raw-data directory
    # as the Browse starting point. A missing/offline experiments.cfg is
    # non-fatal (we fall back to the most-recent directory / home).
    experiments = None
    try:
        from ephys.tools.get_configuration import get_configuration

        _datasets, experiments = get_configuration(
            "config/experiments.cfg", check_completeness=False
        )
    except Exception as exc:  # noqa: BLE001
        CP.cprint(
            "y",
            f"fix_directory_order: experiments.cfg not loaded ({exc}); "
            "falling back to the most-recent directory.",
        )

    FDO = FixDirectoryOrder(config=config, experiments=experiments, app=app)
    app.aboutToQuit.connect(FDO.quit)
    FDO.set_window()

    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
        pg.QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    main()
