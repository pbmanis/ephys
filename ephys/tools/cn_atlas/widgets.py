"""Shared custom widgets and view helpers."""

from __future__ import annotations

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

# All atlas worlds use +X = lateral, +Z = dorsal (up), +Y = rostral/caudal, so a
# camera at azimuth 0 shows the lateral aspect with dorsal up.
LATERAL_VIEW = {"elevation": 12, "azimuth": 0}


def euler_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Rotation matrix for intrinsic X(ML), Y(AP), Z(DV) rotations (degrees)."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


_HANDLE_QSS = """
QSlider::groove:horizontal { height: 6px; background: #4a4d57; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #6f86d6; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 20px; height: 20px; margin: -8px 0; border-radius: 10px;
    background: #d7dae2; border: 1px solid #8a8d96;
}
QSlider::handle:horizontal:hover { background: #ffffff; }
"""


class CollapsibleBox(QtWidgets.QWidget):
    """A titled section that expands/collapses when its header is clicked."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)
        self.toggle.toggled.connect(self._on_toggle)

        self.content = QtWidgets.QWidget()
        self.content.setVisible(expanded)
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(10, 0, 0, 0)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

    def _on_toggle(self, on: bool):
        self.toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if on else QtCore.Qt.ArrowType.RightArrow)
        self.content.setVisible(on)

    def addWidget(self, w):  # noqa: N802 - Qt-style API
        self.content_layout.addWidget(w)

    def addLayout(self, lay):  # noqa: N802 - Qt-style API
        self.content_layout.addLayout(lay)

    def layout_target(self):
        return self.content_layout


class StepSlider(QtWidgets.QSlider):
    """Horizontal slider with a large handle and modifier-scaled arrow keys.

    When focused: Arrow = one singleStep, Shift+Arrow = 3x, Alt/Option+Arrow =
    10x. Left/Down decrease, Right/Up increase.
    """

    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setStyleSheet(_HANDLE_QSS)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, ev):  # noqa: N802 - Qt signature
        key = ev.key()
        dec = key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Down)
        inc = key in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Up)
        if dec or inc:
            mods = ev.modifiers()
            if mods & QtCore.Qt.KeyboardModifier.AltModifier:
                mult = 10
            elif mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
                mult = 3
            else:
                mult = 1
            step = self.singleStep() * mult * (1 if inc else -1)
            self.setValue(self.value() + step)
            ev.accept()
            return
        super().keyPressEvent(ev)
