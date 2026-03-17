#!/usr/bin/env python3
"""
PySide6 color-calibration image viewer.

Widgets
-------
ImageViewerWidget  — full-image view with zoom-rect overlay, drag, wheel
ZoomViewerWidget   — zoomed view with quad annotation, bilinear grid, color sampling
MainWindow         — application shell tying everything together

Bayer debayering
----------------
Algorithm: 2×2 block averaging (nearest-neighbour / subsampling).
Each 2×2 RGGB block becomes one RGB pixel:
  R = top-left,  G = average(top-right, bottom-left),  B = bottom-right
The output image is half the size in each dimension.
No spatial interpolation — the simplest possible approach, no demosaicing artefacts.
Supported patterns: RGGB, BGGR, GRBG, GBRG (selectable by the user).
"""

APP_VERSION = "1.1"

import sys
import os
import json
import math
import queue
import threading
import concurrent.futures
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSizePolicy, QFrame, QSplitter, QGridLayout,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QCheckBox, QMessageBox, QComboBox,
    QProgressDialog,
)
from PySide6.QtGui import (
    QPixmap, QPainter, QColor, QFont, QPen, QBrush, QPolygonF, QIcon,
    QImage, QImageWriter, QTransform, QGuiApplication,
)
from PySide6.QtCore import Qt, QSize, QRectF, QPointF, Signal, QTimer, QSettings, QEvent
from contextlib import contextmanager


@contextmanager
def _wait_cursor():
    """Context manager: show hourglass cursor during a slow operation."""
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


# ──────────────────────────────────────────────────────────────────────────────
# Palette drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_palette_grid(painter: QPainter, rect, palette, corner_labels=None, empty_text=""):
    """
    Draw a palette grid inside `rect`.
    Cells are square with rounded corners; gap = 25% of cell width.
    corner_labels: 4-item list of strings to draw at the palette corners.
    """
    x0, y0, W, H = rect.x(), rect.y(), rect.width(), rect.height()

    if not palette:
        painter.setPen(QColor("#3a3a6a"))
        painter.drawText(rect, Qt.AlignCenter, empty_text or "No palette")
        return

    rows = len(palette)
    cols = max((len(r) for r in palette), default=0)
    if rows == 0 or cols == 0:
        return

    GAP_RATIO = 0.25
    cell_w = W / (cols + GAP_RATIO * max(cols - 1, 0))
    cell_h = H / (rows + GAP_RATIO * max(rows - 1, 0))
    cell   = min(cell_w, cell_h)
    gap    = cell * GAP_RATIO

    total_w = cols * cell + max(cols - 1, 0) * gap
    total_h = rows * cell + max(rows - 1, 0) * gap
    ox      = x0 + (W - total_w) / 2
    oy      = y0 + (H - total_h) / 2
    radius  = cell * 0.18

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)

    for r, row in enumerate(palette):
        for c, rgb in enumerate(row):
            cx = ox + c * (cell + gap)
            cy = oy + r * (cell + gap)
            color = (
                QColor("#1a1a2e") if rgb is None
                else QColor(
                    max(0, min(255, int(rgb[0]))),
                    max(0, min(255, int(rgb[1]))),
                    max(0, min(255, int(rgb[2]))),
                )
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(QRectF(cx, cy, cell, cell), radius, radius)

    # Corner labels (only when cells are large enough to be readable)
    if corner_labels and cell >= 8:
        font      = painter.font()
        font_size = max(7, int(cell * 0.32))
        font.setPointSize(font_size)
        font.setBold(True)
        painter.setFont(font)
        margin   = cell * 0.14
        lbl_size = font_size * 2.4

        corners = [
            (ox + margin,           oy + margin,           Qt.AlignTop    | Qt.AlignLeft,  palette[0][0]   if palette and palette[0]  else None),
            (ox + total_w - margin, oy + margin,           Qt.AlignTop    | Qt.AlignRight, palette[0][-1]  if palette and palette[0]  else None),
            (ox + total_w - margin, oy + total_h - margin, Qt.AlignBottom | Qt.AlignRight, palette[-1][-1] if palette and palette[-1] else None),
            (ox + margin,           oy + total_h - margin, Qt.AlignBottom | Qt.AlignLeft,  palette[-1][0]  if palette and palette[-1] else None),
        ]
        for i, (px, py, align, bg_rgb) in enumerate(corners):
            if i >= len(corner_labels):
                break
            lx = (px - lbl_size) if (align & Qt.AlignRight)  else px
            ly = (py - lbl_size) if (align & Qt.AlignBottom) else py
            # Choose white or black text based on perceived luminance (WCAG formula)
            lum = (0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]) if bg_rgb else 20
            painter.setPen(QColor(0, 0, 0, 230) if lum > 128 else QColor(255, 255, 255, 230))
            painter.drawText(QRectF(lx, ly, lbl_size, lbl_size), Qt.AlignCenter, corner_labels[i])

    painter.restore()


def _palette_cell_at(widget_pos, widget_size, palette) -> tuple[int, int] | None:
    """Return (row, col) of the palette cell under widget_pos, or None."""
    if not palette:
        return None
    rows = len(palette)
    cols = max((len(r) for r in palette), default=0)
    if rows == 0 or cols == 0:
        return None

    W, H      = widget_size
    GAP_RATIO = 0.25
    cell_w    = W / (cols + GAP_RATIO * max(cols - 1, 0))
    cell_h    = H / (rows + GAP_RATIO * max(rows - 1, 0))
    cell      = min(cell_w, cell_h)
    gap       = cell * GAP_RATIO
    total_w   = cols * cell + max(cols - 1, 0) * gap
    total_h   = rows * cell + max(rows - 1, 0) * gap
    ox        = (W - total_w) / 2
    oy        = (H - total_h) / 2

    mx, my = widget_pos.x() - ox, widget_pos.y() - oy
    if mx < 0 or my < 0 or mx > total_w or my > total_h:
        return None

    c  = int(mx / (cell + gap))
    r  = int(my / (cell + gap))
    cx = c * (cell + gap)
    cy = r * (cell + gap)
    # Reject hits inside the gap between cells
    if mx - cx > cell or my - cy > cell:
        return None
    if r < rows and c < cols:
        return (r, c)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# ColorTooltip — frameless popup showing a color swatch + hex + RGB values
# ──────────────────────────────────────────────────────────────────────────────

class ColorTooltip(QWidget):
    """Singleton frameless popup: color swatch on the left, hex and RGB values on the right."""

    _instance: "ColorTooltip | None" = None

    @classmethod
    def show_for(cls, rgb: tuple[int, int, int], global_pos):
        if cls._instance is None:
            cls._instance = cls()
        cls._instance._set_color(rgb)
        cls._instance._reposition(global_pos)
        cls._instance.show()
        cls._instance.raise_()

    @classmethod
    def hide_all(cls):
        if cls._instance is not None:
            cls._instance.hide()

    def __init__(self):
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._rgb = (128, 128, 128)
        self.setFixedSize(150, 52)

    def _set_color(self, rgb: tuple[int, int, int]):
        self._rgb = rgb
        self.update()

    def _reposition(self, global_pos):
        """Place the tooltip slightly below-right of the cursor, clamped to the screen."""
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        sg = screen.availableGeometry()
        x  = global_pos.x() + 16
        y  = global_pos.y() + 16
        if x + self.width()  > sg.right():  x = global_pos.x() - self.width()  - 4
        if y + self.height() > sg.bottom(): y = global_pos.y() - self.height() - 4
        self.move(x, y)

    def paintEvent(self, _event):
        r, g, b = self._rgb
        color   = QColor(r, g, b)
        p       = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H   = self.width(), self.height()
        RADIUS = 9
        SWATCH = 28

        # Drop shadow
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 90))
        p.drawRoundedRect(QRectF(3, 3, W - 2, H - 2), RADIUS, RADIUS)

        # Dark background
        p.setBrush(QColor(22, 22, 40))
        p.drawRoundedRect(QRectF(0, 0, W - 3, H - 3), RADIUS, RADIUS)

        # Color swatch (rounded on the left side only)
        p.setBrush(QBrush(color))
        p.drawRoundedRect(QRectF(0, 0, SWATCH + RADIUS, H - 3), RADIUS, RADIUS)
        p.fillRect(QRectF(SWATCH, 0, RADIUS, H - 3), color)  # square patch to fill right corners

        # Vertical divider
        p.setPen(QPen(QColor(80, 80, 130, 150), 1))
        p.drawLine(QPointF(SWATCH + 6, 6), QPointF(SWATCH + 6, H - 9))

        # Thin border
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(80, 80, 130, 180), 1))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 3.5, H - 3.5), RADIUS, RADIUS)

        # Text (always on dark background — no contrast check needed)
        tx   = SWATCH + 14
        tw   = W - 3 - tx - 6
        half = (H - 3) / 2
        font = p.font()
        font.setFamily("Consolas")

        font.setPointSize(10); font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(220, 220, 255))
        p.drawText(QRectF(tx, 3, tw, half - 1), Qt.AlignLeft | Qt.AlignVCenter, f"#{r:02X}{g:02X}{b:02X}")

        font.setBold(False); font.setPointSize(9)
        p.setFont(font)
        p.setPen(QColor(160, 160, 200))
        p.drawText(QRectF(tx, half, tw, half - 2), Qt.AlignLeft | Qt.AlignVCenter, f"({r}, {g}, {b})")
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# Palette preview widgets
# ──────────────────────────────────────────────────────────────────────────────

class _BasePaletteWidget(QWidget):
    """Base class for palette display widgets with color tooltip on hover."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._palette: list | None = None
        self.setMinimumSize(60, 40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        cell = _palette_cell_at(event.position(), (self.width(), self.height()), self._palette)
        if cell and self._palette:
            r, c = cell
            row  = self._palette[r]
            rgb  = row[c] if c < len(row) else None
            if rgb:
                ColorTooltip.show_for(
                    (int(rgb[0]), int(rgb[1]), int(rgb[2])),
                    self.mapToGlobal(event.position().toPoint()),
                )
                return
        ColorTooltip.hide_all()

    def leaveEvent(self, event):
        ColorTooltip.hide_all()


class PalettePreviewWidget(_BasePaletteWidget):
    """Reference palette loaded from a JSON file — corner labels 1–4."""

    def setPalette(self, palette, name: str = ""):
        self._palette = palette
        self._name    = name
        self.update()

    def clearPalette(self):
        self._palette = None
        self._name    = ""
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d0d1a"))
        _draw_palette_grid(p, QRectF(self.rect()), self._palette,
                           corner_labels=("1", "2", "3", "4"),
                           empty_text="No palette")


class MeasuredPaletteWidget(_BasePaletteWidget):
    """Measured colors sampled from the image — no corner labels."""

    def setMeasured(self, cells: list, rows: int, cols: int):
        """Build the [row][col] = [R, G, B] grid from sampled cell data."""
        if not cells or rows == 0 or cols == 0:
            self._palette = []
            self.update()
            return
        grid = [[None] * cols for _ in range(rows)]
        for cell in cells:
            r, c = cell["row"], cell["col"]
            if cell.get("color_mean"):
                m = cell["color_mean"]
                grid[r][c] = [int(m[0]), int(m[1]), int(m[2])]
        self._palette = grid
        self.update()

    def clearMeasured(self):
        pass  # Keep the last known colors visible

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d0d1a"))
        _draw_palette_grid(p, QRectF(self.rect()), self._palette,
                           corner_labels=None,
                           empty_text="Place 4 points…")


# ──────────────────────────────────────────────────────────────────────────────
# ColorMatrixWidget — 3×3 color matrix + optional offset (least squares)
# ──────────────────────────────────────────────────────────────────────────────

class ColorMatrixWidget(QWidget):
    """
    Displays the least-squares color transformation matrix.
      9-param mode  : rgb_out = M @ rgb_in
      12-param mode : rgb_out = M @ rgb_in + offset
    """

    computeRequested = Signal()
    toggleCorrected  = Signal(bool)
    exportRequested  = Signal()
    importRequested  = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._matrix: np.ndarray | None = None  # shape (3, 3)
        self._offset: np.ndarray | None = None  # shape (3,) or None
        self._residuals: tuple | None   = None
        self._setup_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # Header row: title + offset checkbox + Compute button
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        lbl = QLabel("Matrix M"); lbl.setObjectName("panelTitle")
        hdr.addWidget(lbl, stretch=1)
        self.chk_offset = QCheckBox("+offset")
        self.chk_offset.setToolTip("12-parameter affine transform: M @ rgb + offset\n(corrects a global brightness/colour shift)")
        self.chk_offset.setObjectName("offsetChk")
        hdr.addWidget(self.chk_offset)
        self.btn_calc = QPushButton("⚙  Compute")
        self.btn_calc.setObjectName("coordsBtn")
        self.btn_calc.setCursor(Qt.PointingHandCursor)
        self.btn_calc.setEnabled(False)
        self.btn_calc.setToolTip("Compute the colour correction matrix\nfrom measured and reference colours")
        self.btn_calc.clicked.connect(self.computeRequested)
        hdr.addWidget(self.btn_calc)
        lay.addLayout(hdr)

        # Matrix grid: 3 rows × (channel | ×R | ×G | ×B | +c)
        mat_grid        = QGridLayout()
        mat_grid.setSpacing(3)
        mat_grid.setContentsMargins(0, 2, 0, 2)
        self._cells:     list[list[QLabel]] = []
        self._off_cells: list[QLabel]       = []

        for ci, header in enumerate(["×R", "×G", "×B", "+c"]):
            lbl = QLabel(header); lbl.setObjectName("matChan"); lbl.setAlignment(Qt.AlignCenter)
            mat_grid.addWidget(lbl, 0, ci + 1)

        for r, chan in enumerate(["R", "G", "B"]):
            chan_lbl = QLabel(chan); chan_lbl.setObjectName("matChan")
            chan_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            mat_grid.addWidget(chan_lbl, r + 1, 0)

            row_cells = []
            for c in range(3):
                cell = QLabel("—"); cell.setObjectName("matCell")
                cell.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                cell.setFixedWidth(62)
                mat_grid.addWidget(cell, r + 1, c + 1)
                row_cells.append(cell)
            self._cells.append(row_cells)

            off = QLabel("—"); off.setObjectName("matCell")
            off.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            off.setFixedWidth(56); off.setEnabled(False)
            mat_grid.addWidget(off, r + 1, 4)
            self._off_cells.append(off)

        lay.addLayout(mat_grid)

        self.chk_offset.toggled.connect(self._on_offset_toggled)
        self._on_offset_toggled(False)

        lay.addWidget(_make_separator())

        # Residuals row
        res_grid = QGridLayout()
        res_grid.setSpacing(4); res_grid.setContentsMargins(0, 0, 0, 0)
        self._res_labels: list[QLabel] = []
        for i, name in enumerate(["ΔR", "ΔG", "ΔB", "Δtot"]):
            nl = QLabel(name);  nl.setObjectName("matChan"); nl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            vl = QLabel("—");   vl.setObjectName("matResid"); vl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            res_grid.addWidget(nl, 0, i * 2)
            res_grid.addWidget(vl, 0, i * 2 + 1)
            self._res_labels.append(vl)
        lay.addLayout(res_grid)

        # Error message (hidden by default)
        self.lbl_err = QLabel("")
        self.lbl_err.setObjectName("matError")
        self.lbl_err.setWordWrap(True)
        self.lbl_err.hide()
        lay.addWidget(self.lbl_err)

        lay.addWidget(_make_separator())

        # Action buttons row 1: Preview toggle + Export JSON + Import JSON
        btn_bar = QHBoxLayout(); btn_bar.setSpacing(6)
        self.btn_toggle = QPushButton("👁  Preview")
        self.btn_toggle.setObjectName("coordsBtn")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.setEnabled(False)
        self.btn_toggle.setToolTip("Toggle corrected-colour preview on/off")
        self.btn_toggle.toggled.connect(self._on_toggle)
        btn_bar.addWidget(self.btn_toggle, stretch=1)
        self.btn_export = QPushButton("💾  Export")
        self.btn_export.setObjectName("coordsBtn")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip("Export the correction matrix to a JSON file")
        self.btn_export.clicked.connect(self.exportRequested)
        btn_bar.addWidget(self.btn_export)
        self.btn_import = QPushButton("📂  Import")
        self.btn_import.setObjectName("coordsBtn")
        self.btn_import.setCursor(Qt.PointingHandCursor)
        self.btn_import.setToolTip("Import a correction matrix from a JSON file")
        self.btn_import.clicked.connect(self.importRequested)
        btn_bar.addWidget(self.btn_import)
        lay.addLayout(btn_bar)

        # Source label: shows whether the current matrix is computed or imported
        self.lbl_source = QLabel("—")
        self.lbl_source.setObjectName("matResid")
        self.lbl_source.setAlignment(Qt.AlignCenter)
        self.lbl_source.setWordWrap(True)
        lay.addWidget(self.lbl_source)

        lay.addStretch(1)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_offset_toggled(self, checked: bool):
        """Grey out / restore the offset column; re-enable Compute if a matrix already exists."""
        for lbl in self._off_cells:
            lbl.setEnabled(checked)
            lbl.setStyleSheet("" if checked else "color: #3a3a5a;")
        if self._matrix is not None:
            self.btn_calc.setEnabled(True)

    def _on_toggle(self, checked: bool):
        self.btn_toggle.setText("✅  Corrected" if checked else "👁  Preview")
        self.toggleCorrected.emit(checked)

    # ── Public API ────────────────────────────────────────────────────────────

    def use_offset(self) -> bool:
        return self.chk_offset.isChecked()

    def matrix(self) -> np.ndarray | None:
        return self._matrix

    def offset(self) -> np.ndarray | None:
        return self._offset

    def setResult(self, matrix: np.ndarray, residuals: tuple,
                  offset: np.ndarray | None = None, source_name: str = ""):
        self._matrix    = matrix
        self._offset    = offset
        self._residuals = residuals
        self.lbl_err.hide()
        self.btn_toggle.blockSignals(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("👁  Preview")
        self.btn_toggle.blockSignals(False)
        self.btn_toggle.setEnabled(True)
        self.btn_export.setEnabled(True)

        for r in range(3):
            for c in range(3):
                v       = matrix[r, c]
                is_diag = (r == c)
                if   is_diag and abs(v) >= 0.5:     style = "diag"
                elif not is_diag and abs(v) > 0.15: style = "off"
                else:                               style = "normal"
                cell = self._cells[r][c]
                cell.setText(f"{v:+.4f}")
                cell.setProperty("matStyle", style)
                cell.style().unpolish(cell); cell.style().polish(cell)

        for i, lbl in enumerate(self._off_cells):
            lbl.setText(f"{offset[i]:+.1f}" if offset is not None else "—")

        for lbl, val in zip(self._res_labels, residuals):
            if val != val:  # nan
                lbl.setText("—"); lbl.setStyleSheet("")
            else:
                lbl.setText(f"{val:.2f}")
                color = "#ff5555" if val > 5 else ("#ffaa44" if val > 2 else "#55dd88")
                lbl.setStyleSheet(f"color: {color}; font-family: 'Consolas', monospace; font-size: 12px;")

        # Source label
        if source_name:
            self.lbl_source.setText(f"📂  Imported: {source_name}")
            self.lbl_source.setStyleSheet("color: #ffaa44; font-size: 11px;")
        else:
            self.lbl_source.setText("⚙  Computed")
            self.lbl_source.setStyleSheet("color: #55dd88; font-size: 11px;")

    def setError(self, msg: str):
        self._matrix = None
        self._offset = None
        self.lbl_err.setText(msg)
        self.lbl_err.setVisible(bool(msg))
        self.btn_toggle.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.lbl_source.setText("—")
        self.lbl_source.setStyleSheet("")
        for row in self._cells:
            for cell in row: cell.setText("—")
        for lbl in self._off_cells:  lbl.setText("—")
        for lbl in self._res_labels: lbl.setText("—"); lbl.setStyleSheet("")

    def clear(self):
        self.setError("")
        self.btn_toggle.blockSignals(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("👁  Preview")
        self.btn_toggle.blockSignals(False)


# ──────────────────────────────────────────────────────────────────────────────
# GridSettingsWidget — grid configuration controls
# ──────────────────────────────────────────────────────────────────────────────

class GridSettingsWidget(QWidget):
    """Controls for the sampling grid: columns, rows, cell spacing, std threshold."""

    settingsChanged = Signal(int, int, float)     # cols, rows, gap (fraction [0..1])
    paletteLoaded   = Signal(list, str, int, int) # palette, name, cols, rows

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded_palette_path: str = ""
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        title = QLabel("Grid settings"); title.setObjectName("panelTitle")
        lay.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(5); grid.setContentsMargins(0, 4, 0, 2); grid.setColumnStretch(2, 1)

        def add_row(row, label, widget):
            lbl = QLabel(label); lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(lbl, row, 0); grid.addWidget(widget, row, 1)

        def dbl(lo, hi, val, step, dec):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val)
            s.setSingleStep(step); s.setDecimals(dec)
            s.setObjectName("gridSpin"); s.setFixedWidth(72); return s

        def intspin(lo, hi, val):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val)
            s.setObjectName("gridSpin"); s.setFixedWidth(72); return s

        self.spin_gap  = dbl(0.0, 70.0,  45.0, 0.5, 1)
        self.spin_std  = dbl(0.0, 255.0, 10.0, 1.0, 1)
        self.spin_cols = intspin(1, 32, 6)
        self.spin_rows = intspin(1, 32, 4)
        self.spin_gap .setToolTip("Inter-cell gap (% of cell size)\nIncrease to exclude patch edges from sampling")
        self.spin_std .setToolTip("Std-dev threshold (0–255): cells whose colour variation\nexceeds this value are highlighted in red in the table")
        self.spin_cols.setToolTip("Number of columns in the reference colour chart")
        self.spin_rows.setToolTip("Number of rows in the reference colour chart")
        add_row(0, "Gap (%) :", self.spin_gap)
        add_row(1, "Std thr :", self.spin_std)
        add_row(2, "Columns :", self.spin_cols)
        add_row(3, "Rows :",    self.spin_rows)
        lay.addLayout(grid)

        for spin in (self.spin_cols, self.spin_rows, self.spin_gap, self.spin_std):
            spin.valueChanged.connect(self._emit)

    def _open_palette(self, path: str = ""):
        """Open a JSON palette file. If `path` is given (from prefs), skip the dialog."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open palette JSON", "",
                "JSON files (*.json);;All files (*)"
            )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            name    = str(data.get("name", os.path.basename(path)))
            palette = data["palette"]
            rows    = len(palette)
            cols    = max(len(r) for r in palette) if rows else 0
            if rows == 0 or cols == 0:
                raise ValueError("Empty palette")
            for spin, val in [(self.spin_rows, rows), (self.spin_cols, cols)]:
                spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)
            self._loaded_palette_path = path
            self.paletteLoaded.emit(palette, name, cols, rows)
            self._emit()
        except Exception as e:
            QMessageBox.warning(self, "Palette error", f"Cannot read palette:\n{e}")

    def _emit(self):
        self.settingsChanged.emit(self.spin_cols.value(), self.spin_rows.value(),
                                  self.spin_gap.value() / 100.0)

    def setGapValue(self, gap_abs: float):
        """Update the gap spinbox without double-emitting."""
        self.spin_gap.blockSignals(True)
        self.spin_gap.setValue(round(gap_abs * 100.0, 1))
        self.spin_gap.blockSignals(False)

    def stdThreshold(self) -> float:
        return self.spin_std.value()

    def values(self) -> tuple[int, int, float]:
        return self.spin_cols.value(), self.spin_rows.value(), self.spin_gap.value() / 100.0


# ──────────────────────────────────────────────────────────────────────────────
# CellTableWidget — inline stats table (one row per grid cell)
# ──────────────────────────────────────────────────────────────────────────────

class CellTableWidget(QWidget):
    """Read-only table: row, col, pixel count, color swatch, mean R/G/B, std R/G/B."""

    _HEADERS  = ["Row", "Col", "# px", "Color", "σR", "σG", "σB", "R mean", "G mean", "B mean"]
    _COL_SWAT = 3  # column index of the color swatch

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        self._table = QTableWidget(0, len(self._HEADERS))
        self._table.setHorizontalHeaderLabels(self._HEADERS)
        self._table.setObjectName("coordTable")
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setFocusPolicy(Qt.NoFocus)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(False)
        lay.addWidget(self._table)
        self._std_threshold = 10.0

    def setStdThreshold(self, v: float):
        self._std_threshold = v

    def populate(self, cells: list[dict]):
        tbl = self._table
        tbl.setRowCount(len(cells))
        RED_FG  = QColor("#ff4d4d")
        BG_EVEN = QColor("#1a1a2e")
        BG_ODD  = QColor("#12121e")

        for i, cell in enumerate(cells):
            row, col = cell["row"], cell["col"]
            bg = BG_EVEN if row % 2 == 0 else BG_ODD

            # Approximate cell area via the shoelace formula
            pts  = [cell["tl"], cell["tr"], cell["br"], cell["bl"]]
            area = abs(sum(
                pts[k].x() * pts[(k + 1) % 4].y() - pts[(k + 1) % 4].x() * pts[k].y()
                for k in range(4)
            )) / 2.0

            def make_item(txt, bg_col=None, fg_col=None):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignCenter)
                it.setBackground(bg_col or bg)
                if fg_col: it.setForeground(fg_col)
                return it

            tbl.setItem(i, 0, make_item(str(row + 1)))
            tbl.setItem(i, 1, make_item(str(col + 1)))
            tbl.setItem(i, 2, make_item(f"{area:.0f}"))

            if cell.get("color_mean"):
                r, g, b    = cell["color_mean"]
                sr, sg, sb = cell["color_std"]
                tbl.setItem(i, self._COL_SWAT, make_item("", QColor(int(r), int(g), int(b))))
                thr = self._std_threshold
                for sv, ci in [(sr, 4), (sg, 5), (sb, 6)]:
                    tbl.setItem(i, ci, make_item(f"{sv:.1f}", bg, RED_FG if sv >= thr else None))
                tbl.setItem(i, 7, make_item(f"{r:.1f}", QColor(min(255, bg.red() + 30), bg.green(), bg.blue())))
                tbl.setItem(i, 8, make_item(f"{g:.1f}", QColor(bg.red(), min(255, bg.green() + 30), bg.blue())))
                tbl.setItem(i, 9, make_item(f"{b:.1f}", QColor(bg.red(), bg.green(), min(255, bg.blue() + 30))))
            else:
                for j in range(self._COL_SWAT, len(self._HEADERS)):
                    tbl.setItem(i, j, make_item("—"))

            tbl.setRowHeight(i, 26)

    def clear(self):
        self._table.setRowCount(0)


# ──────────────────────────────────────────────────────────────────────────────
# ImageViewerWidget — full-image view with zoom-rect overlay
# ──────────────────────────────────────────────────────────────────────────────

class ImageViewerWidget(QWidget):
    centerMoved       = Signal(QPointF)
    zoomRequested     = Signal(int, QPointF)  # delta, anchor_norm
    gapShiftRequested = Signal(float)          # gap delta

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap:         QPixmap | None = None
        self._zoom_rect_norm:  QRectF | None = None
        self._dragging = False
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap; self.update()

    def setZoomRect(self, norm_rect: QRectF | None):
        self._zoom_rect_norm = norm_rect; self.update()

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _image_rect(self) -> QRectF:
        w, h   = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        scale  = min(w / iw, h / ih)
        sw, sh = iw * scale, ih * scale
        return QRectF((w - sw) / 2, (h - sh) / 2, sw, sh)

    def _widget_to_norm(self, pos: QPointF) -> QPointF | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        ir = self._image_rect()
        px = max(ir.left(), min(ir.right(),  pos.x()))
        py = max(ir.top(),  min(ir.bottom(), pos.y()))
        return QPointF(
            max(0.0, min(1.0, (px - ir.x()) / ir.width())),
            max(0.0, min(1.0, (py - ir.y()) / ir.height())),
        )

    # ── Events ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap and not self._pixmap.isNull():
            norm = self._widget_to_norm(event.position())
            if norm:
                self._dragging = True
                self.setCursor(Qt.ClosedHandCursor)
                self.centerMoved.emit(norm)

    def mouseMoveEvent(self, event):
        if self._dragging and (event.buttons() & Qt.LeftButton):
            norm = self._widget_to_norm(event.position())
            if norm: self.centerMoved.emit(norm)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False; self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull(): return
        ir = self._image_rect()
        if not ir.contains(event.position()): return
        if event.modifiers() & Qt.ShiftModifier:
            self.gapShiftRequested.emit(0.005 if event.angleDelta().y() > 0 else -0.005)
        else:
            anchor = self._widget_to_norm(event.position())
            if anchor: self.zoomRequested.emit(event.angleDelta().y(), anchor)

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if self._pixmap is None or self._pixmap.isNull():
            painter.fillRect(0, 0, w, h, QColor("#1a1a2e"))
            painter.setPen(QPen(QColor("#4a4a6a"), 2, Qt.DashLine))
            painter.drawRect(20, 20, w - 40, h - 40)
            painter.setPen(QColor("#6a6a9a"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image\nClick « Open »")
            return

        ir     = self._image_rect()
        scaled = self._pixmap.scaled(QSize(int(ir.width()), int(ir.height())),
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.fillRect(0, 0, w, h, QColor("#0d0d1a"))
        painter.drawPixmap(int(ir.x()), int(ir.y()), scaled)
        if self._zoom_rect_norm is not None:
            self._draw_zoom_overlay(painter, ir)

    def _draw_zoom_overlay(self, painter: QPainter, img_rect: QRectF):
        nr   = self._zoom_rect_norm
        rect = QRectF(
            img_rect.x() + nr.x() * img_rect.width(),
            img_rect.y() + nr.y() * img_rect.height(),
            nr.width()  * img_rect.width(),
            nr.height() * img_rect.height(),
        )
        outer = img_rect

        # Dim everything outside the zoom rect
        painter.setBrush(QBrush(QColor(0, 0, 0, 120))); painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(outer.left(),  outer.top(),   outer.width(), rect.top()    - outer.top()))
        painter.drawRect(QRectF(outer.left(),  rect.bottom(), outer.width(), outer.bottom()- rect.bottom()))
        painter.drawRect(QRectF(outer.left(),  rect.top(),    rect.left()   - outer.left(), rect.height()))
        painter.drawRect(QRectF(rect.right(),  rect.top(),    outer.right() - rect.right(), rect.height()))

        painter.setPen(QPen(QColor("#ff6b35") if not self._dragging else QColor("#ffdd57"), 2))
        painter.setBrush(Qt.NoBrush); painter.drawRect(rect)

        # Corner handles
        cs = min(10.0, rect.width() / 4, rect.height() / 4)
        painter.setPen(QPen(QColor("#ffb38a"), 3))
        for cx, cy, dx, dy in [
            (rect.left(),  rect.top(),     cs,  cs), (rect.right(), rect.top(),    -cs,  cs),
            (rect.left(),  rect.bottom(),  cs, -cs), (rect.right(), rect.bottom(), -cs, -cs),
        ]:
            painter.drawLine(QPointF(cx, cy), QPointF(cx + dx, cy))
            painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy))

    def sizeHint(self): return QSize(500, 400)


# ──────────────────────────────────────────────────────────────────────────────
# ZoomViewerWidget — zoomed view with quad annotation and bilinear grid
# ──────────────────────────────────────────────────────────────────────────────

class ZoomViewerWidget(QWidget):
    """
    Zoomed image view.
    - Wheel              : zoom in/out centered on the cursor
    - Left drag          : pan
    - Left click (no drag): add a quadrilateral point (max 4)
    - Right click        : clear all points
    - Shift+wheel        : adjust grid gap
    """

    zoomRectChanged    = Signal(QRectF)
    pointsChanged      = Signal(list)   # list of normalized QPointF [0..1]
    pointsEditingDone  = Signal(list)   # emitted only at end of point drag
    gapChangeRequested = Signal(float)  # new absolute gap [0..0.70]

    ZOOM_MIN       = 1.0
    ZOOM_MAX       = 32.0
    ZOOM_STEP      = 1.15
    DRAG_THRESHOLD = 4   # pixels before a move is considered a pan
    GRAB_RADIUS    = 14  # pixel radius for grabbing a quad point

    POINT_COLORS = [
        QColor("#ff4d6d"), QColor("#4dffb0"),
        QColor("#4db8ff"), QColor("#ffdd57"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._zoom   = 2.0
        self._center = QPointF(0.5, 0.5)

        # Pan state
        self._drag_start_widget: QPointF | None = None
        self._drag_start_center: QPointF | None = None
        self._is_panning = False

        # Point editing (-1 = none)
        self._editing_idx: int = -1

        # Rotation of the displayed pixmap (0 / 90 / 180 / 270)
        self._rotation: int = 0

        # Grid parameters
        self._grid_cols: int   = 6
        self._grid_rows: int   = 4
        self._grid_gap:  float = 0.05

        # Cells whose std dev exceeds the threshold (shown with a red overlay)
        self._std_threshold:     float                = 10.0
        self._highlighted_cells: set[tuple[int, int]] = set()

        # Quad points stored in *original* (un-rotated) image coordinate space [0..1]
        self._quad_points: list[QPointF] = []

        # When True: no point editing (corrected-preview mode is active)
        self._interaction_locked: bool = False

        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    # ── Public API ────────────────────────────────────────────────────────────

    def setPixmap(self, pixmap: QPixmap):
        """Load a new image and reset zoom, center, and quad points."""
        self._pixmap      = pixmap
        self._center      = QPointF(0.5, 0.5)
        self._zoom        = 2.0
        self._quad_points = []
        self.pointsChanged.emit([])
        self._emit_zoom_rect()
        self.update()

    def updatePixmap(self, pixmap: QPixmap, rotation: int = 0):
        """Replace the displayed pixmap WITHOUT resetting zoom, center, or points."""
        old_rotation   = self._rotation
        self._pixmap   = pixmap
        self._rotation = rotation % 360
        # Convert center: old-rotated → original → new-rotated
        center_orig  = self._rotate_norm(self._center, old_rotation, inverse=True)
        self._center = self._rotate_norm(center_orig,  self._rotation, inverse=False)
        self._clamp_center()
        self._emit_zoom_rect()
        self.update()

    def setCenter(self, norm: QPointF):
        self._center = norm; self._clamp_center(); self._emit_zoom_rect(); self.update()

    def clearQuad(self):
        self._quad_points = []; self._highlighted_cells = set()
        self.pointsChanged.emit([]); self.update()

    def setInteractionLocked(self, locked: bool):
        """Lock or unlock point drag / add (used during corrected-preview mode)."""
        self._interaction_locked = locked
        self.setCursor(Qt.ArrowCursor if locked else Qt.CrossCursor)

    def setGrid(self, cols: int, rows: int, gap: float):
        self._grid_cols = max(1, cols); self._grid_rows = max(1, rows)
        self._grid_gap  = max(0.0, min(0.70, gap)); self.update()

    def setHighlightedCells(self, cells: set[tuple[int, int]]):
        self._highlighted_cells = cells; self.update()

    def applyZoom(self, delta: int, anchor_norm: QPointF | None = None):
        """Zoom in/out. If anchor_norm is given, keep that image point fixed on screen."""
        if self._pixmap is None or self._pixmap.isNull(): return
        factor   = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * factor))
        if new_zoom == self._zoom: return
        if anchor_norm is not None:
            ratio        = self._zoom / new_zoom
            self._center = QPointF(
                anchor_norm.x() + (self._center.x() - anchor_norm.x()) * ratio,
                anchor_norm.y() + (self._center.y() - anchor_norm.y()) * ratio,
            )
        self._zoom = new_zoom; self._clamp_center(); self._emit_zoom_rect(); self.update()

    def zoomAtAnchor(self, delta: int, anchor_norm: QPointF):
        """Zoom and move the view center to anchor_norm (used by the full-view wheel)."""
        if self._pixmap is None or self._pixmap.isNull(): return
        self._center = QPointF(anchor_norm.x(), anchor_norm.y())
        self.applyZoom(delta)

    # ── Coordinate transforms ─────────────────────────────────────────────────

    @staticmethod
    def _rotate_norm(p: QPointF, angle: int, inverse: bool) -> QPointF:
        """
        Transform a normalized coordinate by `angle` (0/90/180/270).
        inverse=False : original → rotated
        inverse=True  : rotated  → original
        """
        x, y = p.x(), p.y()
        r    = angle % 360
        if inverse:
            if r == 90:  return QPointF(y, 1.0 - x)
            if r == 180: return QPointF(1.0 - x, 1.0 - y)
            if r == 270: return QPointF(1.0 - y, x)
        else:
            if r == 90:  return QPointF(1.0 - y, x)
            if r == 180: return QPointF(1.0 - x, 1.0 - y)
            if r == 270: return QPointF(y, 1.0 - x)
        return QPointF(x, y)

    def _visible_rect_norm(self) -> QRectF:
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF(0, 0, 1, 1)
        iw, ih = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        wa = ww / wh if wh else 1.0
        ia = iw / ih if ih else 1.0
        if wa > ia:
            frac_h = 1.0 / self._zoom; frac_w = frac_h * wa / ia
        else:
            frac_w = 1.0 / self._zoom; frac_h = frac_w * ia / wa
        return QRectF(self._center.x() - frac_w / 2, self._center.y() - frac_h / 2, frac_w, frac_h)

    def _clamp_center(self):
        rect = self._visible_rect_norm()
        hw, hh = rect.width() / 2, rect.height() / 2
        self._center = QPointF(
            max(hw, min(1.0 - hw, self._center.x())),
            max(hh, min(1.0 - hh, self._center.y())),
        )

    def _widget_to_norm(self, pos: QPointF) -> QPointF:
        rect = self._visible_rect_norm()
        return QPointF(
            rect.x() + (pos.x() / self.width())  * rect.width(),
            rect.y() + (pos.y() / self.height()) * rect.height(),
        )

    def _norm_to_widget(self, norm: QPointF) -> QPointF:
        rect = self._visible_rect_norm()
        return QPointF(
            (norm.x() - rect.x()) / rect.width()  * self.width(),
            (norm.y() - rect.y()) / rect.height() * self.height(),
        )

    def _emit_zoom_rect(self):
        rect = self._visible_rect_norm()
        x  = max(0.0, rect.x());  y  = max(0.0, rect.y())
        x2 = min(1.0, rect.right()); y2 = min(1.0, rect.bottom())
        self.zoomRectChanged.emit(QRectF(x, y, x2 - x, y2 - y))

    # ── Mouse events ──────────────────────────────────────────────────────────

    def _nearest_quad_point(self, widget_pos: QPointF) -> int:
        """Return the index of the quad point within GRAB_RADIUS of widget_pos, or -1."""
        best_idx, best_dist = -1, float("inf")
        for i, p_orig in enumerate(self._quad_points):
            wp   = self._norm_to_widget(self._rotate_norm(p_orig, self._rotation, inverse=False))
            dist = math.hypot(widget_pos.x() - wp.x(), widget_pos.y() - wp.y())
            if dist < self.GRAB_RADIUS and dist < best_dist:
                best_dist, best_idx = dist, i
        return best_idx

    def mousePressEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull(): return
        if event.button() == Qt.LeftButton:
            if not self._interaction_locked:
                idx = self._nearest_quad_point(event.position())
                if idx >= 0:
                    self._editing_idx = idx; self.setCursor(Qt.DragMoveCursor); return
            # Pan is always allowed
            self._drag_start_widget = event.position()
            self._drag_start_center = QPointF(self._center)
            self._is_panning        = False
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton and not self._interaction_locked:
            self.clearQuad()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            self.setCursor(Qt.DragMoveCursor if self._nearest_quad_point(event.position()) >= 0 else Qt.CrossCursor)
            return
        if self._editing_idx >= 0:
            norm_rot = self._widget_to_norm(event.position())
            norm_rot = QPointF(max(0.0, min(1.0, norm_rot.x())), max(0.0, min(1.0, norm_rot.y())))
            self._quad_points[self._editing_idx] = self._rotate_norm(norm_rot, self._rotation, inverse=True)
            self.pointsChanged.emit(list(self._quad_points)); self.update(); return
        if self._drag_start_widget is None: return
        delta = event.position() - self._drag_start_widget
        if not self._is_panning and (abs(delta.x()) > self.DRAG_THRESHOLD or abs(delta.y()) > self.DRAG_THRESHOLD):
            self._is_panning = True
        if self._is_panning:
            rect = self._visible_rect_norm()
            self._center = QPointF(
                self._drag_start_center.x() - delta.x() / self.width()  * rect.width(),
                self._drag_start_center.y() - delta.y() / self.height() * rect.height(),
            )
            self._clamp_center(); self._emit_zoom_rect(); self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton: return
        if self._editing_idx >= 0:
            self._editing_idx = -1
            self.setCursor(Qt.CrossCursor if not self._interaction_locked else Qt.ArrowCursor)
            self.update(); self.pointsEditingDone.emit(list(self._quad_points))
        elif not self._is_panning and not self._interaction_locked:
            self._add_quad_point(event.position()); self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.CrossCursor if not self._interaction_locked else Qt.ArrowCursor)
        self._drag_start_widget = None; self._drag_start_center = None; self._is_panning = False

    def wheelEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull(): return
        if not self._interaction_locked and (event.modifiers() & Qt.ShiftModifier):
            delta   = 0.005 if event.angleDelta().y() > 0 else -0.005
            self.gapChangeRequested.emit(max(0.0, min(0.70, self._grid_gap + delta)))
        else:
            self.applyZoom(event.angleDelta().y(), self._widget_to_norm(event.position()))

    def resizeEvent(self, event):
        super().resizeEvent(event); self._clamp_center(); self._emit_zoom_rect()

    # ── Quad point management ─────────────────────────────────────────────────

    def _add_quad_point(self, widget_pos: QPointF):
        if len(self._quad_points) >= 4: return
        norm_rot = self._widget_to_norm(widget_pos)
        if not (0.0 <= norm_rot.x() <= 1.0 and 0.0 <= norm_rot.y() <= 1.0): return
        self._quad_points.append(self._rotate_norm(norm_rot, self._rotation, inverse=True))
        self.pointsChanged.emit(list(self._quad_points)); self.update()

    # ── Grid geometry ─────────────────────────────────────────────────────────

    @staticmethod
    def _cell_bounds(n: int, gap: float) -> list[tuple[float, float]]:
        """
        Compute [u0, u1] bounds for each of n cells in [0, 1].
        gap is a fraction of one cell width; there are (n-1) gaps total.
        """
        if n == 1: return [(0.0, 1.0)]
        total_gap = gap / n
        cell_size = (1.0 - (n - 1) * total_gap) / n
        bounds, pos = [], 0.0
        for _ in range(n):
            bounds.append((pos, pos + cell_size)); pos += cell_size + total_gap
        return bounds

    def getCellCorners(self, img_w: int, img_h: int,
                       pixmap_orig: "QPixmap | None" = None,
                       quad_override: "list | None" = None) -> list[dict]:
        """
        Return corner coordinates (original-image pixels) for every grid cell.
        Each dict: { 'row', 'col', 'tl', 'tr', 'br', 'bl', 'color_mean', 'color_std' }
        quad_override: if given, use these 4 QPointF (original-image normalised space)
                       instead of self._quad_points.
        """
        pts = quad_override if quad_override is not None else self._quad_points
        if len(pts) != 4: return []
        p0, p1, p2, p3 = pts

        def bilerp_orig(u: float, v: float) -> QPointF:
            top = p0 + (p1 - p0) * u; bot = p3 + (p2 - p3) * u
            pt  = top + (bot - top) * v
            return QPointF(pt.x() * img_w, pt.y() * img_h)

        u_bounds = self._cell_bounds(self._grid_cols, self._grid_gap)
        v_bounds = self._cell_bounds(self._grid_rows, self._grid_gap)

        img_array: np.ndarray | None = None
        if pixmap_orig is not None and not pixmap_orig.isNull():
            qimg      = pixmap_orig.toImage().convertToFormat(QImage.Format_RGBA8888)
            arr       = np.frombuffer(qimg.constBits(), dtype=np.uint8).reshape((qimg.height(), qimg.width(), 4))
            img_array = arr[:, :, :3]

        result = []
        for row in range(self._grid_rows):
            v0, v1 = v_bounds[row]
            for col in range(self._grid_cols):
                u0, u1 = u_bounds[col]
                tl, tr = bilerp_orig(u0, v0), bilerp_orig(u1, v0)
                br, bl = bilerp_orig(u1, v1), bilerp_orig(u0, v1)
                cell = {"row": row, "col": col, "tl": tl, "tr": tr, "br": br, "bl": bl,
                        "color_mean": None, "color_std": None}
                if img_array is not None:
                    cell["color_mean"], cell["color_std"] = self._sample_cell_color(
                        img_array, img_w, img_h, tl, tr, br, bl)
                result.append(cell)
        return result

    @staticmethod
    def _sample_cell_color(img: np.ndarray, img_w: int, img_h: int,
                           tl: QPointF, tr: QPointF, br: QPointF, bl: QPointF
                           ) -> tuple[tuple, tuple]:
        """
        Sample pixels inside the quad cell using a vectorized bilinear UV grid.
        A uniform (n×n) grid of (u,v) coordinates is computed in one numpy pass —
        no Python pixel loop. ~17× faster than the previous loop-based version.
        Returns (mean_rgb, std_rgb) as float tuples.
        """
        side = max(math.hypot(tr.x() - tl.x(), tr.y() - tl.y()),
                   math.hypot(bl.x() - tl.x(), bl.y() - tl.y()))
        n = max(4, min(64, int(side)))

        # n×n UV grid, each in [0,1]
        u = (np.arange(n, dtype=np.float32) + 0.5) / n   # (n,)
        v = (np.arange(n, dtype=np.float32) + 0.5) / n   # (n,)
        uu, vv = np.meshgrid(u, v)                         # (n, n)

        # Bilinear interpolation of pixel coordinates
        tx = tl.x() + (tr.x() - tl.x()) * uu
        ty = tl.y() + (tr.y() - tl.y()) * uu
        bx = bl.x() + (br.x() - bl.x()) * uu
        by = bl.y() + (br.y() - bl.y()) * uu
        px = np.clip((tx + (bx - tx) * vv).round(), 0, img_w - 1).astype(np.int32).ravel()
        py = np.clip((ty + (by - ty) * vv).round(), 0, img_h - 1).astype(np.int32).ravel()

        samples = img[py, px].astype(np.float32)           # (n*n, 3)
        return (tuple(float(v) for v in samples.mean(axis=0)),
                tuple(float(v) for v in samples.std(axis=0)))

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if self._pixmap is None or self._pixmap.isNull():
            painter.fillRect(0, 0, w, h, QColor("#1a1a2e"))
            painter.setPen(QColor("#6a6a9a")); painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, "Zoomed view\n(wheel to zoom)")
            return

        painter.fillRect(0, 0, w, h, QColor("#0d0d1a"))
        rect = self._visible_rect_norm()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        src         = QRectF(rect.x() * iw, rect.y() * ih, rect.width() * iw, rect.height() * ih)
        src_clipped = src.intersected(QRectF(0, 0, iw, ih))
        if not src_clipped.isEmpty():
            dst = QRectF(
                (src_clipped.x() - src.x()) / src.width()  * w,
                (src_clipped.y() - src.y()) / src.height() * h,
                src_clipped.width()  / src.width()  * w,
                src_clipped.height() / src.height() * h,
            )
            painter.drawPixmap(dst, self._pixmap, src_clipped)

        if self._quad_points:
            self._draw_quad(painter)
            if len(self._quad_points) == 4:
                self._draw_grid(painter)

        # Zoom badge
        painter.setPen(Qt.NoPen); painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
        painter.drawRoundedRect(QRectF(8, 8, 72, 22), 5, 5)
        painter.setPen(QColor("#ffb38a")); painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(QRectF(8, 8, 72, 22), Qt.AlignCenter, f"zoom  ×{self._zoom:.1f}")

        # Points-remaining hint
        remaining = 4 - len(self._quad_points)
        if 0 < remaining < 4:
            hint = f"{remaining} point{'s' if remaining > 1 else ''} left"
            painter.setPen(Qt.NoPen); painter.setBrush(QBrush(QColor(0, 0, 0, 140)))
            painter.drawRoundedRect(QRectF(w - 160, 8, 152, 22), 5, 5)
            painter.setPen(QColor("#c8c8e8")); painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QRectF(w - 160, 8, 152, 22), Qt.AlignCenter, hint)

    def _draw_quad(self, painter: QPainter):
        pts = [self._norm_to_widget(self._rotate_norm(p, self._rotation, inverse=False))
               for p in self._quad_points]

        if len(pts) >= 2:
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1.5, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            for i in range(len(pts) - 1): painter.drawLine(pts[i], pts[i + 1])

        if len(pts) == 4:
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
            painter.setBrush(QBrush(QColor(255, 255, 255, 25)))
            painter.drawPolygon(QPolygonF(pts))

        for i, wp in enumerate(pts):
            color      = self.POINT_COLORS[i]
            is_editing = (i == self._editing_idx)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 90 if is_editing else 50)))
            painter.drawEllipse(wp, 18 if is_editing else 12, 18 if is_editing else 12)

            painter.setPen(QPen(color, 3 if is_editing else 2)); painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(wp, 8, 8)

            painter.setPen(QPen(color, 1.5))  # crosshair
            painter.drawLine(QPointF(wp.x() - 5, wp.y()), QPointF(wp.x() + 5, wp.y()))
            painter.drawLine(QPointF(wp.x(), wp.y() - 5), QPointF(wp.x(), wp.y() + 5))

            painter.setPen(Qt.NoPen); painter.setBrush(QBrush(color))
            painter.drawEllipse(wp, 2, 2)  # center dot

            painter.setPen(color); painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(QRectF(wp.x() + 10, wp.y() - 16, 20, 14), Qt.AlignLeft, f"P{i + 1}")

    def _draw_grid(self, painter: QPainter):
        """
        Draw the (cols × rows) deformed grid inside the quadrilateral.
        Points expected in order: P1=top-left, P2=top-right, P3=bottom-right, P4=bottom-left.
        Uses bilinear interpolation to map (u,v) ∈ [0,1]² to widget coordinates.
        """
        pts        = [self._norm_to_widget(self._rotate_norm(p, self._rotation, inverse=False))
                      for p in self._quad_points]
        p0, p1, p2, p3 = pts

        def bilerp(u, v):
            return p0 + (p1 - p0) * u + (p3 + (p2 - p3) * u - p0 - (p1 - p0) * u) * v

        # Dual-layer dashed pens (yellow + black offset) for readability on any background
        pen_y  = QPen(QColor(255, 220, 50, 220), 1.5, Qt.DashLine); pen_y.setDashPattern([4, 3])
        pen_k  = QPen(QColor(0, 0, 0, 160),      1.0, Qt.DashLine); pen_k.setDashPattern([4, 3]); pen_k.setDashOffset(3.5)
        pen_outer = QPen(QColor(255, 220, 80, 80), 0.8, Qt.DashLine)
        pen_alert = QPen(QColor(255, 60, 60, 230), 2.0, Qt.DashLine); pen_alert.setDashPattern([4, 3])
        pen_white = QPen(QColor(255, 255, 255, 180), 1.0, Qt.DashLine); pen_white.setDashPattern([4, 3]); pen_white.setDashOffset(3.5)

        u_bounds = self._cell_bounds(self._grid_cols, self._grid_gap)
        v_bounds = self._cell_bounds(self._grid_rows, self._grid_gap)

        for row in range(self._grid_rows):
            v0, v1 = v_bounds[row]
            for col in range(self._grid_cols):
                u0, u1 = u_bounds[col]
                poly = QPolygonF([bilerp(u0, v0), bilerp(u1, v0), bilerp(u1, v1), bilerp(u0, v1)])
                if (row, col) in self._highlighted_cells:
                    painter.setBrush(QBrush(QColor(255, 40, 40, 35)))
                    painter.setPen(pen_alert); painter.drawPolygon(poly)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(pen_white); painter.drawPolygon(poly)
                else:
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(pen_y); painter.drawPolygon(poly)
                    painter.setPen(pen_k); painter.drawPolygon(poly)

        painter.setPen(pen_outer); painter.drawPolygon(QPolygonF(pts))

    def sizeHint(self): return QSize(400, 400)


# ──────────────────────────────────────────────────────────────────────────────
# Arrow / rotation icon helpers (drawn to QPixmap — avoids Unicode rendering issues)
# ──────────────────────────────────────────────────────────────────────────────

def _make_arrow_pixmap(angle_deg: int, size: int = 14,
                       color: QColor = QColor("#ffdd55")) -> QPixmap:
    """Return a square QPixmap with an upward-pointing arrow rotated by angle_deg."""
    px = QPixmap(size, size); px.fill(Qt.transparent)
    p  = QPainter(px); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(color, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(QBrush(color)); p.translate(size / 2, size / 2); p.rotate(angle_deg)
    s = size / 2 - 1.0
    p.drawPolygon(QPolygonF([
        QPointF(0,           -s * 0.9),
        QPointF( s * 0.3,    -s * 0.9 + s * 0.5),
        QPointF( s * 0.165,  -s * 0.9 + s * 0.5),
        QPointF( s * 0.165,   s * 0.85),
        QPointF(-s * 0.165,   s * 0.85),
        QPointF(-s * 0.165,  -s * 0.9 + s * 0.5),
        QPointF(-s * 0.3,    -s * 0.9 + s * 0.5),
    ]))
    p.end(); return px


def _make_rot_pixmap(clockwise: bool, size: int = 14,
                     color: QColor = QColor("#ffdd55")) -> QPixmap:
    """Return a QPixmap with a curved rotation arrow (↺ or ↻)."""
    px = QPixmap(size, size); px.fill(Qt.transparent)
    p  = QPainter(px); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(color, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)); p.setBrush(Qt.NoBrush)
    m        = size * 0.18
    arc_rect = QRectF(m, m, size - 2 * m, size - 2 * m)
    p.drawArc(arc_rect, (150 if clockwise else 30) * 16, (-270 if clockwise else 270) * 16)
    end_angle = math.radians((150 - 270) if clockwise else (30 + 270))
    cx, cy   = size / 2, size / 2
    r        = (size - 2 * m) / 2
    ex, ey   = cx + r * math.cos(end_angle), cy - r * math.sin(end_angle)
    ah       = size * 0.22
    side     = math.pi / 2 if clockwise else -math.pi / 2
    dx, dy   = math.cos(end_angle + side), -math.sin(end_angle + side)
    p.setBrush(QBrush(color))
    p.drawPolygon(QPolygonF([
        QPointF(ex, ey),
        QPointF(ex - ah * (dx + dy * 0.5), ey - ah * (dy - dx * 0.5)),
        QPointF(ex - ah * (dx - dy * 0.5), ey - ah * (dy + dx * 0.5)),
    ]))
    p.end(); return px


def _make_separator() -> QFrame:
    sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setObjectName("separator")
    return sep


# ──────────────────────────────────────────────────────────────────────────────
# Bayer debayering — three algorithms
# ──────────────────────────────────────────────────────────────────────────────

try:
    import cv2 as _cv2
    _HAVE_CV2 = True
    # Bilinear (fast, standard)
    _CV2_BILINEAR: dict[str, int] = {
        "RGGB": _cv2.COLOR_BAYER_RG2RGB,
        "BGGR": _cv2.COLOR_BAYER_BG2RGB,
        "GRBG": _cv2.COLOR_BAYER_GR2RGB,
        "GBRG": _cv2.COLOR_BAYER_GB2RGB,
    }
    # VNG — Variable Number of Gradients: best quality, avoids moiré/zippering
    _CV2_VNG: dict[str, int] = {
        "RGGB": _cv2.COLOR_BAYER_RG2RGB_VNG,
        "BGGR": _cv2.COLOR_BAYER_BG2RGB_VNG,
        "GRBG": _cv2.COLOR_BAYER_GR2RGB_VNG,
        "GBRG": _cv2.COLOR_BAYER_GB2RGB_VNG,
    }
except ImportError:
    _HAVE_CV2 = False

# Row/column parity of R and B for each supported Bayer pattern.
# Entry: pattern → (R_row%2, R_col%2, B_row%2, B_col%2)
_BAYER_PATTERNS: dict[str, tuple[int, int, int, int]] = {
    "RGGB": (0, 0, 1, 1),
    "BGGR": (1, 1, 0, 0),
    "GRBG": (0, 1, 1, 0),
    "GBRG": (1, 0, 0, 1),
}

# Debayering algorithm names shown in the UI
DEBAYER_ALGOS = ["NN 2×2", "Bilinear 3×3", "VNG (anti-moire)"]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise any 2-D array to uint8, scaling 16-bit to 8-bit if needed."""
    if arr.dtype == np.uint8:
        return arr
    max_val = float(arr.max())
    return (arr.astype(np.float32) * (255.0 / max_val if max_val > 0 else 1.0)
            ).clip(0, 255).astype(np.uint8)


def debayer_nn(arr: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Nearest-neighbour (2×2 block) debayering.
    Each 2×2 RGGB block → one RGB pixel: R, mean(G1,G2), B.
    Output is the SAME resolution as input (each block pixel is upscaled ×2
    with np.repeat so the image keeps its original dimensions).
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {arr.shape}")
    raw = _to_uint8(arr)
    H, W = raw.shape
    # Crop to even dimensions if needed
    raw = raw[:H - H % 2, :W - W % 2]
    H, W = raw.shape

    r_row, r_col, b_row, b_col = _BAYER_PATTERNS[pattern]
    # Determine G positions (the two remaining corners of the 2×2 block)
    all_pos = {(0, 0), (0, 1), (1, 0), (1, 1)}
    g_pos   = all_pos - {(r_row, r_col), (b_row, b_col)}
    g1, g2  = list(g_pos)

    f  = raw.astype(np.float32)
    R  = f[r_row::2, r_col::2]                          # (H/2, W/2)
    G  = (f[g1[0]::2, g1[1]::2] + f[g2[0]::2, g2[1]::2]) * 0.5
    B  = f[b_row::2, b_col::2]

    # Stack and upscale ×2 with nearest-neighbour repeat → original resolution
    half = np.stack([R, G, B], axis=2).clip(0, 255).astype(np.uint8)  # (H/2, W/2, 3)
    full = np.repeat(np.repeat(half, 2, axis=0), 2, axis=1)            # (H, W, 3)
    return full[:H, :W]   # re-crop to original size (in case H or W was odd)


def debayer_bilinear(arr: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Bilinear 3×3 debayering — full resolution.
    Fast path: cv2.cvtColor (C++ bilinear). Fallback: pure numpy.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {arr.shape}")
    raw = _to_uint8(arr)

    if _HAVE_CV2:
        bgr = _cv2.cvtColor(raw, _CV2_BILINEAR[pattern])
        return bgr[:, :, ::-1].copy()   # BGR → RGB, contiguous

    # ── Numpy fallback ───────────────────────────────────────────────────────
    H, W  = raw.shape
    f     = raw.astype(np.float32)
    r_row, r_col, b_row, b_col = _BAYER_PATTERNS[pattern]
    p  = np.pad(f, 1, mode="reflect")
    N  = p[:-2, 1:-1];  S  = p[2:,  1:-1]
    Ww = p[1:-1, :-2];  E  = p[1:-1, 2:]
    NW = p[:-2,  :-2];  NE = p[:-2,  2:]
    SW = p[2:,   :-2];  SE = p[2:,   2:]
    card  = (N + S + Ww + E)    * 0.25
    diag  = (NW + NE + SW + SE) * 0.25
    horiz = (Ww + E)            * 0.5
    vert  = (N  + S)            * 0.5
    rows  = np.arange(H)[:, None] % 2
    cols  = np.arange(W)[None, :] % 2
    mR    = (rows == r_row) & (cols == r_col)
    mB    = (rows == b_row) & (cols == b_col)
    mGR   = ~mR & ~mB & (rows == r_row)
    R = np.where(mR,  f,    np.where(mGR, horiz, np.where(mB,  diag,  vert )))
    G = np.where(mR | mB, card, f)
    B = np.where(mB,  f,    np.where(mGR, vert,  np.where(mR,  diag,  horiz)))
    return np.stack([R, G, B], axis=2).clip(0, 255).astype(np.uint8)


def debayer_vng(arr: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    VNG (Variable Number of Gradients) debayering — best quality, avoids moiré.

    Fast path : cv2.cvtColor (highly optimised C++ SIMD).
    Pure-numpy fallback : full VNG implementation.

    Algorithm (Colour Filter Array Interpolation, Adaptive VNG):
      For each pixel p at position (r,c) in the raw Bayer mosaic:
        1. Compute the 4 cardinal + 4 diagonal raw gradients in a 5×5 window.
        2. Compute a per-direction "colour gradient" that accounts for
           the underlying Bayer pattern (R/G/B identity of each neighbour).
        3. Threshold T = 1.5 × min(gradients).
        4. Average only neighbours whose gradient ≤ T  (Variable Number).
        5. Assign R, G, B from those selected neighbours.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {arr.shape}")
    raw = _to_uint8(arr)

    if _HAVE_CV2:
        bgr = _cv2.cvtColor(raw, _CV2_VNG[pattern])
        return bgr[:, :, ::-1].copy()   # BGR → RGB, contiguous

    # ── Pure-numpy VNG ────────────────────────────────────────────────────────
    H, W = raw.shape
    f    = raw.astype(np.float32)

    r_row, r_col, b_row, b_col = _BAYER_PATTERNS[pattern]

    # Channel mask for every pixel (0=R, 1=G, 2=B)
    rows_idx = np.arange(H)[:, None] % 2
    cols_idx = np.arange(W)[None, :] % 2
    is_R  = (rows_idx == r_row) & (cols_idx == r_col)
    is_B  = (rows_idx == b_row) & (cols_idx == b_col)
    is_G  = ~is_R & ~is_B
    chan  = np.where(is_R, 0, np.where(is_G, 1, 2)).astype(np.uint8)  # (H,W)

    # Pad by 2 for the 5×5 window
    p  = np.pad(f,    2, mode="reflect")
    pc = np.pad(chan, 2, mode="reflect")

    # ── Gather the 8 directional neighbours at offsets ±1 and ±2 ─────────────
    # Direction order: N, NE, E, SE, S, SW, W, NW
    # Each direction uses a near (d=1) and a far (d=2) sample to compute gradient
    dr = np.array([-1, -1,  0,  1,  1,  1,  0, -1], dtype=int)
    dc = np.array([ 0,  1,  1,  1,  0, -1, -1, -1], dtype=int)

    # Precompute known-channel planes (H,W) at offset (dr,dc)*d
    def shifted(pad_arr, dy, dx):
        return pad_arr[2+dy:2+dy+H, 2+dx:2+dx+W]

    # For each of the 8 directions, compute a gradient:
    # grad[d] = |f[p+d1] - f[p]| + |f[p+d2] - f[p+d1]|  (raw intensity gradient)
    # But we must compare only same-channel pixels.  We use the bilinear estimate
    # at p+d1: est(c,pos) = average of same-channel pixels in a 3×3 around pos.
    # For speed, we pre-compute per-channel bilinear planes and index into them.

    # Bilinear interpolated planes (same as debayer_bilinear numpy) → R_bl, G_bl, B_bl
    pp = np.pad(f, 1, mode="reflect")
    N1 = pp[:-2, 1:-1]; S1 = pp[2:,  1:-1]
    W1 = pp[1:-1, :-2]; E1 = pp[1:-1, 2:]
    NW1= pp[:-2,  :-2]; NE1= pp[:-2, 2:]
    SW1= pp[2:,  :-2];  SE1= pp[2:,  2:]
    card  = (N1+S1+W1+E1)      * 0.25
    diag  = (NW1+NE1+SW1+SE1)  * 0.25
    horiz = (W1+E1)            * 0.5
    vert  = (N1+S1)            * 0.5

    mGR   = is_G & (rows_idx == r_row)
    bl_R  = np.where(is_R,  f, np.where(mGR, horiz, np.where(is_B,  diag,  vert)))
    bl_G  = np.where(is_G,  f, card)
    bl_B  = np.where(is_B,  f, np.where(mGR, vert,  np.where(is_R,  diag,  horiz)))
    bl    = np.stack([bl_R, bl_G, bl_B], axis=0)  # (3, H, W)

    # Padded bilinear planes for neighbour lookup
    p_bl = np.pad(bl, ((0,0),(2,2),(2,2)), mode="reflect")  # (3,H+4,W+4)

    # Padded raw for raw gradient
    p_raw = p   # already padded by 2

    # Compute gradient for each of 8 directions
    grads = np.zeros((8, H, W), dtype=np.float32)
    for d in range(8):
        dy1, dx1 = dr[d], dc[d]
        dy2, dx2 = dr[d]*2, dc[d]*2

        # Raw value at neighbour 1 and 2
        n1_raw = shifted(p_raw, dy1, dx1)
        n2_raw = shifted(p_raw, dy2, dx2)

        # Channel at neighbour positions
        n1_chan = shifted(pc, dy1, dx1).astype(int)  # (H,W)

        # Bilinear estimate of same channel as centre pixel at neighbour-1 position
        # We want bl[chan_centre, n1_pos] — but chan varies per pixel.
        # Vectorise: gather from p_bl using chan mask
        n1_bl_all = np.stack([
            p_bl[0, 2+dy1:2+dy1+H, 2+dx1:2+dx1+W],
            p_bl[1, 2+dy1:2+dy1+H, 2+dx1:2+dx1+W],
            p_bl[2, 2+dy1:2+dy1+H, 2+dx1:2+dx1+W],
        ], axis=0)  # (3,H,W)
        # Index by centre channel
        n1_est = np.where(is_R, n1_bl_all[0], np.where(is_G, n1_bl_all[1], n1_bl_all[2]))

        # Gradient = |centre_bilinear_at_n1 - f_centre| + |n2_raw - n1_raw|
        grads[d] = np.abs(n1_est - f) + np.abs(n2_raw - n1_raw)

    # ── Adaptive threshold and selective averaging ────────────────────────────
    T = grads.min(axis=0) * 1.5   # (H,W)

    # For each direction, we want to accumulate R,G,B of neighbour-1 if grad ≤ T
    acc_R = np.zeros((H, W), dtype=np.float32)
    acc_G = np.zeros((H, W), dtype=np.float32)
    acc_B = np.zeros((H, W), dtype=np.float32)
    cnt   = np.zeros((H, W), dtype=np.float32)

    for d in range(8):
        dy1, dx1 = dr[d], dc[d]
        use = (grads[d] <= T)    # (H,W) boolean

        # Bilinear RGB at neighbour-1 position
        n1_R = p_bl[0, 2+dy1:2+dy1+H, 2+dx1:2+dx1+W]
        n1_G = p_bl[1, 2+dy1:2+dy1+H, 2+dx1:2+dx1+W]
        n1_B = p_bl[2, 2+dy1:2+dy1+H, 2+dx1:2+dx1+W]

        acc_R += np.where(use, n1_R, 0.0)
        acc_G += np.where(use, n1_G, 0.0)
        acc_B += np.where(use, n1_B, 0.0)
        cnt   += use.astype(np.float32)

    safe = np.maximum(cnt, 1.0)

    # Known channel is always exact; other channels from adaptive average
    R_vng = np.where(is_R, f, acc_R / safe)
    G_vng = np.where(is_G, f, acc_G / safe)
    B_vng = np.where(is_B, f, acc_B / safe)

    return np.stack([R_vng, G_vng, B_vng], axis=2).clip(0, 255).astype(np.uint8)


def debayer(arr: np.ndarray, pattern: str = "RGGB", algo: str = "Bilinear 3×3") -> np.ndarray:
    """Dispatch to the requested debayering algorithm."""
    if algo == "NN 2×2":
        return debayer_nn(arr, pattern)
    elif algo == "VNG (anti-moire)":
        return debayer_vng(arr, pattern)
    else:
        return debayer_bilinear(arr, pattern)


def is_likely_bayer(pixmap: QPixmap) -> bool:
    """
    Returns True if the image is grayscale (R == G == B for every sampled pixel).
    Any grayscale image is treated as a raw Bayer mosaic.
    We sample a small central patch to keep detection fast.
    """
    img = pixmap.toImage().convertToFormat(QImage.Format_RGB32)
    w, h = img.width(), img.height()
    if w < 2 or h < 2:
        return False
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
    # Sample an 8×8 patch at the centre (Format_RGB32 in memory is BGRA)
    cy, cx  = h // 2, w // 2
    hy, hx  = min(4, cy), min(4, cx)
    patch   = arr[cy - hy:cy + hy, cx - hx:cx + hx]
    r, g, b = patch[:, :, 2], patch[:, :, 1], patch[:, :, 0]
    return bool(np.allclose(r, g, atol=2) and np.allclose(r, b, atol=2))



# ──────────────────────────────────────────────────────────────────────────────
# Automatic quad detection — hybrid colour-matching + homography approach
# ──────────────────────────────────────────────────────────────────────────────

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert (..., 3) float64 RGB [0..255] to CIE Lab (D65 illuminant).
    Pure numpy — no external dependency.
    """
    r = rgb.astype(np.float64) / 255.0
    linear = np.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = linear @ M.T
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    def f(t): return np.where(t > 0.008856, t ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def _kmeans_detect(pixels: np.ndarray, k: int, n_iter: int = 25
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means++ on (N, 3) float32 pixels.
    Fast path: cv2.kmeans.  Fallback: pure numpy Lloyd.
    Returns (labels (N,), centers (k, 3) float32).
    """
    if _HAVE_CV2:
        criteria = (_cv2.TERM_CRITERIA_EPS + _cv2.TERM_CRITERIA_MAX_ITER, n_iter, 1.0)
        _, labels, centers = _cv2.kmeans(
            pixels.astype(np.float32), k, None, criteria, 3, _cv2.KMEANS_PP_CENTERS)
        return labels.ravel(), centers
    # ── Pure numpy fallback ──────────────────────────────────────────────────
    rng     = np.random.default_rng(0)
    centers = pixels[rng.choice(len(pixels), k, replace=False)].copy().astype(np.float32)
    labels  = np.zeros(len(pixels), dtype=np.int32)
    for _ in range(n_iter):
        p2    = (pixels  ** 2).sum(1, keepdims=True)          # (N,1)
        c2    = (centers ** 2).sum(1, keepdims=True).T        # (1,k)
        dists = p2 + c2 - 2.0 * (pixels @ centers.T)         # (N,k)
        labels = np.argmin(dists, axis=1)
        new_c  = np.zeros_like(centers)
        counts = np.zeros(k, dtype=np.int32)
        np.add.at(new_c, labels, pixels)
        np.add.at(counts, labels, 1)
        good = counts > 0
        new_c[good]  /= counts[good, np.newaxis]
        new_c[~good]  = centers[~good]
        centers = new_c
    return labels, centers


def _find_homography_detect(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """
    Compute 3x3 homography H so that dst ~ H * src (homogeneous).
    RANSAC via cv2 when available, otherwise plain normalized DLT.
    src, dst: (N, 2) float64.  Returns H (3,3) or None.
    """
    if len(src) < 4:
        return None
    if _HAVE_CV2:
        H, _ = _cv2.findHomography(src, dst, _cv2.RANSAC, 5.0)
        return H
    # ── Normalized DLT ───────────────────────────────────────────────────────
    def normalize(pts):
        c  = pts.mean(0)
        s  = np.sqrt(2.0) / max(np.std(pts - c), 1e-9)
        T  = np.array([[s, 0, -s*c[0]], [0, s, -s*c[1]], [0, 0, 1.0]])
        ph = np.c_[pts, np.ones(len(pts))] @ T.T
        return ph[:, :2], T
    sn, Ts = normalize(src.astype(np.float64))
    dn, Td = normalize(dst.astype(np.float64))
    A = []
    for (x, y), (xp, yp) in zip(sn, dn):
        A += [[-x,-y,-1, 0, 0, 0, xp*x, xp*y, xp],
              [ 0, 0, 0,-x,-y,-1, yp*x, yp*y, yp]]
    _, _, Vt = np.linalg.svd(np.array(A))
    H = (np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts)
    return None if abs(H[2,2]) < 1e-12 else H / H[2,2]


def _apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N,2). Returns (N,2) after projective transform."""
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    w  = ph[:, 2:3]
    return ph[:, :2] / np.where(np.abs(w) > 1e-12, w, 1e-12)


def auto_detect_quad(
    img_rgb:     np.ndarray,
    ref_palette: list,
    max_px:      int = 1200,
    progress_cb  = None,       # callable(pct: int, msg: str) | None
) -> tuple[list[QPointF] | None, list[dict]]:
    """
    Detect the colour chart via a sliding-grid least-squares search.

    Strategy
    --------
    The fundamental insight: instead of trying to segment the palette from
    the background (unreliable when colours overlap), we *scan* the image
    with a candidate grid and score each placement by how well its cell
    colours match the reference palette after an affine colour correction.

    Steps
    -----
    1. Resize to max_px.
    2. Build a dense colour map: divide the work image into a grid of
       small tiles (e.g. 16×16 px) and compute the mean Lab colour of each.
    3. Multi-scale sliding window search:
       For each candidate (x0, y0, w_grid, h_grid, angle≈0):
         a. Sample the palette grid at cols×rows positions inside the window.
         b. Fit an affine 3×3 Lab colour matrix (least squares) mapping
            sampled colours → reference colours.
         c. Score = mean squared residual after correction.
       Keep the placement with the lowest score.
    4. Refine the best placement with a local optimiser (Nelder-Mead simplex
       over 4 parameters: cx, cy, w, h).
    5. Convert the best quad corners to normalised image coordinates.
    """
    H_img, W_img = img_rgb.shape[:2]
    rows_p = len(ref_palette)
    cols_p = max((len(r) for r in ref_palette), default=0)
    n_cells = rows_p * cols_p
    if n_cells == 0:
        return None, []

    # ── Reference colours ────────────────────────────────────────────────────
    ref_colors, ref_coords = [], []
    for r in range(rows_p):
        for c in range(cols_p):
            val = ref_palette[r][c] if c < len(ref_palette[r]) else None
            if val is not None:
                ref_colors.append(val)
                ref_coords.append((c, r))
    if len(ref_colors) < 4:
        return None, []
    ref_arr = np.array(ref_colors, dtype=np.float64)
    ref_lab = _rgb_to_lab(ref_arr)   # (M, 3)
    M_cells = len(ref_coords)
    ref_col_idx = np.array([c for c, r in ref_coords], dtype=np.float64)
    ref_row_idx = np.array([r for c, r in ref_coords], dtype=np.float64)
    _dbg_pts: list[dict] = []   # [{nx, ny, inlier, col, row}]
    _pcb = progress_cb or (lambda pct, msg: None)

    # ── 1. Resize ─────────────────────────────────────────────────────────────
    _pcb(2, "Resizing…")
    scale = min(1.0, max_px / max(H_img, W_img, 1))
    if scale < 1.0:
        nw = max(1, int(W_img * scale))
        nh = max(1, int(H_img * scale))
        if _HAVE_CV2:
            work = _cv2.resize(img_rgb, (nw, nh), interpolation=_cv2.INTER_AREA)
        else:
            sy = max(1, int(round(1.0 / scale)))
            nh2 = (H_img // sy) * sy; nw2 = (W_img // sy) * sy
            work = img_rgb[:nh2, :nw2].reshape(
                nh2//sy, sy, nw2//sy, sy, 3).mean((1, 3)).astype(np.uint8)
            nh, nw = work.shape[:2]
    else:
        work = img_rgb.copy()
        nh, nw = H_img, W_img

    sx   = nw / W_img
    sy_f = nh / H_img
    work_lab = _rgb_to_lab(work.astype(np.float64))   # (nh, nw, 3) — keep full res

    # ── 2. Dense tile colour map ──────────────────────────────────────────────
    _pcb(8, "Tiled colour map…")
    TILE = max(4, min(16, nw // 40, nh // 40))   # tile size in pixels
    tw = nw // TILE;  th = nh // TILE
    # Mean Lab per tile: reshape into (th, TILE, tw, TILE, 3) then average
    cropped = work_lab[:th*TILE, :tw*TILE]        # (th*TILE, tw*TILE, 3)
    tile_lab = cropped.reshape(th, TILE, tw, TILE, 3).mean(axis=(1, 3))  # (th, tw, 3)


    # ── Helper: score a placement defined in tile coordinates ─────────────────
    def score_placement(x0t, y0t, wt, ht):
        """
        x0t,y0t: top-left in tile coords; wt,ht: size in tiles.
        Returns mean-squared Lab residual after affine correction, or inf.
        """
        if wt < 1 or ht < 1:
            return np.inf
        # Sample each reference cell's position in tile map
        # cell (col, row) → tile (x0t + col/(cols_p-1)*wt, y0t + row/(rows_p-1)*ht)
        denom_c = max(cols_p - 1, 1)
        denom_r = max(rows_p - 1, 1)
        tx = x0t + ref_col_idx / denom_c * wt   # (M,) float tile x
        ty = y0t + ref_row_idx / denom_r * ht   # (M,) float tile y

        # Bilinear interpolation in tile_lab
        tx0 = np.clip(np.floor(tx).astype(int), 0, tw - 1)
        ty0 = np.clip(np.floor(ty).astype(int), 0, th - 1)
        tx1 = np.clip(tx0 + 1, 0, tw - 1)
        ty1 = np.clip(ty0 + 1, 0, th - 1)
        fx  = (tx - tx0).clip(0, 1)[:, None]
        fy  = (ty - ty0).clip(0, 1)[:, None]

        c00 = tile_lab[ty0, tx0]   # (M, 3)
        c10 = tile_lab[ty1, tx0]
        c01 = tile_lab[ty0, tx1]
        c11 = tile_lab[ty1, tx1]
        sampled = (c00*(1-fx)*(1-fy) + c01*fx*(1-fy)
                 + c10*(1-fx)*fy     + c11*fx*fy)   # (M, 3)

        # Affine least-squares correction: sampled @ A + b ≈ ref_lab
        A_aug = np.c_[sampled, np.ones(M_cells)]     # (M, 4)
        try:
            coeffs, res, _, _ = np.linalg.lstsq(A_aug, ref_lab, rcond=None)
            if len(res) == 3:
                return float(res.mean()) / M_cells
            corrected = A_aug @ coeffs
            residuals = np.sum((corrected - ref_lab) ** 2, axis=1)
            return float(residuals.mean())
        except np.linalg.LinAlgError:
            return np.inf

    # ── 3. Multi-scale coarse grid search ────────────────────────────────────
    _pcb(12, "Multi-scale coarse search…")
    best_score  = np.inf
    best_params = None   # (x0t, y0t, wt, ht) floats

    min_wt = max(cols_p,     tw // 20)
    max_wt = max(min_wt + 1, tw * 3 // 4)
    min_ht = max(rows_p,     th // 20)
    max_ht = max(min_ht + 1, th * 3 // 4)

    n_w_steps = 14
    n_h_steps = 10
    w_candidates = np.unique(np.linspace(min_wt, max_wt, n_w_steps).astype(int))
    h_candidates = np.unique(np.linspace(min_ht, max_ht, n_h_steps).astype(int))

    n_w = len(w_candidates)
    for wi, wt in enumerate(w_candidates):
        _pcb(12 + int(38 * wi / max(n_w - 1, 1)), f"Coarse search ({wi+1}/{n_w})…")
        for ht in h_candidates:
            step = max(1, min(wt, ht) // 5)
            for y0t in range(0, th - ht + 1, step):
                for x0t in range(0, tw - wt + 1, step):
                    s = score_placement(x0t, y0t, wt, ht)
                    if s < best_score:
                        best_score  = s
                        best_params = (float(x0t), float(y0t), float(wt), float(ht))


    if best_params is None:
        return None, []

    # ── 4. Pure-numpy Nelder-Mead refinement (no scipy needed) ───────────────
    _pcb(52, "Nelder-Mead refinement…")
    def _nelder_mead(f, x0, max_iter=300, tol=0.05):
        """Minimal Nelder-Mead simplex optimiser. x0: 1-D array-like."""
        n   = len(x0)
        x0  = np.array(x0, dtype=np.float64)
        # Build initial simplex: x0 + perturbations
        step = np.maximum(np.abs(x0) * 0.15, 0.5)
        sim  = np.tile(x0, (n + 1, 1))
        for i in range(n):
            sim[i + 1, i] += step[i]
        fsim = np.array([f(sim[i]) for i in range(n + 1)])

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        for _ in range(max_iter):
            order = np.argsort(fsim)
            sim, fsim = sim[order], fsim[order]
            if fsim[-1] - fsim[0] < tol:
                break
            centroid = sim[:-1].mean(axis=0)
            # Reflect
            xr = centroid + alpha * (centroid - sim[-1])
            fr = f(xr)
            if fr < fsim[0]:                      # expansion
                xe = centroid + gamma * (centroid - sim[-1])
                fe = f(xe)
                if fe < fr:
                    sim[-1], fsim[-1] = xe, fe
                else:
                    sim[-1], fsim[-1] = xr, fr
            elif fr < fsim[-2]:                   # accept reflection
                sim[-1], fsim[-1] = xr, fr
            else:                                 # contraction
                xc = centroid + rho * (sim[-1] - centroid)
                fc = f(xc)
                if fc < fsim[-1]:
                    sim[-1], fsim[-1] = xc, fc
                else:                             # shrink
                    sim[1:] = sim[0] + sigma * (sim[1:] - sim[0])
                    fsim[1:] = np.array([f(sim[i]) for i in range(1, n + 1)])
        return sim[0], fsim[0]

    def _obj(p):
        return score_placement(p[0], p[1], max(1.0, p[2]), max(1.0, p[3]))

    refined, refined_score = _nelder_mead(_obj, list(best_params))
    if refined_score < best_score:
        best_params = tuple(refined)
        best_score  = refined_score

    x0t, y0t, wt, ht = best_params
    wt = max(1.0, wt); ht = max(1.0, ht)

    # ── 5. Fine localisation: find each cell centroid in the ROI ─────────────
    _pcb(65, "Fine cell localisation…")
    # Expand the ROI slightly so border cells are fully included
    margin_t = max(1.0, min(wt, ht) * 0.3)
    roi_x0 = max(0, int((x0t - margin_t) * TILE))
    roi_y0 = max(0, int((y0t - margin_t) * TILE))
    roi_x1 = min(nw,  int((x0t + wt + margin_t) * TILE))
    roi_y1 = min(nh,  int((y0t + ht + margin_t) * TILE))

    roi_lab = work_lab[roi_y0:roi_y1, roi_x0:roi_x1]   # (rh, rw, 3)
    rh, rw  = roi_lab.shape[:2]

    if rh < rows_p or rw < cols_p:
        # ROI too small — fall back to rectangle
        _use_rect = True
    else:
        _use_rect = False

    if not _use_rect:
        # Fit the global affine colour correction on the whole ROI placement
        # (reuse the best params to get the correction matrix)
        denom_c = max(cols_p - 1, 1)
        denom_r = max(rows_p - 1, 1)
        tx_ref = x0t + ref_col_idx / denom_c * wt
        ty_ref = y0t + ref_row_idx / denom_r * ht

        def _interp_tile(tx, ty):
            tx0c = np.clip(np.floor(tx).astype(int), 0, tw-1)
            ty0c = np.clip(np.floor(ty).astype(int), 0, th-1)
            tx1c = np.clip(tx0c+1, 0, tw-1)
            ty1c = np.clip(ty0c+1, 0, th-1)
            fxc  = (tx - tx0c).clip(0,1)[:,None]
            fyc  = (ty - ty0c).clip(0,1)[:,None]
            return (tile_lab[ty0c,tx0c]*(1-fxc)*(1-fyc)
                  + tile_lab[ty0c,tx1c]*fxc*(1-fyc)
                  + tile_lab[ty1c,tx0c]*(1-fxc)*fyc
                  + tile_lab[ty1c,tx1c]*fxc*fyc)

        sampled_ref = _interp_tile(tx_ref, ty_ref)
        A_aug = np.c_[sampled_ref, np.ones(M_cells)]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_aug, ref_lab, rcond=None)  # (4,3)
        except np.linalg.LinAlgError:
            coeffs = None

        # Apply correction to every pixel in ROI, then match each cell
        roi_flat = roi_lab.reshape(-1, 3)
        if coeffs is not None:
            roi_aug  = np.c_[roi_flat, np.ones(len(roi_flat))]
            roi_corr = (roi_aug @ coeffs).reshape(rh, rw, 3)
        else:
            roi_corr = roi_lab

        # ── Fine pass: iterative localisation by connected component ──────────
        # Iteration 1: window centred on coarse position (rectangular)
        # → homographie approx H1
        # Iteration 2: window centred on H1(col,row) → precise centroids
        # Window is large (60% of cell) to absorb perspective,
        # and we take the centroid of the dominant connected component.

        cell_w_px = wt * TILE / max(cols_p, 1)
        cell_h_px = ht * TILE / max(rows_p, 1)

        def _locate_cells(expected_pts):
            """
            For each cell, search the ENTIRE ROI for pixels whose colour
            (after correction) is close to the reference, find all blobs,
            keep the blob whose centroid is closest to the expected centre.
            No rectangular window → no shape bias.
            """
            sp, dp, dbg = [], [], []

            for ci, (ex, ey) in expected_pts.items():
                col_r, row_r = ref_coords[ci]
                target = ref_lab[ci]   # (3,) Lab

                # ΔE sur tout le ROI
                de_roi = np.sqrt(np.sum(
                    (roi_corr.reshape(-1, 3) - target) ** 2, axis=1
                )).reshape(rh, rw)

                # Seuil : percentile 8% des pixels (les plus proches de la couleur)
                # Select pixels that "look like" this cell
                thr = float(np.percentile(de_roi, 8))
                thr = max(thr, 5.0)   # at least 5 Lab units
                mask_roi = (de_roi < thr).astype(np.uint8)

                if mask_roi.sum() < 4:
                    continue

                # Trouver tous les blobs
                if _HAVE_CV2:
                    n_lab, lab_img = _cv2.connectedComponents(mask_roi, connectivity=4)
                    if n_lab <= 1:
                        continue
                    # Centroid of each blob
                    blob_cx, blob_cy, blob_sz = [], [], []
                    for lbl in range(1, n_lab):
                        ys_b, xs_b = np.where(lab_img == lbl)
                        if len(xs_b) < 2:
                            continue
                        blob_cx.append(float(xs_b.mean()) + roi_x0)
                        blob_cy.append(float(ys_b.mean()) + roi_y0)
                        blob_sz.append(len(xs_b))
                else:
                    # BFS pour trouver tous les blobs
                    visited = np.zeros((rh, rw), dtype=bool)
                    blob_cx, blob_cy, blob_sz = [], [], []
                    ys_all, xs_all = np.where(mask_roi)
                    for start_y, start_x in zip(ys_all, xs_all):
                        if visited[start_y, start_x]:
                            continue
                        stack = [(start_y, start_x)]
                        region = []
                        visited[start_y, start_x] = True
                        while stack:
                            r_, c_ = stack.pop()
                            region.append((r_, c_))
                            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr_, nc_ = r_+dr, c_+dc
                                if (0<=nr_<rh and 0<=nc_<rw
                                        and not visited[nr_,nc_]
                                        and mask_roi[nr_,nc_]):
                                    visited[nr_,nc_] = True
                                    stack.append((nr_,nc_))
                        if len(region) >= 2:
                            ys_b = np.array([p[0] for p in region])
                            xs_b = np.array([p[1] for p in region])
                            blob_cx.append(float(xs_b.mean()) + roi_x0)
                            blob_cy.append(float(ys_b.mean()) + roi_y0)
                            blob_sz.append(len(region))

                if not blob_cx:
                    continue

                # Garder le blob le plus proche du centre attendu
                # weighted by size (prefers large nearby blobs)
                blob_cx = np.array(blob_cx)
                blob_cy = np.array(blob_cy)
                blob_sz = np.array(blob_sz, dtype=np.float64)
                dist_to_expected = np.sqrt((blob_cx - ex)**2 + (blob_cy - ey)**2)
                # Score = distance / log(taille+2) — favorise grands blobs proches
                score = dist_to_expected / np.log(blob_sz + 2)
                best = int(np.argmin(score))

                cx_work = float(blob_cx[best])
                cy_work = float(blob_cy[best])
                sp.append([float(col_r), float(row_r)])
                dp.append([cx_work, cy_work])
                dbg.append({'nx': cx_work / (sx * W_img),
                            'ny': cy_work / (sy_f * H_img),
                            'col': col_r, 'row': row_r, 'inlier': None})
            return sp, dp, dbg


        # ── Iteration 1: coarse positions ──────────────────────────────────────
        denom_c = max(cols_p - 1, 1)
        denom_r = max(rows_p - 1, 1)
        exp_pts1 = {ci: ((x0t + ref_coords[ci][0] / denom_c * wt) * TILE,
                         (y0t + ref_coords[ci][1] / denom_r * ht) * TILE)
                    for ci in range(M_cells)}

        sp1, dp1, dbg1 = _locate_cells(exp_pts1)

        # Intermediate homography to re-centre the windows
        H_iter1 = None
        if len(sp1) >= 4:
            src1 = np.array(sp1, dtype=np.float64)
            dst1 = np.array(dp1, dtype=np.float64)
            H_iter1 = _find_homography_detect(src1, dst1)

        # ── Iteration 2: centres predicted by H_iter1 ─────────────────────────
        if H_iter1 is not None:
            # Refit colour correction on the true iter1 centroids
            # (not on the coarse rectangular grid) to remove bias
            all_grid = np.array([[float(c), float(r)]
                                 for c, r in ref_coords], dtype=np.float64)

            # Actual colours at iter1 centroids
            matched_ci  = [i for i, (c,r) in enumerate(ref_coords)
                           if [c,r] in [[int(p[0]),int(p[1])] for p in sp1]]
            # Reconstruire via correspondance src→dst
            ci_map = {tuple(map(int,s)): d for s, d in zip(sp1, dp1)}
            real_colors = []
            real_ref    = []
            for ci2 in range(M_cells):
                key = (int(ref_coords[ci2][0]), int(ref_coords[ci2][1]))
                if key in ci_map:
                    cx2, cy2 = ci_map[key]
                    ix = int(np.clip(round(cx2), 0, nw-1))
                    iy = int(np.clip(round(cy2), 0, nh-1))
                    real_colors.append(work_lab[iy, ix])
                    real_ref.append(ref_lab[ci2])

            if len(real_colors) >= 4:
                A2 = np.c_[np.array(real_colors), np.ones(len(real_colors))]
                try:
                    coeffs2, _, _, _ = np.linalg.lstsq(A2, np.array(real_ref), rcond=None)
                    all_aug2  = np.c_[roi_lab.reshape(-1,3), np.ones(rh*rw)]
                    roi_corr  = (all_aug2 @ coeffs2).reshape(rh, rw, 3)
                except np.linalg.LinAlgError:
                    pass  # keep existing roi_corr

            pred = _apply_H(H_iter1, all_grid)   # (M, 2) work pixels
            exp_pts2 = {ci: (float(pred[ci, 0]), float(pred[ci, 1]))
                        for ci in range(M_cells)}
            sp2, dp2, dbg2 = _locate_cells(exp_pts2)
            if len(sp2) >= len(sp1):
                src_pts, dst_pts = sp2, dp2
                _dbg_pts.extend(dbg2)
            else:
                src_pts, dst_pts = sp1, dp1
                _dbg_pts.extend(dbg1)
        else:
            src_pts, dst_pts = sp1, dp1
            _dbg_pts.extend(dbg1)


        if len(src_pts) >= 4:
            src_arr = np.array(src_pts, dtype=np.float64)
            dst_arr = np.array(dst_pts, dtype=np.float64)

            # ── Helper: median residual of pts against homography H ───────────
            def _residuals(H, sa, da):
                if H is None: return np.full(len(sa), 1e9)
                proj = _apply_H(H, sa)
                return np.sqrt(np.sum((proj - da)**2, axis=1))

            # ── Pass A: fit H on raw found points ─────────────────────────────
            _pcb(78, "Geometric validation…")
            H_a = _find_homography_detect(src_arr, dst_arr)
            if H_a is None:
                return None, [], False, 0

            # Robust threshold: median residual * 3, min 3px, max 15px
            res_a    = _residuals(H_a, src_arr, dst_arr)
            med_res  = float(np.median(res_a))
            thr_geom = float(np.clip(med_res * 3.0, 3.0, 15.0))

            # ── Pass B: detect and replace geometric outliers ─────────────────
            # A point is an outlier if its residual >> median AND it breaks
            # the row/col alignment expected by the homography.
            outlier_mask = res_a > thr_geom
            n_outliers   = int(outlier_mask.sum())

            if n_outliers > 0 and n_outliers < len(src_pts) - 4:
                # Replace outlier dst positions with H-predicted positions
                pred_all = _apply_H(H_a, src_arr)
                dst_arr_corr = dst_arr.copy()
                dst_arr_corr[outlier_mask] = pred_all[outlier_mask]
                # Refit on corrected set
                H_b = _find_homography_detect(src_arr, dst_arr_corr)
                if H_b is not None:
                    # Accept correction only if it reduces overall residual
                    if _residuals(H_b, src_arr, dst_arr_corr).mean() < res_a.mean():
                        dst_arr = dst_arr_corr
                        H_a     = H_b
                        res_a   = _residuals(H_a, src_arr, dst_arr)

            # ── Pass C: interpolate missing cells ─────────────────────────────
            found_keys   = {(int(s[0]), int(s[1])) for s in src_arr.tolist()}
            missing_ci   = [ci for ci in range(M_cells)
                            if (int(ref_coords[ci][0]), int(ref_coords[ci][1]))
                            not in found_keys]
            n_interpolated = 0

            if missing_ci:
                for ci in missing_ci:
                    col_r, row_r = ref_coords[ci]
                    pred_xy = _apply_H(H_a, np.array([[float(col_r), float(row_r)]]))[0]
                    src_arr = np.vstack([src_arr, [float(col_r), float(row_r)]])
                    dst_arr = np.vstack([dst_arr, pred_xy])
                n_interpolated = len(missing_ci)

            # ── Pass D: verify row/col alignment ─────────────────────────────
            # Build a lookup col_r,row_r -> predicted position via H_a
            # then check that each actual found point is within thr_geom of
            # the line defined by its row-neighbours in grid space.
            # Use the final H (refit on all M_cells points) for the check.
            H_final = _find_homography_detect(src_arr, dst_arr)
            alignment_ok = True
            if H_final is not None:
                res_final = _residuals(H_final, src_arr, dst_arr)
                # Alignment fails if more than 20% of originally-found points
                # have residual > 2×thr_geom after correction
                n_bad = int((res_final[:len(src_pts)] > thr_geom * 2).sum())
                if n_bad > max(1, int(len(src_pts) * 0.20)):
                    alignment_ok = False

            all_found = (len(src_arr) == M_cells) and alignment_ok

            # Use H_final for RANSAC step below
            if H_final is not None:
                H_mat     = H_final
                # Mark all as inliers (we already corrected outliers above)
                inliers   = np.ones(len(src_arr), dtype=bool)
            else:
                H_mat     = None
                inliers   = np.ones(len(src_arr), dtype=bool)

            if H_mat is not None:
                m = 0.5
                corners_grid = np.array([
                    [-m,              -m             ],
                    [cols_p - 1 + m,  -m             ],
                    [cols_p - 1 + m,  rows_p - 1 + m],
                    [-m,              rows_p - 1 + m ],
                ], dtype=np.float64)
                corners_work = _apply_H(H_mat, corners_grid)
                corners_norm = np.stack([
                    corners_work[:, 0] / (sx   * W_img),
                    corners_work[:, 1] / (sy_f * H_img),
                ], axis=1).clip(0.0, 1.0)

                def cross2d(o, a, b):
                    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                signs = [cross2d(corners_norm[i], corners_norm[(i+1)%4],
                                 corners_norm[(i+2)%4]) for i in range(4)]
                if all(s > 0 for s in signs) or all(s < 0 for s in signs):
                    _pcb(100, "Done.")
                    return ([QPointF(float(c[0]), float(c[1])) for c in corners_norm],
                            _dbg_pts, all_found, n_interpolated)


    # ── 6. Fallback: axis-aligned rectangle from coarse params ───────────────
    def tile_to_norm(tx, ty):
        px = tx * TILE / (sx * W_img)
        py = ty * TILE / (sy_f * H_img)
        return float(np.clip(px, 0, 1)), float(np.clip(py, 0, 1))

    mt_x = 0.5 / max(cols_p - 1, 1) * wt
    mt_y = 0.5 / max(rows_p - 1, 1) * ht
    tl = tile_to_norm(x0t - mt_x,      y0t - mt_y)
    tr = tile_to_norm(x0t + wt + mt_x, y0t - mt_y)
    br = tile_to_norm(x0t + wt + mt_x, y0t + ht + mt_y)
    bl = tile_to_norm(x0t - mt_x,      y0t + ht + mt_y)
    corners_norm = np.array([tl, tr, br, bl])

    def cross2d(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    signs = [cross2d(corners_norm[i], corners_norm[(i+1)%4], corners_norm[(i+2)%4])
             for i in range(4)]
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return None, [], False, 0

    return [QPointF(c[0], c[1]) for c in corners_norm], [], False, 0



# ──────────────────────────────────────────────────────────────────────────────
# MainWindow
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QWidget):

    _ROT_LABELS       = ["0°", "90°", "180°", "270°"]
    _ROT_ARROW_ANGLES = [0, 90, 180, 270]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"colcal  v{APP_VERSION}")
        self.setMinimumSize(700, 450)
        self.resize(1200, 720)

        self._pixmap_orig:     QPixmap | None = None
        self._rotation:        int            = 0
        self._img_orig_size:   tuple          = (0, 0)
        self._last_image_dir:  str            = ""
        self._last_matrix_dir: str            = ""
        self._last_image_path: str            = ""
        self._corrected_mode:  bool           = False
        self._ref_palette:     list | None    = None

        # Bayer state
        self._bayer_raw_arr:   np.ndarray | None = None  # original raw grey array
        self._is_bayer:        bool              = False  # detected as Bayer mosaic
        self._bayer_debayered: bool              = False  # True = show debayered image
        self._bayer_pattern:   str               = "RGGB" # current Bayer pattern
        self._bayer_algo:      str               = DEBAYER_ALGOS[1]  # default: bilinear

        # Debounce timer: avoids recomputing the stats table on every wheel tick
        self._table_debounce = QTimer(self)
        self._table_debounce.setSingleShot(True)
        self._table_debounce.setInterval(600)
        self._table_debounce.timeout.connect(self._refresh_cell_table)

        self._setup_ui()
        self._apply_styles()
        self.setFocusPolicy(Qt.StrongFocus)
        self._load_prefs()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Toolbar ──
        toolbar = QHBoxLayout(); toolbar.setSpacing(8)
        self.btn_open = QPushButton("📂  Open image")
        self.btn_open.setCursor(Qt.PointingHandCursor)
        self.btn_open.setToolTip("Open an image to correct\n(PNG, JPG, TIFF, RAW/DNG…)")
        self.btn_open.clicked.connect(self._open_image)
        self.btn_save_corrected = QPushButton("💾  Save corrected")
        self.btn_save_corrected.setObjectName("saveCorrectedBtn")
        self.btn_save_corrected.setCursor(Qt.PointingHandCursor)
        self.btn_save_corrected.setEnabled(False)
        self.btn_save_corrected.setToolTip("Save the image with the colour correction applied")
        self.btn_save_corrected.clicked.connect(self._save_corrected_image)
        self.btn_batch = QPushButton("✨  Correct batch…")
        self.btn_batch.setObjectName("saveCorrectedBtn")
        self.btn_batch.setCursor(Qt.PointingHandCursor)
        self.btn_batch.setEnabled(False)
        self.btn_batch.setToolTip("Apply the current colour correction to a batch of images at once")
        self.btn_batch.clicked.connect(self._batch_correct)
        self.btn_auto_quad = QPushButton("🎯  Auto-detect")
        self.btn_auto_quad.setObjectName("autoQuadBtn")
        self.btn_auto_quad.setCursor(Qt.PointingHandCursor)
        self.btn_auto_quad.setEnabled(False)
        self.btn_auto_quad.setToolTip(
            "Automatically locate the colour chart in the full image.\n"
            "Rotate the image first so the chart is roughly upright.")
        self.btn_auto_quad.clicked.connect(self._auto_detect_quad)
        self.btn_auto_quad_roi = QPushButton("🔍  Detect in zoom")
        self.btn_auto_quad_roi.setObjectName("autoQuadBtn")
        self.btn_auto_quad_roi.setCursor(Qt.PointingHandCursor)
        self.btn_auto_quad_roi.setEnabled(False)
        self.btn_auto_quad_roi.setToolTip(
            "Search for the colour chart only inside the current zoom view.\n"
            "Zoom in on the chart area first to restrict the search zone.")
        self.btn_auto_quad_roi.clicked.connect(self._auto_detect_quad_roi)
        lbl_title = QLabel(f"Manu's lifelong dream  v{APP_VERSION}"); lbl_title.setObjectName("appTitle")
        lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_info = QLabel("No image loaded")
        self.lbl_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        toolbar.addWidget(self.btn_open); toolbar.addWidget(self.btn_save_corrected)
        toolbar.addWidget(self.btn_batch)
        toolbar.addWidget(self.btn_auto_quad); toolbar.addWidget(self.btn_auto_quad_roi)
        toolbar.addStretch(1); toolbar.addWidget(lbl_title); toolbar.addStretch(1)
        toolbar.addWidget(self.lbl_info)

        # ── Widgets ──
        self.viewer           = ImageViewerWidget()
        self.zoom_viewer      = ZoomViewerWidget()
        self.grid_settings    = GridSettingsWidget()
        self.cell_table       = CellTableWidget()
        self.palette_preview  = PalettePreviewWidget()
        self.measured_palette = MeasuredPaletteWidget()
        self.color_matrix     = ColorMatrixWidget()

        # ── Left column: zoomed view (stretches) + bottom bar ──
        left_col = QWidget()
        left_lay = QVBoxLayout(left_col); left_lay.setContentsMargins(0, 0, 0, 0); left_lay.setSpacing(4)
        left_lay.addWidget(self._labeled(
            self.zoom_viewer,
            "Zoomed view  —  wheel/↑↓=zoom  |  Shift+wheel=gap  |  drag=pan  |  click=point  |  right-click/Del=clear"
        ), stretch=1)
        left_lay.addWidget(_make_separator())

        # Bottom bar: grid settings | measured palette | color matrix | reference palette
        bottom_bar = QHBoxLayout(); bottom_bar.setContentsMargins(0, 0, 0, 0); bottom_bar.setSpacing(0)
        bottom_bar.addWidget(self.grid_settings)

        def vline():
            sep = _make_separator(); sep.setFrameShape(QFrame.VLine); return sep

        def palette_panel(widget, title_attr, title_text, with_open_btn=False):
            wrap = QWidget(); wrap.setMinimumWidth(70)
            wl   = QVBoxLayout(wrap); wl.setContentsMargins(4, 4, 4, 4); wl.setSpacing(2)
            hdr  = QHBoxLayout(); hdr.setSpacing(4)
            lbl  = QLabel(title_text); lbl.setObjectName("panelTitle"); lbl.setAlignment(Qt.AlignCenter)
            setattr(self, title_attr, lbl); hdr.addWidget(lbl, stretch=1)
            if with_open_btn:
                btn = QPushButton("📂  Open…"); btn.setObjectName("coordsBtn")
                btn.setCursor(Qt.PointingHandCursor); btn.setToolTip("Open a reference colour palette file (JSON)…")
                btn.clicked.connect(self.grid_settings._open_palette); hdr.addWidget(btn)
            wl.addLayout(hdr); wl.addWidget(widget, stretch=1)
            return wrap

        bottom_bar.addWidget(vline())
        bottom_bar.addWidget(palette_panel(self.measured_palette, "lbl_measured_title", "Measured"), stretch=1)
        bottom_bar.addWidget(vline()); bottom_bar.addWidget(self.color_matrix)
        bottom_bar.addWidget(vline())
        bottom_bar.addWidget(palette_panel(self.palette_preview, "lbl_palette_name", "No palette", with_open_btn=True), stretch=1)
        left_lay.addLayout(bottom_bar)

        # ── Right column: full view (small) + stats table (large) ──
        right_col = QWidget()
        right_lay = QVBoxLayout(right_col); right_lay.setContentsMargins(0, 0, 0, 0); right_lay.setSpacing(4)

        lbl_right = QLabel("Full view  —  click/drag=pan  |  wheel=zoom  |  Shift+wheel=gap  |  ←/→=rotate")
        lbl_right.setObjectName("panelTitle"); right_lay.addWidget(lbl_right)

        # Rotation bar
        rot_bar = QHBoxLayout(); rot_bar.setSpacing(6)
        icon_color = QColor("#ffdd55")
        self.btn_rot_left  = QPushButton("  Left")
        self.btn_rot_right = QPushButton("  Right")
        self.btn_rot_reset = QPushButton("  Origin")
        self.btn_rot_left .setIcon(QIcon(_make_rot_pixmap(False, 14, icon_color)))
        self.btn_rot_right.setIcon(QIcon(_make_rot_pixmap(True,  14, icon_color)))
        self.btn_rot_reset.setIcon(QIcon(_make_arrow_pixmap(0, 14, icon_color)))
        self.btn_rot_left .setToolTip("Rotate image 90° counter-clockwise\nKeyboard shortcut: ←")
        self.btn_rot_right.setToolTip("Rotate image 90° clockwise\nKeyboard shortcut: →")
        self.btn_rot_reset.setToolTip("Reset rotation to 0°")
        for btn in (self.btn_rot_left, self.btn_rot_right, self.btn_rot_reset):
            btn.setCursor(Qt.PointingHandCursor); btn.setObjectName("rotBtn"); btn.setFixedHeight(26)
        self.btn_rot_left .clicked.connect(lambda: self._rotate(-90))
        self.btn_rot_right.clicked.connect(lambda: self._rotate(+90))
        self.btn_rot_reset.clicked.connect(lambda: self._set_rotation(0.0))

        # Rotation spinbox — shows current angle (replaces separate orient indicator)
        # A small icon label is placed right before it so it stays compact.
        self.lbl_orient_icon = QLabel()
        self.lbl_orient_icon.setFixedSize(14, 14)
        self.spin_rot_fine = QSpinBox()
        self.spin_rot_fine.setRange(0, 359)
        self.spin_rot_fine.setSingleStep(1)
        self.spin_rot_fine.setSuffix("°")
        self.spin_rot_fine.setValue(0)
        self.spin_rot_fine.setFixedHeight(26)
        self.spin_rot_fine.setFixedWidth(72)
        self.spin_rot_fine.setWrapping(True)
        self.spin_rot_fine.setToolTip(
            "Current rotation (0–359°)\n"
            "─────────────────────────\n"
            "Scroll wheel         : ±1°\n"
            "Shift + scroll       : ±5°\n"
            "Right-click + scroll : ±5°\n"
            "Widget arrows ↑↓     : ±1°"
        )
        self.spin_rot_fine.valueChanged.connect(
            lambda v: self._set_rotation(float(v)))
        self.spin_rot_fine.setContextMenuPolicy(Qt.NoContextMenu)
        self.spin_rot_fine.installEventFilter(self)

        rot_bar.addWidget(self.btn_rot_left); rot_bar.addWidget(self.btn_rot_right)
        rot_bar.addWidget(self.btn_rot_reset)
        rot_bar.addSpacing(6)
        rot_bar.addWidget(self.lbl_orient_icon)
        rot_bar.addWidget(self.spin_rot_fine)
        rot_bar.addStretch()
        right_lay.addLayout(rot_bar)

        # Vertical splitter: small full view on top, large table on bottom
        right_vsplit = QSplitter(Qt.Vertical); right_vsplit.setHandleWidth(5)
        right_vsplit.setChildrenCollapsible(False); right_vsplit.addWidget(self.viewer)
        cell_wrap = QWidget()
        cell_lay  = QVBoxLayout(cell_wrap); cell_lay.setContentsMargins(0, 2, 0, 0); cell_lay.setSpacing(2)
        lbl_table = QLabel("Cell statistics  —  row, col, # px, color, σ"); lbl_table.setObjectName("panelTitle")
        cell_lay.addWidget(lbl_table); cell_lay.addWidget(self.cell_table)
        right_vsplit.addWidget(cell_wrap); right_vsplit.setSizes([220, 460])
        right_vsplit.setStretchFactor(0, 1); right_vsplit.setStretchFactor(1, 3)
        right_lay.addWidget(right_vsplit, stretch=1)
        self._right_vsplit = right_vsplit

        # Main horizontal splitter (left = zoom priority, right = full view)
        main_splitter = QSplitter(Qt.Horizontal); main_splitter.setHandleWidth(6)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(left_col); main_splitter.addWidget(right_col)
        main_splitter.setSizes([780, 420]); main_splitter.setStretchFactor(0, 3); main_splitter.setStretchFactor(1, 1)
        right_col.setMinimumWidth(120); self._main_splitter = main_splitter

        # ── Signal connections ──
        self.zoom_viewer.zoomRectChanged.connect(self.viewer.setZoomRect)
        self.viewer.centerMoved.connect(self.zoom_viewer.setCenter)
        self.viewer.zoomRequested.connect(self.zoom_viewer.zoomAtAnchor)
        self.viewer.gapShiftRequested.connect(
            lambda d: self._on_gap_changed(max(0.0, min(0.70, self.zoom_viewer._grid_gap + d)))
        )
        self.zoom_viewer.pointsChanged.connect(self._on_points_changed)
        self.zoom_viewer.pointsChanged.connect(lambda _: self._refresh_calc_btn())
        self.zoom_viewer.pointsEditingDone.connect(self._on_points_editing_done)
        self.zoom_viewer.gapChangeRequested.connect(self._on_gap_changed)
        self.zoom_viewer.gapChangeRequested.connect(lambda _: self._refresh_calc_btn())
        self.grid_settings.settingsChanged.connect(self._on_grid_settings_changed)
        self.grid_settings.settingsChanged.connect(lambda *_: self._refresh_calc_btn())
        self.grid_settings.paletteLoaded.connect(self._on_palette_loaded)
        self.color_matrix.computeRequested.connect(self._compute_color_matrix)
        self.color_matrix.toggleCorrected.connect(self._on_toggle_corrected)
        self.color_matrix.exportRequested.connect(self._export_matrix_json)
        self.color_matrix.importRequested.connect(self._import_matrix_json)

        root.addLayout(toolbar); root.addWidget(_make_separator())

        # ── Bayer toolbar (hidden until a Bayer image is detected) ──
        self._bayer_bar = QWidget()
        self._bayer_bar.setObjectName("bayerBar")
        bayer_lay = QHBoxLayout(self._bayer_bar)
        bayer_lay.setContentsMargins(8, 4, 8, 4); bayer_lay.setSpacing(8)
        bayer_lay.addWidget(QLabel("🔬  Raw Bayer detected:"))
        self.btn_bayer_toggle = QPushButton("Show debayered")
        self.btn_bayer_toggle.setObjectName("coordsBtn")
        self.btn_bayer_toggle.setCheckable(True)
        self.btn_bayer_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_bayer_toggle.setToolTip("Show the debayered image instead of the raw Bayer mosaic")
        self.btn_bayer_toggle.toggled.connect(self._on_bayer_toggle)
        bayer_lay.addWidget(self.btn_bayer_toggle)
        lbl_pat = QLabel("Pattern:")
        bayer_lay.addWidget(lbl_pat)
        self._bayer_combo = QComboBox()
        self._bayer_combo.addItems(list(_BAYER_PATTERNS.keys()))
        self._bayer_combo.setObjectName("bayerCombo")
        self._bayer_combo.setToolTip("Bayer pattern of the camera sensor (RGGB, BGGR…)")
        self._bayer_combo.currentTextChanged.connect(self._on_bayer_pattern_changed)
        bayer_lay.addWidget(self._bayer_combo)
        bayer_lay.addWidget(QLabel("Algo:"))
        self._bayer_algo_combo = QComboBox()
        self._bayer_algo_combo.addItems(DEBAYER_ALGOS)
        self._bayer_algo_combo.setCurrentText(DEBAYER_ALGOS[1])
        self._bayer_algo_combo.setObjectName("bayerCombo")
        self._bayer_algo_combo.setToolTip("Debayering algorithm: Bilinear (fast) or VNG (best quality, anti-moire)")
        self._bayer_algo_combo.currentTextChanged.connect(self._on_bayer_algo_changed)
        bayer_lay.addWidget(self._bayer_algo_combo)
        bayer_lay.addStretch()
        self._bayer_bar.hide()
        root.addWidget(self._bayer_bar)

        root.addWidget(main_splitter, stretch=1)

    @staticmethod
    def _labeled(child: QWidget, title: str) -> QWidget:
        """Wrap a widget with a panelTitle label above it."""
        wrap = QWidget(); lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)
        lbl = QLabel(title); lbl.setObjectName("panelTitle")
        lay.addWidget(lbl); lay.addWidget(child, stretch=1)
        return wrap

    # ── Rotation ──────────────────────────────────────────────────────────────

    def _rotate(self, delta_deg: int):
        self._set_rotation((self._rotation + delta_deg) % 360)

    def _set_rotation(self, angle: float):
        self._rotation = float(angle) % 360
        self._apply_rotation()
        self._update_rotation_ui()

    # ── Coordinate helpers: displayed-pixmap ↔ original-pixmap ───────────────

    def _display_to_orig_pts(self, pts: list, rot: float) -> list:
        """
        Convert normalised points from *displayed* pixmap space (after Qt rotation
        by `rot` degrees) back to *original* pixmap (pixmap_orig) normalised space.
        Handles any angle exactly via Qt's own bounding-rect geometry.
        """
        if not pts or self._pixmap_orig is None:
            return list(pts)
        W0, H0 = self._pixmap_orig.width(), self._pixmap_orig.height()
        if W0 == 0 or H0 == 0:
            return list(pts)
        rot = float(rot) % 360
        if abs(rot) < 1e-4:
            return list(pts)
        t  = QTransform(); t.rotate(rot)
        r  = t.mapRect(QRectF(0, 0, W0, H0))
        Wd = r.width(); Hd = r.height()
        dx = -r.left(); dy  = -r.top()
        rad   = math.radians(rot)
        cos_a = math.cos(rad); sin_a = math.sin(rad)
        result = []
        for p in pts:
            xd = p.x() * Wd - dx
            yd = p.y() * Hd - dy
            xo =  cos_a * xd + sin_a * yd
            yo = -sin_a * xd + cos_a * yd
            result.append(QPointF(float(np.clip(xo / W0, 0.0, 1.0)),
                                   float(np.clip(yo / H0, 0.0, 1.0))))
        return result

    def _orig_to_display_pts(self, pts: list, rot: float) -> list:
        """Inverse of _display_to_orig_pts."""
        if not pts or self._pixmap_orig is None:
            return list(pts)
        W0, H0 = self._pixmap_orig.width(), self._pixmap_orig.height()
        if W0 == 0 or H0 == 0:
            return list(pts)
        rot = float(rot) % 360
        if abs(rot) < 1e-4:
            return list(pts)
        t  = QTransform(); t.rotate(rot)
        r  = t.mapRect(QRectF(0, 0, W0, H0))
        Wd = r.width(); Hd = r.height()
        dx = -r.left(); dy  = -r.top()
        rad   = math.radians(rot)
        cos_a = math.cos(rad); sin_a = math.sin(rad)
        result = []
        for p in pts:
            xo = p.x() * W0; yo = p.y() * H0
            xd =  cos_a * xo - sin_a * yo + dx
            yd =  sin_a * xo + cos_a * yo + dy
            result.append(QPointF(float(np.clip(xd / Wd, 0.0, 1.0)),
                                   float(np.clip(yd / Hd, 0.0, 1.0))))
        return result

    def _quad_pts_to_orig(self) -> list:
        """Return current quad points converted to original-image normalised space."""
        pts = list(self.zoom_viewer._quad_points)
        return self._display_to_orig_pts(pts, self._rotation)

    def _apply_rotation(self):
        if self._pixmap_orig is None: return

        zv       = self.zoom_viewer
        prev_rot = getattr(self, '_prev_rotation', 0.0)

        # Convert center: old-display → original → new-display
        center_orig  = self._display_to_orig_pts([zv._center], prev_rot)[0]
        zv._center   = self._orig_to_display_pts([center_orig], self._rotation)[0]

        # Convert existing quad points: old-display → original → new-display
        if zv._quad_points:
            orig_pts = self._display_to_orig_pts(list(zv._quad_points), prev_rot)
            zv._quad_points = self._orig_to_display_pts(orig_pts, self._rotation)

        self._prev_rotation = self._rotation

        # Build the rotated display pixmap
        if self._rotation == 0.0:
            px = self._pixmap_orig
        else:
            t = QTransform(); t.rotate(self._rotation)
            px = self._pixmap_orig.transformed(t, Qt.SmoothTransformation)
        px = self._apply_correction_if_needed(px)

        # Always pass rotation=0: quad points and center are already in display space
        self.viewer.setPixmap(px)
        self.zoom_viewer.updatePixmap(px, 0)

    def _update_rotation_ui(self):
        snap = int(round(self._rotation / 90)) % 4
        self.lbl_orient_icon.setPixmap(
            _make_arrow_pixmap(self._ROT_ARROW_ANGLES[snap], 14, QColor("#ffdd55")))
        is_rotated = self._rotation % 360 >= 0.5
        self.btn_rot_reset.setProperty("active", is_rotated)
        self.btn_rot_reset.style().unpolish(self.btn_rot_reset)
        self.btn_rot_reset.style().polish(self.btn_rot_reset)
        # Keep spinbox in sync without triggering valueChanged→_set_rotation loop
        self.spin_rot_fine.blockSignals(True)
        self.spin_rot_fine.setValue(int(round(self._rotation)) % 360)
        self.spin_rot_fine.blockSignals(False)

    def eventFilter(self, obj, event):
        """Shift+wheel or RightButton+wheel on the rotation spinbox → ±5° steps."""
        if obj is self.spin_rot_fine and event.type() == QEvent.Type.Wheel:
            delta    = event.angleDelta().y()
            big_step = bool(event.modifiers() & Qt.ShiftModifier) or \
                       bool(event.buttons()   & Qt.RightButton)
            if big_step:
                step = 5 if delta > 0 else -5
                self._set_rotation((self._rotation + step) % 360)
                return True     # consumed — don't pass to spinbox default handler
        return super().eventFilter(obj, event)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if   key == Qt.Key_Left:                       self._rotate(-90)
        elif key == Qt.Key_Right:                      self._rotate(+90)
        elif key == Qt.Key_Up:                         self.zoom_viewer.applyZoom(+120)
        elif key == Qt.Key_Down:                       self.zoom_viewer.applyZoom(-120)
        elif key in (Qt.Key_Delete, Qt.Key_Backspace): self.zoom_viewer.clearQuad()
        else:                                          super().keyPressEvent(event)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_palette_loaded(self, palette: list, name: str, cols: int, rows: int):
        self._ref_palette = palette
        self.palette_preview.setPalette(palette, name)
        self.lbl_palette_name.setText(name)
        self.color_matrix.clear()
        self._refresh_auto_quad_btn()

    def _on_points_changed(self, points: list):
        """Called on every point change; skip table update while dragging."""
        if self.zoom_viewer._editing_idx < 0:
            self._refresh_cell_table()

    def _on_points_editing_done(self, _points: list):
        """Called on mouse release after dragging a point."""
        self._refresh_cell_table()

    def _on_grid_settings_changed(self, cols: int, rows: int, gap: float):
        self.zoom_viewer.setGrid(cols, rows, gap); self._table_debounce.start()

    def _on_gap_changed(self, new_gap: float):
        """Called by Shift+wheel on either view."""
        self.zoom_viewer.setGrid(self.zoom_viewer._grid_cols, self.zoom_viewer._grid_rows, new_gap)
        self.grid_settings.setGapValue(new_gap); self._table_debounce.start()

    # ── Button state ──────────────────────────────────────────────────────────

    def _refresh_calc_btn(self):
        """Enable Compute only when image + complete quad (4 pts) are both available."""
        self.color_matrix.btn_calc.setEnabled(
            self._pixmap_orig is not None and len(self.zoom_viewer._quad_points) == 4
        )

    def _refresh_save_corrected_btn(self):
        """Enable 'Save corrected' and 'Correct batch' only when a matrix exists."""
        has_matrix = self.color_matrix.matrix() is not None
        self.btn_save_corrected.setEnabled(
            self._pixmap_orig is not None and has_matrix
        )
        self.btn_batch.setEnabled(has_matrix)

    def _refresh_auto_quad_btn(self):
        raw_bayer = self._is_bayer and not self._bayer_debayered
        enabled = (self._pixmap_orig is not None
                   and self._ref_palette is not None
                   and not raw_bayer)
        tip_off = "Raw Bayer image — enable debayering first"
        self.btn_auto_quad.setEnabled(enabled)
        self.btn_auto_quad.setToolTip(
            tip_off if raw_bayer else
            "Automatically locate the colour chart in the full image.\n"
            "Rotate the image first so the chart is roughly upright.")
        self.btn_auto_quad_roi.setEnabled(enabled)
        self.btn_auto_quad_roi.setToolTip(
            tip_off if raw_bayer else
            "Search for the colour chart only inside the current zoom view.\n"
            "Zoom in on the chart area first to restrict the search zone.")

    def _auto_detect_quad(self):
        """
        Run the hybrid colour-matching + homography auto-detection.
        The detection runs on the *display* image (after applying the user's
        rotation), so that the chart is approximately upright as the algorithm
        expects.  Detected corners are then mapped back to original-image
        normalised coordinates via _rotate_norm(inverse=True).
        On success, sets the 4 quad points and refreshes stats.
        On failure or incomplete detection, shows a warning dialog.
        """
        if self._pixmap_orig is None or self._ref_palette is None:
            return

        # Progress dialog — shown immediately, before the heavy computation
        prog = QProgressDialog("Preparing…", None, 0, 100, self)
        prog.setWindowTitle("🎯  Auto-detect")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        QApplication.processEvents()

        def _progress(pct: int, msg: str):
            prog.setValue(pct)
            prog.setLabelText(msg)
            QApplication.processEvents()

        try:
            with _wait_cursor():
                # Build the rotated display pixmap (same as shown to the user)
                rot = self._rotation
                if rot == 0:
                    px_display = self._pixmap_orig
                else:
                    t = QTransform(); t.rotate(rot)
                    px_display = self._pixmap_orig.transformed(t, Qt.SmoothTransformation)

                qimg = px_display.toImage().convertToFormat(QImage.Format_RGB888)
                w, h = qimg.width(), qimg.height()
                bpl  = qimg.bytesPerLine()
                raw  = np.frombuffer(qimg.constBits(), dtype=np.uint8).reshape((h, bpl))
                arr  = np.ascontiguousarray(raw[:, :w * 3].reshape(h, w, 3))

                result_rot, dbg_pts, all_found, n_interpolated = auto_detect_quad(
                    arr, self._ref_palette, progress_cb=_progress)
        finally:
            prog.setValue(100)
            prog.close()

        if result_rot is None:
            QMessageBox.warning(
                self, "Auto-detect failed",
                "Could not automatically locate the colour chart.\n\n"
                "Possible reasons:\n"
                "• The chart colours are too far from the palette reference\n"
                "• The chart occupies too little of the image\n"
                "• The background has similar colours to the chart\n\n"
                "Please place the 4 points manually."
            )
            return

        if not all_found:
            rows_r = len(self._ref_palette)
            cols_r = max((len(r) for r in self._ref_palette), default=0)
            n_ref  = rows_r * cols_r
            n_located = len(dbg_pts)          # cells with real centroids
            n_total   = n_located + n_interpolated  # + geometrically interpolated
            if n_total < n_ref:
                # Genuinely missing cells — warn and abort
                QMessageBox.warning(
                    self, "Incomplete detection",
                    f"Only {n_total} out of {n_ref} chart colours could be located "
                    f"({n_located} measured, {n_interpolated} interpolated).\n\n"
                    "This usually means some chart colours are too close to the background "
                    "or gap colour.\n\n"
                    "Please place the 4 points manually."
                )
                return
            # all cells found but alignment check was borderline — continue anyway

        # result_rot is in normalised display-image space — store directly in viewer
        # (viewer always uses rotation=0, points live in display space).
        result = list(result_rot)

        # dbg_pts are also in display space; convert to original space for _optimise_gap
        # which samples pixmap_orig.
        dbg_pts_orig = []
        for d in dbg_pts:
            p_disp = QPointF(d['nx'], d['ny'])
            p_orig = self._display_to_orig_pts([p_disp], self._rotation)[0]
            dbg_pts_orig.append({**d, 'nx': p_orig.x(), 'ny': p_orig.y()})
        quad_orig = self._display_to_orig_pts(result, self._rotation)

        # Inject the 4 detected points into the zoom viewer (display space)
        self.zoom_viewer._quad_points = result
        self.zoom_viewer.pointsChanged.emit(list(result))
        self.zoom_viewer.update()
        self._refresh_calc_btn()

        # ── Pan/zoom so the quad is visible ───────────────────────────────────
        self._frame_quad_in_zoom_viewer(result)

        # ── Optimise gap (works in original-image space) ──────────────────────
        if dbg_pts_orig:
            self._optimise_gap(quad_orig, dbg_pts_orig)
        else:
            self._refresh_cell_table()

    def _auto_detect_quad_roi(self):
        """
        Run auto-detection restricted to the current zoom view.

        1. Get the visible rect from the zoom viewer (normalised coords in the
           displayed/rotated pixmap space).
        2. Crop the displayed image to that rect.
        3. Run auto_detect_quad on the crop.
        4. Remap the resulting normalised corners from crop-space back to
           full-display-image space, then store them in the zoom viewer.
        """
        if self._pixmap_orig is None or self._ref_palette is None:
            return

        zv  = self.zoom_viewer
        vis = zv._visible_rect_norm()   # in displayed-image normalised coords

        # Build the displayed (rotated) pixmap — same logic as _auto_detect_quad
        rot = self._rotation
        if rot == 0.0:
            px_display = self._pixmap_orig
        else:
            t = QTransform(); t.rotate(rot)
            px_display = self._pixmap_orig.transformed(t, Qt.SmoothTransformation)

        dw, dh = px_display.width(), px_display.height()

        # Clamp rect to [0,1]
        rx0 = max(0.0, vis.left());  ry0 = max(0.0, vis.top())
        rx1 = min(1.0, vis.right()); ry1 = min(1.0, vis.bottom())
        if rx1 <= rx0 or ry1 <= ry0:
            return

        # Pixel coords of the crop
        px0 = int(rx0 * dw); py0 = int(ry0 * dh)
        px1 = int(rx1 * dw); py1 = int(ry1 * dh)
        if px1 - px0 < 8 or py1 - py0 < 8:
            return

        prog = QProgressDialog("Preparing…", None, 0, 100, self)
        prog.setWindowTitle("🔍  Detect in zoom")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        QApplication.processEvents()

        def _progress(pct: int, msg: str):
            prog.setValue(pct); prog.setLabelText(msg)
            QApplication.processEvents()

        try:
            with _wait_cursor():
                qimg = px_display.toImage().convertToFormat(QImage.Format_RGB888)
                w, h = qimg.width(), qimg.height()
                bpl  = qimg.bytesPerLine()
                raw  = np.frombuffer(qimg.constBits(), dtype=np.uint8).reshape((h, bpl))
                full_arr = np.ascontiguousarray(raw[:, :w * 3].reshape(h, w, 3))
                crop_arr = np.ascontiguousarray(full_arr[py0:py1, px0:px1])

                result_crop, dbg_pts, all_found, n_interpolated = auto_detect_quad(
                    crop_arr, self._ref_palette, progress_cb=_progress)
        finally:
            prog.setValue(100); prog.close()

        if result_crop is None:
            QMessageBox.warning(
                self, "Detect in zoom — failed",
                "Could not locate the colour chart in the current view.\n\n"
                "Try zooming in closer to the chart, or use 🎯 Auto-detect\n"
                "to search the full image."
            )
            return

        if not all_found:
            rows_r = len(self._ref_palette)
            cols_r = max((len(r) for r in self._ref_palette), default=0)
            n_ref  = rows_r * cols_r
            n_total = len(dbg_pts) + n_interpolated
            if n_total < n_ref:
                QMessageBox.warning(
                    self, "Detect in zoom — incomplete",
                    f"Only {n_total} out of {n_ref} chart colours could be located "
                    f"({len(dbg_pts)} measured, {n_interpolated} interpolated).\n\n"
                    "Try zooming in closer to the chart."
                )
                return

        # Remap from crop-normalised → full-display-image normalised
        crop_w = rx1 - rx0; crop_h = ry1 - ry0

        def _remap(p: QPointF) -> QPointF:
            return QPointF(rx0 + p.x() * crop_w, ry0 + p.y() * crop_h)

        # All coords stay in display space
        result = [_remap(p) for p in result_crop]
        for d in dbg_pts:
            rp = _remap(QPointF(d['nx'], d['ny']))
            d['nx'] = rp.x(); d['ny'] = rp.y()

        # Convert to original space for _optimise_gap
        dbg_pts_orig = []
        for d in dbg_pts:
            p_orig = self._display_to_orig_pts([QPointF(d['nx'], d['ny'])], rot)[0]
            dbg_pts_orig.append({**d, 'nx': p_orig.x(), 'ny': p_orig.y()})
        quad_orig = self._display_to_orig_pts(result, rot)

        self.zoom_viewer._quad_points = result
        self.zoom_viewer.pointsChanged.emit(list(result))
        self.zoom_viewer.update()
        self._refresh_calc_btn()
        self._frame_quad_in_zoom_viewer(result)
        if dbg_pts_orig:
            self._optimise_gap(quad_orig, dbg_pts_orig)
        else:
            self._refresh_cell_table()

    def _frame_quad_in_zoom_viewer(self, quad: list[QPointF]):
        """
        Pan (and only if necessary, zoom out) the ZoomViewerWidget so that all
        4 quad points are visible.  The zoom is never *increased* — only reduced
        if the quad bounding box exceeds the current view.
        quad points are in display (viewer) normalised space.
        """
        if not quad or len(quad) != 4:
            return

        zv = self.zoom_viewer
        # Points are already in display space (viewer._rotation is always 0)
        xs = [p.x() for p in quad]; ys = [p.y() for p in quad]
        qx0, qx1 = min(xs), max(xs)
        qy0, qy1 = min(ys), max(ys)
        qw = qx1 - qx0
        qh = qy1 - qy0
        qcx = (qx0 + qx1) * 0.5
        qcy = (qy0 + qy1) * 0.5

        # Current visible rect
        vis = zv._visible_rect_norm()

        # Check if all points already visible
        if (vis.left()  <= qx0 and vis.right()  >= qx1 and
                vis.top() <= qy0 and vis.bottom() >= qy1):
            return   # nothing to do

        # Margin: 20% of the quad bounding box on each side
        margin = 0.20
        needed_w = qw * (1.0 + 2 * margin)
        needed_h = qh * (1.0 + 2 * margin)

        # If the quad bbox is larger than the current view, reduce zoom
        if needed_w > vis.width() or needed_h > vis.height():
            iw = zv._pixmap.width() if zv._pixmap else 1
            ih = zv._pixmap.height() if zv._pixmap else 1
            ww, wh = zv.width(), zv.height()
            wa = ww / wh if wh else 1.0
            ia = iw / ih if ih else 1.0
            if wa > ia:
                zoom_needed = (1.0 / needed_h) if needed_h > 0 else zv._zoom
            else:
                zoom_needed = (1.0 / needed_w) if needed_w > 0 else zv._zoom
            zoom_needed = max(zv.ZOOM_MIN, min(zv._zoom, zoom_needed))
            zv._zoom = zoom_needed

        # Now pan to centre the quad
        zv._center = QPointF(qcx, qcy)
        zv._clamp_center()
        zv._emit_zoom_rect()
        zv.update()

    def _optimise_gap(self, quad_init: list[QPointF], dbg_pts: list[dict]):
        """
        Jointly optimise the 4 quad corners (8 coords) + gap (1 value) = 9 params.

        Objective (minimise, all terms on the same ~1 scale):
          alignment : mean_sq_dist(grid_centres, detected_centroids)  [normalised 0..1]
          uniformity: mean_RGB_std / 255                               [normalised 0..1]
          gap_pen   : quadratic penalty below GAP_MIN and above GAP_MAX

        Weights chosen so alignment dominates, std is a tiebreaker,
        and gap is kept in a sensible range [GAP_MIN, GAP_MAX].
        """
        iw, ih = self._img_orig_size
        if iw == 0 or ih == 0:
            self._refresh_cell_table(); return

        rows = self.zoom_viewer._grid_rows
        cols = self.zoom_viewer._grid_cols

        # Detected centroids: {(col,row): (nx, ny)}
        det: dict[tuple, tuple] = {}
        for pt in dbg_pts:
            det[(int(pt['col']), int(pt['row']))] = (pt['nx'], pt['ny'])
        if len(det) < 4:
            self._refresh_cell_table(); return

        # Image array for std computation
        qimg     = self._pixmap_orig.toImage().convertToFormat(QImage.Format_RGBA8888)
        bpl      = qimg.bytesPerLine()
        raw      = np.frombuffer(qimg.constBits(), dtype=np.uint8).reshape((ih, bpl))
        img_arr  = np.ascontiguousarray(raw[:, :iw * 4].reshape(ih, iw, 4)[:, :, :3])

        GAP_MIN = 0.10   # never go below 10%
        GAP_MAX = 0.65

        # Initial parameter vector: [x0,y0, x1,y1, x2,y2, x3,y3, gap]
        x0_init = np.array([
            quad_init[0].x(), quad_init[0].y(),
            quad_init[1].x(), quad_init[1].y(),
            quad_init[2].x(), quad_init[2].y(),
            quad_init[3].x(), quad_init[3].y(),
            float(np.clip(self.zoom_viewer._grid_gap, GAP_MIN, GAP_MAX)),
        ], dtype=np.float64)

        def _eval(params):
            p = [QPointF(float(params[i*2]), float(params[i*2+1])) for i in range(4)]
            gap = float(np.clip(params[8], GAP_MIN, GAP_MAX))

            u_b = ZoomViewerWidget._cell_bounds(cols, gap)
            v_b = ZoomViewerWidget._cell_bounds(rows, gap)

            # ── Alignment error (normalised: distances already in [0..1] coords) ─
            pos_sq = []
            for row in range(rows):
                v0, v1 = v_b[row]; vm = (v0 + v1) * 0.5
                for col in range(cols):
                    key = (col, row)
                    if key not in det:
                        continue
                    u0, u1 = u_b[col]; um = (u0 + u1) * 0.5
                    top = p[0] + (p[1] - p[0]) * um
                    bot = p[3] + (p[2] - p[3]) * um
                    gpt = top + (bot - top) * vm
                    nx_d, ny_d = det[key]
                    pos_sq.append((gpt.x() - nx_d)**2 + (gpt.y() - ny_d)**2)
            # already in [0..1]² coords, typical value ~0.0001..0.01
            pos_err = float(np.mean(pos_sq)) * 1000.0 if pos_sq else 1e6

            # ── RGB std error (normalised to [0..1]) ─────────────────────────
            stds = []
            for row in range(rows):
                v0, v1 = v_b[row]
                for col in range(cols):
                    u0, u1 = u_b[col]
                    def bp(u, v, p=p):
                        top = p[0] + (p[1]-p[0])*u
                        bot = p[3] + (p[2]-p[3])*u
                        return top + (bot-top)*v
                    _, std = ZoomViewerWidget._sample_cell_color(
                        img_arr, iw, ih,
                        bp(u0,v0), bp(u1,v0), bp(u1,v1), bp(u0,v1))
                    stds.append(sum(std) / (3.0 * 255.0))   # normalised [0..1]
            std_err = float(np.mean(stds)) if stds else 0.0

            # ── Gap range penalty ────────────────────────────────────────────
            gap_pen  = max(0.0, GAP_MIN - gap)**2 * 5000.0
            gap_pen += max(0.0, gap - GAP_MAX)**2 * 5000.0

            # Weights: alignment dominates, std is tiebreaker
            return 1.0 * pos_err + 0.3 * std_err + gap_pen

        # ── Nelder-Mead ──────────────────────────────────────────────────────
        def _nelder_mead_9(f, x0, max_iter=800, tol=1e-5):
            n    = len(x0)
            x0   = np.array(x0, dtype=np.float64)
            # Step sizes: ~2% of image for corners, 8% for gap
            step = np.concatenate([np.full(8, 0.02), [0.08]])
            sim  = np.tile(x0, (n+1, 1))
            for i in range(n):
                sim[i+1, i] += step[i]
            fsim = np.array([f(sim[i]) for i in range(n+1)])
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            for _ in range(max_iter):
                order = np.argsort(fsim)
                sim, fsim = sim[order], fsim[order]
                if fsim[-1] - fsim[0] < tol:
                    break
                c = sim[:-1].mean(0)
                xr = c + alpha*(c - sim[-1]); fr = f(xr)
                if fr < fsim[0]:
                    xe = c + gamma*(c - sim[-1]); fe = f(xe)
                    sim[-1], fsim[-1] = (xe, fe) if fe < fr else (xr, fr)
                elif fr < fsim[-2]:
                    sim[-1], fsim[-1] = xr, fr
                else:
                    xc = c + rho*(sim[-1]-c); fc = f(xc)
                    if fc < fsim[-1]:
                        sim[-1], fsim[-1] = xc, fc
                    else:
                        sim[1:] = sim[0] + sigma*(sim[1:]-sim[0])
                        fsim[1:] = np.array([f(sim[i]) for i in range(1, n+1)])
            return sim[0], fsim[0]

        best_p, best_score = _nelder_mead_9(_eval, x0_init)

        gap_final  = float(np.clip(best_p[8], GAP_MIN, GAP_MAX))
        quad_orig_final = [QPointF(float(np.clip(best_p[i*2], 0, 1)),
                              float(np.clip(best_p[i*2+1], 0, 1)))
                      for i in range(4)]

        # Convert optimised result from original space → display space for the viewer
        quad_final = self._orig_to_display_pts(quad_orig_final, self._rotation)

        # Apply
        self.zoom_viewer._quad_points = quad_final
        self.zoom_viewer.pointsChanged.emit(list(quad_final))
        self.zoom_viewer.update()
        self._on_gap_changed(gap_final)
        self._refresh_calc_btn()
        self._refresh_cell_table()


    def _refresh_cell_table(self):
        iw, ih = self._img_orig_size
        if iw == 0 or ih == 0 or len(self.zoom_viewer._quad_points) != 4:
            self.cell_table.clear()
            self.measured_palette.setMeasured([], 0, 0)
            self.zoom_viewer.setHighlightedCells(set())
            return
        with _wait_cursor():
            thr      = self.grid_settings.stdThreshold()
            # Convert quad points from display space to original-image space for sampling
            orig_pts = self._quad_pts_to_orig()
            cells    = self.zoom_viewer.getCellCorners(iw, ih, self._pixmap_orig,
                                                       quad_override=orig_pts)
            self.cell_table.setStdThreshold(thr); self.cell_table.populate(cells)
            self.measured_palette.setMeasured(cells, self.zoom_viewer._grid_rows, self.zoom_viewer._grid_cols)
            highlighted = {(c["row"], c["col"]) for c in cells
                           if c["color_std"] and any(v >= thr for v in c["color_std"])}
            self.zoom_viewer.setHighlightedCells(highlighted)

    # ── Color matrix computation ──────────────────────────────────────────────

    def _compute_color_matrix(self):
        """
        Least-squares color transform.
          9-param  : A (N×3) @ X ≈ B  →  M = X.T
          12-param : A_aug (N×4) @ X ≈ B  →  M = X[:3].T, offset = X[3]
        Always samples raw colours from pixmap_orig, regardless of preview mode.
        """
        def fail(msg):
            self.color_matrix.setError(msg); self._refresh_save_corrected_btn()

        if self._ref_palette is None:
            return fail("No reference palette loaded.")
        if self._pixmap_orig is None:
            return fail("No image loaded.")

        # Always re-sample from pixmap_orig to get raw (uncorrected) cell colours
        iw, ih = self._img_orig_size
        if iw == 0 or ih == 0 or len(self.zoom_viewer._quad_points) != 4:
            return fail("No measured colors (place 4 points).")
        orig_pts = self._quad_pts_to_orig()
        cells    = self.zoom_viewer.getCellCorners(iw, ih, self._pixmap_orig,
                                                   quad_override=orig_pts)
        # Build measured palette from raw samples
        rows_g = self.zoom_viewer._grid_rows
        cols_g = self.zoom_viewer._grid_cols
        measured = [[None] * cols_g for _ in range(rows_g)]
        for cell in cells:
            r, c = cell["row"], cell["col"]
            if cell["color_mean"] is not None:
                measured[r][c] = cell["color_mean"]

        ref    = self._ref_palette
        rows_m = len(measured); cols_m = max((len(r) for r in measured), default=0)
        rows_r = len(ref);      cols_r = max((len(r) for r in ref),      default=0)
        if rows_m != rows_r or cols_m != cols_r:
            return fail(f"Dimension mismatch: measured {rows_m}×{cols_m} ≠ ref {rows_r}×{cols_r}")

        A_rows, B_rows = [], []
        for r in range(rows_m):
            for c in range(cols_m):
                m = measured[r][c] if c < len(measured[r]) else None
                b = ref[r][c]      if c < len(ref[r])      else None
                if m is not None and b is not None:
                    A_rows.append([float(m[0]), float(m[1]), float(m[2])])
                    B_rows.append([float(b[0]), float(b[1]), float(b[2])])

        use_offset = self.color_matrix.use_offset()
        n_params   = 4 if use_offset else 3
        if len(A_rows) < n_params:
            return fail(f"Not enough observations ({len(A_rows)} < {n_params}).")

        with _wait_cursor():
            A = np.array(A_rows, dtype=np.float64)
            B = np.array(B_rows, dtype=np.float64)
            if use_offset:
                A_aug      = np.hstack([A, np.ones((len(A), 1))])
                X, _, _, _ = np.linalg.lstsq(A_aug, B, rcond=None)
                M, offset  = X[:3, :].T, X[3, :]
                B_pred     = A_aug @ X
            else:
                X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                M, offset  = X.T, None
                B_pred     = A @ X
            diff     = B - B_pred
            rms_chan = np.sqrt(np.mean(diff ** 2, axis=0))
            self.color_matrix.setResult(
                M,
                (float(rms_chan[0]), float(rms_chan[1]), float(rms_chan[2]), float(np.sqrt(np.mean(diff ** 2)))),
                offset=offset,
            )
        self.color_matrix.btn_calc.setEnabled(False)
        self._refresh_save_corrected_btn()
        # If preview was active, keep it active and refresh with the new matrix
        if self._corrected_mode:
            self.color_matrix.btn_toggle.blockSignals(True)
            self.color_matrix.btn_toggle.setChecked(True)
            self.color_matrix.btn_toggle.setText("✅  Corrected")
            self.color_matrix.btn_toggle.blockSignals(False)
            self._on_toggle_corrected(True)

    # ── Color matrix application ──────────────────────────────────────────────

    def _apply_correction_if_needed(self, pixmap: QPixmap) -> QPixmap:
        """
        Return the matrix-corrected version of pixmap if corrected_mode is active
        and a matrix is available, otherwise return pixmap unchanged.
        Always uses the latest computed matrix.
        """
        if not self._corrected_mode:
            return pixmap
        M = self.color_matrix.matrix()
        if M is None:
            return pixmap
        return self._apply_matrix_to_pixmap(pixmap, M, self.color_matrix.offset())

    def _apply_matrix_to_pixmap(self, pixmap: QPixmap, M: np.ndarray,
                                 offset: np.ndarray | None = None) -> QPixmap:
        """Apply M (3×3) [+ offset (3)] to every pixel. Returns a new QPixmap."""
        img  = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
        w, h = img.width(), img.height()
        # Qt Format_RGB32 stores pixels as BGRA in memory
        arr  = np.frombuffer(img.bits(), dtype=np.uint8).reshape((h, w, 4)).copy()
        rgb  = arr[:, :, :3][:, :, ::-1].astype(np.float32)  # (H, W, 3) → R, G, B

        corrected = rgb.reshape(-1, 3) @ M.T
        if offset is not None: corrected += offset.astype(np.float32)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8).reshape(h, w, 3)

        arr[:, :, 0] = corrected[:, :, 2]  # B
        arr[:, :, 1] = corrected[:, :, 1]  # G
        arr[:, :, 2] = corrected[:, :, 0]  # R
        arr[:, :, 3] = 255
        return QPixmap.fromImage(QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_RGB32))

    def _on_toggle_corrected(self, on: bool):
        """Switch corrected-preview on/off while preserving zoom, center, quad, and stats."""
        self._corrected_mode = on
        self.zoom_viewer.setInteractionLocked(on)
        self.grid_settings.setEnabled(not on)

        M = self.color_matrix.matrix()
        with _wait_cursor():
            if self._pixmap_orig is not None:
                if self._rotation == 0.0:
                    px_rotated = self._pixmap_orig
                else:
                    t = QTransform(); t.rotate(self._rotation)
                    px_rotated = self._pixmap_orig.transformed(t, Qt.SmoothTransformation)
                if on and M is not None:
                    offset     = self.color_matrix.offset()
                    display_px = self._apply_matrix_to_pixmap(px_rotated, M, offset)
                else:
                    display_px = px_rotated
                self.zoom_viewer.updatePixmap(display_px, 0)
                self.viewer.setPixmap(display_px)

            if on and M is not None:
                # Always build corrected palette from raw pixmap_orig samples
                iw, ih = self._img_orig_size
                if iw > 0 and ih > 0 and len(self.zoom_viewer._quad_points) == 4:
                    orig_pts = self._quad_pts_to_orig()
                    cells    = self.zoom_viewer.getCellCorners(iw, ih, self._pixmap_orig,
                                                               quad_override=orig_pts)
                    offset   = self.color_matrix.offset()
                    rows_g   = self.zoom_viewer._grid_rows
                    cols_g   = self.zoom_viewer._grid_cols
                    corrected_pal = [[None] * cols_g for _ in range(rows_g)]
                    for cell in cells:
                        raw = cell["color_mean"]
                        if raw is not None:
                            v  = np.array([[raw[0], raw[1], raw[2]]], dtype=np.float32)
                            cv = np.clip(v @ M.T + (offset.astype(np.float32)
                                                    if offset is not None else 0), 0, 255)[0]
                            corrected_pal[cell["row"]][cell["col"]] = [int(cv[0]), int(cv[1]), int(cv[2])]
                    self.measured_palette._palette = corrected_pal
                    self.measured_palette.update()
            else:
                self._refresh_cell_table()   # restores raw measured colours
        self._refresh_save_corrected_btn()

    # ── File operations ───────────────────────────────────────────────────────

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", self._last_image_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff *.tif *.ppm *.pgm *.raw *.dng *.nef *.cr2 *.arw);;All files (*)"
        )
        if not path: return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.lbl_info.setText("❌ Cannot load image"); return

        self._last_image_dir  = os.path.dirname(path)
        self._last_image_path = path

        # Detect Bayer mosaic
        self._is_bayer = is_likely_bayer(pixmap)
        self._bayer_raw_arr = None

        if self._is_bayer:
            # Extract the raw grayscale array for debayering
            img = pixmap.toImage().convertToFormat(QImage.Format_RGB32)
            arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(
                (img.height(), img.width(), 4))
            self._bayer_raw_arr = arr[:, :, 2].copy()  # R channel = luma for grey image
            self._bayer_bar.show()
            # Restore previous toggle state if user re-opens an image
            # (keep btn_bayer_toggle state from prefs / previous session)
        else:
            self._bayer_bar.hide()
            self._bayer_debayered = False
            self.btn_bayer_toggle.blockSignals(True)
            self.btn_bayer_toggle.setChecked(False)
            self.btn_bayer_toggle.setText("Show debayered")
            self.btn_bayer_toggle.blockSignals(False)

        # Build the working pixmap
        if self._is_bayer and self._bayer_debayered:
            rgb = debayer(self._bayer_raw_arr, self._bayer_pattern, self._bayer_algo)
            pixmap = qpixmap_from_numpy_rgb(rgb)

        self._load_pixmap(pixmap, path)

    def _load_pixmap(self, pixmap: QPixmap, path: str):
        """Common finalisation after any pixmap is ready (normal or debayered)."""
        self._pixmap_orig   = pixmap
        self._rotation      = 0.0
        self._img_orig_size = (pixmap.width(), pixmap.height())
        self._apply_rotation(); self._update_rotation_ui()
        self.zoom_viewer.setPixmap(pixmap)
        self._refresh_cell_table()   # clears table + measured palette (no quad yet)
        self._refresh_calc_btn(); self._refresh_save_corrected_btn()
        self._refresh_auto_quad_btn()
        filename = os.path.basename(path)
        bayer_tag = "  [Bayer" + (f" {self._bayer_pattern} debayered]" if self._bayer_debayered else " raw]") if self._is_bayer else ""
        self.lbl_info.setText(f"{filename}{bayer_tag}  •  {pixmap.width()} × {pixmap.height()} px")

    def _on_bayer_toggle(self, checked: bool):
        """Switch between raw (grayscale mosaic) and debayered display."""
        self._bayer_debayered = checked
        self.btn_bayer_toggle.setText("Debayered ✓" if checked else "Show debayered")
        self._refresh_auto_quad_btn()
        self._rebuild_bayer_pixmap()

    def _on_bayer_pattern_changed(self, pattern: str):
        """Re-debayer with the newly selected Bayer pattern."""
        self._bayer_pattern = pattern
        if self._bayer_debayered:
            self._rebuild_bayer_pixmap()

    def _on_bayer_algo_changed(self, algo: str):
        """Re-debayer with the newly selected algorithm."""
        self._bayer_algo = algo
        if self._bayer_debayered:
            self._rebuild_bayer_pixmap()

    def _rebuild_bayer_pixmap(self):
        """
        Rebuild _pixmap_orig from the raw array after a toggle/pattern/algo change.
        Preserves zoom, center and quad points — only the pixel content changes.
        If corrected mode is active, the color matrix is re-applied automatically.
        """
        if self._bayer_raw_arr is None:
            return
        with _wait_cursor():
            if self._bayer_debayered:
                rgb    = debayer(self._bayer_raw_arr, self._bayer_pattern, self._bayer_algo)
                pixmap = qpixmap_from_numpy_rgb(rgb)
            else:
                pixmap = QPixmap(self._last_image_path)  # original raw mosaic (grayscale)

            self._pixmap_orig   = pixmap
            self._img_orig_size = (pixmap.width(), pixmap.height())

            # If corrected mode is on, show the matrix-corrected version
            display_px = self._apply_correction_if_needed(pixmap)

            # updatePixmap preserves zoom, center and quad points
            self.zoom_viewer.updatePixmap(display_px, self._rotation)
            self.viewer.setPixmap(display_px)

            self._refresh_calc_btn()
            self._refresh_save_corrected_btn()
            self._refresh_cell_table()

        filename  = os.path.basename(self._last_image_path)
        algo_tag  = f" {self._bayer_algo}" if self._bayer_debayered else ""
        bayer_tag = f"  [Bayer {self._bayer_pattern}{algo_tag} debayered]" if self._bayer_debayered else "  [Bayer raw]"
        self.lbl_info.setText(f"{filename}{bayer_tag}  •  {pixmap.width()} × {pixmap.height()} px")

    def _export_matrix_json(self):
        M = self.color_matrix.matrix()
        if M is None: return
        base         = os.path.splitext(os.path.basename(self._last_image_path))[0] if self._last_image_path else "matrix"
        default_name = os.path.join(self._last_image_dir, f"matrix_{base}.json")
        path, _      = QFileDialog.getSaveFileName(self, "Export matrix", default_name, "JSON (*.json);;All (*)")
        if not path: return
        offset = self.color_matrix.offset()
        data   = {
            "description": "Color correction matrix (least squares)",
            "image":   os.path.basename(self._last_image_path) if self._last_image_path else "",
            "palette": os.path.basename(getattr(self.grid_settings, "_loaded_palette_path", "")) or "",
            "mode":    "affine_12" if offset is not None else "linear_9",
            "matrix":  M.tolist(),
        }
        if offset is not None: data["offset"] = offset.tolist()
        data["formula"] = "rgb_out = M @ rgb_in + offset" if offset is not None else "rgb_out = M @ rgb_in"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._save_prefs()
        except Exception as e:
            QMessageBox.warning(self, "Export error", str(e))

    def _import_matrix_json(self):
        """Load a previously-exported matrix JSON and make it the active correction."""
        default_dir = self._last_matrix_dir or self._last_image_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Import matrix", default_dir, "JSON (*.json);;All (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            M_list = data.get("matrix")
            if M_list is None:
                raise ValueError("No 'matrix' key found in JSON.")
            M = np.array(M_list, dtype=np.float64)
            if M.shape != (3, 3):
                raise ValueError(f"Expected 3×3 matrix, got shape {M.shape}.")
            offset_list = data.get("offset")
            offset = np.array(offset_list, dtype=np.float64) if offset_list is not None else None
            if offset is not None and offset.shape != (3,):
                raise ValueError(f"Expected offset of length 3, got shape {offset.shape}.")
        except Exception as e:
            QMessageBox.warning(self, "Import error", str(e))
            return

        self._last_matrix_dir = os.path.dirname(path)
        source_name = os.path.basename(path)

        # Residuals are unknown for an imported matrix — show dashes
        residuals = (float("nan"), float("nan"), float("nan"))
        self.color_matrix.setResult(M, residuals, offset, source_name=source_name)
        self._refresh_save_corrected_btn()
        self._save_prefs()


    def _save_corrected_image(self):
        M = self.color_matrix.matrix()
        if M is None or self._pixmap_orig is None: return
        base         = os.path.splitext(os.path.basename(self._last_image_path))[0] if self._last_image_path else "image"
        default_name = os.path.join(self._last_image_dir, f"{base}_corrected.png")
        path, _      = QFileDialog.getSaveFileName(
            self, "Save corrected image", default_name,
            "PNG lossless (*.png);;TIFF lossless (*.tiff *.tif);;BMP lossless (*.bmp);;"
            "JPEG (lossy, not recommended) (*.jpg *.jpeg)"
        )
        if not path: return
        with _wait_cursor():
            px_corrected = self._apply_matrix_to_pixmap(self._pixmap_orig, M, self.color_matrix.offset())
            ext = os.path.splitext(path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                rep = QMessageBox.warning(self, "Lossy format",
                                          "JPEG is a lossy format.\nExact pixel values will be altered.\n\nContinue anyway?",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if rep != QMessageBox.Yes: return
                ok = px_corrected.save(path, "JPEG", 98)
            elif ext in (".tiff", ".tif"):
                writer = QImageWriter(path, b"TIFF"); writer.setCompression(1)
                ok = writer.write(px_corrected.toImage())
            elif ext == ".bmp":
                ok = px_corrected.save(path, "BMP")
            else:
                ok = px_corrected.save(path, "PNG", 0)
        if not ok:
            QMessageBox.warning(self, "Error", f"Cannot write file:\n{path}")

    def _batch_correct(self):
        """
        Apply the current colour correction matrix to a set of image files.
        • Files are chosen with a native dialog.
        • The pixel maths runs in parallel (ThreadPoolExecutor) — one worker
          per CPU core — so large images benefit from multi-core processing.
        • Output files are written next to the originals with '_corrected'
          suffix; existing files are never overwritten.
        • JPEG inputs are silently upgraded to PNG to avoid lossy roundtrips.
        """
        M = self.color_matrix.matrix()
        if M is None:
            return
        offset = self.color_matrix.offset()

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select images to correct",
            self._last_image_dir,
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp "
            "*.PNG *.JPG *.JPEG *.TIFF *.TIF *.BMP);;"
            "All files (*)"
        )
        if not paths:
            return

        # Force the file-chooser to close before showing the progress dialog
        QApplication.processEvents()

        n = len(paths)
        prog = QProgressDialog("Preparing…", "Cancel", 0, n, self)
        prog.setWindowTitle("✨ Correct batch")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        QApplication.processEvents()

        # ── Build output path (collision-free) ────────────────────────────────
        def _out_path(src: str) -> str:
            base, ext = os.path.splitext(src)
            out_ext = ".png" if ext.lower() in (".jpg", ".jpeg") else ext.lower()
            candidate = f"{base}_corrected{out_ext}"
            counter = 0
            while os.path.exists(candidate):
                counter += 1
                candidate = f"{base}_corrected_{counter}{out_ext}"
            return candidate

        # ── Worker: pure numpy — no Qt objects ────────────────────────────────
        M_copy      = M.copy()
        off_copy    = offset.copy() if offset is not None else None

        def _process_one(src_path: str) -> tuple[str, str | None]:
            """Returns (src_path, error_msg_or_None)."""
            dst_path = _out_path(src_path)
            try:
                # Load via PIL/numpy if available for thread safety; fall back
                # to a temporary QImage created in a local QGuiApplication context.
                # We use only numpy here — no Qt pixel objects.
                if _HAVE_CV2:
                    bgr = _cv2.imread(src_path, _cv2.IMREAD_COLOR)
                    if bgr is None:
                        return src_path, "cannot load"
                    rgb = bgr[:, :, ::-1].astype(np.float32)
                else:
                    # PIL fallback
                    try:
                        from PIL import Image as _PILImage
                        pil = _PILImage.open(src_path).convert("RGB")
                        rgb = np.array(pil, dtype=np.float32)
                    except Exception:
                        return src_path, "cannot load (need cv2 or Pillow)"

                H, W = rgb.shape[:2]
                corrected = rgb.reshape(-1, 3) @ M_copy.T
                if off_copy is not None:
                    corrected += off_copy.astype(np.float32)
                corrected = np.clip(corrected, 0, 255).astype(np.uint8).reshape(H, W, 3)

                _, ext = os.path.splitext(dst_path)
                ext = ext.lower()
                if _HAVE_CV2:
                    bgr_out = corrected[:, :, ::-1]
                    params = []
                    if ext in (".jpg", ".jpeg"):
                        params = [_cv2.IMWRITE_JPEG_QUALITY, 98]
                    elif ext == ".png":
                        params = [_cv2.IMWRITE_PNG_COMPRESSION, 0]
                    ok = _cv2.imwrite(dst_path, bgr_out, params)
                    if not ok:
                        return src_path, f"write failed → {os.path.basename(dst_path)}"
                else:
                    from PIL import Image as _PILImage
                    pil_out = _PILImage.fromarray(corrected)
                    pil_out.save(dst_path)
            except Exception as e:
                return src_path, str(e)
            return src_path, None

        # ── Run in thread pool ─────────────────────────────────────────────────
        errors    = []
        n_done    = 0
        cancelled = False

        # Use a results queue so the main thread can poll progress
        result_queue: queue.Queue = queue.Queue()

        def _worker_wrapper(p):
            r = _process_one(p)
            result_queue.put(r)

        max_workers = max(1, min(os.cpu_count() or 1, n))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker_wrapper, p) for p in paths]
            completed = 0
            while completed < n:
                try:
                    src, err = result_queue.get(timeout=0.05)
                    completed += 1
                    if err:
                        errors.append(f"{os.path.basename(src)}: {err}")
                    else:
                        n_done += 1
                    prog.setValue(completed)
                    prog.setLabelText(
                        f"{completed} / {n}  —  {os.path.basename(src)}")
                    QApplication.processEvents()
                except queue.Empty:
                    QApplication.processEvents()
                if prog.wasCanceled():
                    cancelled = True
                    for f in futures:
                        f.cancel()
                    break

        prog.setValue(n)

        # ── Summary ────────────────────────────────────────────────────────────
        n_skipped = len(errors)
        msg = f"✅  {n_done} image{'s' if n_done != 1 else ''} corrected and saved."
        if cancelled:
            msg += "\n⚠  Processing interrupted."
        if n_skipped:
            msg += f"\n❌  {n_skipped} file{'s' if n_skipped != 1 else ''} failed."
        if errors:
            detail = "\n".join(errors[:10])
            if len(errors) > 10:
                detail += f"\n… and {len(errors)-10} more."
            msg += f"\n\nDetails:\n{detail}"
        QMessageBox.information(self, "✨ Correct batch — done", msg)

    def _settings(self) -> QSettings:
        return QSettings(QSettings.IniFormat, QSettings.UserScope, "JoeSoft", "colcal")

    def _save_prefs(self):
        s = self._settings()
        s.setValue("window/geometry",     self.saveGeometry())
        s.setValue("splitter/main",        self._main_splitter.saveState())
        s.setValue("splitter/right",       self._right_vsplit.saveState())
        s.setValue("grid/gap_pct",         self.grid_settings.spin_gap.value())
        s.setValue("grid/std_thr",         self.grid_settings.spin_std.value())
        s.setValue("grid/cols",            self.grid_settings.spin_cols.value())
        s.setValue("grid/rows",            self.grid_settings.spin_rows.value())
        s.setValue("matrix/use_offset",    self.color_matrix.chk_offset.isChecked())
        s.setValue("paths/last_image_dir", self._last_image_dir)
        s.setValue("paths/last_palette_path", getattr(self.grid_settings, "_loaded_palette_path", ""))
        s.setValue("paths/last_matrix_dir", self._last_matrix_dir)
        s.setValue("bayer/pattern",  self._bayer_pattern)
        s.setValue("bayer/algo",     self._bayer_algo)

    def _load_prefs(self):
        s = self._settings()
        geom = s.value("window/geometry")
        if geom: self.restoreGeometry(geom)
        for key, splitter in [("splitter/main", self._main_splitter), ("splitter/right", self._right_vsplit)]:
            state = s.value(key)
            if state: splitter.restoreState(state)

        def restore_spin(spin, key, typ):
            val = s.value(key, None)
            if val is not None:
                spin.blockSignals(True); spin.setValue(typ(val)); spin.blockSignals(False)

        restore_spin(self.grid_settings.spin_gap,  "grid/gap_pct", float)
        restore_spin(self.grid_settings.spin_std,  "grid/std_thr", float)
        restore_spin(self.grid_settings.spin_cols, "grid/cols",    int)
        restore_spin(self.grid_settings.spin_rows, "grid/rows",    int)
        self.zoom_viewer.setGrid(self.grid_settings.spin_cols.value(),
                                 self.grid_settings.spin_rows.value(),
                                 self.grid_settings.spin_gap.value() / 100.0)
        self._last_image_dir  = s.value("paths/last_image_dir", "")
        self._last_matrix_dir = s.value("paths/last_matrix_dir", "")
        last_palette = s.value("paths/last_palette_path", "")
        if last_palette and os.path.isfile(last_palette):
            self.grid_settings._open_palette(last_palette)

        pattern = s.value("bayer/pattern", "RGGB")
        if pattern in _BAYER_PATTERNS:
            self._bayer_pattern = pattern
            self._bayer_combo.blockSignals(True)
            self._bayer_combo.setCurrentText(pattern)
            self._bayer_combo.blockSignals(False)
        algo = s.value("bayer/algo", DEBAYER_ALGOS[1])
        if algo in DEBAYER_ALGOS:
            self._bayer_algo = algo
            self._bayer_algo_combo.blockSignals(True)
            self._bayer_algo_combo.setCurrentText(algo)
            self._bayer_algo_combo.blockSignals(False)
        use_offset = s.value("matrix/use_offset", False)
        if isinstance(use_offset, str): use_offset = use_offset.lower() == "true"
        self.color_matrix.chk_offset.blockSignals(True)
        self.color_matrix.chk_offset.setChecked(bool(use_offset))
        self.color_matrix.chk_offset.blockSignals(False)
        self.color_matrix._on_offset_toggled(bool(use_offset))

    # ── Stylesheet ────────────────────────────────────────────────────────────

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget#bayerBar     { background-color: #1a1230; border-bottom: 1px solid #4a2a8a; }
            QComboBox#bayerCombo {
                background-color: #1e2a3a; color: #7ab8ff;
                border: 1px solid #3a5a7a; border-radius: 4px;
                padding: 2px 6px; font-size: 12px; min-width: 70px;
            }
            QComboBox#bayerCombo::drop-down { border: none; width: 16px; }
            QComboBox#bayerCombo QAbstractItemView {
                background-color: #1a1a32; color: #c8c8e8;
                selection-background-color: #2a3a5a;
            }
            QWidget {
                background-color: #0d0d1a;
                color: #c8c8e8;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            QLabel                { color: #7a7aaa; font-size: 13px; }
            QLabel#panelTitle     { color: #9a9abf; font-size: 13px; letter-spacing: 0.5px;
                                    padding: 2px 4px; border-bottom: 1px solid #2a2a4a; }
            QLabel#orientText     { color: #ffdd55; font-size: 14px; font-weight: bold; background: transparent; }
            QLabel#appTitle       { color: #c8b8ff; font-size: 16px; font-weight: bold; letter-spacing: 1px; }
            QWidget#orientLabel   { background-color: #1a1230; border: 1px solid #6644aa; border-radius: 5px; }

            QPushButton {
                background-color: #2a2a4a; color: #c8c8e8;
                border: 1px solid #4a4a7a; border-radius: 6px;
                padding: 6px 14px; min-height: 26px;
            }
            QPushButton:hover    { background-color: #3a3a6a; border-color: #7a7aaa; }
            QPushButton:pressed  { background-color: #1a1a3a; }
            QPushButton:disabled { color: #4a4a6a; border-color: #252540; background-color: #181828; }

            QPushButton#saveCorrectedBtn:disabled {
                color: #3a3a58; border-color: #1e1e38; background-color: #111120;
            }

            QPushButton#autoQuadBtn {
                color: #a8d8a8; border-color: #3a6a3a; background-color: #1a2e1a;
            }
            QPushButton#autoQuadBtn:hover    { background-color: #243824; border-color: #4a8a4a; }
            QPushButton#autoQuadBtn:disabled { color: #3a4a3a; border-color: #1e2e1e; background-color: #111811; }

            QPushButton#rotBtn         { color: #ffdd55; }
            QPushButton#rotBtn:hover   { color: #ffee88; }
            QPushButton#rotBtn[active="true"] {
                background-color: #2a1a4a; border-color: #cc99ff; color: #ffdd55;
            }

            QPushButton#coordsBtn {
                background-color: #1e2a3a; color: #7ab8ff;
                border: 1px solid #3a5a7a; border-radius: 6px;
                padding: 4px 8px; font-size: 12px; min-height: 24px;
            }
            QPushButton#coordsBtn:hover    { background-color: #2a3a5a; border-color: #6a9aff; }
            QPushButton#coordsBtn:pressed  { background-color: #141e2a; }
            QPushButton#coordsBtn:disabled { color: #2e4455; border-color: #1a2e3a; background-color: #0e1820; }
            QPushButton#coordsBtn:checked  { background-color: #1a3a2a; border-color: #44cc88; color: #55ff99; }
            QPushButton#coordsBtn:checked:hover { background-color: #1e4a32; }

            QFrame#separator      { background-color: #2a2a4a; max-height: 1px; border: none; }
            QSplitter::handle     { background-color: #2a2a4a; }
            QSplitter::handle:hover { background-color: #4a4a7a; }

            QSpinBox, QDoubleSpinBox {
                background-color: #1a1a32; color: #e0e0ff;
                border: 1px solid #4a4a7a; border-radius: 4px; padding: 2px 4px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus { border-color: #8888cc; }

            QLabel#matCell {
                color: #c8d8ff; font-family: 'Consolas', monospace; font-size: 12px;
                background-color: #131325; border: 1px solid #2a2a4a;
                border-radius: 3px; padding: 1px 4px;
            }
            QLabel#matCell[matStyle="diag"]   { color: #66eebb; border-color: #336644; background-color: #0d1f16; }
            QLabel#matCell[matStyle="off"]    { color: #ffaa55; border-color: #664422; background-color: #1f140d; }
            QLabel#matCell[matStyle="normal"] { color: #9999cc; }
            QLabel#matChan  { color: #7a7aaa; font-family: 'Consolas', monospace;
                              font-size: 12px; font-weight: bold; min-width: 14px; }
            QLabel#matResid { color: #55dd88; font-family: 'Consolas', monospace; font-size: 12px; }
            QLabel#matError { color: #ff6666; font-size: 12px; }

            QCheckBox { color: #9a9abf; font-size: 13px; spacing: 5px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #4a4a7a; border-radius: 3px; background: #1a1a2e;
            }
            QCheckBox::indicator:checked { background: #5555aa; border-color: #8888dd; }
            QCheckBox::indicator:hover   { border-color: #7a7abf; }

            QTableWidget#coordTable {
                background-color: #12121e; alternate-background-color: #1a1a2e;
                color: #e0e0ff; gridline-color: #2a2a4a;
                border: 1px solid #2a2a4a;
                font-family: 'Consolas', monospace; font-size: 13px;
            }
            QTableWidget#coordTable QHeaderView::section {
                background-color: #2a2a4a; color: #9a9abf;
                padding: 3px; border: none; font-size: 12px;
            }
        """)

    def closeEvent(self, event):
        self._save_prefs(); super().closeEvent(event)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("JoeSoft")
    app.setApplicationName("colcal")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
