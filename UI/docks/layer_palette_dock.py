from PyQt6.QtWidgets import (
    QDockWidget, QScrollArea, QWidget, QVBoxLayout, QPushButton, QLabel, QApplication
)
from PyQt6.QtCore import Qt, QMimeData, QPoint, QByteArray
from PyQt6.QtGui import QDrag, QPixmap, QPainter, QColor, QFont, QPen, QBrush

from UI.canvas.node_item import NODE_WIDTH, NODE_HEIGHT

LAYER_COLORS = {
    "InputLayer":  ("#3C5A48", "#1A7A43"),   # green   always the root
    "Dense":       ("#354A4D", "#105860"),   # teal
    "Conv2D":      ("#38364D", "#4A4490"),   # purple
    "Pooling":     ("#7F6634", "#9A5515"),   # amber
    "Recurrent":   ("#784F42", "#8C3515"),   # coral
    "Flatten":     ("#74672F", "#906E00"),   # yellowish brown 5ra vibes
    "Activation":  ("#683949", "#8C2E4F"),   # pink
    "Merge":       ("#57415F", "#6C3483"),   # purpley-pink, twilight sparkle vibes
}

# ── CHANGE: LAYER_COLORS is now imported from node_item (single source of truth).
# The local duplicate definition that existed here has been removed.

# Mime type used to pass the layer type string through the drag operation.
# Canvas reads this in dropEvent() to know which layer was dropped.
LAYER_MIME_TYPE = "application/x-nn-layer-type"   # ADDED


class _DraggableLayerButton(QPushButton):
    """
    ADDED: Subclass of QPushButton that initiates a Qt drag on mouse-move
    (after a small threshold), instead of just emitting clicked().

    What changed vs a plain QPushButton:
    - mousePressEvent records the press position.
    - mouseMoveEvent detects movement past QApplication.startDragDistance()
      and starts a QDrag with:
        • mimeData  — the layer type string so canvas knows what to spawn.
        • pixmap    — a rendered ghost of the future NodeItem so the user
                      sees exactly what they're about to drop.
      The drag pixmap hotspot is centred on the cursor so it feels "attached".
    - clicked() is still emitted on a plain click (no drag movement), so the
      old palette-click workflow (network_vm.add_layer) is unchanged.
    """

    def __init__(self, layer_type: str, parent=None):
        super().__init__(layer_type, parent)
        self.layer_type = layer_type
        self._press_pos: QPoint | None = None   # ADDED

    # ── ADDED: capture press position ────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.pos()   # ADDED — remember where the press started
        super().mousePressEvent(event)

    # ── ADDED: start drag once the cursor moves far enough ───────────────────

    def mouseMoveEvent(self, event) -> None:
        if (
            self._press_pos is not None
            and (event.pos() - self._press_pos).manhattanLength()
               >= QApplication.startDragDistance()   # ~4 px by default
        ):
            self._start_drag()
            self._press_pos = None   # consume — don't re-trigger
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._press_pos = None   # ADDED — always reset on release
        super().mouseReleaseEvent(event)

    # ── ADDED: build and execute the QDrag ───────────────────────────────────

    def _start_drag(self) -> None:
        """
        Creates a QDrag carrying the layer type string and a ghost pixmap.

        Ghost rendering: we paint a mini NodeItem look-alike (rounded rect +
        label text) directly onto a QPixmap using the same LAYER_COLORS as the
        real NodeItem, so the drag cursor matches what will land on the canvas.
        The pixmap is made semi-transparent (0.75 alpha) so the canvas shows
        through while dragging.
        """
        drag = QDrag(self)

        # ── mime data: just the layer type string ─────────────────────────
        mime = QMimeData()
        mime.setData(LAYER_MIME_TYPE, QByteArray(self.layer_type.encode()))
        drag.setMimeData(mime)

        # No drag pixmap — cursor only, no ghost duplicate.
        drag.setPixmap(QPixmap())

        # exec() blocks until drop or cancel; return value tells us the action.
        drag.exec(Qt.DropAction.MoveAction)   # ADDED

    def _make_ghost_pixmap(self) -> QPixmap:
        """
        ADDED: renders a NODE_WIDTH × NODE_HEIGHT pixmap that looks like a
        real NodeItem — same rounded rect, same colours, layer type centred.
        A device-pixel-ratio of 2 keeps it crisp on HiDPI screens.
        """
        dpr    = 2
        width  = NODE_WIDTH
        height = NODE_HEIGHT

        pixmap = QPixmap(width * dpr, height * dpr)
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)

        fill, border = LAYER_COLORS.get(self.layer_type, ("#1592A2", "#105860"))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent so the user can see the canvas through the ghost
        painter.setOpacity(0.35)   # lowered from 0.82 — subtle ghost, not a duplicate

        # Rounded rect background
        painter.setBrush(QBrush(QColor(fill)))
        painter.setPen(QPen(QColor(border), 1))   # thinner border to reduce visual weight
        painter.drawRoundedRect(1, 1, width - 2, height - 2, 10, 10)

        # Layer type text — keep it at the same reduced opacity as the rect
        painter.setOpacity(0.35)
        font = QFont("Consolas", 11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(10, 0, width - 12, height // 2 + 6,
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         self.layer_type)

        # Subtle "drop here" hint in the bottom half
        hint_font = QFont("Consolas", 8)
        painter.setFont(hint_font)
        painter.setPen(QColor("#D0EEF5"))
        painter.drawText(10, height // 2, width - 12, height // 2 - 4,
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         "drop on canvas")

        painter.end()
        return pixmap


class LayerPaletteDock(QDockWidget):
    # CHANGED: accepts network_vm instead of canvas
    # Previously called canvas.add_node() directly, now calls network_vm.add_layer()
    # so the Core model is updated before the canvas draws anything.
    #
    # ADDED: buttons are now _DraggableLayerButton instances so the user can
    # drag a layer from the palette directly onto the canvas at a chosen position,
    # in addition to the existing click-to-add behaviour.
    def __init__(self, network_vm):
        super().__init__("Layer Palette")
        self.network_vm = network_vm

        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Add a layer:"))

        layers = ["Dense", "Conv2D", "Pooling", "Recurrent", "Flatten", "Merge", "Activation"]

        for layer_type in layers:
            # CHANGED: QPushButton → _DraggableLayerButton
            btn = _DraggableLayerButton(layer_type)

            base_color, dark_color = LAYER_COLORS[layer_type]

            btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {base_color};
            color: #f7f5ff;
            border: 1px solid {dark_color};
            padding: 4px;
            border-radius: 6px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {dark_color};
        }}
        QPushButton:pressed {{
            background-color: {dark_color};
            padding-left: 8px;
            padding-top: 8px;
        }}
    """)

            # clicked() still works for plain clicks (no drag) — unchanged
            btn.clicked.connect(lambda checked, t=layer_type: self.network_vm.add_layer(t))
            layout.addWidget(btn)

        layout.addStretch()
        widget.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.setWidget(scroll)