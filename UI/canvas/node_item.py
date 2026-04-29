from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem
from PyQt6.QtGui import QBrush, QPen, QColor, QFont, QPainter, QPainterPath
from PyQt6.QtCore import Qt, QRectF


# Color per layer type — makes different layers visually distinct on the canvas
LAYER_COLORS = {
    "InputLayer":  ("#24A359", "#1A7A43"),   # green   always the root
    "Dense":       ("#1592A2", "#105860"),   # teal
    "Conv2D":      ("#7F77DD", "#4A4490"),   # purple
    "Pooling":     ("#E67E22", "#9A5515"),   # amber
    "Recurrent":   ("#D85A30", "#8C3515"),   # coral
    "Flatten":     ("#B29928", "#906E00"),   # yellowish brown 5ra vibes
    "Activation":  ("#D4537E", "#8C2E4F"),   # pink
    "Merge":       ("#956AA6", "#6C3483"),   # purpley-pink, twilight sparkle vibes
}
DEFAULT_COLOR = ("#1592A2", "#105860")

# Layout constants — change these to tweak spacing across all nodes
NODE_WIDTH  = 145
NODE_HEIGHT = 75

_Y_TYPE  = 6    # top of type label
_Y_NAME  = 28   # top of custom name label
_Y_SHAPE = 52   # top of output shape label


class NodeItem(QGraphicsRectItem):
    def __init__(self, layer_id: str, title: str, layer_type: str, x: float, y: float,
                 width: int = NODE_WIDTH, height: int = NODE_HEIGHT):
        super().__init__(0, 0, width, height)
        self.layer_id   = layer_id
        self.layer_type = layer_type
        self._activation_type = None

        self.connections = []
        self.setPos(x, y)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        fill, border = LAYER_COLORS.get(layer_type, DEFAULT_COLOR)
        self.setBrush(QBrush(QColor(fill)))
        self.setPen(QPen(QColor(border), 2))

        # ── Line 1: layer type (e.g. "Dense", "Conv2D") ──────────────────────
        type_font = QFont("Consolas", 11)
        type_font.setBold(True)
        self.type_label = QGraphicsTextItem(layer_type, self)
        self.type_label.setDefaultTextColor(Qt.GlobalColor.white)
        self.type_label.setFont(type_font)
        self.type_label.setPos(10, _Y_TYPE)

        # ── Line 2: custom name (e.g. "Dense 1", user-renamed) ───────────────
        name_font = QFont("Consolas", 9)
        self.label = QGraphicsTextItem(title, self)
        self.label.setDefaultTextColor(QColor("#D0EEF5"))   # lighter tint so it reads as secondary
        self.label.setFont(name_font)
        self.label.setPos(10, _Y_NAME)

        # ── Line 3: output shape — empty until propagate_shapes() runs ───────
        shape_font = QFont("Consolas", 8)
        self.shape_label = QGraphicsTextItem("", self)
        self.shape_label.setDefaultTextColor(QColor("#E7F9FF"))  # muted blue-white
        self.shape_label.setFont(shape_font)
        self.shape_label.setPos(10, _Y_SHAPE)

    # ── public update methods ─────────────────────────────────────────────────

    def update_shape_label(self, shape_str: str) -> None:
        """Called by canvas after network_changed fires and shapes are known."""
        self.shape_label.setPlainText(shape_str)

    def update_name_label(self, name: str) -> None:
        """Called when the user renames the layer in LayerConfigDock."""
        self.label.setPlainText(name)

    def update_activation_type(self, act_type: str) -> None:
        """For Activation layers: show the activation function on line 1."""
        if self.layer_type == "Activation":
            self._activation_type = act_type
            # Keep type_label as "Activation", update name label to show function
            self.label.setPlainText(act_type)

    # ── movement -> update connections ────────────────────────────────────────

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for conn in self.connections:
                conn.update_position()
        return super().itemChange(change, value)

    # ── paint: rounded rect only — Qt draws children automatically ───────────

    def paint(self, painter, option, widget=None):
        """Custom paint with selection glow + clean border"""

        rect = QRectF(0, 0, self.rect().width(), self.rect().height())

        # Get layer colors
        fill_color, border_color = LAYER_COLORS.get(self.layer_type, DEFAULT_COLOR)

        base_color = QColor(fill_color)
        border_qcolor = QColor(border_color)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ── 1. Glow effect when selected ─────────────────────────────
        # if self.isSelected():
        #     glow_color = QColor(fill_color)
        #     glow_color.setAlpha(80)  # transparency

        #     for i in range(3):  # 3 layers of glow
        #         pen = QPen(glow_color, 6 + i * 2)
        #         painter.setPen(pen)
        #         painter.setBrush(Qt.BrushStyle.NoBrush)
        #         painter.drawRoundedRect(rect, 10, 10)

        # ── 2. Main rectangle ────────────────────────────────────────
        painter.setBrush(QBrush(base_color))

        if self.isSelected():
            # Bright selection border
            painter.setPen(QPen(QColor("#9ED3FFFF"), 3))
        else:
            painter.setPen(QPen(border_qcolor, 2))

        painter.drawRoundedRect(rect, 10, 10)