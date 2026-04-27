from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem
from PyQt6.QtGui import QBrush, QPen, QColor
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QRectF


# Color per layer type — makes different layers visually distinct on the canvas
LAYER_COLORS = {
    "InputLayer":  ("#2ECC71", "#1A7A43"),   # green   always the root
    "Dense":       ("#1592A2", "#105860"),   # teal
    "Conv2D":      ("#7F77DD", "#4A4490"),   # purple
    "Pooling":     ("#E67E22", "#9A5515"),   # amber
    "Recurrent":   ("#D85A30", "#8C3515"),   # coral
    "Flatten":     ("#B29928", "#906E00"),   # yellowish , 5ra vibezzz
    "Activation":  ("#D4537E", "#8C2E4F"),   # pink
    "Merge":       ("#956AA6", "#6C3483"),   #purpley-pink, twilight sparkle vibes
}
DEFAULT_COLOR = ("#1592A2", "#105860")


class NodeItem(QGraphicsRectItem):
    # ADDED: layer_id and layer_type params.
    # layer_id links this visual node to the Core layer object inside NetworkVM.
    # layer_type drives the color so users can tell layers apart at a glance.
    def __init__(self, layer_id: str, title: str, layer_type: str, x: float, y: float,
                 width: int = 160, height: int = 60):
        super().__init__(0, 0, width, height)
        self.layer_id = layer_id       # ADDED: used by canvas to call network_vm methods
        self.layer_type = layer_type   # ADDED: used for color lookup
        self._activation_type = None  # Store activation type

        self.connections = []
        self.setPos(x, y)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        # CHANGED: color now depends on layer type instead of always being teal
        fill, border = LAYER_COLORS.get(layer_type, DEFAULT_COLOR)
        self.setBrush(QBrush(QColor(fill)))
        self.setPen(QPen(QColor(border), 2))

        self.label = QGraphicsTextItem(title, self)
        self.label.setDefaultTextColor(Qt.GlobalColor.white)
        self.label.setPos(10, 10)

        # ADDED: shape label shown below the title once propagate_shapes() runs.
        # Stays empty until NetworkVM pushes a shape string via update_shape_label().
        self.shape_label = QGraphicsTextItem("", self)
        self.shape_label.setDefaultTextColor(QColor("#CCCCCC"))
        self.shape_label.setPos(10, 32)
        font = self.shape_label.font()
        font.setPointSize(7)
        self.shape_label.setFont(font)

    def update_shape_label(self, shape_str: str) -> None:
        # ADDED: called by canvas after network_changed fires and shapes are known
        self.shape_label.setPlainText(shape_str)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for conn in self.connections:
                conn.update_position()
        return super().itemChange(change, value)
    
    def update_activation_type(self, act_type: str):
        """Update the label for activation layers."""
        if self.layer_type == "Activation":
            self._activation_type = act_type
            self.label.setPlainText(f"{self.layer_type}\n({act_type})")


   # Override paint method in NodeItem for rounded corners

    def paint(self, painter, option, widget=None):
        """Override to draw rounded rectangle."""
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
    
        # Draw rounded rectangle
        rect = QRectF(0, 0, self.rect().width(), self.rect().height())
        painter.drawRoundedRect(rect, 10, 10)  # 10px radius
    
        # Draw children (labels, etc.)
        for child in self.childItems():
            painter.save()
            painter.translate(child.pos())
            child.paint(painter, option, widget)
            painter.restore()