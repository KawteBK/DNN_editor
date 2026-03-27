from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem
from PyQt6.QtGui import QBrush, QPen, QColor
from PyQt6.QtCore import Qt

class NodeItem(QGraphicsRectItem):
    def __init__(self, title, x, y, width=150, height=60):
        super().__init__(0, 0, width, height)
        self.setPos(x, y)
        self.connections = []
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setBrush(QBrush(QColor("#1592A2")))
        self.setPen(QPen(QColor("#105860"), 2))

        self.label = QGraphicsTextItem(title, self)
        self.label.setDefaultTextColor(Qt.GlobalColor.white)
        self.label.setPos(10, 15)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for conn in self.connections:
                conn.update_position()
        return super().itemChange(change, value)