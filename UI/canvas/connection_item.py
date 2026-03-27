from PyQt6.QtWidgets import QGraphicsLineItem
from PyQt6.QtGui import QPen, QColor

class ConnectionItem(QGraphicsLineItem):
    def __init__(self, start_node, end_node):
        super().__init__()
        self.start_node = start_node
        self.end_node = end_node
        self.setPen(QPen(QColor("#FFFFFF"), 2))
        start_node.connections.append(self)
        end_node.connections.append(self)
        self.update_position()

    def update_position(self):
        start = self.start_node.sceneBoundingRect().center()
        end = self.end_node.sceneBoundingRect().center()
        self.setLine(start.x(), start.y(), end.x(), end.y())