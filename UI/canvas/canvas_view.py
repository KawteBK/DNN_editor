from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QPainter, QBrush, QColor
from PyQt6.QtCore import Qt
from canvas.node_item import NodeItem
from canvas.connection_item import ConnectionItem

class CanvasView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))

        node1 = self.add_node("Linear", 100, 100)
        node2 = self.add_node("ReLU", 350, 100)
        node3 = self.add_node("Conv2D", 600, 100)

        self.add_connection(node1, node2)
        self.add_connection(node2, node3)

    def add_node(self, title, x, y):
        node = NodeItem(title, x, y)
        self.scene.addItem(node)
        return node

    def add_connection(self, start_node, end_node):
        connection = ConnectionItem(start_node, end_node)
        self.scene.addItem(connection)