from PyQt6.QtWidgets import QGraphicsLineItem
from PyQt6.QtGui import QPen, QColor, QBrush, QPainterPath, QPainter
from PyQt6.QtCore import Qt, QTimer, QPointF, QLineF


class ConnectionItem(QGraphicsLineItem):
    def __init__(self, start_node, end_node):
        super().__init__()
        self.start_node = start_node
        self.end_node = end_node
        self._animation_offset = 0
        self._is_animating = False
        self._timer = None
        
        # Normal style (not animating)
        self.normal_pen = QPen(QColor("#81D4FA"), 2)
        self.setPen(self.normal_pen)
        
        start_node.connections.append(self)
        end_node.connections.append(self)
        self.update_position()

    def update_position(self):
        start = self.start_node.sceneBoundingRect().center()
        end = self.end_node.sceneBoundingRect().center()
        self.setLine(start.x(), start.y(), end.x(), end.y())
        
    def start_training_animation(self):
        """Start the animated dashed line effect."""
        if self._is_animating:
            return
        
        self._is_animating = True
        self._animation_offset = 0
        
        # Create dashed pen for animation
        self.animation_pen = QPen(QColor("#FFA500"), 3)  # Orange
        self.animation_pen.setStyle(Qt.PenStyle.DashLine)
        self.animation_pen.setDashOffset(self._animation_offset)
        
        self.setPen(self.animation_pen)
        
        # Start timer to animate
        self._timer = QTimer()
        self._timer.timeout.connect(self._animate)
        self._timer.start(50)  # Update every 50ms
        
    def stop_training_animation(self):
        """Stop animation and restore normal style."""
        if self._timer:
            self._timer.stop()
            self._timer = None
        self._is_animating = False
        self.setPen(self.normal_pen)
        
    def _animate(self):
        """Update dash offset to create moving dots effect."""
        self._animation_offset = (self._animation_offset + 5) % 20
        self.animation_pen.setDashOffset(self._animation_offset)
        self.setPen(self.animation_pen)