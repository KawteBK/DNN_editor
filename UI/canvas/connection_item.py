from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsItem, QStyleOptionGraphicsItem
from PyQt6.QtGui import QPen, QColor, QPainterPath, QPainterPathStroker, QPainter
from PyQt6.QtCore import Qt, QTimer


class ConnectionItem(QGraphicsLineItem):
    _PEN_NORMAL   = QPen(QColor("#FFFFFF"), 2)
    _PEN_SELECTED = QPen(QColor("#F97316"), 2)   # orange when selected

    def __init__(self, start_node, end_node):
        super().__init__()
        self.start_node = start_node
        self.end_node   = end_node
        self._animation_offset = 0
        self._is_animating     = False
        self._timer            = None

        self.normal_pen    = QPen(self._PEN_NORMAL)
        self.animation_pen = QPen(QColor("#FFA500"), 3)  # initialised here so paint() never crashes
        self.animation_pen.setStyle(Qt.PenStyle.DashLine)

        self.setPen(self.normal_pen)

        # ── make it clickable ────────────────────────────────────────────────
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        # Sit below nodes visually but above the dark background
        self.setZValue(-1)

        start_node.connections.append(self)
        end_node.connections.append(self)
        self.update_position()

    # ── geometry ──────────────────────────────────────────────────────────────

    def update_position(self) -> None:
        start = self.start_node.sceneBoundingRect().center()
        end   = self.end_node.sceneBoundingRect().center()
        self.setLine(start.x(), start.y(), end.x(), end.y())

    def shape(self) -> QPainterPath:
        """
        Return a 12-px-wide stroked path as the hit area.
        Without this, clicking a 2-px line is nearly impossible.
        """
        path = QPainterPath()
        path.moveTo(self.line().p1())
        path.lineTo(self.line().p2())
        stroker = QPainterPathStroker()
        stroker.setWidth(12)
        return stroker.createStroke(path)

    # ── paint ─────────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        """Draw the line; orange tint when selected, animated when training."""
        if self._is_animating:
            painter.setPen(self.animation_pen)
        elif self.isSelected():
            painter.setPen(self._PEN_SELECTED)
        else:
            painter.setPen(self.normal_pen)
        painter.drawLine(self.line())

    # ── training animation ────────────────────────────────────────────────────

    def start_training_animation(self) -> None:
        """Start the animated dashed line effect."""
        if self._is_animating:
            return
        self._is_animating     = True
        self._animation_offset = 0
        self.animation_pen.setDashOffset(self._animation_offset)
        self._timer = QTimer()
        self._timer.timeout.connect(self._animate)
        self._timer.start(50)

    def stop_training_animation(self) -> None:
        """Stop animation and restore normal style."""
        if self._timer:
            self._timer.stop()
            self._timer = None
        self._is_animating = False
        self.update()   # trigger repaint so normal_pen is used immediately

    def _animate(self) -> None:
        self._animation_offset = (self._animation_offset + 5) % 20
        self.animation_pen.setDashOffset(self._animation_offset)
        self.update()