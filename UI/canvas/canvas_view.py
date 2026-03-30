from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen
from PyQt6.QtCore import Qt, QPointF

from canvas.node_item import NodeItem
from canvas.connection_item import ConnectionItem


class CanvasView(QGraphicsView):
    # CHANGED: accepts network_vm instead of building hardcoded nodes.
    # The VM owns all graph state; the canvas only owns visuals.
    def __init__(self, network_vm):
        super().__init__()
        self.network_vm = network_vm

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))

        # ADDED: id → NodeItem lookup so signals can find the right visual node
        self._nodes: dict[str, NodeItem] = {}
        # ADDED: (from_id, to_id) → ConnectionItem lookup for removal
        self._connections: dict[tuple, ConnectionItem] = {}

        # ADDED: rubber-band line drawn while the user drags a new connection
        self._pending_line: QGraphicsLineItem | None = None
        self._drag_source_node: NodeItem | None = None

        # ADDED: wire NetworkVM signals → canvas slots
        self.network_vm.layer_added.connect(self._on_layer_added)
        self.network_vm.layer_removed.connect(self._on_layer_removed)
        self.network_vm.connection_added.connect(self._on_connection_added)
        self.network_vm.connection_removed.connect(self._on_connection_removed)
        self.network_vm.network_changed.connect(self._on_network_changed)

    # ── NetworkVM signal handlers ─────────────────────────────────────────────

    def _on_layer_added(self, layer_id: str, layer_type: str, name: str) -> None:
        # ADDED: creates the visual NodeItem when NetworkVM confirms a layer was added.
        # Position is auto-staggered so new nodes don't pile on top of each other.
        offset = len(self._nodes) * 180
        x = 60 + (offset % (180 * 4))
        y = 80 + (offset // (180 * 4)) * 120
        node = NodeItem(layer_id, name, layer_type, x, y)
        self.scene.addItem(node)
        self._nodes[layer_id] = node

    def _on_layer_removed(self, layer_id: str) -> None:
        # ADDED: removes the NodeItem when NetworkVM confirms deletion.
        node = self._nodes.pop(layer_id, None)
        if node:
            self.scene.removeItem(node)

    def _on_connection_added(self, from_id: str, to_id: str) -> None:
        # ADDED: draws a ConnectionItem between two existing NodeItems.
        from_node = self._nodes.get(from_id)
        to_node   = self._nodes.get(to_id)
        if from_node and to_node:
            conn = ConnectionItem(from_node, to_node)
            self.scene.addItem(conn)
            self._connections[(from_id, to_id)] = conn

    def _on_connection_removed(self, from_id: str, to_id: str) -> None:
        # ADDED: removes the visual line when NetworkVM confirms disconnection.
        conn = self._connections.pop((from_id, to_id), None)
        if conn:
            self.scene.removeItem(conn)

    def _on_network_changed(self) -> None:
        # ADDED: refreshes the output shape label on every node after any edit.
        # NetworkVM has already called propagate_shapes() internally.
        if self.network_vm.network is None:
            return
        for layer_id, node in self._nodes.items():
            layer = self.network_vm.get_layer(layer_id)
            if layer and layer._outputShape:
                node.update_shape_label(str(layer._outputShape))

    # ── canvas reset (called when a project is loaded) ────────────────────────

    def clear_canvas(self) -> None:
        # ADDED: wipes all visuals. Called by NetworkVM.set_network() indirectly —
        # MainWindow calls this before set_network() re-emits layer_added signals.
        self.scene.clear()
        self._nodes.clear()
        self._connections.clear()

    # ── mouse events for drag-to-connect ─────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        # ADDED: right-click on a node starts a connection drag.
        # Left-click is handled by Qt's built-in scroll/select drag mode.
        if event.button() == Qt.MouseButton.RightButton:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene.itemAt(scene_pos, self.transform())
            # Walk up to find the NodeItem (click might land on the text label child)
            while item and not isinstance(item, NodeItem):
                item = item.parentItem()
            if isinstance(item, NodeItem):
                self._drag_source_node = item
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                # Draw a rubber-band line from the node centre
                start = item.sceneBoundingRect().center()
                pen = QPen(QColor("#FFFFFF"), 1, Qt.PenStyle.DashLine)
                self._pending_line = self.scene.addLine(
                    start.x(), start.y(), start.x(), start.y(), pen
                )
                return
        # ADDED: left-click selects node and tells NetworkVM
        elif event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene.itemAt(scene_pos, self.transform())
            while item and not isinstance(item, NodeItem):
                item = item.parentItem()
            if isinstance(item, NodeItem):
                self.network_vm.select_layer(item.layer_id)
            else:
                self.network_vm.select_layer(None)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        # ADDED: updates the rubber-band line endpoint while dragging
        if self._pending_line and self._drag_source_node:
            scene_pos = self.mapToScene(event.pos())
            line = self._pending_line.line()
            self._pending_line.setLine(line.x1(), line.y1(),
                                       scene_pos.x(), scene_pos.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        # ADDED: if released on a different node, ask NetworkVM to add a connection.
        # NetworkVM validates it (cycle check, shape check) and emits
        # connection_added or operation_failed accordingly.
        if event.button() == Qt.MouseButton.RightButton and self._drag_source_node:
            if self._pending_line:
                self.scene.removeItem(self._pending_line)
                self._pending_line = None

            scene_pos = self.mapToScene(event.pos())
            item = self.scene.itemAt(scene_pos, self.transform())
            while item and not isinstance(item, NodeItem):
                item = item.parentItem()

            if isinstance(item, NodeItem) and item is not self._drag_source_node:
                self.network_vm.add_connection(
                    self._drag_source_node.layer_id, item.layer_id
                )

            self._drag_source_node = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            return

        super().mouseReleaseEvent(event)

    # ── keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        # ADDED: Delete key removes the selected node via NetworkVM.
        # NetworkVM also removes all its connections and emits the right signals.
        if event.key() == Qt.Key.Key_Delete:
            for item in self.scene.selectedItems():
                if isinstance(item, NodeItem):
                    self.network_vm.remove_layer(item.layer_id)
        super().keyPressEvent(event)