from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen
from PyQt6.QtCore import Qt, QPointF

from canvas.node_item import NodeItem
from canvas.connection_item import ConnectionItem
from Core.layer import Activation


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

        # Monotonically increasing spawn counter — never reset even when nodes
        # are deleted, so a freshly added layer never reuses a slot occupied by
        # a surviving node.
        self._spawn_counter: int = 0

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
        # Creates the visual NodeItem when NetworkVM confirms a layer was added.
        x, y = self._next_spawn_position()
        node = NodeItem(layer_id, name, layer_type, x, y)
        self.scene.addItem(node)
        self._nodes[layer_id] = node
        self._spawn_counter += 1

    # ── spawn position helpers ────────────────────────────────────────────────

    # Padding around each cell so nodes never touch each other.
    _CELL_PAD_X = 30   # horizontal gap between columns
    _CELL_PAD_Y = 30   # vertical gap between rows
    # How many columns before wrapping to the next row.
    # Wide enough for a typical canvas; can be changed without touching anything else.
    _GRID_COLS  = 4

    def _next_spawn_position(self) -> tuple[float, float]:
        """
        Return (x, y) for the next new node using a two-pass strategy:

        Pass 1 — find the first grid cell (in reading order) that is free,
                  i.e. no existing node's bounding rect overlaps it.
        Pass 2 — if every cell up to the current spawn count is occupied
                  (unlikely but possible after many add/delete cycles),
                  fall back to appending a new cell beyond all existing ones.

        Grid origin is (60, 60).  Cell size = node footprint + padding.
        """
        from canvas.node_item import NODE_WIDTH, NODE_HEIGHT

        cell_w = NODE_WIDTH  + self._CELL_PAD_X
        cell_h = NODE_HEIGHT + self._CELL_PAD_Y
        origin_x, origin_y = 60, 60

        # Collect bounding rects of every existing node (scene coordinates).
        occupied = [n.sceneBoundingRect() for n in self._nodes.values()]

        def _cell_xy(index: int) -> tuple[float, float]:
            col = index % self._GRID_COLS
            row = index // self._GRID_COLS
            return origin_x + col * cell_w, origin_y + row * cell_h

        def _is_free(cx: float, cy: float) -> bool:
            """True when no existing node overlaps the candidate cell."""
            from PyQt6.QtCore import QRectF
            candidate = QRectF(cx, cy, NODE_WIDTH, NODE_HEIGHT)
            return not any(candidate.intersects(r) for r in occupied)

        # Pass 1: scan grid cells in reading order up to spawn_counter + slack.
        search_limit = max(self._spawn_counter + 1, len(self._nodes) + self._GRID_COLS)
        for i in range(search_limit):
            cx, cy = _cell_xy(i)
            if _is_free(cx, cy):
                return cx, cy

        # Pass 2: just append after the last used cell index.
        return _cell_xy(self._spawn_counter)

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
            if layer:
                if layer._outputShape:
                    node.update_shape_label(str(layer._outputShape))
                    # Update activation type if applicable
                if isinstance(layer, Activation) and hasattr(node, 'update_activation_type'):
                    node.update_activation_type(layer._activationType.value)

            
    # ── canvas reset (called when a project is loaded) ────────────────────────

    def clear_canvas(self) -> None:
        # Wipes all visuals. Called by MainWindow before set_network() re-emits
        # layer_added signals for the freshly loaded project.
        self.scene.clear()
        self._nodes.clear()
        self._connections.clear()
        self._spawn_counter = 0   # reset so loaded projects start from cell (0,0)

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
        # Delete key removes selected nodes AND selected connections.
        if event.key() == Qt.Key.Key_Delete:
            for item in self.scene.selectedItems():
                if isinstance(item, NodeItem):
                    self.network_vm.remove_layer(item.layer_id)
                elif isinstance(item, ConnectionItem):
                    self.network_vm.remove_connection(
                        item.start_node.layer_id,
                        item.end_node.layer_id,
                    )
        super().keyPressEvent(event)

        #for the animated connection

    def start_training_animation(self):
        """Start animation on all connections."""
        for conn in self._connections.values():
            conn.start_training_animation()

    def stop_training_animation(self):
        """Stop animation on all connections."""
        for conn in self._connections.values():
            conn.stop_training_animation()