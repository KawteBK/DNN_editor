from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen
from PyQt6.QtCore import Qt, QPointF

from canvas.node_item import NodeItem
from canvas.connection_item import ConnectionItem
from Core.layer import Activation

# ADDED: mime type constant — must match the one in layer_palette_dock.py
LAYER_MIME_TYPE = "application/x-nn-layer-type"


class CanvasView(QGraphicsView):
    # CHANGED: accepts network_vm instead of building hardcoded nodes.
    # The VM owns all graph state; the canvas only owns visuals.
    #
    # ADDED: drag-and-drop support from LayerPaletteDock.
    #   dragEnterEvent  — accepts drops carrying LAYER_MIME_TYPE.
    #   dragMoveEvent   — moves a semi-transparent ghost NodeItem under the cursor.
    #   dragLeaveEvent  — removes the ghost when the cursor leaves the canvas.
    #   dropEvent       — finalises the drop: calls network_vm.add_layer() then
    #                     immediately repositions the spawned node to the drop point
    #                     rather than the default grid cell.
    def __init__(self, network_vm):
        super().__init__()
        self.network_vm = network_vm

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))

        # ADDED: enable drop events on this widget
        self.setAcceptDrops(True)   # ADDED

        # ADDED: id → NodeItem lookup so signals can find the right visual node
        self._nodes: dict[str, NodeItem] = {}
        # ADDED: (from_id, to_id) → ConnectionItem lookup for removal
        self._connections: dict[tuple, ConnectionItem] = {}

        # Monotonically increasing spawn counter — never reset on individual
        # node deletions, so a freshly added layer never reuses a slot occupied
        # by a surviving node.  It IS reset to 0 by clear_canvas() so that
        # nodes in a freshly loaded project are placed from cell (0, 0).
        # Pass 1 of _next_spawn_position() guards against overlaps even after a
        # reset by scanning for actually-free cells before committing.
        self._spawn_counter: int = 0

        # ── connect mode state ────────────────────────────────────────────────
        # Two-click flow: click node A (stored in _connect_first_node, gets a
        # cyan ring), then click node B → connection fires automatically.
        # Right-click drag is unchanged and works regardless of mode.
        self._connect_mode: bool = False
        self._connect_first_node: NodeItem | None = None
        self._hovered_node: NodeItem | None = None

        # rubber-band line drawn while the user right-click-drags a connection
        self._pending_line: QGraphicsLineItem | None = None
        self._drag_source_node: NodeItem | None = None

        # ── ADDED: palette drag state ─────────────────────────────────────────
        # _ghost_node is a temporary NodeItem rendered at 50% opacity that
        # follows the cursor during a palette drag so the user can see where
        # the new layer will land.  It is never registered in self._nodes and
        # is removed as soon as the drag ends (drop or cancel).
        self._ghost_node: NodeItem | None = None           # ADDED
        self._ghost_layer_type: str | None = None          # ADDED — type being dragged

        # ADDED: wire NetworkVM signals → canvas slots
        self.network_vm.layer_added.connect(self._on_layer_added)
        self.network_vm.layer_removed.connect(self._on_layer_removed)
        self.network_vm.connection_added.connect(self._on_connection_added)
        self.network_vm.connection_removed.connect(self._on_connection_removed)
        self.network_vm.network_changed.connect(self._on_network_changed)

    # ── connect mode public API ───────────────────────────────────────────────

    def set_connect_mode(self, active: bool) -> None:
        """
        Toggle the two-click connect mode from MainWindow's Connect button.
        When active:
          - First left-click on a node locks it as source (cyan ring stays on).
          - Second left-click on any other node fires the connection.
          - Clicking the same node again or clicking empty space resets the
            first selection so the user can pick again.
          - Escape (or toggling the button off) cancels everything.
        Right-click drag still works independently regardless of mode.
        """
        self._connect_mode = active
        if active:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._cancel_connect_selection()

    def _cancel_connect_selection(self) -> None:
        """Clear the first-click selection without creating a connection."""
        if self._connect_first_node is not None:
            self._connect_first_node.set_connect_hover(False)
            self._connect_first_node = None

    def _clear_hover_highlight(self) -> None:
        """Remove the transient hover ring (never clears the locked source node)."""
        if self._hovered_node is not None:
            if self._hovered_node is not self._connect_first_node:
                self._hovered_node.set_connect_hover(False)
            self._hovered_node = None

    # ── palette drag-and-drop ─────────────────────────────────────────────────
    # These three methods implement the receive side of the drag that
    # _DraggableLayerButton initiates in layer_palette_dock.py.

    def dragEnterEvent(self, event) -> None:                              # ADDED
        """
        ADDED: Accept the drag only if it carries our custom mime type.
        This guards against accepting file-system drags or other Qt drags.
        Also creates the ghost NodeItem at the entry position.
        """
        if event.mimeData().hasFormat(LAYER_MIME_TYPE):
            event.acceptProposedAction()
            layer_type = event.mimeData().data(LAYER_MIME_TYPE).data().decode()
            self._show_ghost(layer_type, self.mapToScene(event.position().toPoint()))
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:                               # ADDED
        """
        ADDED: Keep accepting and move the ghost node to follow the cursor.
        Qt requires dragMoveEvent to be accepted continuously or the drop
        cursor reverts to the "forbidden" icon.
        """
        if event.mimeData().hasFormat(LAYER_MIME_TYPE):
            event.acceptProposedAction()
            scene_pos = self.mapToScene(event.position().toPoint())
            self._move_ghost(scene_pos)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:                              # ADDED
        """ADDED: Cursor left the canvas without dropping — remove ghost."""
        self._remove_ghost()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:                                   # ADDED
        """
        ADDED: The user released the button over the canvas.

        Steps:
        1. Decode the layer type from mime data.
        2. Remove the ghost (it will be replaced by the real node).
        3. Record the scene-space drop position BEFORE calling add_layer(),
           because _on_layer_added() uses _next_spawn_position() to place
           the node in the grid by default.
        4. Call network_vm.add_layer() — this fires layer_added which calls
           _on_layer_added() synchronously (Qt direct connection), so the
           new NodeItem is already in self._nodes by the time we reach step 5.
        5. Grab the newest node (last item added) and move it to the drop
           position so the layer lands exactly where the user dropped it.
        """
        if not event.mimeData().hasFormat(LAYER_MIME_TYPE):
            event.ignore()
            return

        event.acceptProposedAction()

        layer_type = event.mimeData().data(LAYER_MIME_TYPE).data().decode()
        drop_scene_pos = self.mapToScene(event.position().toPoint())

        self._remove_ghost()   # ADDED — clean up ghost before real node appears

        # Remember how many nodes exist before the add so we can find the new one
        ids_before = set(self._nodes.keys())

        # network_vm.add_layer() → layer_added signal → _on_layer_added()
        # all happens synchronously in the same call stack.
        self.network_vm.add_layer(layer_type)

        # Find the newly created node (the one whose id wasn't there before)
        new_ids = set(self._nodes.keys()) - ids_before
        if new_ids:
            new_id   = next(iter(new_ids))
            new_node = self._nodes[new_id]

            # Centre the node on the cursor's drop point                    # ADDED
            from canvas.node_item import NODE_WIDTH, NODE_HEIGHT
            new_node.setPos(
                drop_scene_pos.x() - NODE_WIDTH  / 2,
                drop_scene_pos.y() - NODE_HEIGHT / 2,
            )

    # ── ghost helpers ─────────────────────────────────────────────────────────

    def _show_ghost(self, layer_type: str, scene_pos: QPointF) -> None:   # ADDED
        """
        ADDED: Create a semi-transparent NodeItem and add it to the scene.
        It is NOT registered in self._nodes so it is invisible to the VM.
        """
        self._remove_ghost()   # safety: never have two ghosts at once
        from canvas.node_item import NODE_WIDTH, NODE_HEIGHT
        ghost = NodeItem(
            layer_id   = "__ghost__",   # sentinel id — never sent to VM
            title      = layer_type,
            layer_type = layer_type,
            x          = scene_pos.x() - NODE_WIDTH  / 2,
            y          = scene_pos.y() - NODE_HEIGHT / 2,
        )
        ghost.setOpacity(0.50)          # ADDED — half-transparent ghost
        ghost.setFlag(NodeItem.GraphicsItemFlag.ItemIsMovable,   False)
        ghost.setFlag(NodeItem.GraphicsItemFlag.ItemIsSelectable, False)
        ghost.setZValue(100)            # ADDED — always on top during drag
        self.scene.addItem(ghost)
        self._ghost_node        = ghost
        self._ghost_layer_type  = layer_type

    def _move_ghost(self, scene_pos: QPointF) -> None:                    # ADDED
        """ADDED: Reposition the ghost to stay centred under the cursor."""
        if self._ghost_node is None:
            return
        from canvas.node_item import NODE_WIDTH, NODE_HEIGHT
        self._ghost_node.setPos(
            scene_pos.x() - NODE_WIDTH  / 2,
            scene_pos.y() - NODE_HEIGHT / 2,
        )

    def _remove_ghost(self) -> None:                                       # ADDED
        """ADDED: Remove the ghost from the scene and release the reference."""
        if self._ghost_node is not None:
            self.scene.removeItem(self._ghost_node)
            self._ghost_node       = None
            self._ghost_layer_type = None

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

        NOTE: the ghost node is NOT in self._nodes so it never pollutes the
        occupied-cell scan — the real node still picks a proper grid slot,
        which dropEvent() then overrides with the exact drop position.
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
            # connection_removed signals fire before layer_removed (per network_vm),
            # so all ConnectionItems are already gone from the scene by now.
            # Clear the node's own connections list to drop the stale references
            # and prevent memory leaks or crashes if anything iterates it later.
            node.connections.clear()
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
        self._remove_ghost()   # ADDED — don't leave a ghost behind on project load
        self.scene.clear()
        self._nodes.clear()
        self._connections.clear()
        self._spawn_counter = 0   # reset so loaded projects start from cell (0,0)

    # ── mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        # ── right-click: always starts a rubber-band drag (power-user flow) ──
        if event.button() == Qt.MouseButton.RightButton:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene.itemAt(scene_pos, self.transform())
            while item and not isinstance(item, NodeItem):
                item = item.parentItem()
            if isinstance(item, NodeItem):
                self._drag_source_node = item
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                start = item.sceneBoundingRect().center()
                pen = QPen(QColor("#00E5FF"), 2, Qt.PenStyle.DashLine)
                self._pending_line = self.scene.addLine(
                    start.x(), start.y(), start.x(), start.y(), pen
                )
                return

        # ── left-click in connect mode: two-click state machine ───────────────
        elif event.button() == Qt.MouseButton.LeftButton and self._connect_mode:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene.itemAt(scene_pos, self.transform())
            while item and not isinstance(item, NodeItem):
                item = item.parentItem()

            if self._connect_first_node is None:
                # ── STATE 1: no source yet — lock this node as source ──────────
                if isinstance(item, NodeItem):
                    self._connect_first_node = item
                    item.set_connect_hover(True)   # stays lit until resolved
                    # Tell MainWindow to update the status bar hint
                    main = self.window()
                    if hasattr(main, "statusBar"):
                        main.statusBar().showMessage(
                            f"Source: '{item.layer_id}' — now click the target node.  "
                            "Click the same node or empty space to cancel.", 0
                        )
                # clicking empty space while no source selected → nothing to do
            else:
                # ── STATE 2: source is locked — resolve the second click ────────
                if isinstance(item, NodeItem) and item is not self._connect_first_node:
                    # Valid target — fire the connection
                    self.network_vm.add_connection(
                        self._connect_first_node.layer_id, item.layer_id
                    )
                    self._cancel_connect_selection()
                else:
                    # Clicked same node or empty space → cancel selection
                    self._cancel_connect_selection()
                    main = self.window()
                    if hasattr(main, "statusBar"):
                        main.statusBar().showMessage(
                            "Selection cleared — click a source node to start again.", 3000
                        )
            return   # don't pass to super — we own this click in connect mode

        # ── left-click in normal mode: select node ────────────────────────────
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
        # Only active during right-click rubber-band drag.
        if self._pending_line and self._drag_source_node:
            scene_pos = self.mapToScene(event.pos())
            line = self._pending_line.line()
            self._pending_line.setLine(line.x1(), line.y1(),
                                       scene_pos.x(), scene_pos.y())

            # Highlight the node under the cursor as a drop target
            item = self.scene.itemAt(scene_pos, self.transform())
            while item and not isinstance(item, NodeItem):
                item = item.parentItem()
            target = item if isinstance(item, NodeItem) else None
            if target is not self._hovered_node:
                self._clear_hover_highlight()
                if target is not None and target is not self._drag_source_node:
                    target.set_connect_hover(True)
                    self._hovered_node = target
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        # Only right-click drag ends here — two-click connect mode is handled
        # entirely inside mousePressEvent, no release needed.
        if event.button() == Qt.MouseButton.RightButton and self._drag_source_node:
            if self._pending_line:
                self.scene.removeItem(self._pending_line)
                self._pending_line = None

            self._clear_hover_highlight()

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
        # Key_Delete = Windows/Linux "Delete" key.
        # Key_Backspace = macOS "Delete" key (the only delete key on most Mac keyboards).
        # Both must be handled so the shortcut works cross-platform.
        _DELETE_KEYS = (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
        if event.key() in _DELETE_KEYS:
            main = self.window()
            if hasattr(main, "_on_delete_selected"):
                main._on_delete_selected()

        elif event.key() == Qt.Key.Key_Escape and self._connect_mode:
            # If a source node is already selected, just cancel the selection
            # without exiting connect mode — lets the user re-pick.
            if self._connect_first_node is not None:
                self._cancel_connect_selection()
                main = self.window()
                if hasattr(main, "statusBar"):
                    main.statusBar().showMessage(
                        "Selection cleared — click a source node to start again.", 3000
                    )
            else:
                # No selection pending — Escape exits connect mode entirely
                main = self.window()
                if hasattr(main, "btn_connect"):
                    main.btn_connect.setChecked(False)
                    main._on_toggle_connect_mode()

        super().keyPressEvent(event)

    # ── training animation ────────────────────────────────────────────────────

    def start_training_animation(self):
        """Start animation on all connections."""
        for conn in self._connections.values():
            conn.start_training_animation()

    def stop_training_animation(self):
        """Stop animation on all connections."""
        for conn in self._connections.values():
            conn.stop_training_animation()