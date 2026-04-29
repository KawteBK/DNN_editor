from PyQt6.QtCore import QObject, pyqtSignal

from Core.project import Project
from Core.network import Network
from Core.layer import (
    Layer, InputLayer, Dense, Conv2D, Pooling, Recurrent,
    Flatten, Activation,
    ActivationType, PoolingType, RecurrentType, DropoutStyle,Merge, MergeMode
)


class NetworkVM(QObject):
    """
    Handles all network graph editing operations.

    The canvas (View) calls methods here instead of touching Core directly.
    This VM validates operations against the Core Network model and emits
    signals so the canvas and other docks stay in sync.

    Signals
    -------
    layer_added      — (layer_id: str, layer_type: str, name: str)
                       Canvas creates a NodeItem when it receives this.
    layer_removed    — (layer_id: str)
                       Canvas removes the matching NodeItem.
    connection_added — (from_id: str, to_id: str)
                       Canvas draws a ConnectionItem.
    connection_removed — (from_id: str, to_id: str)
                       Canvas removes the matching ConnectionItem.
    layer_selected   — (layer_id: str | None)
                       LayerConfigDock rebuilds its form for this layer.
    network_changed  — ()
                       General-purpose signal: anything that should trigger
                       a code regeneration or a validation badge update.
    operation_failed — (reason: str)
                       The attempted operation was rejected (e.g. cycle,
                       shape mismatch). MainWindow shows this as a status
                       bar message — not a blocking dialog.
    """

    layer_added        = pyqtSignal(str, str, str)   # id, type, name
    layer_removed      = pyqtSignal(str)
    connection_added   = pyqtSignal(str, str)
    connection_removed = pyqtSignal(str, str)
    layer_selected     = pyqtSignal(object)          # Layer | None
    network_changed    = pyqtSignal()
    operation_failed   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._network: Network | None = None
        self._selected_layer_id: str | None = None
        self._node_counter: dict[str, int] = {}   # type -> count, for auto IDs and stuff

    #  called by ProjectVM when a project is loaded / created 

    def set_network(self, network: Network | None) -> None:
        """
        Replaces the current network. Called when ProjectVM emits
        project_changed. The canvas should clear itself first, then
        re-add all nodes and connections from the new network.
        """
        self._network = network
        self._selected_layer_id = None
        self._node_counter.clear()

        if network is not None:
            # Seed counters from IDs already present in the loaded network so
            # the next add_layer() call never produces a duplicate ID.
            # IDs are expected to follow the pattern "<type>_<int>", e.g.
            # "dense_3", "conv2d_1".  Any ID that doesn't match is skipped
            # safely — it just won't influence the counter for that type.
            for layer_id in network.layers:                                     #added for appa
                parts = layer_id.rsplit("_", 1)                                 #added for appa
                if len(parts) == 2 and parts[1].isdigit():                      #added for appa
                    key   = parts[0]                                            #added for appa
                    index = int(parts[1])                                       #added for appa
                    if index > self._node_counter.get(key, 0):                  #added for appa
                        self._node_counter[key] = index                         #added for appa

            # Ensure shapes are propagated before visual nodes are created
            try:
               network.propagate_shapes()
            except Exception:
               pass  # Silent fail — shapes will propagate when network is complete
            for layer in network.topological_sort():
                self.layer_added.emit(
                    layer._id,
                    type(layer).__name__,
                    layer._name,
                )
            for conn in network.connections:
                self.connection_added.emit(conn._fromLayer, conn._toLayer)

    # ── layer operations (called by LayerPaletteDock / canvas) ───────────────

    def add_layer(self, layer_type: str) -> str | None:
        """
        Creates a Core layer of the given type with sensible defaults,
        adds it to the network, and emits layer_added.
        Returns the new layer's id, or None if the operation failed.

        layer_type must be one of:
            "Dense", "Conv2D", "Pooling", "Recurrent",
            "Flatten", "Activation"
        (InputLayer is created automatically when a project is created,
        not through the palette.)
        """
        if self._network is None:
            self.operation_failed.emit("No active network.")
            return None

        layer = self._build_default_layer(layer_type)
        if layer is None:
            self.operation_failed.emit(f"Unknown layer type: {layer_type}")
            return None

        success = self._network.addLayer(layer)
        if not success:
            self.operation_failed.emit(
                f"Layer id '{layer._id}' already exists in the network."
            )
            return None

        self.layer_added.emit(layer._id, layer_type, layer._name)
        self.network_changed.emit()
        return layer._id

    def remove_layer(self, layer_id: str) -> None:
        """
        Removes a layer and all its connections from the network.
        Emits layer_removed on success, operation_failed if the layer
        is locked (e.g. InputLayer) or not found.
        """
        if self._network is None:
            return

        # Snapshot connections involving this layer before removal
        affected_conns = [
            (c._fromLayer, c._toLayer)
            for c in self._network.connections
            if c._fromLayer == layer_id or c._toLayer == layer_id
        ]

        success = self._network.removeLayer(layer_id)
        if not success:
            if layer_id not in self._network.layers:
                self.operation_failed.emit(f"Layer '{layer_id}' not found.")
            else:
                self.operation_failed.emit(
                    f"Layer '{layer_id}' is locked and cannot be removed."
                )
            return

        # Notify canvas to remove affected connection visuals
        for from_id, to_id in affected_conns:
            self.connection_removed.emit(from_id, to_id)

        self.layer_removed.emit(layer_id)

        if self._selected_layer_id == layer_id:
            self._selected_layer_id = None
            self.layer_selected.emit(None)

        self.network_changed.emit()

    # ── connection operations (called by canvas drag) ─────────────────────────

    def add_connection(self, from_id: str, to_id: str) -> bool:
        """
        Attempts to add a connection. Returns True on success.
        Emits connection_added or operation_failed accordingly.
        """
        if self._network is None:
            return False

        success = self._network.addConnect(from_id, to_id)
        if success:
            self.connection_added.emit(from_id, to_id)
            self.network_changed.emit()
        else:
            # Give the user a useful reason
            if from_id == to_id:
                reason = "A layer cannot connect to itself."
            elif to_id in self._network._outgoing.get(from_id, []):
                reason = "Connection already exists."
            elif to_id in self._network.layers and self._network.layers[to_id]._isInput:
                reason = "Cannot connect to the Input layer."
            else:
                reason = (
                    "Connection rejected — this would create a cycle "
                    "or an incompatible shape."
                )
            self.operation_failed.emit(reason)

        return success

    def remove_connection(self, from_id: str, to_id: str) -> None:
        if self._network is None:
            return
        success = self._network.removeConnect(from_id, to_id)
        if success:
            self.connection_removed.emit(from_id, to_id)
            self.network_changed.emit()

    # ── selection (called by canvas click) ────────────────────────────────────

    def select_layer(self, layer_id: str | None) -> None:
        """
        Called when the user clicks a node on the canvas.
        Emits layer_selected with the actual Layer object so
        LayerConfigDock can build the right form fields.
        """
        self._selected_layer_id = layer_id
        if layer_id is None or self._network is None:
            self.layer_selected.emit(None)
        else:
            layer = self._network.layers.get(layer_id)
            self.layer_selected.emit(layer)

    # ── layer config update (called by LayerConfigDock) ───────────────────────

    def update_layer_params(self, layer_id: str, params: dict) -> None:
        """
        Applies parameter changes from LayerConfigDock to the Core layer.
        `params` is a flat dict of field_name → value, e.g.:
            {"unit": 128, "useBias": True, "dropout": 0.5}

        After updating, re-propagates shapes so output shape labels
        on nodes stay accurate.
        """
        if self._network is None:
            return
        layer = self._network.layers.get(layer_id)
        if layer is None:
            return
       # if it's an activation layer, notify canvas
        if layer and isinstance(layer, Activation):
            if "activationType" in params:
                # Emit to canvas to update label
                 self.network_changed.emit()  # Will trigger refresh

        for key, value in params.items():
            attr = f"_{key}"
            if hasattr(layer, attr):
                setattr(layer, attr, value)

        # Re-propagate shapes — a unit count change affects all downstream layers
        try:
            self._network.propagate_shapes()
        except Exception:
            pass   # shape errors are shown passively; don't block the edit

        self.network_changed.emit()

    # ── read-only access for docks ────────────────────────────────────────────

    @property
    def network(self) -> Network | None:
        return self._network

    def get_layer(self, layer_id: str) -> Layer | None:
        if self._network is None:
            return None
        return self._network.layers.get(layer_id)

    def is_valid(self) -> bool:
        return self._network is not None and self._network.validate()

    # ── private helpers ───────────────────────────────────────────────────────

    def _next_id(self, layer_type: str) -> str:
        """Generates sequential IDs: dense_1, dense_2, conv2d_1, ..."""
        key = layer_type.lower()
        self._node_counter[key] = self._node_counter.get(key, 0) + 1
        return f"{key}_{self._node_counter[key]}"

    def _build_default_layer(self, layer_type: str) -> Layer | None:
        """
        Constructs a Core layer with sensible defaults for each type.
        These defaults are what the user sees when they first drop a layer
        onto the canvas — they can change them in LayerConfigDock.
        """
        lid = self._next_id(layer_type)
        name = f"{layer_type} {self._node_counter[layer_type.lower()]}"

        if layer_type == "Dense":
            return Dense(unit=128, useBias=True, dropout=0.0,
                         id=lid, name=name, role="hidden")

        elif layer_type == "Conv2D":
            return Conv2D(filters=32, kernelSize=(3, 3), strides=(1, 1),
                          padding=(1, 1), dropout=0.0, dropoutStyle=None,
                          id=lid, name=name, role="hidden")

        elif layer_type == "Pooling":
            return Pooling(poolSize=(2, 2), strides=(2, 2), padding=(0, 0),
                           id=lid, name=name, role="hidden",
                           poolingType=PoolingType.MAX)

        elif layer_type == "Recurrent":
            return Recurrent(hiddenSize=64, numLayer=1, dropout=0.0,
                             id=lid, name=name, role="hidden",
                             bidirection=False, returnSeq=False,
                             recurrentType=RecurrentType.LSTM)

        elif layer_type == "Flatten":
            return Flatten(id=lid, name=name, role="hidden")

        elif layer_type == "Activation":
            return Activation(activationType=ActivationType.RELU,
                              id=lid, name=name, role="hidden")

        elif layer_type == "Merge":
             return Merge(
            id=lid, 
            name=name, 
            role="merge",
            mergeMode=MergeMode.ADD  # Default to ADD
    )

        return None