from .layer import (Activation, ActivationType, Conv2D, DropoutStyle, Dense, Flatten, InputLayer, Layer,
                    Merge, MergeMode, Pooling, PoolingType, Recurrent, RecurrentType)
from .connection import Connection
from collections import deque


class Network:
    def __init__(self, id: str):
        self.id = id
        self.layers: dict[str, Layer] = {}
        self.connections: list[Connection] = []

        self._outgoing: dict[str, list[str]] = {}  # layerId → [children ids]
        self._incoming: dict[str, list[str]] = {}  # layerId → [parent ids]

    def addLayer(self, layer: Layer):
        if layer._id in self.layers:
            return False
        self.layers[layer._id] = layer
        self._outgoing[layer._id] = []
        self._incoming[layer._id] = []
        return True

    def removeLayer(self, layerId: str):
        if layerId not in self.layers:
            return False
        if self.layers[layerId]._isLocked:
            return False

        del self.layers[layerId]
        for child in self._outgoing[layerId]:
            self._incoming[child].remove(layerId)
        for parent in self._incoming[layerId]:
            self._outgoing[parent].remove(layerId)

        self.connections = [
            conn for conn in self.connections
            if conn._fromLayer != layerId and conn._toLayer != layerId
        ]
        del self._outgoing[layerId]
        del self._incoming[layerId]
        return True

    def addConnect(self, fromLayerId: str, toLayerId: str):
        if fromLayerId not in self.layers or toLayerId not in self.layers:
            return False
        if fromLayerId == toLayerId:
            return False
        if toLayerId in self._outgoing[fromLayerId]:
            return False
        if self.layers[toLayerId]._isInput:
            return False

        self._outgoing[fromLayerId].append(toLayerId)
        self._incoming[toLayerId].append(fromLayerId)

        connection = Connection(f"{fromLayerId}->{toLayerId}", fromLayerId, toLayerId)
        self.connections.append(connection)

        try:
            # Validate DAG (raises on cycle)
            self.topological_sort()
            # Re-propagate shapes for the whole graph now that topology changed.
            # propagate_shapes() handles both single- and multi-parent layers,
            # so this is safe for non-linear (functional-API) graphs too.
            self.propagate_shapes()
        except Exception:
            self._outgoing[fromLayerId].remove(toLayerId)
            self._incoming[toLayerId].remove(fromLayerId)
            self.connections.pop()
            return False

        return True

    def removeConnect(self, fromLayerId: str, toLayerId: str):
        if fromLayerId not in self.layers or toLayerId not in self.layers:
            return False
        if toLayerId not in self._outgoing[fromLayerId]:
            return False

        self._outgoing[fromLayerId].remove(toLayerId)
        self._incoming[toLayerId].remove(fromLayerId)

        self.connections = [
            conn for conn in self.connections
            if not (conn._fromLayer == fromLayerId and conn._toLayer == toLayerId)
        ]
        return True

    def getInputLayers(self) -> list[Layer]:
        return [self.layers[lid] for lid, parents in self._incoming.items() if not parents]

    def getOutputLayers(self):
        return [self.layers[lid] for lid, children in self._outgoing.items() if not children]

    def get_parents(self, layerId):
        return self._incoming.get(layerId, [])

    def get_children(self, layerId):
        return self._outgoing.get(layerId, [])

    def topological_sort(self):
        in_degree = {lid: len(parents) for lid, parents in self._incoming.items()}
        queue = deque([lid for lid, deg in in_degree.items() if deg == 0])
        sorted_layers = []

        while queue:
            current = queue.popleft()
            sorted_layers.append(self.layers[current])
            for child in self._outgoing[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_layers) != len(self.layers):
            raise Exception("Cycle detected in network")

        return sorted_layers

    def propagate_shapes(self):
        """
        Propagates output shapes through the graph.

        Supports both:
          - Single-parent layers (all existing types)
          - Multi-parent layers (Merge/Add/Concatenate) for non-linear graphs
        """
        ordered = self.topological_sort()

        for layer in ordered:
            if layer._isInput:
                continue

            parents = self.get_parents(layer._id)
            if len(parents) == 0:
                continue

            # ── multi-parent path (non-linear graphs, e.g. Merge layers) ──────
            if len(parents) > 1:
                parent_shapes = []
                for pid in parents:
                    parent_layer = self.layers[pid]
                    if parent_layer._outputShape is None:
                        raise Exception(
                            f"Parent layer '{pid}' has undefined output shape"
                        )
                    parent_shapes.append(parent_layer._outputShape)
                layer._inputShape = parent_shapes[0]   # convention: first parent's shape
                layer._outputShape = layer.getOutputShapeMulti(parent_shapes)

            # ── single-parent path (original behaviour) ───────────────────────
            else:
                parent_layer = self.layers[parents[0]]
                if parent_layer._outputShape is None:
                    raise Exception(
                        f"Parent layer '{parent_layer._id}' has undefined output shape"
                    )
                layer._inputShape = parent_layer._outputShape
                layer._outputShape = layer.getOutputShape(parent_layer._outputShape)

    def validate(self) -> bool:
        """
        Returns True only if the network is structurally valid AND
        has no unconnected (puppy) layers that would silently break training.
        """
        if not self.layers:
            return False

        try:
            self.topological_sort()
        except Exception:
            return False

        has_input  = any(not parents  for parents  in self._incoming.values())
        has_output = any(not children for children in self._outgoing.values())

        if not (has_input and has_output):
            return False

        # ── puppy-layer check: no floating nodes allowed ───────────────────────
        # A puppy layer has no connections at all (neither incoming nor outgoing)
        # but is not the sole node in the network. These silently never train.
        if len(self.layers) > 1:
            for lid in self.layers:
                if not self._incoming[lid] and not self._outgoing[lid]:
                    return False   # unconnected node — reject

        return True

    def get_orphaned_layers(self) -> list[str]:
        """
        Returns ids of layers that are completely unconnected
        (no parents AND no children) and are not input layers.
        Used by the UI to highlight orphan / puppy layers before training.
        """
        orphans = []
        for lid, layer in self.layers.items():
            if layer._isInput:
                continue
            if not self._incoming[lid] and not self._outgoing[lid]:
                orphans.append(lid)
        return orphans

    def generateSummary(self) -> dict[str, str]:
        return {
            "Network ID":        self.id,
            "Total Layers":      str(len(self.layers)),
            "Total Connections": str(len(self.connections)),
            "Input Layers":      ", ".join(l._id for l in self.getInputLayers()),
            "Output Layers":     ", ".join(l._id for l in self.getOutputLayers()),
        }

    # ── serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def fromDict(cls, data: dict) -> "Network":
        if not data:
            return None

        network = cls(id=data["id"])

        def to_tuple(val):
            return tuple(val) if val is not None else None

        for lid, ldata in data["layers"].items():
            layer_type = ldata["type"]

            if layer_type == "InputLayer":
                layer = InputLayer(inputShape=to_tuple(ldata["outputShape"]))

            elif layer_type == "Flatten":
                layer = Flatten(id=ldata["id"], name=ldata["name"], role=ldata["role"])

            elif layer_type == "Dense":
                layer = Dense(unit=ldata["unit"], useBias=ldata["useBias"],
                              dropout=ldata["dropout"], id=ldata["id"],
                              name=ldata["name"], role=ldata["role"])

            elif layer_type == "Conv2D":
                raw_ds = ldata.get("dropoutStyle")
                dropout_style = DropoutStyle(raw_ds) if raw_ds else None
                layer = Conv2D(
                    filters=ldata["filters"],
                    kernelSize=tuple(ldata["kernelSize"]),
                    strides=tuple(ldata["strides"]),
                    padding=tuple(ldata["padding"]),
                    dropout=ldata["dropout"],
                    dropoutStyle=dropout_style,
                    id=ldata["id"], name=ldata["name"], role=ldata["role"],
                )

            elif layer_type == "Recurrent":
                layer = Recurrent(
                    hiddenSize=ldata["hiddenSize"],
                    numLayer=ldata["numLayer"],
                    dropout=ldata["dropout"],
                    id=ldata["id"], name=ldata["name"], role=ldata["role"],
                    bidirection=ldata["bidirection"],
                    returnSeq=ldata["returnSeq"],
                    recurrentType=RecurrentType(ldata.get("recurrentType", "lstm")),
                )

            elif layer_type == "Pooling":
                layer = Pooling(
                    poolSize=tuple(ldata["poolSize"]),
                    strides=tuple(ldata["strides"]),
                    padding=tuple(ldata["padding"]),
                    id=ldata["id"], name=ldata["name"], role=ldata["role"],
                    poolingType=PoolingType(ldata.get("poolingType", "max")),
                )

            elif layer_type == "Activation":
                layer = Activation(
                    activationType=ActivationType(ldata["activationType"]),
                    id=ldata["id"], name=ldata["name"], role=ldata["role"]
                )

            elif layer_type == "Merge":
                layer = Merge(
                    id=ldata["id"], name=ldata["name"], role=ldata["role"],
                    mergeMode=MergeMode(ldata.get("mergeMode", "add")),
                )

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            if layer_type != "InputLayer":
                layer._inputShape  = to_tuple(ldata.get("inputShape"))
                layer._outputShape = to_tuple(ldata.get("outputShape"))

            network.addLayer(layer)

        # ── Phase 2: wire up connections directly (bypass addConnect's
        # incremental propagation so that non-linear graphs with multi-parent
        # Merge layers don't fail on partially-built topologies).
        # A single propagate_shapes() call at the end handles everything.
        from .connection import Connection as _Conn
        for cdata in data["connections"]:
            from_id, to_id = cdata["from"], cdata["to"]
            if from_id not in network.layers or to_id not in network.layers:
                continue
            if to_id in network._outgoing[from_id]:
                continue  # duplicate edge in serialised data — skip
            network._outgoing[from_id].append(to_id)
            network._incoming[to_id].append(from_id)
            network.connections.append(_Conn(f"{from_id}->{to_id}", from_id, to_id))

        # Re-derive shapes from scratch so they're always consistent.
        try:
            network.propagate_shapes()
        except Exception:
            # Shapes may be incomplete (e.g. partially configured network saved
            # mid-edit). The network is still usable for UI display; the trainer
            # will catch shape errors before training starts.
            pass

        return network

    def toDict(self) -> dict:
        return {
            "id": self.id,
            "layers": {lid: self._serializeLayer(layer) for lid, layer in self.layers.items()},
            "connections": [
                {"id": conn._id, "from": conn._fromLayer, "to": conn._toLayer}
                for conn in self.connections
            ]
        }

    def _serializeLayer(self, layer) -> dict:
        base = {
            "id":          layer._id,
            "name":        layer._name,
            "role":        layer._role,
            "type":        type(layer).__name__,
            "inputShape":  layer._inputShape,
            "outputShape": layer._outputShape,
        }

        if isinstance(layer, InputLayer):
            pass

        elif isinstance(layer, Flatten):
            pass

        elif isinstance(layer, Dense):
            base.update({
                "unit":    layer._unit,
                "useBias": layer._useBias,
                "dropout": layer._dropout,
            })

        elif isinstance(layer, Conv2D):
            base.update({
                "filters":      layer._filters,
                "kernelSize":   layer._kernelSize,
                "strides":      layer._strides,
                "padding":      layer._padding,
                "dropout":      layer._dropout,
                "dropoutStyle": layer._dropoutStyle.value if layer._dropoutStyle else None,
            })

        elif isinstance(layer, Recurrent):
            base.update({
                "hiddenSize":    layer._hiddenSize,
                "numLayer":      layer._numLayer,
                "dropout":       layer._dropout,
                "bidirection":   layer._bidirection,
                "returnSeq":     layer._returnSeq,
                "recurrentType": layer._recurrentType.value,
            })

        elif isinstance(layer, Pooling):
            base.update({
                "poolSize":    layer._poolSize,
                "strides":     layer._strides,
                "padding":     layer._padding,
                "poolingType": layer._poolingType.value,
            })

        elif isinstance(layer, Activation):
            base.update({
                "activationType": layer._activationType.value,
            })

        elif isinstance(layer, Merge):
            base.update({
                "mergeMode": layer._mergeMode.value,
            })

        return base