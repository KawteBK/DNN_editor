import torch
import torch.nn as nn

from Core.layer import (
    Activation, ActivationType,
    Conv2D, DropoutStyle,
    Dense,
    Flatten,
    InputLayer, Merge, MergeMode,
    Pooling, PoolingType,
    Recurrent, RecurrentType,
)
from Core.network import Network


# ── Inner wrappers ────────────────────────────────────────────────────────────

class RecurrentWrapper(nn.Module):
    """
    Wraps nn.LSTM or nn.GRU so it can live inside a sequential graph.

    LSTM/GRU return (output, hidden_state).  This wrapper unpacks the tuple
    and applies returnSeq logic so downstream layers always receive a plain
    tensor:
        returnSeq=True  → [batch, seq_len, hidden]
        returnSeq=False → [batch, hidden]   (last timestep only)
    """
    def __init__(self, rnn: nn.Module, returnSeq: bool):
        super().__init__()
        self.rnn = rnn
        self.returnSeq = returnSeq

    def forward(self, x):
        output, _ = self.rnn(x)
        return output if self.returnSeq else output[:, -1, :]


class GraphModule(nn.Module):
    """
    A custom nn.Module that executes a Network graph in topological order,
    supporting both linear (sequential) AND non-linear (Merge/skip-connection)
    topologies.

    How it works
    ────────────
    At construction time we receive:
      • node_modules  — dict[layer_id → nn.Module | None]
                        None entries represent InputLayer and no-op layers
                        (linear Activation, Merge — handled inline).
      • exec_order    — list of layer_ids in topological order.
      • parents       — dict[layer_id → list[layer_id]]
      • merge_modes   — dict[layer_id → MergeMode]  for Merge layers.

    forward() walks exec_order, computes each layer's output tensor, and
    stores it in a cache keyed by layer_id.  Merge layers combine their
    parent tensors before applying (or skipping) their registered module.
    """

    def __init__(self, node_modules: dict, exec_order: list,
                 parents: dict, merge_modes: dict):
        super().__init__()
        self._exec_order  = exec_order
        self._parents     = parents
        self._merge_modes = merge_modes   # layer_id → MergeMode (only for Merge layers)

        # Register every non-None module so PyTorch tracks parameters
        self._mods = nn.ModuleDict()
        for lid, mod in node_modules.items():
            if mod is not None:
                # ModuleDict keys cannot contain arbitrary chars — sanitise.
                # Spaces, hyphens and dots are all replaced with underscores so
                # the key is a valid Python identifier and stays stable across
                # train → save → load → predict round-trips.
                self._mods[lid.replace("-", "_").replace(".", "_").replace(" ", "_")] = mod
        self._lid_to_key = {
            lid: lid.replace("-", "_").replace(".", "_").replace(" ", "_")
            for lid in node_modules if node_modules[lid] is not None
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache: dict[str, torch.Tensor] = {}

        for lid in self._exec_order:
            pids = self._parents.get(lid, [])

            # ── InputLayer: seed from the batch tensor ────────────────────────
            if not pids:
                cache[lid] = x
                continue

            # ── gather parent tensors ─────────────────────────────────────────
            parent_tensors = [cache[pid] for pid in pids]

            # ── Merge layer: combine parents first ────────────────────────────
            if lid in self._merge_modes:
                mode = self._merge_modes[lid]
                if mode == MergeMode.ADD:
                    t = parent_tensors[0]
                    for pt in parent_tensors[1:]:
                        t = t + pt
                else:  # CONCATENATE
                    t = torch.cat(parent_tensors, dim=-1)
                # Merge has no learnable parameters → no module to apply
                cache[lid] = t
                continue

            # ── normal layer: single parent input ────────────────────────────
            t = parent_tensors[0]
            key = self._lid_to_key.get(lid)
            if key is not None:
                t = self._mods[key](t)
            cache[lid] = t

        # Return the output of the last node in topological order
        return cache[self._exec_order[-1]]


# ── Main builder ──────────────────────────────────────────────────────────────

class PyTorchModelBuilder:

    @staticmethod
    def build(network: Network) -> nn.Module:
        """
        Converts the Network abstraction into a PyTorch nn.Module.

        For purely linear graphs the result is behaviourally identical to the
        old nn.Sequential approach.  For graphs that contain Merge (Add /
        Concatenate) layers — residual connections, skip connections, multi-
        branch architectures — a GraphModule is returned instead, which
        implements the proper graph traversal in its forward() method.
        """
        if not network.validate():
            raise ValueError(
                "Network is invalid — call network.validate() before building."
            )

        network.propagate_shapes()
        ordered = network.topological_sort()

        exec_order: list[str]             = []
        node_modules: dict[str, nn.Module | None] = {}
        parents: dict[str, list[str]]     = {}
        merge_modes: dict[str, MergeMode] = {}

        for layer in ordered:
            lid = layer._id
            exec_order.append(lid)
            parents[lid] = network.get_parents(lid)

            if isinstance(layer, InputLayer):
                node_modules[lid] = None
                continue

            if isinstance(layer, Merge):
                node_modules[lid] = None          # no learnable params
                merge_modes[lid]  = layer._mergeMode
                continue

            mods = PyTorchModelBuilder._build_module(layer)
            if len(mods) == 0:
                node_modules[lid] = None          # e.g. LINEAR activation
            elif len(mods) == 1:
                node_modules[lid] = mods[0]
            else:
                # e.g. Conv2D + Dropout — wrap in a tiny Sequential
                node_modules[lid] = nn.Sequential(*mods)

        return GraphModule(node_modules, exec_order, parents, merge_modes)

    # ── per-type builders ─────────────────────────────────────────────────────

    @staticmethod
    def _build_module(layer) -> list[nn.Module]:
        if isinstance(layer, Dense):
            return PyTorchModelBuilder._build_dense(layer)
        elif isinstance(layer, Conv2D):
            return PyTorchModelBuilder._build_conv2d(layer)
        elif isinstance(layer, Recurrent):
            return PyTorchModelBuilder._build_recurrent(layer)
        elif isinstance(layer, Pooling):
            return PyTorchModelBuilder._build_pooling(layer)
        elif isinstance(layer, Flatten):
            return PyTorchModelBuilder._build_flatten(layer)
        elif isinstance(layer, Activation):
            return PyTorchModelBuilder._build_activation(layer)
        else:
            raise ValueError(f"Unknown layer type: {type(layer).__name__}")

    @staticmethod
    def _build_dense(layer: Dense) -> list[nn.Module]:
        in_features = layer._inputShape[0]
        modules = [nn.Linear(in_features, layer._unit, bias=layer._useBias)]
        if layer._dropout and layer._dropout > 0.0:
            modules.append(nn.Dropout(p=layer._dropout))
        return modules

    @staticmethod
    def _build_conv2d(layer: Conv2D) -> list[nn.Module]:
        in_channels = layer._inputShape[0]
        modules = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer._filters,
                kernel_size=layer._kernelSize,
                stride=layer._strides,
                padding=layer._padding,
            )
        ]
        if layer._dropout and layer._dropout > 0.0 and layer._dropoutStyle is not None:
            if layer._dropoutStyle == DropoutStyle.SPATIAL:
                modules.append(nn.Dropout2d(p=layer._dropout))
            else:
                modules.append(nn.Dropout(p=layer._dropout))
        return modules

    @staticmethod
    def _build_recurrent(layer: Recurrent) -> list[nn.Module]:
        input_size = layer._inputShape[1]
        rnn_kwargs = dict(
            input_size=input_size,
            hidden_size=layer._hiddenSize,
            num_layers=layer._numLayer,
            batch_first=True,
            dropout=layer._dropout if layer._numLayer > 1 else 0.0,
            bidirectional=layer._bidirection,
        )
        rnn = (nn.GRU if layer._recurrentType == RecurrentType.GRU
               else nn.LSTM)(**rnn_kwargs)
        return [RecurrentWrapper(rnn, returnSeq=layer._returnSeq)]

    @staticmethod
    def _build_pooling(layer: Pooling) -> list[nn.Module]:
        kwargs = dict(
            kernel_size=layer._poolSize,
            stride=layer._strides,
            padding=layer._padding,
        )
        if layer._poolingType == PoolingType.AVERAGE:
            return [nn.AvgPool2d(**kwargs)]
        return [nn.MaxPool2d(**kwargs)]

    @staticmethod
    def _build_flatten(layer: Flatten) -> list[nn.Module]:
        return [nn.Flatten(start_dim=1)]

    @staticmethod
    def _build_activation(layer: Activation) -> list[nn.Module]:
        t = layer._activationType
        if t == ActivationType.RELU:
            return [nn.ReLU()]
        elif t == ActivationType.SIGMOID:
            return [nn.Sigmoid()]
        elif t == ActivationType.TANH:
            return [nn.Tanh()]
        elif t == ActivationType.SOFTMAX:
            return [nn.Softmax(dim=1)]
        elif t == ActivationType.LINEAR:
            return []   # identity — no module needed
        raise ValueError(f"Unknown activation type: {t}")