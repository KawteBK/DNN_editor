import torch.nn as nn # what i want to know in all this, where is the forward function and how is it implemented ???

from Core.layer import (
    Activation, ActivationType,
    Conv2D, DropoutStyle,
    Dense,
    Flatten,
    InputLayer,
    Pooling, PoolingType,
    Recurrent, RecurrentType,
)
from Core.network import Network


# ── Inner wrapper , lives here, not in layer.py 

class RecurrentWrapper(nn.Module):
    """
    Wraps nn.LSTM or nn.GRU so it can live inside nn.Sequential.

    The problem: PyTorch's LSTM/GRU return (output, hidden_state) — a tuple.
    nn.Sequential blindly passes whatever it receives to the next layer, so
    the next layer would receive a tuple instead of a tensor and crash.

    This wrapper:
      1. Calls the inner RNN and unpacks the tuple.
      2. Applies returnSeq logic:
           returnSeq=True  → return all timesteps  [batch, seq_len, hidden]
           returnSeq=False → return last timestep only  [batch, hidden]
         This matches exactly what Recurrent.getOutputShape() already promises.
    """

    def __init__(self, rnn: nn.Module, returnSeq: bool):
        super().__init__()
        self.rnn = rnn
        self.returnSeq = returnSeq

    def forward(self, x):
        output, _ = self.rnn(x)   # _ discards (h_n, c_n) for LSTM or h_n for GRU
        if self.returnSeq:
            return output          # [batch, seq_len, hidden_size]
        else:
            return output[:, -1, :]  # [batch, hidden_size] — last timestep only


# ─ Main builder 

class PyTorchModelBuilder:

    # ─ public 
    @staticmethod
    def build(network: Network) -> nn.Module:
        """
        Converts the Network abstraction into a PyTorch nn.Module.

        Steps:
          1. Call network.topological_sort() — gets layers in execution order
          2. Skip InputLayer — it's a shape anchor, not a real PyTorch layer
          3. Call _build_module() on each remaining layer → list of nn.Modules
          4. Flatten all those lists into one sequence
          5. Wrap in nn.Sequential and return

        The returned model accepts batches of shape
        (batch_size, *inputLayer._outputShape) and is ready to pass to the trainer.
        """
        if not network.validate():
            raise ValueError("Network is invalid — call network.validate() before building.")

        # Make sure shapes are propagated (safe to call multiple times)
        network.propagate_shapes()

        ordered = network.topological_sort()
        modules = []

        for layer in ordered:
            if isinstance(layer, InputLayer):
                continue   # not a real layer ;just tells us the input shape

            layer_modules = PyTorchModelBuilder._build_module(layer)
            modules.extend(layer_modules)

        if not modules:
            raise ValueError("Network produced no PyTorch modules after skipping InputLayer.")

        return nn.Sequential(*modules)

    #  private dispatcher 

    @staticmethod
    def _build_module(layer) -> list[nn.Module]:
        """
        Dispatcher — routes each layer type to the right builder method.

        Returns a LIST of nn.Modules, not a single one, because some layers
        expand into two (e.g. Conv2D with dropout → [nn.Conv2d, nn.Dropout2d]).
        Keeping the return type consistent (always a list) means build() can
        just call .extend() on every result without any special casing.
        """
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

    # ── per-type builders 

    @staticmethod
    def _build_dense(layer: Dense) -> list[nn.Module]:
        """
        Dense → nn.Linear, optionally followed by nn.Dropout.

        layer._unit     → out_features
        layer._useBias  → bias (True/False)
        layer._dropout  → dropout probability (0.0 means no dropout layer added)

        in_features is read from layer._inputShape[0], which was set by
        network.propagate_shapes() before build() ran.
        """
        in_features = layer._inputShape[0]
        modules = [nn.Linear(in_features, layer._unit, bias=layer._useBias)]

        if layer._dropout and layer._dropout > 0.0:
            modules.append(nn.Dropout(p=layer._dropout))

        return modules

    @staticmethod
    def _build_conv2d(layer: Conv2D) -> list[nn.Module]:
        """
        Conv2D → nn.Conv2d, optionally followed by nn.Dropout or nn.Dropout2d.

        layer._filters      → out_channels
        layer._kernelSize   → kernel_size  (tuple e.g. (3,3))
        layer._strides      → stride
        layer._padding      → padding
        layer._dropout      → dropout probability
        layer._dropoutStyle → DropoutStyle.STANDARD → nn.Dropout
                              DropoutStyle.SPATIAL  → nn.Dropout2d
                              None                  → no dropout

        in_channels comes from layer._inputShape[0]  (C in C,H,W).
        """
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
        """
        Recurrent → RecurrentWrapper(nn.LSTM or nn.GRU).

        Always returns a single-item list containing the wrapper — the wrapper
        itself handles returnSeq logic and tuple unpacking (see RecurrentWrapper).

        layer._hiddenSize  → hidden_size
        layer._numLayer    → num_layers  (stacked RNN layers inside the block)
        layer._dropout     → dropout between stacked layers (only active if numLayer > 1)
        layer._bidirection → bidirectional
        layer._returnSeq   → passed to RecurrentWrapper
        layer._recurrentType → LSTM or GRU

        batch_first=True is set so the tensor shape is (batch, seq_len, features)
        which matches the (seq_len, features) convention in your abstraction
        once a batch dimension is prepended — without this PyTorch expects
        (seq_len, batch, features) and your shape math breaks.
        """
        input_size = layer._inputShape[1]   # inputShape is (seq_len, features) → features

        rnn_kwargs = dict(
            input_size=input_size,
            hidden_size=layer._hiddenSize,
            num_layers=layer._numLayer,
            batch_first=True,
            dropout=layer._dropout if layer._numLayer > 1 else 0.0,
            bidirectional=layer._bidirection,
        )

        if layer._recurrentType == RecurrentType.GRU:
            rnn = nn.GRU(**rnn_kwargs)
        else:
            rnn = nn.LSTM(**rnn_kwargs)

        return [RecurrentWrapper(rnn, returnSeq=layer._returnSeq)]

    @staticmethod
    def _build_pooling(layer: Pooling) -> list[nn.Module]:
        """
        Pooling → nn.MaxPool2d or nn.AvgPool2d.

        layer._poolSize   → kernel_size
        layer._strides    → stride
        layer._padding    → padding
        layer._poolingType → MAX → nn.MaxPool2d
                             AVERAGE → nn.AvgPool2d
        """
        kwargs = dict(
            kernel_size=layer._poolSize,
            stride=layer._strides,
            padding=layer._padding,
        )

        if layer._poolingType == PoolingType.AVERAGE:
            return [nn.AvgPool2d(**kwargs)]
        else:
            return [nn.MaxPool2d(**kwargs)]

    @staticmethod
    def _build_flatten(layer: Flatten) -> list[nn.Module]:
        """
        Flatten → nn.Flatten(start_dim=1).

        start_dim=1 means we flatten everything EXCEPT the batch dimension.
        A tensor of shape (batch, C, H, W) becomes (batch, C*H*W).
        start_dim=0 would also flatten the batch dimension — never what you want.
        """
        return [nn.Flatten(start_dim=1)]

    @staticmethod
    def _build_activation(layer: Activation) -> list[nn.Module]:
        """
        Activation → the corresponding nn activation module.

        LINEAR is a no-op — returns an empty list (no module added).
        This is intentional: linear activation means "do nothing", so adding
        an nn.Identity() would just waste a forward pass.

        Softmax uses dim=1 — applies across the class dimension for a
        (batch, num_classes) output tensor, which is the standard setup.
        If you ever add sequence outputs, this dim will need revisiting.
        """
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
            return []   # no-op — linear activation = identity, no module needed
        else:
            raise ValueError(f"Unknown activation type: {t}")