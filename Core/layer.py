#girl, woman, person, WE NEED TO ADDA FLATTEN LAYER, fuuuuck, and i thought i was done adding layer
from enum import Enum


class Layer:
    def __init__(self, id, name, role=None) -> None:
        self._id = id
        self._name = name
        self._role = role
        self._inputShape = None
        self._outputShape = None
        self._isInput = False
        self._isLocked = False

    def getOutputShape(self, inputShape):
        raise NotImplementedError("Subclasses must implement getOutputShape()")

    def getOutputShapeMulti(self, inputShapes: list) -> tuple:
        """
        Called when this layer has multiple parents (non-linear graph).
        Default: reject — layers that support merging override this.
        """
        raise ValueError(
            f"Layer '{type(self).__name__}' does not support multiple inputs. "
            "Use an Add or Concatenate merge layer."
        )


class InputLayer(Layer):
    def __init__(self, inputShape):
        super().__init__(id="input", name="Input", role="input")
        self._outputShape = inputShape
        self._isInput = True
        self._isLocked = True


class Dense(Layer):
    def __init__(self, unit, useBias, dropout, id, name, role) -> None:
        super().__init__(id, name, role)
        self._unit = unit
        self._useBias = useBias
        self._dropout = dropout

    def getOutputShape(self, inputShape):
        if len(inputShape) != 1:
            raise ValueError(
                f"Dense expects 1D input e.g. (512,) but received {inputShape} — add a Flatten layer first"
            )
        return (self._unit,)


# ── Merge layers — gateway to non-linear / functional-API graphs ──────────────

class MergeMode(Enum):
    ADD         = "add"          # element-wise sum — shapes must match exactly
    CONCATENATE = "concatenate"  # concat along last axis


class Merge(Layer):
    """
    Combines outputs of two or more parent layers.

    MergeMode.ADD:         element-wise sum; all parents must share identical shapes.
    MergeMode.CONCATENATE: concat along the last dimension; only last dim may differ.

    Enables residual connections, skip connections, multi-branch networks, etc.
    """
    def __init__(self, id, name, role=None, mergeMode: MergeMode = MergeMode.ADD):
        super().__init__(id, name, role)
        self._mergeMode = mergeMode

    def getOutputShape(self, inputShape):
        # Single-parent fallback (passthrough)
        return inputShape

    def getOutputShapeMulti(self, inputShapes: list) -> tuple:
        if not inputShapes:
            raise ValueError("Merge layer received no input shapes.")

        if self._mergeMode == MergeMode.ADD:
            ref = inputShapes[0]
            for s in inputShapes[1:]:
                if s != ref:
                    raise ValueError(
                        f"Add merge requires identical shapes from all parents, got {inputShapes}"
                    )
            return ref

        elif self._mergeMode == MergeMode.CONCATENATE:
            ref = inputShapes[0]
            if len(ref) == 0:
                raise ValueError("Cannot concatenate scalar shapes.")
            for s in inputShapes[1:]:
                if len(s) != len(ref) or s[:-1] != ref[:-1]:
                    raise ValueError(
                        f"Concatenate: all dims except last must match, got {inputShapes}"
                    )
            last_dim = sum(s[-1] for s in inputShapes)
            return ref[:-1] + (last_dim,)

        raise ValueError(f"Unknown MergeMode: {self._mergeMode}")


class DropoutStyle(Enum):
    STANDARD = "standard"   # nn.Dropout
    SPATIAL  = "spatial"    # nn.Dropout2d


class Conv2D(Layer):
    def __init__(self, filters, kernelSize, strides, padding, dropout, dropoutStyle, id, name, role) -> None:
        super().__init__(id, name, role)
        self._filters = filters
        self._kernelSize = kernelSize
        self._strides = strides
        self._padding = padding
        self._dropout = dropout
        self._dropoutStyle = dropoutStyle

    def getOutputShape(self, inputShape):
        if len(inputShape) != 3:
            raise ValueError("Conv2D expects input shape (C, H, W)")
        C, H, W = inputShape
        H_out = int((H - self._kernelSize[0] + 2 * self._padding[0]) / self._strides[0]) + 1
        W_out = int((W - self._kernelSize[1] + 2 * self._padding[1]) / self._strides[1]) + 1
        return self._filters, H_out, W_out


class RecurrentType(Enum):
    LSTM = "lstm"
    GRU  = "gru"


class Recurrent(Layer):
    def __init__(self, hiddenSize, numLayer, dropout, id, name, role,
                 bidirection, returnSeq, recurrentType: RecurrentType = RecurrentType.LSTM) -> None:
        super().__init__(id, name, role)
        self._hiddenSize = hiddenSize
        self._numLayer = numLayer
        self._dropout = dropout
        self._bidirection = bidirection
        self._returnSeq = returnSeq
        self._recurrentType = recurrentType

    def getOutputShape(self, inputShape):
        if len(inputShape) != 2:
            raise ValueError("Recurrent expects input shape (seq_len, features)")
        seq_len, features = inputShape
        hiddenSize = self._hiddenSize * 2 if self._bidirection else self._hiddenSize
        if self._returnSeq:
            return (seq_len, hiddenSize)
        else:
            return (hiddenSize,)


class PoolingType(Enum):
    MAX     = "max"
    AVERAGE = "average"


class Pooling(Layer):
    def __init__(self, poolSize, strides, padding, id, name, role,
                 poolingType: PoolingType = PoolingType.MAX) -> None:
        super().__init__(id, name, role)
        self._poolSize = poolSize
        self._strides = strides
        self._padding = padding
        self._poolingType = poolingType

    def getOutputShape(self, inputShape):
        if len(inputShape) != 3:
            raise ValueError("Pooling expects input shape (C, H, W)")
        C, H, W = inputShape
        H_out = int((H - self._poolSize[0] + 2 * self._padding[0]) / self._strides[0]) + 1
        W_out = int((W - self._poolSize[1] + 2 * self._padding[1]) / self._strides[1]) + 1
        return (C, H_out, W_out)


class Flatten(Layer):
    """Collapses any multi-dimensional input (C, H, W) → (C*H*W,)."""
    def __init__(self, id, name, role=None):
        super().__init__(id, name, role)

    def getOutputShape(self, inputShape):
        if len(inputShape) == 1:
            return inputShape
        flat_size = 1
        for dim in inputShape:
            flat_size *= dim
        return (flat_size,)


class ActivationType(Enum):
    RELU    = "relu"
    SIGMOID = "sigmoid"
    TANH    = "tanh"
    SOFTMAX = "softmax"
    LINEAR  = "linear"


class Activation(Layer):
    def __init__(self, activationType, id, name, role=None):
        super().__init__(id, name, role)
        self._activationType = activationType

    def getOutputShape(self, inputShape):
        return inputShape