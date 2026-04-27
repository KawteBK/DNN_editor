from enum import Enum
import os


class DatasetSource(Enum):
    BUILTIN = "builtin"
    CUSTOM  = "custom"   # user uploads / provides a folder path


class BuiltinDataset(Enum):
    MNIST         = "mnist"
    FASHION_MNIST = "fashion_mnist"
    KMNIST        = "kmnist"
    CIFAR10       = "cifar10"
    CIFAR100      = "cifar100"
    SVHN          = "svhn"
    STL10         = "stl10"


BUILTIN_METADATA = {
    BuiltinDataset.MNIST:         {"inputShape": (1, 28, 28),  "numClasses": 10},
    BuiltinDataset.FASHION_MNIST: {"inputShape": (1, 28, 28),  "numClasses": 10},
    BuiltinDataset.KMNIST:        {"inputShape": (1, 28, 28),  "numClasses": 10},
    BuiltinDataset.CIFAR10:       {"inputShape": (3, 32, 32),  "numClasses": 10},
    BuiltinDataset.CIFAR100:      {"inputShape": (3, 32, 32),  "numClasses": 100},
    BuiltinDataset.SVHN:          {"inputShape": (3, 32, 32),  "numClasses": 10},
    BuiltinDataset.STL10:         {"inputShape": (3, 96, 96),  "numClasses": 10},
}


class Dataset:
    def __init__(self, id: str, name: str, inputShape: tuple, numClasses: int,
                 source: DatasetSource = DatasetSource.BUILTIN,
                 builtinType: BuiltinDataset = None,
                 dataPath: str = None):
        self._id = id
        self._name = name
        self._inputShape = inputShape
        self._numClasses = numClasses
        self._source = source
        self._builtinType = builtinType
        # ── custom dataset support ────────────────────────────────────────────
        # dataPath points to a root folder organised as ImageFolder expects:
        #   <dataPath>/class_a/img1.png
        #   <dataPath>/class_b/img2.png
        # For CSV/NPZ custom datasets the path is the file itself.
        self._dataPath = dataPath

    # ── factories ─────────────────────────────────────────────────────────────

    @classmethod
    def fromBuiltin(cls, builtinType: BuiltinDataset) -> "Dataset":
        """Factory for built-in torchvision datasets."""
        meta = BUILTIN_METADATA[builtinType]
        return cls(
            id=builtinType.value,
            name=builtinType.value,
            inputShape=meta["inputShape"],
            numClasses=meta["numClasses"],
            source=DatasetSource.BUILTIN,
            builtinType=builtinType,
        )

    @classmethod
    def fromCustomPath(cls, path: str, inputShape: tuple, numClasses: int,
                       name: str = None) -> "Dataset":
        """
        Factory for user-supplied datasets.

        path        — absolute path to an ImageFolder-style directory
                      OR a .csv/.npz file.
        inputShape  — (C, H, W) or (features,) — must be declared by the user.
        numClasses  — number of output classes — must be declared by the user.
        name        — human-readable label; defaults to the folder/file basename.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Custom dataset path does not exist: '{path}'")

        resolved_name = name or os.path.basename(path.rstrip("/\\"))
        dataset_id = resolved_name.lower().replace(" ", "_")

        return cls(
            id=dataset_id,
            name=resolved_name,
            inputShape=inputShape,
            numClasses=numClasses,
            source=DatasetSource.CUSTOM,
            builtinType=None,
            dataPath=path,
        )

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self) -> bool:
        if not self._id or not self._name:
            return False
        if not self._inputShape or len(self._inputShape) == 0:
            return False
        if self._numClasses < 1:
            return False
        if self._source == DatasetSource.CUSTOM:
            if not self._dataPath:
                return False
            # Path may be absent on a different machine after load; we warn but
            # still consider the Dataset object itself structurally valid.
            # The trainer will raise at data-load time if the path is missing.
        return True

    def isDataAvailable(self) -> bool:
        """
        Returns True when the underlying data can actually be read.
        For builtin datasets this is always True (torchvision downloads them).
        For custom datasets the declared path must exist on the current machine.
        """
        if self._source == DatasetSource.BUILTIN:
            return True
        return bool(self._dataPath and os.path.exists(self._dataPath))

    # ── serialisation ─────────────────────────────────────────────────────────

    def toDict(self) -> dict:
        return {
            "id":          self._id,
            "name":        self._name,
            "inputShape":  list(self._inputShape),
            "numClasses":  self._numClasses,
            "source":      self._source.value,
            "builtinType": self._builtinType.value if self._builtinType else None,
            "dataPath":    self._dataPath,
        }

    @classmethod
    def fromDict(cls, data: dict) -> "Dataset | None":
        """
        Reconstruct a Dataset from a serialised dict.
        Returns None when data is empty or None (e.g. project saved without a dataset).
        """
        if not data:
            return None
        source      = DatasetSource(data.get("source", "builtin"))
        builtin_raw = data.get("builtinType")
        builtinType = BuiltinDataset(builtin_raw) if builtin_raw else None
        return cls(
            id=data["id"],
            name=data["name"],
            inputShape=tuple(data["inputShape"]),
            numClasses=data["numClasses"],
            source=source,
            builtinType=builtinType,
            dataPath=data.get("dataPath"),
        )