from enum import Enum


class DatasetSource(Enum):
    BUILTIN = "builtin"
    CUSTOM = "custom"    #eeeh, for the future, when the user will be able to upload his owna datasets, for now we will only have builtin datasets
class BuiltinDataset(Enum):
    MNIST          = "mnist"
    FASHION_MNIST  = "fashion_mnist"
    KMNIST         = "kmnist"
    CIFAR10        = "cifar10"
    CIFAR100       = "cifar100"
    SVHN           = "svhn"
    STL10          = "stl10"

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
                 builtinType: BuiltinDataset = None):
        self._id = id
        self._name = name
        self._inputShape = inputShape
        self._numClasses = numClasses
        self._source = source
        self._builtinType = builtinType

    @classmethod
    def fromBuiltin(cls, builtinType: BuiltinDataset) -> "Dataset":
        """Factory method — the easy way to create a built-in dataset."""
        meta = BUILTIN_METADATA[builtinType]
        return cls(
            id=builtinType.value,
            name=builtinType.value,
            inputShape=meta["inputShape"],
            numClasses=meta["numClasses"],
            source=DatasetSource.BUILTIN,
            builtinType=builtinType,
        )

    def validate(self) -> bool:
        if not self._id or not self._name:
            return False
        if not self._inputShape or len(self._inputShape) == 0:
            return False
        if self._numClasses < 1:
            return False
        return True

    def toDict(self) -> dict:
        return {
            "id": self._id,
            "name": self._name,
            "inputShape": list(self._inputShape),
            "numClasses": self._numClasses,
            "source": self._source.value,
            "builtinType": self._builtinType.value if self._builtinType else None,
        }

    @classmethod
    def fromDict(cls, data: dict) -> "Dataset":
        source = DatasetSource(data.get("source", "builtin"))
        builtin_raw = data.get("builtinType")
        builtinType = BuiltinDataset(builtin_raw) if builtin_raw else None
        return cls(
            id=data["id"],
            name=data["name"],
            inputShape=tuple(data["inputShape"]),
            numClasses=data["numClasses"],
            source=source,
            builtinType=builtinType,
        )