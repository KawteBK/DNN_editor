import os #yeah for file operations, creating folders type shit
import torch
from torch.utils.data import DataLoader, random_split, Dataset as TorchDataset  #data loader is for loading datasets in batches (tgasmhoum not load everything at once) and random_split is for splitting the dataset into train and validation sets
from torchvision import datasets, transforms #transform is for transforming images before training

from Core.dataset import Dataset, DatasetSource, BuiltinDataset
from Core.trainingConfig import TrainingConfig


# Per-dataset normalization stats (mean, std).
# These are the canonical values computed over each full training set.
_NORMALIZATION_STATS = {
    BuiltinDataset.MNIST:         ((0.1307,),                     (0.3081,)),
    BuiltinDataset.FASHION_MNIST: ((0.2860,),                     (0.3530,)),
    BuiltinDataset.KMNIST:        ((0.1918,),                     (0.3483,)),
    BuiltinDataset.CIFAR10:       ((0.4914, 0.4822, 0.4465),      (0.2470, 0.2435, 0.2616)),
    BuiltinDataset.CIFAR100:      ((0.5071, 0.4867, 0.4408),      (0.2675, 0.2565, 0.2761)),
    BuiltinDataset.SVHN:          ((0.4377, 0.4438, 0.4728),      (0.1980, 0.2010, 0.1970)),
    BuiltinDataset.STL10:         ((0.4467, 0.4398, 0.4066),      (0.2242, 0.2215, 0.2239)),
    # sequence variants share the same pixel stats as their image counterparts
    # (same underlying data, just reshaped) so we reuse those values.
    BuiltinDataset.MNIST_SEQ:     ((0.1307,),                     (0.3081,)),
    BuiltinDataset.CIFAR10_SEQ:   ((0.4914, 0.4822, 0.4465),      (0.2470, 0.2435, 0.2616)),
}

# benifets of noramlizing are : faster training ; stable gradients; better convergence


# ── SequenceDataset wrapper ───────────────────────────────────────────────────

class SequenceDataset(TorchDataset):
    """
    Wraps an existing image dataset and reshapes each sample into a 2-D
    sequence tensor that Recurrent layers (LSTM / GRU) can consume.

    Reshaping logic
    ───────────────
    Image tensor coming in:  (C, H, W)   — standard torchvision output

    MNIST / grayscale  (C=1):  squeeze channel → (H, W) = (28, 28)
                               → 28 timesteps, each with 28 features (pixel columns)

    CIFAR10 / RGB      (C=3):  flatten channel + width → (H, C*W) = (32, 96)
                               → 32 timesteps, each with 96 features (3 channels × 32 cols)

    This matches the inputShape declared in BUILTIN_METADATA exactly:
        MNIST_SEQ   → (28, 28)
        CIFAR10_SEQ → (32, 96)

    Why this reshaping? Each row of the image becomes one timestep. The RNN
    reads the image top-to-bottom, one row at a time — a classic benchmark
    called "sequential MNIST" / "row-wise MNIST".
    """

    def __init__(self, base_dataset: TorchDataset):
        self._base = base_dataset

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx):
        image, label = self._base[idx]          # image: (C, H, W) float tensor
        C, H, W = image.shape

        if C == 1:
            # Grayscale: drop the channel dim → (H, W)
            seq = image.squeeze(0)              # (28, 28)
        else:
            # RGB: permute to (H, C, W) then flatten last two dims → (H, C*W)
            seq = image.permute(1, 0, 2).reshape(H, C * W)   # (32, 96)

        return seq, label





# Default folder where torchvision will download datasets.
_DEFAULT_DATA_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "nndl", "datasets")


class PyTorchDatasets:

    #public 

    @staticmethod
    def get_dataloaders(
        dataset: Dataset,
        config: TrainingConfig,
        dataRoot: str = _DEFAULT_DATA_ROOT,
    ) -> tuple[DataLoader, DataLoader | None]:
        """
        Main entry point — the only method pytorch_trainer.py needs to call.

        Takes your Dataset abstraction + TrainingConfig and returns two
        ready-to-iterate DataLoaders: (train_loader, val_loader).
        val_loader is None when config._validationSplit == 0.

        All the downloading, normalizing, and splitting happens here.
        """
        if dataset._source == DatasetSource.CUSTOM: #for now only built in, cuz wtf we runningout of time
            raise NotImplementedError(
                "Custom dataset loading is not yet implemented. "
                "Use a BuiltinDataset for now."
            )

        transform  = PyTorchDatasets._get_transforms(dataset._builtinType)
        full_data  = PyTorchDatasets._load_builtin(dataset._builtinType, transform, dataRoot)
        train_set, val_set = PyTorchDatasets._split_dataset(
            full_data, config._validationSplit, config._shuffle
        )

        train_loader = DataLoader(
            train_set,
            batch_size=config._batchSize,
            shuffle=config._shuffle,
            num_workers=2,       # 2 background workers, safe default on most machines
            pin_memory=True,     # speeds up CPU→GPU transfer when a GPU is available
        )

        val_loader = None
        if val_set is not None:
            val_loader = DataLoader(
                val_set,
                batch_size=config._batchSize,
                shuffle=False,   # never shuffle validation — order doesn't matter and it's slower
                num_workers=2,
                pin_memory=True,
            )

        return train_loader, val_loader

    # private 
    @staticmethod
    def _get_transforms(builtinType: BuiltinDataset) -> transforms.Compose:
        """
        Builds the transform pipeline for a dataset.

        Every dataset gets at minimum:
          1. ToTensor()   — converts PIL image (H, W, C) uint8 → (C, H, W) float32 in [0,1]
          2. Normalize()  — shifts to roughly zero-mean, unit-variance using
                            the canonical stats for that dataset.

        Why normalize? Without it, pixel values sit in [0, 1] or [0, 255].
        The network's first layer receives wildly different scales per channel,
        gradients become unbalanced, and training is slow / unstable.
        These stats make each channel ~ N(0, 1) which is what weight-init assumes.

        Separated from get_dataloaders so we can add augmentation here later
        (RandomHorizontalFlip, RandomCrop, etc.) without touching anything else.

        Note for sequence variants: the image-to-tensor transform is applied first
        (producing a standard (C, H, W) tensor), then SequenceDataset reshapes it
        into (seq_len, features) at __getitem__ time. So we use the same transform
        pipeline here — only the wrapping layer differs.
        """
        mean, std = _NORMALIZATION_STATS[builtinType]
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def _load_builtin(
        builtinType: BuiltinDataset,
        transform: transforms.Compose,
        dataRoot: str,
    ):
        """
        Downloads (if needed) and loads the full training split of a built-in dataset.

        torchvision caches downloads — calling this twice doesn't re-download.

        SVHN is the odd one out: it uses split="train" instead of train=True.
        Every other dataset uses the train=True / train=False boolean API.
        """
        os.makedirs(dataRoot, exist_ok=True)
        kwargs = dict(root=dataRoot, download=True, transform=transform)

        if builtinType == BuiltinDataset.MNIST:
            return datasets.MNIST(**kwargs, train=True)

        elif builtinType == BuiltinDataset.FASHION_MNIST:
            return datasets.FashionMNIST(**kwargs, train=True)

        elif builtinType == BuiltinDataset.KMNIST:
            return datasets.KMNIST(**kwargs, train=True)

        elif builtinType == BuiltinDataset.CIFAR10:
            return datasets.CIFAR10(**kwargs, train=True)

        elif builtinType == BuiltinDataset.CIFAR100:
            return datasets.CIFAR100(**kwargs, train=True)

        elif builtinType == BuiltinDataset.SVHN:
            # SVHN uses split="train" not train=True — different API, same idea.
            return datasets.SVHN(**kwargs, split="train")

        elif builtinType == BuiltinDataset.STL10:
            return datasets.STL10(**kwargs, split="train")

        elif builtinType == BuiltinDataset.MNIST_SEQ:
            # Load standard MNIST then wrap it so each (1,28,28) image
            # becomes a (28,28) sequence tensor at __getitem__ time.
            base = datasets.MNIST(**kwargs, train=True)
            return SequenceDataset(base)

        elif builtinType == BuiltinDataset.CIFAR10_SEQ:
            # Load standard CIFAR-10 then wrap it so each (3,32,32) image
            # becomes a (32,96) sequence tensor at __getitem__ time.
            base = datasets.CIFAR10(**kwargs, train=True)
            return SequenceDataset(base)

        else:
            raise ValueError(f"Unsupported built-in dataset: {builtinType}")

    @staticmethod
    def _load_builtin_test(
        builtinType: BuiltinDataset,
        transform: transforms.Compose,
        dataRoot: str,
    ):
        """
        added this for aang — loads the official held-out TEST split for each dataset.

        This is completely separate from the training data that _load_builtin returns.
        The model never sees these samples during training, so evaluation scores are honest.

        SVHN and STL10 use split="test" instead of train=False — same asymmetry as
        their training counterparts.
        """
        os.makedirs(dataRoot, exist_ok=True)
        kwargs = dict(root=dataRoot, download=True, transform=transform)

        if builtinType == BuiltinDataset.MNIST:
            return datasets.MNIST(**kwargs, train=False)  # added this for aang

        elif builtinType == BuiltinDataset.FASHION_MNIST:
            return datasets.FashionMNIST(**kwargs, train=False)  # added this for aang

        elif builtinType == BuiltinDataset.KMNIST:
            return datasets.KMNIST(**kwargs, train=False)  # added this for aang

        elif builtinType == BuiltinDataset.CIFAR10:
            return datasets.CIFAR10(**kwargs, train=False)  # added this for aang

        elif builtinType == BuiltinDataset.CIFAR100:
            return datasets.CIFAR100(**kwargs, train=False)  # added this for aang

        elif builtinType == BuiltinDataset.SVHN:
            return datasets.SVHN(**kwargs, split="test")  # added this for aang

        elif builtinType == BuiltinDataset.STL10:
            return datasets.STL10(**kwargs, split="test")  # added this for aang

        elif builtinType == BuiltinDataset.MNIST_SEQ:
            base = datasets.MNIST(**kwargs, train=False)
            return SequenceDataset(base)

        elif builtinType == BuiltinDataset.CIFAR10_SEQ:
            base = datasets.CIFAR10(**kwargs, train=False)
            return SequenceDataset(base)

        else:
            raise ValueError(f"Unsupported built-in dataset: {builtinType}")

    @staticmethod
    def _split_dataset(full_dataset, validationSplit: float, shuffle: bool):
        """
        Splits a full dataset into train and validation subsets.

        Uses torch.utils.data.random_split which gives you proper non-overlapping
        Subset objects — no sample ever appears in both splits.

        If validationSplit is 0, the full dataset is returned as-is for training
        and val_set is None. The trainer handles None val_loader gracefully.

        Why pass shuffle here? random_split always randomizes the split indices
        regardless, but we document the intent — the DataLoader shuffle flag
        (set in get_dataloaders) is what actually controls per-epoch order.
        """
        if validationSplit <= 0.0:
            return full_dataset, None

        total     = len(full_dataset)
        val_size  = int(total * validationSplit)
        train_size = total - val_size

        if val_size == 0:
            return full_dataset, None

        train_set, val_set = random_split(full_dataset, [train_size, val_size])
        return train_set, val_set