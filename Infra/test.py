"""
test_infrastructure.py
──────────────────────
Tests for the full PyTorch infrastructure layer:
  - PyTorchModelBuilder  (build, all layer types, forward pass shapes)
  - PyTorchDatasets      (transforms, split, dataloaders — no real download)
  - PyTorchTrainer       (full train loop, history, weights, early stop, callback)
  - PyTorchCodeGenerator (generated code structure and content)

All tests use tiny synthetic networks and fake/minimal data so they run fast
with no internet access and no GPU required.
"""

import sys, traceback, os, tempfile, importlib, types
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ── project-layer imports 

from Core.layer import (
    InputLayer, Flatten, Dense, Conv2D, Recurrent, Pooling, Activation,
    ActivationType, RecurrentType, PoolingType, DropoutStyle,
)
from Core.network import Network
from Core.dataset import Dataset, DatasetSource, BuiltinDataset
from Core.trainingConfig import TrainingConfig
from Core.networkConfig import NetworkConfig, OptimizerType, LossType, MetricType
from Core.project import Project

# ── infrastructure imports ────────────────────────────────────────────────────
from .PytorchModelBuilder import PyTorchModelBuilder, RecurrentWrapper
from .pytorchDataset import PyTorchDatasets
from .trainer import PyTorchTrainer
from .codeGenerator import PyTorchCodeGenerator

# ── test harness ──────────────────────────────────────────────────────────────
passed = 0
failed = 0

def check(label, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {label}")
        passed += 1
    else:
        print(f"  ✗ {label}  ← FAILED")
        failed += 1

def section(title):
    print(f"\n── {title} {'─' * max(1, 52 - len(title))}")

def expect_exception(label, fn, exc_type=Exception):
    global passed, failed
    try:
        fn()
        print(f"  ✗ {label}  ← FAILED (no exception raised)")
        failed += 1
    except exc_type:
        print(f"  ✓ {label}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {label}  ← FAILED (wrong exc: {type(e).__name__}: {e})")
        failed += 1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build standard networks used across multiple tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_dense_network():
    """InputLayer(784,) → Dense(128) → ReLU → Dense(10)"""
    net = Network("dense_net")
    net.addLayer(InputLayer((784,)))
    net.addLayer(Dense(unit=128, useBias=True, dropout=0.0, id="d1", name="D1", role="hidden"))
    net.addLayer(Activation(ActivationType.RELU, id="a1", name="ReLU", role="hidden"))
    net.addLayer(Dense(unit=10, useBias=True, dropout=0.0, id="d2", name="D2", role="output"))
    net.addConnect("input", "d1")
    net.addConnect("d1", "a1")
    net.addConnect("a1", "d2")
    return net


def _make_cnn_network():
    """InputLayer(1,28,28) → Conv2D → MaxPool → Flatten → Dense(10)"""
    net = Network("cnn_net")
    net.addLayer(InputLayer((1, 28, 28)))
    net.addLayer(Conv2D(filters=8, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                        dropout=0.0, dropoutStyle=None, id="c1", name="Conv1", role="hidden"))
    net.addLayer(Pooling(poolSize=(2,2), strides=(2,2), padding=(0,0),
                         id="p1", name="Pool1", role="hidden"))
    net.addLayer(Flatten(id="fl", name="Flat", role="hidden"))
    net.addLayer(Dense(unit=10, useBias=True, dropout=0.0, id="d1", name="D1", role="output"))
    net.addConnect("input", "c1")
    net.addConnect("c1", "p1")
    net.addConnect("p1", "fl")
    net.addConnect("fl", "d1")
    return net


def _make_rnn_network():
    """InputLayer(10,32) → LSTM(returnSeq=False) → Dense(5)"""
    net = Network("rnn_net")
    net.addLayer(InputLayer((10, 32)))
    net.addLayer(Recurrent(hiddenSize=64, numLayer=1, dropout=0.0,
                            id="r1", name="LSTM1", role="hidden",
                            bidirection=False, returnSeq=False,
                            recurrentType=RecurrentType.LSTM))
    net.addLayer(Dense(unit=5, useBias=True, dropout=0.0, id="d1", name="D1", role="output"))
    net.addConnect("input", "r1")
    net.addConnect("r1", "d1")
    return net


def _make_full_project(net, dataset_type=BuiltinDataset.MNIST, tmp_path="/tmp/test_project"):
    """Wraps a network in a complete Project with all configs populated."""
    proj = Project(name="test_project", path=tmp_path)
    proj.network = net
    proj.dataset = Dataset.fromBuiltin(dataset_type)
    proj.trainingConfig = TrainingConfig(
        batchSize=32, epochs=1, validationSplit=0.1, shuffle=True, earlyStop=False
    )
    proj.networkConfig = NetworkConfig(
        optimizer=OptimizerType.ADAM,
        loss=LossType.CROSS_ENTROPY,
        metric=[MetricType.ACCURACY],
        learningRate=0.001,
    )
    return proj


# ─────────────────────────────────────────────────────────────────────────────
# 1. PyTorchModelBuilder
# ─────────────────────────────────────────────────────────────────────────────

def test_builder_dense_network():
    section("ModelBuilder — dense network")
    net   = _make_dense_network()
    model = PyTorchModelBuilder.build(net)

    check("returns nn.Sequential",          isinstance(model, nn.Sequential))
    check("correct number of sub-modules",  len(list(model.children())) == 3)  # Linear, ReLU, Linear

    # forward pass with a real batch
    x   = torch.randn(4, 784)
    out = model(x)
    check("output shape (4, 10)",           out.shape == torch.Size([4, 10]))


def test_builder_cnn_network():
    section("ModelBuilder — CNN network")
    net   = _make_cnn_network()
    model = PyTorchModelBuilder.build(net)

    check("returns nn.Sequential",          isinstance(model, nn.Sequential))

    x   = torch.randn(2, 1, 28, 28)
    out = model(x)
    check("output shape (2, 10)",           out.shape == torch.Size([2, 10]))


def test_builder_conv_with_dropout():
    section("ModelBuilder — Conv2D with dropout styles")
    net = Network("drop_net")
    net.addLayer(InputLayer((3, 16, 16)))
    net.addLayer(Conv2D(filters=4, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                        dropout=0.3, dropoutStyle=DropoutStyle.SPATIAL,
                        id="c1", name="C1", role="hidden"))
    net.addLayer(Flatten(id="fl", name="Flat", role="hidden"))
    net.addLayer(Dense(unit=5, useBias=True, dropout=0.2, id="d1", name="D1", role="output"))
    net.addConnect("input", "c1")
    net.addConnect("c1", "fl")
    net.addConnect("fl", "d1")
    model = PyTorchModelBuilder.build(net)

    # Conv + Dropout2d + Flatten + Linear + Dropout
    module_types = [type(m).__name__ for m in model.children()]
    check("Conv2d present",                 "Conv2d" in module_types)
    check("Dropout2d present",              "Dropout2d" in module_types)
    check("Dropout present after Linear",   "Dropout" in module_types)

    x   = torch.randn(2, 3, 16, 16)
    out = model(x)
    check("forward pass shape (2, 5)",      out.shape == torch.Size([2, 5]))


def test_builder_rnn_network():
    section("ModelBuilder — LSTM network")
    net   = _make_rnn_network()
    model = PyTorchModelBuilder.build(net)

    # RecurrentWrapper + Linear
    children = list(model.children())
    check("RecurrentWrapper is first",      isinstance(children[0], RecurrentWrapper))
    check("Linear is second",               isinstance(children[1], nn.Linear))

    x   = torch.randn(3, 10, 32)   # (batch, seq_len, features)
    out = model(x)
    check("output shape (3, 5)",            out.shape == torch.Size([3, 5]))


def test_builder_gru_bidirectional():
    section("ModelBuilder — GRU bidirectional returnSeq=True")
    net = Network("gru_net")
    net.addLayer(InputLayer((10, 16)))
    net.addLayer(Recurrent(hiddenSize=32, numLayer=1, dropout=0.0,
                            id="r1", name="GRU1", role="hidden",
                            bidirection=True, returnSeq=True,
                            recurrentType=RecurrentType.GRU))
    net.addLayer(Flatten(id="fl", name="Flat", role="hidden"))
    net.addLayer(Dense(unit=4, useBias=True, dropout=0.0, id="d1", name="D1", role="output"))
    net.addConnect("input", "r1")
    net.addConnect("r1", "fl")
    net.addConnect("fl", "d1")
    model = PyTorchModelBuilder.build(net)

    x   = torch.randn(2, 10, 16)
    out = model(x)
    check("GRU bidirectional output shape (2,4)", out.shape == torch.Size([2, 4]))


def test_builder_all_activations():
    section("ModelBuilder — all activation types")
    for act_type in [ActivationType.RELU, ActivationType.SIGMOID,
                     ActivationType.TANH, ActivationType.SOFTMAX]:
        net = Network(f"act_{act_type.value}")
        net.addLayer(InputLayer((8,)))
        net.addLayer(Dense(unit=4, useBias=True, dropout=0.0, id="d1", name="D1", role="hidden"))
        net.addLayer(Activation(act_type, id="a1", name="Act", role="hidden"))
        net.addLayer(Dense(unit=2, useBias=True, dropout=0.0, id="d2", name="D2", role="output"))
        net.addConnect("input", "d1")
        net.addConnect("d1", "a1")
        net.addConnect("a1", "d2")
        model = PyTorchModelBuilder.build(net)
        x = torch.randn(3, 8)
        out = model(x)
        check(f"{act_type.value} forward shape (3,2)", out.shape == torch.Size([3, 2]))

    # LINEAR is a no-op — no module should be added
    net = Network("linear_act")
    net.addLayer(InputLayer((8,)))
    net.addLayer(Dense(unit=4, useBias=True, dropout=0.0, id="d1", name="D1", role="hidden"))
    net.addLayer(Activation(ActivationType.LINEAR, id="a1", name="Linear", role="hidden"))
    net.addLayer(Dense(unit=2, useBias=True, dropout=0.0, id="d2", name="D2", role="output"))
    net.addConnect("input", "d1")
    net.addConnect("d1", "a1")
    net.addConnect("a1", "d2")
    model = PyTorchModelBuilder.build(net)
    module_types = [type(m).__name__ for m in model.children()]
    check("LINEAR adds no module",          "Identity" not in module_types and len(module_types) == 2)


def test_builder_pooling_types():
    section("ModelBuilder — MaxPool and AvgPool")
    for pool_type, expected_cls in [(PoolingType.MAX, "MaxPool2d"), (PoolingType.AVERAGE, "AvgPool2d")]:
        net = Network(f"pool_{pool_type.value}")
        net.addLayer(InputLayer((1, 16, 16)))
        net.addLayer(Pooling(poolSize=(2,2), strides=(2,2), padding=(0,0),
                             id="p1", name="P1", role="hidden", poolingType=pool_type))
        net.addLayer(Flatten(id="fl", name="Flat", role="hidden"))
        net.addLayer(Dense(unit=3, useBias=True, dropout=0.0, id="d1", name="D1", role="output"))
        net.addConnect("input", "p1")
        net.addConnect("p1", "fl")
        net.addConnect("fl", "d1")
        model = PyTorchModelBuilder.build(net)
        module_types = [type(m).__name__ for m in model.children()]
        check(f"{pool_type.value} → {expected_cls}", expected_cls in module_types)


def test_builder_invalid_network():
    section("ModelBuilder — rejects invalid network")
    net = Network("empty")
    expect_exception("empty network raises ValueError",
                     lambda: PyTorchModelBuilder.build(net), ValueError)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PyTorchDatasets
# ─────────────────────────────────────────────────────────────────────────────

def test_datasets_transforms():
    section("Datasets — _get_transforms returns Compose")
    from torchvision import transforms as T
    for bt in [BuiltinDataset.MNIST, BuiltinDataset.CIFAR10]:
        t = PyTorchDatasets._get_transforms(bt)
        check(f"{bt.value} transform is Compose", isinstance(t, T.Compose))
        check(f"{bt.value} has ToTensor",
              any(isinstance(x, T.ToTensor) for x in t.transforms))
        check(f"{bt.value} has Normalize",
              any(isinstance(x, T.Normalize) for x in t.transforms))


def test_datasets_split_no_val():
    section("Datasets — split with validationSplit=0")
    fake = TensorDataset(torch.randn(100, 4), torch.zeros(100, dtype=torch.long))
    train_set, val_set = PyTorchDatasets._split_dataset(fake, 0.0, True)
    check("val_set is None",                val_set is None)
    check("train_set is full dataset",      train_set is fake)


def test_datasets_split_with_val():
    section("Datasets — split with validationSplit=0.2")
    fake = TensorDataset(torch.randn(100, 4), torch.zeros(100, dtype=torch.long))
    train_set, val_set = PyTorchDatasets._split_dataset(fake, 0.2, True)
    check("val_set is not None",            val_set is not None)
    check("train size = 80",                len(train_set) == 80)
    check("val size = 20",                  len(val_set) == 20)
    check("no overlap (sizes sum to 100)",  len(train_set) + len(val_set) == 100)


def test_datasets_custom_raises():
    section("Datasets — CUSTOM source raises NotImplementedError")
    ds     = Dataset(id="custom", name="custom", inputShape=(1,28,28),
                     numClasses=10, source=DatasetSource.CUSTOM)
    config = TrainingConfig(batchSize=32, epochs=1, validationSplit=0.1,
                            shuffle=True, earlyStop=False)
    expect_exception("CUSTOM raises NotImplementedError",
                     lambda: PyTorchDatasets.get_dataloaders(ds, config),
                     NotImplementedError)


def test_datasets_dataloader_shapes():
    section("Datasets — DataLoader yields correct batch shapes")
    # Monkey-patch _load_builtin to return a fake dataset — no real download
    fake_data = TensorDataset(
        torch.randn(200, 1, 28, 28),
        torch.randint(0, 10, (200,))
    )
    original = PyTorchDatasets._load_builtin.__func__ if hasattr(PyTorchDatasets._load_builtin, '__func__') else None

    import unittest.mock as mock
    with mock.patch.object(PyTorchDatasets, '_load_builtin', return_value=fake_data):
        ds     = Dataset.fromBuiltin(BuiltinDataset.MNIST)
        config = TrainingConfig(batchSize=16, epochs=1, validationSplit=0.1,
                                shuffle=False, earlyStop=False)
        train_loader, val_loader = PyTorchDatasets.get_dataloaders(ds, config)

        check("train_loader is DataLoader",  isinstance(train_loader, DataLoader))
        check("val_loader is DataLoader",    isinstance(val_loader, DataLoader))

        inputs, labels = next(iter(train_loader))
        check("batch input shape (16,1,28,28)", inputs.shape == torch.Size([16, 1, 28, 28]))
        check("batch labels shape (16,)",       labels.shape == torch.Size([16]))


# ─────────────────────────────────────────────────────────────────────────────
# 3. PyTorchTrainer
# ─────────────────────────────────────────────────────────────────────────────

def _fake_dataloaders(input_shape, num_classes, n_samples=64, batch_size=16):
    """Returns (train_loader, val_loader) backed by random tensors — no download."""
    x = torch.randn(n_samples, *input_shape)
    y = torch.randint(0, num_classes, (n_samples,))
    ds = TensorDataset(x, y)
    split = n_samples // 5                       # 20% val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_samples - split, split])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _run_trainer_with_fake_data(net, input_shape, num_classes, tmp_dir, **config_overrides):
    """Patches get_dataloaders so trainer uses fake tensors instead of real datasets."""
    import unittest.mock as mock

    train_loader, val_loader = _fake_dataloaders(input_shape, num_classes)
    proj = _make_full_project(net, tmp_path=tmp_dir)

    for key, val in config_overrides.items():
        setattr(proj.trainingConfig, f"_{key}", val)

    with mock.patch.object(PyTorchDatasets, 'get_dataloaders',
                            return_value=(train_loader, val_loader)):
        result = PyTorchTrainer.train(proj)
    return result


def test_trainer_dense_completes():
    section("Trainer — dense network trains and returns Project")
    with tempfile.TemporaryDirectory() as tmp:
        net    = _make_dense_network()
        result = _run_trainer_with_fake_data(net, (784,), 10, tmp)

        check("returns a Project",              isinstance(result, Project))
        check("trainingLogs populated",         bool(result.trainingLogs))
        check("train_loss key exists",          "train_loss" in result.trainingLogs)
        check("train_loss has 1 entry (1 epoch)", len(result.trainingLogs["train_loss"]) == 1)
        check("val_loss key exists",            "val_loss" in result.trainingLogs)
        check("val_accuracy key exists",        "val_accuracy" in result.trainingLogs)
        check("weightsPath is set",             result.weightsPath is not None)
        check("weights file exists on disk",    os.path.isfile(result.weightsPath))


def test_trainer_cnn_completes():
    section("Trainer — CNN network trains")
    with tempfile.TemporaryDirectory() as tmp:
        net    = _make_cnn_network()
        result = _run_trainer_with_fake_data(net, (1, 28, 28), 10, tmp)

        check("returns a Project",              isinstance(result, Project))
        check("train_loss has 1 entry",         len(result.trainingLogs["train_loss"]) == 1)
        check("val_loss is a float",            isinstance(result.trainingLogs["val_loss"][0], float))


def test_trainer_history_grows_over_epochs():
    section("Trainer — history list grows one entry per epoch")
    with tempfile.TemporaryDirectory() as tmp:
        net  = _make_dense_network()
        proj = _make_full_project(net, tmp_path=tmp)
        proj.trainingConfig._epochs = 3

        import unittest.mock as mock
        train_loader, val_loader = _fake_dataloaders((784,), 10)
        with mock.patch.object(PyTorchDatasets, 'get_dataloaders',
                                return_value=(train_loader, val_loader)):
            result = PyTorchTrainer.train(proj)

        check("train_loss has 3 entries",       len(result.trainingLogs["train_loss"]) == 3)
        check("val_loss has 3 entries",         len(result.trainingLogs["val_loss"]) == 3)
        check("val_accuracy has 3 entries",     len(result.trainingLogs["val_accuracy"]) == 3)


def test_trainer_callback_fires():
    section("Trainer — on_epoch callback called once per epoch")
    with tempfile.TemporaryDirectory() as tmp:
        net  = _make_dense_network()
        proj = _make_full_project(net, tmp_path=tmp)
        proj.trainingConfig._epochs = 3

        calls = []
        def on_epoch(epoch, total, train_loss, val_metrics):
            calls.append((epoch, total))

        import unittest.mock as mock
        train_loader, val_loader = _fake_dataloaders((784,), 10)
        with mock.patch.object(PyTorchDatasets, 'get_dataloaders',
                                return_value=(train_loader, val_loader)):
            PyTorchTrainer.train(proj, on_epoch=on_epoch)

        check("callback fired 3 times",         len(calls) == 3)
        check("epoch numbers are 1,2,3",        [c[0] for c in calls] == [1, 2, 3])
        check("total is always 3",              all(c[1] == 3 for c in calls))


def test_trainer_no_val_split():
    section("Trainer — no validation split → no val keys in history")
    with tempfile.TemporaryDirectory() as tmp:
        net  = _make_dense_network()
        proj = _make_full_project(net, tmp_path=tmp)
        proj.trainingConfig._validationSplit = 0.0

        import unittest.mock as mock
        x = torch.randn(64, 784)
        y = torch.randint(0, 10, (64,))
        train_loader = DataLoader(TensorDataset(x, y), batch_size=16)
        with mock.patch.object(PyTorchDatasets, 'get_dataloaders',
                                return_value=(train_loader, None)):
            result = PyTorchTrainer.train(proj)

        check("train_loss exists",              "train_loss" in result.trainingLogs)
        check("val_loss absent",                "val_loss" not in result.trainingLogs)
        check("val_accuracy absent",            "val_accuracy" not in result.trainingLogs)


def test_trainer_early_stopping():
    section("Trainer — early stopping halts before max epochs")
    with tempfile.TemporaryDirectory() as tmp:
        net  = _make_dense_network()
        proj = _make_full_project(net, tmp_path=tmp)
        proj.trainingConfig._epochs    = 50   # would take forever if not stopped
        proj.trainingConfig._earlyStop = True

        import unittest.mock as mock

        # Return a val_loader that always reports increasing loss
        # by giving the model data it can't learn (random noise → random labels)
        train_loader, val_loader = _fake_dataloaders((784,), 10, n_samples=100)
        with mock.patch.object(PyTorchDatasets, 'get_dataloaders',
                                return_value=(train_loader, val_loader)):
            result = PyTorchTrainer.train(proj)

        epochs_run = len(result.trainingLogs["train_loss"])
        check("stopped before 50 epochs",      epochs_run < 50)
        check("ran at least 1 epoch",          epochs_run >= 1)


def test_trainer_all_optimizers():
    section("Trainer — all optimizer types build without error")
    for opt_type in OptimizerType:
        net  = _make_dense_network()
        net.propagate_shapes()
        model = PyTorchModelBuilder.build(net)
        proj = type('FakeProject', (), {})()
        proj.networkConfig = NetworkConfig(
            optimizer=opt_type, loss=LossType.CROSS_ENTROPY,
            metric=[MetricType.ACCURACY], learningRate=0.001
        )
        try:
            optimizer = PyTorchTrainer._build_optimizer(model, proj.networkConfig)
            check(f"{opt_type.value} optimizer builds", isinstance(optimizer, torch.optim.Optimizer))
        except Exception as e:
            check(f"{opt_type.value} optimizer builds", False)


def test_trainer_all_losses():
    section("Trainer — all loss types build without error")
    for loss_type in LossType:
        proj = type('FakeProject', (), {})()
        proj.networkConfig = NetworkConfig(
            optimizer=OptimizerType.ADAM, loss=loss_type,
            metric=[MetricType.ACCURACY], learningRate=0.001
        )
        try:
            criterion = PyTorchTrainer._build_loss(proj.networkConfig)
            check(f"{loss_type.value} loss builds", isinstance(criterion, nn.Module))
        except Exception as e:
            check(f"{loss_type.value} loss builds", False)


def test_trainer_not_ready_raises():
    section("Trainer — raises if project not ready")
    proj = Project(name="empty", path="/tmp/empty")
    expect_exception("missing network raises ValueError",
                     lambda: PyTorchTrainer.train(proj), ValueError)


def test_trainer_weights_loadable():
    section("Trainer — saved weights can be reloaded")
    with tempfile.TemporaryDirectory() as tmp:
        net    = _make_dense_network()
        result = _run_trainer_with_fake_data(net, (784,), 10, tmp)

        # Rebuild identical architecture and load saved weights
        net2   = _make_dense_network()
        model2 = PyTorchModelBuilder.build(net2)
        state  = torch.load(result.weightsPath, weights_only=True)
        try:
            model2.load_state_dict(state)
            check("weights load into fresh model", True)
        except Exception as e:
            check("weights load into fresh model", False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PyTorchCodeGenerator
# ─────────────────────────────────────────────────────────────────────────────

def _make_gen_project(net, dataset_type=BuiltinDataset.MNIST):
    proj = _make_full_project(net, dataset_type=dataset_type)
    proj.path = "/tmp/gen_test"
    return proj


def test_generator_produces_string():
    section("CodeGenerator — returns a non-empty string")
    proj = _make_gen_project(_make_dense_network())
    code = PyTorchCodeGenerator.generate(proj)
    check("output is a str",                isinstance(code, str))
    check("output is non-empty",            len(code) > 100)


def test_generator_header():
    section("CodeGenerator — header contains project metadata")
    proj = _make_gen_project(_make_dense_network())
    code = PyTorchCodeGenerator.generate(proj)
    check("project name in header",         "test_project" in code)
    check("optimizer in header",            "adam" in code)
    check("loss in header",                 "cross_entropy" in code)
    check("epochs in header",               "epochs" in code.lower())


def test_generator_imports():
    section("CodeGenerator — imports block is correct")
    proj = _make_gen_project(_make_dense_network())
    code = PyTorchCodeGenerator.generate(proj)
    check("imports torch",                  "import torch" in code)
    check("imports nn",                     "import torch.nn as nn" in code)
    check("imports optim",                  "import torch.optim as optim" in code)
    check("imports DataLoader",             "DataLoader" in code)
    check("imports torchvision for builtin","from torchvision import datasets" in code)


def test_generator_model_class():
    section("CodeGenerator — Model class structure")
    proj = _make_gen_project(_make_dense_network())
    code = PyTorchCodeGenerator.generate(proj)
    check("class Model defined",            "class Model(nn.Module):" in code)
    check("__init__ present",               "def __init__(self):" in code)
    check("forward present",                "def forward(self, x):" in code)
    check("nn.Sequential used",             "nn.Sequential" in code)
    check("nn.Linear present",              "nn.Linear" in code)
    check("nn.ReLU present",                "nn.ReLU" in code)


def test_generator_cnn_model_class():
    section("CodeGenerator — CNN model class has correct layers")
    proj = _make_gen_project(_make_cnn_network())
    code = PyTorchCodeGenerator.generate(proj)
    check("nn.Conv2d present",              "nn.Conv2d" in code)
    check("nn.MaxPool2d present",           "nn.MaxPool2d" in code)
    check("nn.Flatten present",             "nn.Flatten" in code)
    check("nn.Linear present",              "nn.Linear" in code)


def test_generator_recurrent_wrapper_emitted():
    section("CodeGenerator — RecurrentWrapper emitted for RNN networks")
    proj_rnn   = _make_gen_project(_make_rnn_network())
    proj_dense = _make_gen_project(_make_dense_network())
    code_rnn   = PyTorchCodeGenerator.generate(proj_rnn)
    code_dense = PyTorchCodeGenerator.generate(proj_dense)

    check("RNN code has RecurrentWrapper",  "class RecurrentWrapper" in code_rnn)
    check("Dense code has no wrapper",      "class RecurrentWrapper" not in code_dense)
    check("nn.LSTM in RNN code",            "nn.LSTM" in code_rnn)


def test_generator_dataset_block():
    section("CodeGenerator — dataset block correct for MNIST")
    proj = _make_gen_project(_make_dense_network())
    code = PyTorchCodeGenerator.generate(proj)
    check("get_dataloaders defined",        "def get_dataloaders" in code)
    check("MNIST class used",               "datasets.MNIST" in code)
    check("correct mean in normalize",      "0.1307" in code)
    check("train=True for MNIST",           "train=True" in code)
    check("random_split present",           "random_split" in code)


def test_generator_training_block():
    section("CodeGenerator — training block correct")
    proj = _make_gen_project(_make_dense_network())
    code = PyTorchCodeGenerator.generate(proj)
    check("train_one_epoch defined",        "def train_one_epoch" in code)
    check("evaluate defined",               "def evaluate" in code)
    check("build_optimizer defined",        "def build_optimizer_and_loss" in code)
    check("Adam optimizer in code",         "optim.Adam" in code)
    check("CrossEntropyLoss in code",       "nn.CrossEntropyLoss" in code)


def test_generator_main_block():
    section("CodeGenerator — __main__ block present")
    proj = _make_gen_project(_make_dense_network())
    code = PyTorchCodeGenerator.generate(proj)
    check("__main__ guard present",         'if __name__ == "__main__"' in code)
    check("device detection present",       "torch.device" in code)
    check("epoch loop present",             "for epoch in range" in code)
    check("torch.save present",             "torch.save" in code)


def test_generator_early_stop_block():
    section("CodeGenerator — early stop block emitted when configured")
    proj_es     = _make_gen_project(_make_dense_network())
    proj_no_es  = _make_gen_project(_make_dense_network())
    proj_es.trainingConfig._earlyStop    = True
    proj_no_es.trainingConfig._earlyStop = False

    code_es    = PyTorchCodeGenerator.generate(proj_es)
    code_no_es = PyTorchCodeGenerator.generate(proj_no_es)
    check("early stop code present when enabled",  "patience" in code_es)
    check("early stop code absent when disabled",  "patience" not in code_no_es)


def test_generator_cifar10_stats():
    section("CodeGenerator — CIFAR10 gets 3-channel normalize stats")
    net  = Network("cifar_net")
    net.addLayer(InputLayer((3, 32, 32)))
    net.addLayer(Conv2D(filters=4, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                        dropout=0.0, dropoutStyle=None, id="c1", name="C1", role="hidden"))
    net.addLayer(Flatten(id="fl", name="Flat", role="hidden"))
    net.addLayer(Dense(unit=10, useBias=True, dropout=0.0, id="d1", name="D1", role="output"))
    net.addConnect("input","c1"); net.addConnect("c1","fl"); net.addConnect("fl","d1")
    proj = _make_gen_project(net, dataset_type=BuiltinDataset.CIFAR10)
    code = PyTorchCodeGenerator.generate(proj)
    check("datasets.CIFAR10 used",          "datasets.CIFAR10" in code)
    check("3-channel mean in code",         "0.4914" in code)


def test_generator_no_duplicate_topological_helper():
    section("CodeGenerator — no _topological_order method (removed)")
    check("method removed from class",
          not hasattr(PyTorchCodeGenerator, '_topological_order'))


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # ModelBuilder
        test_builder_dense_network,
        test_builder_cnn_network,
        test_builder_conv_with_dropout,
        test_builder_rnn_network,
        test_builder_gru_bidirectional,
        test_builder_all_activations,
        test_builder_pooling_types,
        test_builder_invalid_network,
        # Datasets
        test_datasets_transforms,
        test_datasets_split_no_val,
        test_datasets_split_with_val,
        test_datasets_custom_raises,
        test_datasets_dataloader_shapes,
        # Trainer
        test_trainer_dense_completes,
        test_trainer_cnn_completes,
        test_trainer_history_grows_over_epochs,
        test_trainer_callback_fires,
        test_trainer_no_val_split,
        test_trainer_early_stopping,
        test_trainer_all_optimizers,
        test_trainer_all_losses,
        test_trainer_not_ready_raises,
        test_trainer_weights_loadable,
        # CodeGenerator
        test_generator_produces_string,
        test_generator_header,
        test_generator_imports,
        test_generator_model_class,
        test_generator_cnn_model_class,
        test_generator_recurrent_wrapper_emitted,
        test_generator_dataset_block,
        test_generator_training_block,
        test_generator_main_block,
        test_generator_early_stop_block,
        test_generator_cifar10_stats,
        test_generator_no_duplicate_topological_helper,
    ]

    for test in tests:
        try:
            test()
        except Exception:
            section(f"EXCEPTION in {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'═' * 54}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ✅ All infrastructure tests passed")
    else:
        print("  ❌ Some tests failed — check output above")
    print(f"{'═' * 54}")
    sys.exit(0 if failed == 0 else 1)