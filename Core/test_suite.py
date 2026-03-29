"""
test_suite.py
=============
Integration + unit tests for the full pipeline:
  - Layer shape propagation
  - Network graph operations
  - PyTorchModelBuilder  (architecture)
  - PyTorchTrainer       (training loop)
  - ProjectManager       (save / load / file I/O)

Run with:
    pytest test_suite.py -v
or:
    python test_suite.py
"""

import os
import json
import tempfile
import unittest

import torch
import torch.nn as nn

# ── project modules ───────────────────────────────────────────────────────────
from .layer import (
    Activation, ActivationType,
    Conv2D, Dense, Flatten, InputLayer,
    Pooling, PoolingType,
    Recurrent, RecurrentType,
    DropoutStyle,
)
from .connection import Connection
from .network import Network
from .dataset import Dataset, BuiltinDataset
from .trainingConfig import TrainingConfig
from .networkConfig import NetworkConfig, OptimizerType, LossType, MetricType
from .project import Project
from Infra.PytorchModelBuilder import PyTorchModelBuilder
from Infra.trainer import PyTorchTrainer
from .projectManager import ProjectManager


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cnn_network() -> Network:
    """
    Minimal CNN: Input(1,28,28) → Conv2D → ReLU → Pooling
                → Flatten → Dense → Softmax
    Matches MNIST shape.
    """
    net = Network("cnn-test")

    net.addLayer(InputLayer(inputShape=(1, 28, 28)))
    net.addLayer(Conv2D(filters=8, kernelSize=(3, 3), strides=(1, 1),
                        padding=(0, 0), dropout=0.0, dropoutStyle=None,
                        id="conv1", name="Conv1", role=None))
    net.addLayer(Activation(activationType=ActivationType.RELU,
                            id="relu1", name="ReLU1"))
    net.addLayer(Pooling(poolSize=(2, 2), strides=(2, 2), padding=(0, 0),
                         id="pool1", name="Pool1", role=None,
                         poolingType=PoolingType.MAX))
    net.addLayer(Flatten(id="flat1", name="Flatten1"))
    net.addLayer(Dense(unit=32, useBias=True, dropout=0.0,
                       id="dense1", name="Dense1", role=None))
    net.addLayer(Activation(activationType=ActivationType.RELU,
                            id="relu2", name="ReLU2"))
    net.addLayer(Dense(unit=10, useBias=True, dropout=0.0,
                       id="out", name="Output", role=None))
    net.addLayer(Activation(activationType=ActivationType.SOFTMAX,
                            id="softmax", name="Softmax"))

    net.addConnect("input", "conv1")
    net.addConnect("conv1", "relu1")
    net.addConnect("relu1", "pool1")
    net.addConnect("pool1", "flat1")
    net.addConnect("flat1", "dense1")
    net.addConnect("dense1", "relu2")
    net.addConnect("relu2", "out")
    net.addConnect("out",   "softmax")

    return net


def _make_mlp_network() -> Network:
    """
    Simple MLP: Input(784,) → Dense(128) → ReLU → Dense(10) → Softmax
    """
    net = Network("mlp-test")

    net.addLayer(InputLayer(inputShape=(784,)))
    net.addLayer(Dense(unit=128, useBias=True, dropout=0.0,
                       id="d1", name="Dense1", role=None))
    net.addLayer(Activation(activationType=ActivationType.RELU,
                            id="r1", name="ReLU1"))
    net.addLayer(Dense(unit=10, useBias=True, dropout=0.0,
                       id="out", name="Output", role=None))
    net.addLayer(Activation(activationType=ActivationType.SOFTMAX,
                            id="sm", name="Softmax"))

    net.addConnect("input", "d1")
    net.addConnect("d1",    "r1")
    net.addConnect("r1",    "out")
    net.addConnect("out",   "sm")

    return net


def _make_project(network: Network, tmpdir: str) -> Project:
    p = Project(name="test-project", path=tmpdir)
    p.network = network
    p.dataset = Dataset.fromBuiltin(BuiltinDataset.MNIST)
    p.trainingConfig = TrainingConfig(
        batchSize=64,
        epochs=1,
        validationSplit=0.1,
        shuffle=True,
        earlyStop=False,
    )
    p.networkConfig = NetworkConfig(
        optimizer=OptimizerType.ADAM,
        loss=LossType.CROSS_ENTROPY,
        metric=[MetricType.ACCURACY],
        learningRate=1e-3,
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1. Layer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLayers(unittest.TestCase):

    def test_dense_output_shape(self):
        layer = Dense(unit=64, useBias=True, dropout=0.0,
                      id="d", name="d", role=None)
        self.assertEqual(layer.getOutputShape((128,)), (64,))

    def test_dense_rejects_2d_input(self):
        layer = Dense(unit=64, useBias=True, dropout=0.0,
                      id="d", name="d", role=None)
        with self.assertRaises(ValueError):
            layer.getOutputShape((3, 32, 32))

    def test_conv2d_output_shape(self):
        layer = Conv2D(filters=16, kernelSize=(3, 3), strides=(1, 1),
                       padding=(0, 0), dropout=0.0, dropoutStyle=None,
                       id="c", name="c", role=None)
        # (1, 28, 28) → (16, 26, 26)
        self.assertEqual(layer.getOutputShape((1, 28, 28)), (16, 26, 26))

    def test_conv2d_with_padding(self):
        layer = Conv2D(filters=8, kernelSize=(3, 3), strides=(1, 1),
                       padding=(1, 1), dropout=0.0, dropoutStyle=None,
                       id="c", name="c", role=None)
        # same padding: output H/W unchanged
        self.assertEqual(layer.getOutputShape((1, 28, 28)), (8, 28, 28))

    def test_pooling_output_shape(self):
        layer = Pooling(poolSize=(2, 2), strides=(2, 2), padding=(0, 0),
                        id="p", name="p", role=None)
        self.assertEqual(layer.getOutputShape((8, 26, 26)), (8, 13, 13))

    def test_flatten_output_shape(self):
        layer = Flatten(id="f", name="f")
        self.assertEqual(layer.getOutputShape((8, 13, 13)), (8 * 13 * 13,))

    def test_flatten_already_1d(self):
        layer = Flatten(id="f", name="f")
        self.assertEqual(layer.getOutputShape((512,)), (512,))

    def test_activation_passthrough(self):
        layer = Activation(activationType=ActivationType.RELU,
                           id="a", name="a")
        self.assertEqual(layer.getOutputShape((64,)), (64,))

    def test_recurrent_return_seq_false(self):
        layer = Recurrent(hiddenSize=32, numLayer=1, dropout=0.0,
                          id="r", name="r", role=None,
                          bidirection=False, returnSeq=False)
        self.assertEqual(layer.getOutputShape((10, 16)), (32,))

    def test_recurrent_return_seq_true(self):
        layer = Recurrent(hiddenSize=32, numLayer=1, dropout=0.0,
                          id="r", name="r", role=None,
                          bidirection=False, returnSeq=True)
        self.assertEqual(layer.getOutputShape((10, 16)), (10, 32))

    def test_recurrent_bidirectional(self):
        layer = Recurrent(hiddenSize=32, numLayer=1, dropout=0.0,
                          id="r", name="r", role=None,
                          bidirection=True, returnSeq=False)
        self.assertEqual(layer.getOutputShape((10, 16)), (64,))

    def test_input_layer_shape(self):
        layer = InputLayer(inputShape=(3, 32, 32))
        self.assertEqual(layer._outputShape, (3, 32, 32))
        self.assertTrue(layer._isInput)
        self.assertTrue(layer._isLocked)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Network / graph tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNetwork(unittest.TestCase):

    def test_add_layer(self):
        net = Network("n")
        net.addLayer(InputLayer(inputShape=(1, 28, 28)))
        self.assertIn("input", net.layers)

    def test_duplicate_layer_rejected(self):
        net = Network("n")
        net.addLayer(InputLayer(inputShape=(1, 28, 28)))
        result = net.addLayer(InputLayer(inputShape=(1, 28, 28)))
        self.assertFalse(result)

    def test_remove_layer(self):
        net = Network("n")
        net.addLayer(Dense(unit=10, useBias=True, dropout=0.0,
                           id="d", name="d", role=None))
        self.assertTrue(net.removeLayer("d"))
        self.assertNotIn("d", net.layers)

    def test_locked_layer_not_removed(self):
        net = Network("n")
        net.addLayer(InputLayer(inputShape=(784,)))
        self.assertFalse(net.removeLayer("input"))

    def test_add_connection(self):
        net = Network("n")
        net.addLayer(InputLayer(inputShape=(784,)))
        net.addLayer(Dense(unit=10, useBias=True, dropout=0.0,
                           id="d", name="d", role=None))
        self.assertTrue(net.addConnect("input", "d"))

    def test_self_loop_rejected(self):
        net = Network("n")
        net.addLayer(InputLayer(inputShape=(784,)))
        self.assertFalse(net.addConnect("input", "input"))

    def test_cycle_rejected(self):
        net = Network("n")
        net.addLayer(InputLayer(inputShape=(784,)))
        net.addLayer(Dense(unit=64, useBias=True, dropout=0.0,
                           id="d1", name="d1", role=None))
        net.addLayer(Dense(unit=10, useBias=True, dropout=0.0,
                           id="d2", name="d2", role=None))
        net.addConnect("input", "d1")
        net.addConnect("d1", "d2")
        # closing a cycle d2 → d1 must be rejected
        self.assertFalse(net.addConnect("d2", "d1"))

    def test_topological_sort_order(self):
        net = _make_mlp_network()
        order = [l._id for l in net.topological_sort()]
        self.assertLess(order.index("input"), order.index("d1"))
        self.assertLess(order.index("d1"),    order.index("r1"))
        self.assertLess(order.index("r1"),    order.index("out"))

    def test_propagate_shapes_mlp(self):
        net = _make_mlp_network()
        net.propagate_shapes()
        self.assertEqual(net.layers["d1"]._inputShape,  (784,))
        self.assertEqual(net.layers["d1"]._outputShape, (128,))
        self.assertEqual(net.layers["out"]._outputShape, (10,))

    def test_propagate_shapes_cnn(self):
        net = _make_cnn_network()
        net.propagate_shapes()
        # after Conv(8,3x3,no-pad): (1,28,28) → (8,26,26)
        self.assertEqual(net.layers["conv1"]._outputShape, (8, 26, 26))
        # after MaxPool(2x2,stride2): (8,26,26) → (8,13,13)
        self.assertEqual(net.layers["pool1"]._outputShape, (8, 13, 13))
        # after Flatten: 8*13*13 = 1352
        self.assertEqual(net.layers["flat1"]._outputShape, (1352,))

    def test_validate_valid_network(self):
        self.assertTrue(_make_mlp_network().validate())

    def test_validate_empty_network(self):
        self.assertFalse(Network("empty").validate())

    def test_get_input_layers(self):
        net = _make_mlp_network()
        inputs = net.getInputLayers()
        self.assertEqual(len(inputs), 1)
        self.assertTrue(inputs[0]._isInput)

    def test_get_output_layers(self):
        net = _make_mlp_network()
        outputs = net.getOutputLayers()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]._id, "sm")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Serialization tests  (Network + Project round-trip)
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialization(unittest.TestCase):

    def _round_trip(self, net: Network) -> Network:
        return Network.fromDict(net.toDict())

    def test_mlp_round_trip(self):
        net = _make_mlp_network()
        net.propagate_shapes()
        restored = self._round_trip(net)
        self.assertEqual(set(net.layers.keys()), set(restored.layers.keys()))
        self.assertEqual(len(net.connections), len(restored.connections))

    def test_cnn_round_trip(self):
        net = _make_cnn_network()
        net.propagate_shapes()
        restored = self._round_trip(net)
        self.assertEqual(set(net.layers.keys()), set(restored.layers.keys()))

    def test_shapes_preserved_after_round_trip(self):
        net = _make_mlp_network()
        net.propagate_shapes()
        restored = self._round_trip(net)
        self.assertEqual(
            restored.layers["d1"]._outputShape,
            net.layers["d1"]._outputShape,
        )

    def test_project_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_project(_make_mlp_network(), tmp)
            p.network.propagate_shapes()
            restored = Project.fromDict(p.toDict())
            self.assertEqual(p.name, restored.name)
            self.assertIsNotNone(restored.network)
            self.assertIsNotNone(restored.dataset)
            self.assertIsNotNone(restored.trainingConfig)
            self.assertIsNotNone(restored.networkConfig)

    def test_dataset_round_trip(self):
        ds = Dataset.fromBuiltin(BuiltinDataset.CIFAR10)
        restored = Dataset.fromDict(ds.toDict())
        self.assertEqual(ds._inputShape, restored._inputShape)
        self.assertEqual(ds._numClasses, restored._numClasses)

    def test_training_config_round_trip(self):
        tc = TrainingConfig(batchSize=32, epochs=5, validationSplit=0.2,
                            shuffle=True, earlyStop=True)
        restored = TrainingConfig.fromDict(tc.toDict())
        self.assertEqual(tc._batchSize, restored._batchSize)
        self.assertEqual(tc._epochs,    restored._epochs)

    def test_network_config_round_trip(self):
        nc = NetworkConfig(optimizer=OptimizerType.SGD,
                           loss=LossType.CROSS_ENTROPY,
                           metric=[MetricType.ACCURACY, MetricType.F1_SCORE],
                           learningRate=0.01)
        restored = NetworkConfig.fromDict(nc.toDict())
        self.assertEqual(nc._optimizer, restored._optimizer)
        self.assertEqual(nc._metric,    restored._metric)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PyTorchModelBuilder tests  (architecture)
# ─────────────────────────────────────────────────────────────────────────────

class TestModelBuilder(unittest.TestCase):

    def test_mlp_builds_sequential(self):
        net = _make_mlp_network()
        model = PyTorchModelBuilder.build(net)
        self.assertIsInstance(model, nn.Sequential)

    def test_cnn_builds_sequential(self):
        net = _make_cnn_network()
        model = PyTorchModelBuilder.build(net)
        self.assertIsInstance(model, nn.Sequential)

    def test_mlp_forward_pass(self):
        net = _make_mlp_network()
        model = PyTorchModelBuilder.build(net)
        x = torch.randn(4, 784)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_cnn_forward_pass(self):
        net = _make_cnn_network()
        model = PyTorchModelBuilder.build(net)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_linear_activation_is_no_op(self):
        """LINEAR activation must not add a module."""
        net = Network("linear-test")
        net.addLayer(InputLayer(inputShape=(64,)))
        net.addLayer(Dense(unit=10, useBias=True, dropout=0.0,
                           id="d", name="d", role=None))
        net.addLayer(Activation(activationType=ActivationType.LINEAR,
                                id="lin", name="lin"))
        net.addConnect("input", "d")
        net.addConnect("d",     "lin")
        model = PyTorchModelBuilder.build(net)
        # only nn.Linear, no extra activation module
        self.assertEqual(len(list(model.children())), 1)
        self.assertIsInstance(list(model.children())[0], nn.Linear)

    def test_dense_with_dropout(self):
        net = Network("dropout-test")
        net.addLayer(InputLayer(inputShape=(128,)))
        net.addLayer(Dense(unit=64, useBias=True, dropout=0.5,
                           id="d", name="d", role=None))
        net.addConnect("input", "d")
        model = PyTorchModelBuilder.build(net)
        children = list(model.children())
        self.assertEqual(len(children), 2)
        self.assertIsInstance(children[1], nn.Dropout)

    def test_conv2d_spatial_dropout(self):
        net = Network("spatialdrop-test")
        net.addLayer(InputLayer(inputShape=(1, 28, 28)))
        net.addLayer(Conv2D(filters=8, kernelSize=(3, 3), strides=(1, 1),
                            padding=(0, 0), dropout=0.3,
                            dropoutStyle=DropoutStyle.SPATIAL,
                            id="c", name="c", role=None))
        net.addConnect("input", "c")
        model = PyTorchModelBuilder.build(net)
        children = list(model.children())
        self.assertIsInstance(children[1], nn.Dropout2d)

    def test_recurrent_wrapper_last_timestep(self):
        """returnSeq=False must return shape (batch, hidden), not (batch, seq, hidden)."""
        net = Network("rnn-test")
        net.addLayer(InputLayer(inputShape=(10, 16)))
        net.addLayer(Recurrent(hiddenSize=32, numLayer=1, dropout=0.0,
                               id="rnn", name="rnn", role=None,
                               bidirection=False, returnSeq=False,
                               recurrentType=RecurrentType.LSTM))
        net.addConnect("input", "rnn")
        model = PyTorchModelBuilder.build(net)
        x = torch.randn(4, 10, 16)
        out = model(x)
        self.assertEqual(out.shape, (4, 32))

    def test_invalid_network_raises(self):
        with self.assertRaises(ValueError):
            PyTorchModelBuilder.build(Network("empty"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. ProjectManager tests  (infra / file I/O)
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectManager(unittest.TestCase):

    def _saved_project(self, tmpdir):
        """Helper: create, populate, and save a project, then return its path."""
        p = _make_project(_make_mlp_network(), tmpdir)
        p.network.propagate_shapes()
        p.updateLogs({"train_loss": [0.5, 0.4], "val_accuracy": [80.0, 85.0]})
        ProjectManager.save(p)
        return p

    def test_save_creates_project_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._saved_project(tmp)
            self.assertTrue(
                os.path.isfile(os.path.join(tmp, Project.ARCHITECTURE_FILENAME))
            )

    def test_save_creates_logs_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._saved_project(tmp)
            self.assertTrue(
                os.path.isfile(os.path.join(tmp, Project.LOGS_FILENAME))
            )

    def test_project_json_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._saved_project(tmp)
            path = os.path.join(tmp, Project.ARCHITECTURE_FILENAME)
            with open(path) as f:
                data = json.load(f)
            self.assertIn("network", data)
            self.assertIn("dataset", data)

    def test_load_restores_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = self._saved_project(tmp)
            restored = ProjectManager.load(tmp)
            self.assertEqual(original.name, restored.name)
            self.assertIsNotNone(restored.network)
            self.assertIsNotNone(restored.dataset)

    def test_load_restores_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._saved_project(tmp)
            restored = ProjectManager.load(tmp)
            self.assertIn("train_loss", restored.trainingLogs)

    def test_load_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            ProjectManager.load("/tmp/this_does_not_exist_xyz")

    def test_exists_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._saved_project(tmp)
            self.assertTrue(ProjectManager.exists(tmp))

    def test_exists_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(ProjectManager.exists(tmp))

    def test_has_logs_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._saved_project(tmp)
            self.assertTrue(ProjectManager.has_logs(tmp))

    def test_has_weights_false_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._saved_project(tmp)
            # weights.pt not written by ProjectManager.save(), only by trainer
            self.assertFalse(ProjectManager.has_weights(tmp))

    def test_network_survives_full_serialization_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = self._saved_project(tmp)
            restored = ProjectManager.load(tmp)
            orig_layers  = set(original.network.layers.keys())
            rest_layers  = set(restored.network.layers.keys())
            self.assertEqual(orig_layers, rest_layers)


# ─────────────────────────────────────────────────────────────────────────────
# 6. End-to-end training test  (slow — downloads MNIST on first run)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):
    """
    Runs one real training epoch on MNIST.
    Skipped automatically if torch.cuda is unavailable and the environment
    variable SKIP_TRAINING_TEST is set to 1 (useful for fast CI).
    """

    def setUp(self):
        if os.environ.get("SKIP_TRAINING_TEST") == "1":
            self.skipTest("SKIP_TRAINING_TEST=1")

    def test_full_pipeline_cnn(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = _make_project(_make_cnn_network(), tmp)

            epochs_seen = []
            def on_epoch(epoch, total, train_loss, val_metrics):
                epochs_seen.append(epoch)
                self.assertGreater(train_loss, 0.0)

            project = PyTorchTrainer.train(project, on_epoch=on_epoch)

            # ── training ran ──────────────────────────────────────────────────
            self.assertEqual(epochs_seen, [1])

            # ── weights saved to disk ─────────────────────────────────────────
            self.assertIsNotNone(project.weightsPath)
            self.assertTrue(os.path.isfile(project.weightsPath))

            # ── logs populated ────────────────────────────────────────────────
            self.assertIn("train_loss", project.trainingLogs)
            self.assertEqual(len(project.trainingLogs["train_loss"]), 1)

            # ── ProjectManager saves + loads correctly ────────────────────────
            ProjectManager.save(project)
            restored = ProjectManager.load(tmp)

            self.assertEqual(restored.name, project.name)
            self.assertIn("train_loss", restored.trainingLogs)
            self.assertTrue(ProjectManager.has_weights(tmp))

            # ── restored model produces same output shapes ────────────────────
            model = PyTorchModelBuilder.build(restored.network)
            x = torch.randn(2, 1, 28, 28)
            out = model(x)
            self.assertEqual(out.shape, (2, 10))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)