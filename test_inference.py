"""
verify_predict.py
─────────────────
Tests PyTorchTrainer.predict() and _to_tensor() — the parts not covered
by the existing test.py (which only tests the training loop).

Run from the project ROOT (the folder containing Core/ and PyTorch/):

    python verify_predict.py

No internet, no GPU, no real dataset downloads needed.
"""

import sys, os, tempfile, traceback
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from Core.layer import InputLayer, Flatten, Dense, Activation, ActivationType, Conv2D
from Core.network import Network
from Core.dataset import Dataset, BuiltinDataset
from Core.trainingConfig import TrainingConfig
from Core.networkConfig import NetworkConfig, OptimizerType, LossType, MetricType
from Core.project import Project
from Infra.trainer import PyTorchTrainer
from Infra.PytorchModelBuilder import PyTorchModelBuilder

# ── test harness ──────────────────────────────────────────────────────────────
passed = 0
failed = 0

def check(label, condition):
    global passed, failed
    if condition:
        print(f"  ✓  {label}")
        passed += 1
    else:
        print(f"  ✗  {label}   ← FAILED")
        failed += 1

def expect_exception(label, fn, exc_type):
    global passed, failed
    try:
        fn()
        print(f"  ✗  {label}   ← FAILED (no exception raised)")
        failed += 1
    except exc_type:
        print(f"  ✓  {label}")
        passed += 1
    except Exception as e:
        print(f"  ✗  {label}   ← FAILED (wrong exc {type(e).__name__}: {e})")
        failed += 1

def section(title):
    print(f"\n── {title} {'─' * max(1, 50 - len(title))}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_dense_net():
    """Simple 784 → 128 → 10 network."""
    net = Network("pred_dense")
    net.addLayer(InputLayer((784,)))
    net.addLayer(Dense(unit=128, useBias=True, dropout=0.0, id="d1", name="D1", role="hidden"))
    net.addLayer(Activation(ActivationType.RELU, id="a1", name="ReLU", role="hidden"))
    net.addLayer(Dense(unit=10, useBias=True, dropout=0.0, id="d2", name="D2", role="output"))
    net.addConnect("input", "d1")
    net.addConnect("d1", "a1")
    net.addConnect("a1", "d2")
    return net

def _make_cnn_net():
    """1×28×28 → Conv → Flatten → Dense(10)."""
    net = Network("pred_cnn")
    net.addLayer(InputLayer((1, 28, 28)))
    net.addLayer(Conv2D(filters=4, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                        dropout=0.0, dropoutStyle=None, id="c1", name="C1", role="hidden"))
    net.addLayer(Flatten(id="fl", name="Flat", role="hidden"))
    net.addLayer(Dense(unit=10, useBias=True, dropout=0.0, id="d1", name="D1", role="output"))
    net.addConnect("input", "c1")
    net.addConnect("c1", "fl")
    net.addConnect("fl", "d1")
    return net

def _save_weights(net, tmp_dir) -> Project:
    """
    Builds the model, saves random weights to disk, returns a Project
    with weightsPath set — simulating a post-training state without
    actually running a full training loop.
    """
    net.propagate_shapes()
    model = PyTorchModelBuilder.build(net)

    weights_path = os.path.join(tmp_dir, Project.WEIGHTS_FILENAME)
    torch.save(model.state_dict(), weights_path)

    proj = Project(name="pred_test", path=tmp_dir)
    proj.network = net
    proj.dataset = Dataset.fromBuiltin(BuiltinDataset.MNIST)
    proj.trainingConfig = TrainingConfig(32, 1, 0.1, True, False)
    proj.networkConfig  = NetworkConfig(OptimizerType.ADAM, LossType.CROSS_ENTROPY,
                                        [MetricType.ACCURACY], 0.001)
    proj.setWeightsPath(weights_path)
    return proj


# ── 1. _to_tensor ─────────────────────────────────────────────────────────────

def test_to_tensor_torch_3d():
    section("_to_tensor — 3D torch.Tensor gets batch dim")
    t = torch.randn(1, 28, 28)           # single image, no batch dim
    out = PyTorchTrainer._to_tensor(t)
    check("output is float32",  out.dtype == torch.float32)
    check("shape is (1,1,28,28)", out.shape == torch.Size([1, 1, 28, 28]))

def test_to_tensor_torch_4d():
    section("_to_tensor — 4D torch.Tensor passed through as-is")
    t = torch.randn(5, 1, 28, 28)        # already batched
    out = PyTorchTrainer._to_tensor(t)
    check("shape unchanged (5,1,28,28)", out.shape == torch.Size([5, 1, 28, 28]))

def test_to_tensor_torch_1d():
    section("_to_tensor — 1D torch.Tensor (flat vector) gets batch dim")
    t = torch.randn(784)
    out = PyTorchTrainer._to_tensor(t)
    check("shape is (1, 784)", out.shape == torch.Size([1, 784]))

def test_to_tensor_numpy_3d():
    section("_to_tensor — 3D numpy array converted correctly")
    arr = np.random.rand(1, 28, 28).astype(np.float32)
    out = PyTorchTrainer._to_tensor(arr)
    check("is torch.Tensor",   isinstance(out, torch.Tensor))
    check("dtype float32",     out.dtype == torch.float32)
    check("shape (1,1,28,28)", out.shape == torch.Size([1, 1, 28, 28]))

def test_to_tensor_numpy_4d():
    section("_to_tensor — 4D numpy array (already batched) passed through")
    arr = np.random.rand(3, 1, 28, 28).astype(np.float32)
    out = PyTorchTrainer._to_tensor(arr)
    check("shape (3,1,28,28)", out.shape == torch.Size([3, 1, 28, 28]))

def test_to_tensor_numpy_float64():
    section("_to_tensor — float64 numpy array cast to float32")
    arr = np.random.rand(784).astype(np.float64)
    out = PyTorchTrainer._to_tensor(arr)
    check("dtype is float32",  out.dtype == torch.float32)

def test_to_tensor_unsupported_type():
    section("_to_tensor — unsupported type raises TypeError")
    expect_exception(
        "list input raises TypeError",
        lambda: PyTorchTrainer._to_tensor([[1, 2, 3]]),
        TypeError
    )
    expect_exception(
        "string input raises TypeError",
        lambda: PyTorchTrainer._to_tensor("image.png"),
        TypeError
    )


# ── 2. predict() — output structure ──────────────────────────────────────────

def test_predict_output_keys():
    section("predict() — result dict has correct keys")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)
        inputs = torch.randn(4, 784)
        result = PyTorchTrainer.predict(proj, inputs)

        check("'classes' key present",       "classes"       in result)
        check("'probabilities' key present", "probabilities" in result)
        check("'confidence' key present",    "confidence"    in result)

def test_predict_output_lengths():
    section("predict() — output lists match batch size")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)
        n = 6
        inputs = torch.randn(n, 784)
        result = PyTorchTrainer.predict(proj, inputs)

        check(f"classes length = {n}",       len(result["classes"])       == n)
        check(f"probabilities length = {n}", len(result["probabilities"]) == n)
        check(f"confidence length = {n}",    len(result["confidence"])    == n)

def test_predict_class_indices_valid():
    section("predict() — class indices are valid integers in [0, num_classes)")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)
        inputs = torch.randn(8, 784)
        result = PyTorchTrainer.predict(proj, inputs)

        num_classes = 10
        check("all classes are ints",
              all(isinstance(c, int) for c in result["classes"]))
        check("all classes in [0, 9]",
              all(0 <= c < num_classes for c in result["classes"]))

def test_predict_probabilities_sum_to_one():
    section("predict() — probabilities sum to ~1.0 per sample")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)
        inputs = torch.randn(5, 784)
        result = PyTorchTrainer.predict(proj, inputs)

        for i, probs in enumerate(result["probabilities"]):
            check(f"sample {i} probs length = 10",   len(probs) == 10)
            check(f"sample {i} probs sum ≈ 1.0",
                  abs(sum(probs) - 1.0) < 1e-4)

def test_predict_confidence_is_max_prob():
    section("predict() — confidence equals max of probabilities")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)
        inputs = torch.randn(4, 784)
        result = PyTorchTrainer.predict(proj, inputs)

        for i, (conf, probs) in enumerate(
                zip(result["confidence"], result["probabilities"])):
            check(f"sample {i}: confidence == max(probs)",
                  abs(conf - max(probs)) < 1e-5)

def test_predict_confidence_range():
    section("predict() — confidence values are in [0, 1]")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)
        inputs = torch.randn(10, 784)
        result = PyTorchTrainer.predict(proj, inputs)
        check("all confidences in [0,1]",
              all(0.0 <= c <= 1.0 for c in result["confidence"]))


# ── 3. predict() — input type handling ───────────────────────────────────────

def test_predict_accepts_torch_tensor():
    section("predict() — accepts torch.Tensor input")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)
        t = torch.randn(3, 784)
        result = PyTorchTrainer.predict(proj, t)
        check("returns 3 predictions", len(result["classes"]) == 3)

def test_predict_accepts_numpy_array():
    section("predict() — accepts numpy array input")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)
        arr = np.random.rand(3, 784).astype(np.float32)
        result = PyTorchTrainer.predict(proj, arr)
        check("returns 3 predictions", len(result["classes"]) == 3)

def test_predict_single_sample_tensor():
    section("predict() — single sample (no batch dim) returns 1 prediction")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)
        single = torch.randn(784)        # no batch dim
        result = PyTorchTrainer.predict(proj, single)
        check("classes length = 1",       len(result["classes"])       == 1)
        check("confidence length = 1",    len(result["confidence"])    == 1)
        check("probabilities length = 1", len(result["probabilities"]) == 1)

def test_predict_cnn_input():
    section("predict() — CNN network with image input")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_cnn_net(), tmp)
        imgs = torch.randn(4, 1, 28, 28)
        result = PyTorchTrainer.predict(proj, imgs)
        check("returns 4 predictions",   len(result["classes"]) == 4)
        check("10 classes output",       len(result["probabilities"][0]) == 10)


# ── 4. predict() — error cases ────────────────────────────────────────────────

def test_predict_no_weights_raises():
    section("predict() — raises if no weights file exists")
    with tempfile.TemporaryDirectory() as tmp:
        proj = Project(name="no_weights", path=tmp)
        proj.network = _make_dense_net()
        proj.network.propagate_shapes()
        # weightsPath is None, and weights.pt doesn't exist on disk

        expect_exception(
            "missing weights raises ValueError",
            lambda: PyTorchTrainer.predict(proj, torch.randn(1, 784)),
            ValueError
        )

def test_predict_no_network_raises():
    section("predict() — raises if project has no network")
    with tempfile.TemporaryDirectory() as tmp:
        proj = Project(name="no_net", path=tmp)
        # network is None

        expect_exception(
            "None network raises ValueError",
            lambda: PyTorchTrainer.predict(proj, torch.randn(1, 784)),
            ValueError
        )

def test_predict_wrong_input_type_raises():
    section("predict() — raises TypeError for unsupported input type")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)

        expect_exception(
            "list input raises TypeError",
            lambda: PyTorchTrainer.predict(proj, [[0.1] * 784]),
            TypeError
        )

def test_predict_explicit_weights_path():
    section("predict() — respects explicit weightsPath over default path")
    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(_make_dense_net(), tmp)

        # Move the weights file to a non-default name to confirm
        # predict() uses weightsPath rather than guessing project.path/weights.pt
        custom_path = os.path.join(tmp, "custom_weights.pt")
        os.rename(proj.weightsPath, custom_path)
        proj.setWeightsPath(custom_path)

        result = PyTorchTrainer.predict(proj, torch.randn(2, 784))
        check("works with custom weights path", len(result["classes"]) == 2)


# ── 5. predict() — model is in eval mode ─────────────────────────────────────

def test_predict_deterministic_with_dropout():
    section("predict() — dropout off during inference (eval mode)")
    # Build a network with dropout and confirm two identical calls return
    # the same result (dropout would make them differ if training mode)
    net = Network("dropout_pred")
    net.addLayer(InputLayer((64,)))
    net.addLayer(Dense(unit=32, useBias=True, dropout=0.5, id="d1", name="D1", role="hidden"))
    net.addLayer(Dense(unit=4,  useBias=True, dropout=0.0, id="d2", name="D2", role="output"))
    net.addConnect("input", "d1")
    net.addConnect("d1", "d2")

    with tempfile.TemporaryDirectory() as tmp:
        proj = _save_weights(net, tmp)
        inputs = torch.randn(3, 64)

        result1 = PyTorchTrainer.predict(proj, inputs)
        result2 = PyTorchTrainer.predict(proj, inputs)

        check("classes identical across two calls",
              result1["classes"] == result2["classes"])
        check("confidence identical across two calls",
              result1["confidence"] == result2["confidence"])


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # _to_tensor
        test_to_tensor_torch_3d,
        test_to_tensor_torch_4d,
        test_to_tensor_torch_1d,
        test_to_tensor_numpy_3d,
        test_to_tensor_numpy_4d,
        test_to_tensor_numpy_float64,
        test_to_tensor_unsupported_type,
        # predict() output structure
        test_predict_output_keys,
        test_predict_output_lengths,
        test_predict_class_indices_valid,
        test_predict_probabilities_sum_to_one,
        test_predict_confidence_is_max_prob,
        test_predict_confidence_range,
        # predict() input types
        test_predict_accepts_torch_tensor,
        test_predict_accepts_numpy_array,
        test_predict_single_sample_tensor,
        test_predict_cnn_input,
        # predict() error cases
        test_predict_no_weights_raises,
        test_predict_no_network_raises,
        test_predict_wrong_input_type_raises,
        test_predict_explicit_weights_path,
        # eval mode
        test_predict_deterministic_with_dropout,
    ]

    print("=" * 54)
    print("  predict() verification")
    print("=" * 54)

    for test in tests:
        try:
            test()
        except Exception:
            section(f"EXCEPTION in {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 54}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  All predict() checks passed")
    else:
        print("  Some checks failed — see output above")
    print(f"{'=' * 54}\n")

    sys.exit(0 if failed == 0 else 1)