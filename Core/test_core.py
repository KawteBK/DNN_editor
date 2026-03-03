import sys
import traceback

from layer import (InputLayer, Dense, Conv2D, Recurrent,
                   Pooling, Activation, ActivationType)
from connection import Connection
from network import Network
from networkConfig import NetworkConfig, OptimizerType, LossType, MetricType
from trainingConfig import TrainingConfig
from dataset import Dataset, DatasetSource, BuiltinDataset, BUILTIN_METADATA
from project import Project


# ── Test runner ───────────────────────────────────────────────────────────────

passed = 0
failed = 0

def check(label: str, condition: bool):
    global passed, failed
    if condition:
        print(f"  ✓ {label}")
        passed += 1
    else:
        print(f"  ✗ {label}  ← FAILED")
        failed += 1

def section(title: str):
    print(f"\n── {title} {'─' * max(1, 52 - len(title))}")

def expect_exception(label: str, fn, exc_type=Exception):
    global passed, failed
    try:
        fn()
        print(f"  ✗ {label}  ← FAILED (no exception raised)")
        failed += 1
    except exc_type:
        print(f"  ✓ {label}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {label}  ← FAILED (wrong exception: {type(e).__name__}: {e})")
        failed += 1

# ── Layer constructors ────────────────────────────────────────────────────────
# After the design change:
# - inputShape and outputShape are NOT constructor parameters (except InputLayer)
# - they start as None and are set by propagate_shapes()
# - only InputLayer takes inputShape since it has no parent

def make_dense(unit=128, id="d1", name="D1", role="hidden"):
    return Dense(unit=unit, useBias=True, dropout=0.0, id=id, name=name, role=role)

def make_conv(filters=16, id="c1", name="C1", role="hidden"):
    return Conv2D(filters=filters, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                  dropout=0.0, dropoutStyle=None, id=id, name=name, role=role)

def make_pool(id="p1", name="P1", role="hidden"):
    return Pooling(poolSize=(2,2), strides=(2,2), padding=(0,0),
                   id=id, name=name, role=role)

def make_rnn(hidden=64, bi=False, seq=True, id="r1", name="R1", role="hidden"):
    return Recurrent(hiddenSize=hidden, numLayer=1, dropout=0.0,
                     id=id, name=name, role=role, bidirection=bi, returnSeq=seq)

def make_act(act_type=ActivationType.RELU, id="a1", name="A1", role="hidden"):
    return Activation(activationType=act_type, id=id, name=name, role=role)


# ═════════════════════════════════════════════════════════════════════════════
# 1. LAYER TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_input_layer():
    section("InputLayer")
    inp = InputLayer(inputShape=(3, 224, 224))
    check("outputShape set correctly",          inp._outputShape == (3, 224, 224))
    check("inputShape is None",                 inp._inputShape is None)
    check("_isInput is True",                   inp._isInput == True)
    check("_isLocked is True",                  inp._isLocked == True)
    check("id is 'input'",                      inp._id == "input")


def test_dense_layer():
    section("Dense Layer — getOutputShape")
    d = make_dense(unit=128)
    check("shapes start as None",               d._inputShape is None and d._outputShape is None)
    check("1D: (512,) → (128,)",                d.getOutputShape((512,)) == (128,))
    check("_isInput is False",                  d._isInput == False)
    check("_isLocked is False",                 d._isLocked == False)

    # Dense should reject non-1D input since we enforce flat input
    expect_exception("3D input raises ValueError",
                     lambda: make_dense().getOutputShape((1, 28, 28)), ValueError)
    expect_exception("2D input raises ValueError",
                     lambda: make_dense().getOutputShape((10, 512)), ValueError)
    expect_exception("empty tuple raises ValueError",
                     lambda: make_dense().getOutputShape(()), ValueError)


def test_conv2d_layer():
    section("Conv2D Layer — getOutputShape")
    check("shapes start as None",
          make_conv()._inputShape is None and make_conv()._outputShape is None)
    check("same padding preserves dims",
          make_conv(16).getOutputShape((3, 224, 224)) == (16, 224, 224))
    check("no padding reduces dims",
          Conv2D(32,(3,3),(1,1),(0,0),0.0,None,"c","C","hidden").getOutputShape((3,224,224)) == (32,222,222))
    check("stride=2 halves dims",
          Conv2D(64,(3,3),(2,2),(1,1),0.0,None,"c","C","hidden").getOutputShape((3,224,224)) == (64,112,112))

    expect_exception("1D input raises ValueError",
                     lambda: make_conv().getOutputShape((224,)), ValueError)
    expect_exception("2D input raises ValueError",
                     lambda: make_conv().getOutputShape((224, 224)), ValueError)


def test_pooling_layer():
    section("Pooling Layer — getOutputShape")
    check("shapes start as None",
          make_pool()._inputShape is None and make_pool()._outputShape is None)
    check("2x2 pool halves dims",
          make_pool().getOutputShape((32, 224, 224)) == (32, 112, 112))
    check("channels preserved",
          make_pool().getOutputShape((64, 56, 56))[0] == 64)
    check("stride=1 padded no size change",
          Pooling((3,3),(1,1),(1,1),"p","P","hidden").getOutputShape((16,32,32)) == (16,32,32))

    expect_exception("2D input raises ValueError",
                     lambda: make_pool().getOutputShape((224, 224)), ValueError)


def test_recurrent_layer():
    section("Recurrent Layer — getOutputShape")
    check("shapes start as None",
          make_rnn()._inputShape is None and make_rnn()._outputShape is None)
    check("uni + returnSeq=True",
          make_rnn(64, False, True).getOutputShape((10, 32)) == (10, 64))
    check("uni + returnSeq=False",
          make_rnn(64, False, False).getOutputShape((10, 32)) == (64,))
    check("bi + returnSeq=False doubles hidden",
          make_rnn(64, True, False).getOutputShape((10, 32)) == (128,))
    check("bi + returnSeq=True",
          make_rnn(64, True, True).getOutputShape((10, 32)) == (10, 128))

    expect_exception("1D input raises ValueError",
                     lambda: make_rnn().getOutputShape((10,)), ValueError)
    expect_exception("3D input raises ValueError",
                     lambda: make_rnn().getOutputShape((10, 32, 4)), ValueError)


def test_activation_layer():
    section("Activation Layer — getOutputShape")
    for act_type in ActivationType:
        act = make_act(act_type)
        check(f"{act_type.value}: shapes start as None",
              act._inputShape is None and act._outputShape is None)
        check(f"{act_type.value}: passthrough 1D",
              act.getOutputShape((128,)) == (128,))
        check(f"{act_type.value}: passthrough 3D",
              act.getOutputShape((32, 16, 16)) == (32, 16, 16))


def test_layer_instance_independence():
    section("Layer instance independence")
    inp   = InputLayer(inputShape=(10,))
    dense = make_dense()
    check("_isInput doesn't bleed to Dense",    dense._isInput == False)
    check("_isLocked doesn't bleed to Dense",   dense._isLocked == False)
    check("Two InputLayers are independent",
          InputLayer((3,32,32))._outputShape != InputLayer((1,28,28))._outputShape)
    check("Dense shapes are None by default",
          dense._inputShape is None and dense._outputShape is None)


# ═════════════════════════════════════════════════════════════════════════════
# 2. CONNECTION TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_connection():
    section("Connection")
    c = Connection("a->b", "a", "b")
    check("id stored",                          c._id == "a->b")
    check("fromLayer stored",                   c._fromLayer == "a")
    check("toLayer stored",                     c._toLayer == "b")
    check("isValid with both layers",           c.isValid() == True)
    check("isValid with None fromLayer",        Connection("x", None, "b").isValid() == False)


# ═════════════════════════════════════════════════════════════════════════════
# 3. NETWORK STRUCTURE TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_network_add_layers():
    section("Network — addLayer")
    net = Network("n1")
    inp = InputLayer((3, 32, 32))
    d1  = make_dense(id="d1")

    check("addLayer succeeds",                  net.addLayer(inp) == True)
    check("addLayer second succeeds",           net.addLayer(d1) == True)
    check("duplicate rejected",                 net.addLayer(inp) == False)
    check("_outgoing initialized",              "input" in net._outgoing)
    check("_incoming initialized",              "input" in net._incoming)
    check("new node _outgoing empty",           net._outgoing["input"] == [])
    check("new node _incoming empty",           net._incoming["d1"] == [])


def test_network_remove_layers():
    section("Network — removeLayer")
    net = Network("n2")
    inp = InputLayer((512,))
    d1  = make_dense(id="d1")
    d2  = make_dense(64, id="d2")

    net.addLayer(inp); net.addLayer(d1); net.addLayer(d2)
    net.addConnect("input", "d1")
    net.addConnect("d1", "d2")

    check("locked layer rejected",              net.removeLayer("input") == False)
    check("unknown layer rejected",             net.removeLayer("ghost") == False)
    check("removeLayer succeeds",               net.removeLayer("d1") == True)
    check("removed from layers",                "d1" not in net.layers)
    check("removed from _outgoing",             "d1" not in net._outgoing)
    check("removed from _incoming",             "d1" not in net._incoming)
    check("input _outgoing cleaned",            "d1" not in net._outgoing.get("input", []))
    check("d2 _incoming cleaned",               "d1" not in net._incoming.get("d2", []))
    check("flat connections cleaned",
          all(c._fromLayer != "d1" and c._toLayer != "d1" for c in net.connections))


def test_network_add_connections():
    section("Network — addConnect")
    net = Network("n3")
    inp = InputLayer((512,))
    d1  = make_dense(128, id="d1")
    d2  = make_dense(64,  id="d2")

    net.addLayer(inp); net.addLayer(d1); net.addLayer(d2)

    check("valid connection succeeds",          net.addConnect("input", "d1") == True)
    check("duplicate rejected",                 net.addConnect("input", "d1") == False)
    check("self-loop rejected",                 net.addConnect("d1", "d1") == False)
    check("connect TO input rejected",          net.addConnect("d1", "input") == False)
    check("unknown from-layer rejected",        net.addConnect("ghost", "d1") == False)
    check("unknown to-layer rejected",          net.addConnect("d1", "ghost") == False)

    net.addConnect("d1", "d2")
    check("_outgoing updated",                  net._outgoing["input"] == ["d1"])
    check("_incoming updated",                  net._incoming["d1"] == ["input"])
    check("connection in flat list",
          any(c._fromLayer == "d1" and c._toLayer == "d2" for c in net.connections))


def test_network_shape_compatibility():
    section("Network — shape compatibility via addConnect")
    net = Network("n_compat")

    # Conv2D → Dense should be rejected (3D → 1D mismatch)
    inp  = InputLayer((3, 32, 32))
    conv = make_conv(16, id="conv1")
    d1   = make_dense(128, id="d1")

    net.addLayer(inp); net.addLayer(conv); net.addLayer(d1)
    net.addConnect("input", "conv1")

    check("Conv→Dense rejected (shape mismatch)",
          net.addConnect("conv1", "d1") == False)
    check("adjacency rolled back after rejection",
          "d1" not in net._outgoing.get("conv1", []))
    check("connections list clean after rejection",
          not any(c._fromLayer == "conv1" and c._toLayer == "d1" for c in net.connections))

    # InputLayer(512,) → Dense should be accepted (1D → 1D match)
    net2 = Network("n_compat2")
    inp2 = InputLayer((512,))
    d2   = make_dense(128, id="d2")
    net2.addLayer(inp2); net2.addLayer(d2)
    check("flat input → Dense accepted",        net2.addConnect("input", "d2") == True)

    # InputLayer(3,32,32) → Conv2D should be accepted
    net3 = Network("n_compat3")
    inp3 = InputLayer((3, 32, 32))
    conv3 = make_conv(16, id="conv3")
    net3.addLayer(inp3); net3.addLayer(conv3)
    check("3D input → Conv2D accepted",         net3.addConnect("input", "conv3") == True)

    # Conv2D → Pooling should be accepted
    net4 = Network("n_compat4")
    inp4  = InputLayer((3, 32, 32))
    conv4 = make_conv(16, id="conv4")
    pool4 = make_pool(id="pool4")
    net4.addLayer(inp4); net4.addLayer(conv4); net4.addLayer(pool4)
    net4.addConnect("input", "conv4")
    check("Conv2D → Pooling accepted",          net4.addConnect("conv4", "pool4") == True)


def test_network_remove_connections():
    section("Network — removeConnect")
    net = Network("n4")
    inp = InputLayer((512,))
    d1  = make_dense(id="d1")

    net.addLayer(inp); net.addLayer(d1)
    net.addConnect("input", "d1")

    check("removeConnect succeeds",             net.removeConnect("input", "d1") == True)
    check("nonexistent rejected",               net.removeConnect("input", "d1") == False)
    check("_outgoing cleaned",                  "d1" not in net._outgoing["input"])
    check("_incoming cleaned",                  "input" not in net._incoming["d1"])
    check("flat connections cleaned",
          not any(c._fromLayer == "input" and c._toLayer == "d1" for c in net.connections))


def test_network_cycle_detection():
    section("Network — Cycle Detection")
    net = Network("n5")
    # use flat inputs so shape compatibility doesn't interfere
    inp = InputLayer((64,))
    a   = make_dense(64, id="a")
    b   = make_dense(64, id="b")
    c   = make_dense(64, id="c")

    net.addLayer(inp); net.addLayer(a); net.addLayer(b); net.addLayer(c)
    net.addConnect("input", "a")
    net.addConnect("a", "b")
    net.addConnect("b", "c")

    check("3-node cycle rejected",              net.addConnect("c", "a") == False)
    check("_outgoing rolled back",              "a" not in net._outgoing.get("c", []))
    check("_incoming rolled back",              "c" not in net._incoming.get("a", []))
    check("connections list rolled back",
          not any(x._fromLayer == "c" and x._toLayer == "a" for x in net.connections))
    check("self-cycle rejected",                net.addConnect("a", "a") == False)


def test_network_input_output_layers():
    section("Network — getInputLayers / getOutputLayers")
    net = Network("n6")
    inp = InputLayer((512,))
    d1  = make_dense(128, id="d1")
    d2  = make_dense(10,  id="d2")

    net.addLayer(inp); net.addLayer(d1); net.addLayer(d2)
    net.addConnect("input", "d1")
    net.addConnect("d1", "d2")

    input_ids  = [l._id for l in net.getInputLayers()]
    output_ids = [l._id for l in net.getOutputLayers()]

    check("input layer identified",             "input" in input_ids)
    check("output layer identified",            "d2" in output_ids)
    check("middle not an input",                "d1" not in input_ids)
    check("middle not an output",               "d1" not in output_ids)
    check("input not an output",                "input" not in output_ids)


def test_network_validate():
    section("Network — validate")
    net = Network("n7")
    check("empty network → invalid",            net.validate() == False)

    inp = InputLayer((512,))
    net.addLayer(inp)
    check("single node → valid",                net.validate() == True)

    d1 = make_dense(id="d1")
    net.addLayer(d1)
    net.addConnect("input", "d1")
    check("linear chain → valid",               net.validate() == True)


def test_topological_sort():
    section("Network — topological_sort")
    net = Network("n8")
    inp = InputLayer((512,))
    d1  = make_dense(128, id="d1")
    d2  = make_dense(10,  id="d2")

    net.addLayer(inp); net.addLayer(d1); net.addLayer(d2)
    net.addConnect("input", "d1")
    net.addConnect("d1", "d2")

    order = [l._id for l in net.topological_sort()]
    check("input before d1",                    order.index("input") < order.index("d1"))
    check("d1 before d2",                       order.index("d1") < order.index("d2"))
    check("all layers present",                 len(order) == 3)


def test_get_parents_children():
    section("Network — get_parents / get_children")
    net = Network("n9")
    inp = InputLayer((512,))
    d1  = make_dense(128, id="d1")
    d2  = make_dense(10,  id="d2")

    net.addLayer(inp); net.addLayer(d1); net.addLayer(d2)
    net.addConnect("input", "d1")
    net.addConnect("d1", "d2")

    check("get_parents of d1",                  net.get_parents("d1") == ["input"])
    check("get_parents of input is empty",      net.get_parents("input") == [])
    check("get_children of d1",                 net.get_children("d1") == ["d2"])
    check("get_children of d2 is empty",        net.get_children("d2") == [])
    check("unknown key returns []",             net.get_parents("ghost") == [])


# ═════════════════════════════════════════════════════════════════════════════
# 4. SHAPE PROPAGATION
# ═════════════════════════════════════════════════════════════════════════════

def test_shape_propagation_dense():
    section("Shape Propagation — Dense chain (flat input)")
    net = Network("sp1")
    inp = InputLayer((512,))
    d1  = make_dense(128, id="d1")
    d2  = make_dense(10,  id="d2")

    net.addLayer(inp); net.addLayer(d1); net.addLayer(d2)
    net.addConnect("input", "d1")
    net.addConnect("d1", "d2")
    net.propagate_shapes()

    check("d1 inputShape set",                  d1._inputShape == (512,))
    check("d1 outputShape correct",             d1._outputShape == (128,))
    check("d2 inputShape set",                  d2._inputShape == (128,))
    check("d2 outputShape correct",             d2._outputShape == (10,))


def test_shape_propagation_conv_pool():
    section("Shape Propagation — Conv2D → Pooling")
    net  = Network("sp2")
    inp  = InputLayer((3, 32, 32))
    conv = make_conv(16, id="conv1")
    pool = make_pool(id="pool1")

    net.addLayer(inp); net.addLayer(conv); net.addLayer(pool)
    net.addConnect("input", "conv1")
    net.addConnect("conv1", "pool1")
    net.propagate_shapes()

    check("conv inputShape set",                conv._inputShape == (3, 32, 32))
    check("conv outputShape correct",           conv._outputShape == (16, 32, 32))
    check("pool inputShape set",                pool._inputShape == (16, 32, 32))
    check("pool outputShape correct",           pool._outputShape == (16, 16, 16))


def test_shape_propagation_recurrent():
    section("Shape Propagation — Recurrent")
    net = Network("sp3")
    inp = InputLayer((10, 32))
    rnn = make_rnn(64, False, True, id="rnn1")

    net.addLayer(inp); net.addLayer(rnn)
    net.addConnect("input", "rnn1")
    net.propagate_shapes()

    check("rnn inputShape set",                 rnn._inputShape == (10, 32))
    check("rnn outputShape correct",            rnn._outputShape == (10, 64))


def test_shape_propagation_shapes_none_before():
    section("Shape Propagation — shapes are None before propagation")
    net = Network("sp4")
    inp = InputLayer((512,))
    d1  = make_dense(128, id="d1")

    net.addLayer(inp); net.addLayer(d1)
    net.addConnect("input", "d1")

    check("d1 outputShape None before propagate",  d1._outputShape is None)
    check("d1 inputShape None before propagate",   d1._inputShape is None)

    net.propagate_shapes()

    check("d1 outputShape set after propagate",    d1._outputShape == (128,))
    check("d1 inputShape set after propagate",     d1._inputShape == (512,))


# ═════════════════════════════════════════════════════════════════════════════
# 5. DATASET TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_dataset_validate():
    section("Dataset — validate")
    ds = Dataset(id="ds1", name="CIFAR10", inputShape=(3,32,32), numClasses=10)
    check("valid dataset",                      ds.validate() == True)
    check("empty id → invalid",                 Dataset("", "X", (3,32,32), 10).validate() == False)
    check("empty name → invalid",               Dataset("ds1", "", (3,32,32), 10).validate() == False)
    check("empty inputShape → invalid",         Dataset("ds1", "X", (), 10).validate() == False)
    check("numClasses=0 → invalid",             Dataset("ds1", "X", (3,32,32), 0).validate() == False)
    check("numClasses<0 → invalid",             Dataset("ds1", "X", (3,32,32), -1).validate() == False)


def test_dataset_builtin_factory():
    section("Dataset — fromBuiltin (all datasets)")
    for builtin in BuiltinDataset:
        ds   = Dataset.fromBuiltin(builtin)
        meta = BUILTIN_METADATA[builtin]
        check(f"{builtin.value}: inputShape",   ds._inputShape == meta["inputShape"])
        check(f"{builtin.value}: numClasses",   ds._numClasses == meta["numClasses"])
        check(f"{builtin.value}: source",       ds._source == DatasetSource.BUILTIN)
        check(f"{builtin.value}: builtinType",  ds._builtinType == builtin)
        check(f"{builtin.value}: validates",    ds.validate() == True)


def test_dataset_serialization():
    section("Dataset — toDict / fromDict")
    ds = Dataset.fromBuiltin(BuiltinDataset.CIFAR10)
    restored = Dataset.fromDict(ds.toDict())

    check("id preserved",                       restored._id == ds._id)
    check("name preserved",                     restored._name == ds._name)
    check("inputShape preserved as tuple",      restored._inputShape == (3, 32, 32))
    check("numClasses preserved",               restored._numClasses == 10)
    check("source preserved",                   restored._source == DatasetSource.BUILTIN)
    check("builtinType preserved",              restored._builtinType == BuiltinDataset.CIFAR10)

    for builtin in BuiltinDataset:
        ds2 = Dataset.fromBuiltin(builtin)
        check(f"{builtin.value}: round-trip",
              Dataset.fromDict(ds2.toDict())._builtinType == builtin)


# ═════════════════════════════════════════════════════════════════════════════
# 6. TRAININGCONFIG TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_training_config():
    section("TrainingConfig — toDict / fromDict")
    tc = TrainingConfig(batchSize=32, epochs=50, validationSplit=0.2,
                        shuffle=True, earlyStop=True)

    check("_batchSize stored",                  tc._batchSize == 32)
    check("_epochs stored",                     tc._epochs == 50)
    check("_validationSplit stored",            tc._validationSplit == 0.2)
    check("_shuffle stored",                    tc._shuffle == True)
    check("_earlyStop stored",                  tc._earlyStop == True)

    try:
        d = tc.toDict()
        check("toDict has batchSize key",       "batchSize" in d)
        restored = TrainingConfig.fromDict(d)
        check("batchSize round-trips",          restored._batchSize == 32)
        check("epochs round-trips",             restored._epochs == 50)
        check("validationSplit round-trips",    restored._validationSplit == 0.2)
        check("shuffle round-trips",            restored._shuffle == True)
        check("earlyStop round-trips",          restored._earlyStop == True)
    except AttributeError as e:
        check(f"BUG — attribute error: {e}",    False)


# ═════════════════════════════════════════════════════════════════════════════
# 7. NETWORKCONFIG TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_network_config():
    section("NetworkConfig — toDict / fromDict")
    nc = NetworkConfig(
        optimizer=OptimizerType.ADAM,
        loss=LossType.CROSS_ENTROPY,
        metric=[MetricType.ACCURACY, MetricType.F1_SCORE],
        learningRate=0.001
    )

    check("optimizer stored",                   nc._optimizer == OptimizerType.ADAM)
    check("loss stored",                        nc._loss == LossType.CROSS_ENTROPY)
    check("metrics stored",                     nc._metric == [MetricType.ACCURACY, MetricType.F1_SCORE])
    check("learningRate stored",                nc._learningRate == 0.001)

    restored = NetworkConfig.fromDict(nc.toDict())
    check("optimizer round-trips",              restored._optimizer == OptimizerType.ADAM)
    check("loss round-trips",                   restored._loss == LossType.CROSS_ENTROPY)
    check("metrics round-trips",               restored._metric == [MetricType.ACCURACY, MetricType.F1_SCORE])
    check("learningRate round-trips",           restored._learningRate == 0.001)

    for opt in OptimizerType:
        nc2 = NetworkConfig(opt, LossType.MSE, [MetricType.ACCURACY], 0.01)
        check(f"OptimizerType.{opt.name} round-trips",
              NetworkConfig.fromDict(nc2.toDict())._optimizer == opt)

    for loss in LossType:
        nc3 = NetworkConfig(OptimizerType.SGD, loss, [MetricType.ACCURACY], 0.01)
        check(f"LossType.{loss.name} round-trips",
              NetworkConfig.fromDict(nc3.toDict())._loss == loss)


# ═════════════════════════════════════════════════════════════════════════════
# 8. NETWORK SERIALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def test_network_serialization():
    section("Network — full serialization round-trip")
    net  = Network("net_serial")
    inp  = InputLayer((3, 32, 32))
    conv = make_conv(16, id="conv1")
    pool = make_pool(id="pool1")
    act  = make_act(ActivationType.RELU, id="act1")

    net.addLayer(inp); net.addLayer(conv); net.addLayer(pool); net.addLayer(act)
    net.addConnect("input", "conv1")
    net.addConnect("conv1", "pool1")
    net.addConnect("pool1", "act1")
    net.propagate_shapes()

    restored = Network.fromDict(net.toDict())

    check("layer count preserved",              len(restored.layers) == 4)
    check("connection count preserved",         len(restored.connections) == 3)
    check("adjacency rebuilt (input→conv1)",
          "conv1" in restored._outgoing.get("input", []))
    check("adjacency rebuilt (conv1→pool1)",
          "pool1" in restored._outgoing.get("conv1", []))
    check("conv1 outputShape preserved",
          restored.layers["conv1"]._outputShape == (16, 32, 32))
    check("pool1 outputShape preserved",
          restored.layers["pool1"]._outputShape == (16, 16, 16))
    check("conv1 inputShape preserved",
          restored.layers["conv1"]._inputShape == (3, 32, 32))
    check("restored network validates",         restored.validate() == True)

    conv_r = restored.layers["conv1"]
    check("Conv2D filters preserved",           conv_r._filters == 16)
    check("Conv2D kernelSize preserved",        conv_r._kernelSize == (3, 3))
    check("Conv2D padding preserved",           conv_r._padding == (1, 1))

    pool_r = restored.layers["pool1"]
    check("Pooling poolSize preserved",         pool_r._poolSize == (2, 2))
    check("Pooling strides preserved",          pool_r._strides == (2, 2))

    act_r = restored.layers["act1"]
    check("Activation type preserved",          act_r._activationType == ActivationType.RELU)


def test_network_serialization_dense():
    section("Network — Dense serialization round-trip")
    net = Network("net_dense")
    inp = InputLayer((512,))
    d1  = make_dense(128, id="d1")
    d2  = make_dense(10,  id="d2")

    net.addLayer(inp); net.addLayer(d1); net.addLayer(d2)
    net.addConnect("input", "d1")
    net.addConnect("d1", "d2")
    net.propagate_shapes()

    restored = Network.fromDict(net.toDict())

    check("d1 outputShape preserved as tuple",
          restored.layers["d1"]._outputShape == (128,))
    check("d2 outputShape preserved as tuple",
          restored.layers["d2"]._outputShape == (10,))
    check("d1 inputShape preserved as tuple",
          restored.layers["d1"]._inputShape == (512,))
    check("Dense unit preserved",               restored.layers["d1"]._unit == 128)
    check("Dense useBias preserved",            restored.layers["d1"]._useBias == True)


# ═════════════════════════════════════════════════════════════════════════════
# 9. PROJECT TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_project_is_ready_to_train():
    section("Project — isReadyToTrain")
    p = Project("p", "/tmp")
    check("nothing → not ready",                p.isReadyToTrain() == False)

    p.network = Network("n")
    p.network.addLayer(InputLayer((512,)))
    check("no dataset → not ready",             p.isReadyToTrain() == False)

    p.dataset = Dataset.fromBuiltin(BuiltinDataset.MNIST)
    check("no trainingConfig → not ready",      p.isReadyToTrain() == False)

    p.trainingConfig = TrainingConfig(32, 10, 0.2, True, False)
    check("no networkConfig → not ready",       p.isReadyToTrain() == False)

    p.networkConfig = NetworkConfig(OptimizerType.ADAM, LossType.CROSS_ENTROPY,
                                    [MetricType.ACCURACY], 0.001)
    check("all set → ready",                    p.isReadyToTrain() == True)


def test_project_logs():
    section("Project — training logs")
    p = Project("p", "/tmp")
    history = {"loss": [0.9, 0.7, 0.5], "accuracy": [0.6, 0.75, 0.85]}

    p.updateLogs(history)
    check("logs stored",                        p.trainingLogs == history)
    check("getLogsDict returns logs",           p.getLogsDict() == history)

    p.loadLogs({"loss": [0.3]})
    check("loadLogs overwrites",                p.trainingLogs == {"loss": [0.3]})
    expect_exception("updateLogs rejects non-dict",
                     lambda: p.updateLogs("bad"), ValueError)


def test_project_weights_path():
    section("Project — weights path")
    p = Project("p", "/tmp")
    check("weightsPath initially None",         p.getWeightsPath() is None)
    p.setWeightsPath("/tmp/weights.pt")
    check("weightsPath set correctly",          p.getWeightsPath() == "/tmp/weights.pt")


def test_project_full_round_trip():
    section("Project — full toDict / fromDict round-trip")
    p   = Project(name="MyProject", path="/tmp/myproject")

    # use flat input for Dense chain — architecturally correct
    net = Network("net1")
    inp = InputLayer((512,))
    d1  = make_dense(128, id="d1")
    d2  = make_dense(10,  id="d2")

    net.addLayer(inp); net.addLayer(d1); net.addLayer(d2)
    net.addConnect("input", "d1")
    net.addConnect("d1", "d2")
    net.propagate_shapes()
    p.network = net

    p.dataset        = Dataset.fromBuiltin(BuiltinDataset.MNIST)
    p.trainingConfig = TrainingConfig(32, 20, 0.2, True, True)
    p.networkConfig  = NetworkConfig(OptimizerType.ADAM, LossType.CROSS_ENTROPY,
                                     [MetricType.ACCURACY, MetricType.F1_SCORE], 0.001)
    p.updateLogs({"loss": [0.9, 0.5], "accuracy": [0.6, 0.85]})

    restored = Project.fromDict(p.toDict())

    check("name preserved",                     restored.name == "MyProject")
    check("path preserved",                     restored.path == "/tmp/myproject")
    check("network restored",                   restored.network is not None)
    check("network validates",                  restored.network.validate() == True)
    check("layer count correct",                len(restored.network.layers) == 3)
    check("connection count correct",           len(restored.network.connections) == 2)
    check("d1 outputShape preserved as tuple",
          restored.network.layers["d1"]._outputShape == (128,))
    check("d1 inputShape preserved as tuple",
          restored.network.layers["d1"]._inputShape == (512,))
    check("dataset builtinType preserved",
          restored.dataset._builtinType == BuiltinDataset.MNIST)
    check("trainingConfig epochs preserved",    restored.trainingConfig._epochs == 20)
    check("networkConfig optimizer preserved",
          restored.networkConfig._optimizer == OptimizerType.ADAM)
    check("networkConfig metrics preserved",
          MetricType.F1_SCORE in restored.networkConfig._metric)
    check("logs preserved",                     restored.trainingLogs["loss"] == [0.9, 0.5])
    check("restored project ready to train",    restored.isReadyToTrain() == True)


def test_project_filename_constants():
    section("Project — filename constants")
    check("ARCHITECTURE_FILENAME",              Project.ARCHITECTURE_FILENAME == "project.json")
    check("WEIGHTS_FILENAME",                   Project.WEIGHTS_FILENAME == "weights.pt")
    check("LOGS_FILENAME",                      Project.LOGS_FILENAME == "training_logs.json")


# ═════════════════════════════════════════════════════════════════════════════
# Run all
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # Layers
        test_input_layer,
        test_dense_layer,
        test_conv2d_layer,
        test_pooling_layer,
        test_recurrent_layer,
        test_activation_layer,
        test_layer_instance_independence,
        # Connection
        test_connection,
        # Network
        test_network_add_layers,
        test_network_remove_layers,
        test_network_add_connections,
        test_network_shape_compatibility,
        test_network_remove_connections,
        test_network_cycle_detection,
        test_network_input_output_layers,
        test_network_validate,
        test_topological_sort,
        test_get_parents_children,
        # Shape propagation
        test_shape_propagation_dense,
        test_shape_propagation_conv_pool,
        test_shape_propagation_recurrent,
        test_shape_propagation_shapes_none_before,
        # Dataset
        test_dataset_validate,
        test_dataset_builtin_factory,
        test_dataset_serialization,
        # Configs
        test_training_config,
        test_network_config,
        # Serialization
        test_network_serialization,
        test_network_serialization_dense,
        # Project
        test_project_is_ready_to_train,
        test_project_logs,
        test_project_weights_path,
        test_project_full_round_trip,
        test_project_filename_constants,
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
        print("  ✅ All tests passed — core is solid")
    else:
        print("  ❌ Some tests failed — check output above")
    print(f"{'═' * 54}")
    sys.exit(0 if failed == 0 else 1)