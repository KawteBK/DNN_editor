from layer import InputLayer, Dense, Conv2D, Recurrent, Pooling, Activation, ActivationType
from connection import Connection
from network import Network
from networkConfig import NetworkConfig, OptimizerType, LossType, MetricType
from trainingConfig import TrainingConfig
from dataset import Dataset
from project import Project


def test_layers():
    print("\n── Layer Tests ──")

    # InputLayer
    inp = InputLayer(inputShape=(3, 224, 224))
    assert inp._outputShape == (3, 224, 224)
    assert inp._isInput == True
    assert inp._isLocked == True
    print("✓ InputLayer")

    # Dense
    dense = Dense(unit=128, useBias=True, dropout=0.2,
                  id="d1", name="Dense1", role="hidden",
                  inputShape=(512,), outputShape=None)
    assert dense.getOutputShape((512,)) == (128,)
    print("✓ Dense")

    # Conv2D
    conv = Conv2D(filters=32, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                  dropout=0.0, droupoutStyle=None,
                  id="c1", name="Conv1", role="hidden",
                  inputShape=(3,224,224), outputShape=None)
    assert conv.getOutputShape((3, 224, 224)) == (32, 224, 224)
    print("✓ Conv2D")

    # Pooling
    pool = Pooling(poolSize=(2,2), strides=(2,2), padding=(0,0),
                   id="p1", name="Pool1", role="hidden",
                   inputShape=(32,224,224), outputShape=None)
    assert pool.getOutputShape((32, 224, 224)) == (32, 112, 112)
    print("✓ Pooling")

    # Recurrent
    rnn = Recurrent(hiddenSize=64, numLayer=1, dropout=0.0,
                    id="r1", name="RNN1", role="hidden",
                    inputShape=(10, 32), outputShape=None,
                    bidirection=False, returnSeq=True)
    assert rnn.getOutputShape((10, 32)) == (10, 64)

    rnn_bi = Recurrent(hiddenSize=64, numLayer=1, dropout=0.0,
                       id="r2", name="RNN2", role="hidden",
                       inputShape=(10, 32), outputShape=None,
                       bidirection=True, returnSeq=False)
    assert rnn_bi.getOutputShape((10, 32)) == (128,)
    print("✓ Recurrent (unidirectional + bidirectional)")

    # Activation
    act = Activation(activationType=ActivationType.RELU,
                     id="a1", name="ReLU", role="hidden")
    assert act.getOutputShape((128,)) == (128,)
    print("✓ Activation")


def test_network_basic():
    print("\n── Network Basic Tests ──")

    net = Network("net1")

    inp = InputLayer(inputShape=(3, 224, 224))
    dense = Dense(unit=128, useBias=True, dropout=0.0,
                  id="d1", name="Dense1", role="hidden",
                  inputShape=None, outputShape=None)

    assert net.addLayer(inp) == True
    assert net.addLayer(dense) == True
    assert net.addLayer(inp) == False        # duplicate
    print("✓ addLayer / duplicate prevention")

    assert net.addConnect("input", "d1") == True
    assert net.addConnect("input", "d1") == False   # duplicate connection
    assert net.addConnect("d1", "d1") == False      # self-loop
    assert net.addConnect("d1", "input") == False   # connecting TO input layer
    print("✓ addConnect / duplicate / self-loop / input protection")

    assert net.getInputLayers()[0]._id == "input"
    assert net.getOutputLayers()[0]._id == "d1"
    print("✓ getInputLayers / getOutputLayers")

    assert net.validate() == True
    print("✓ validate")


def test_network_cycle():
    print("\n── Cycle Detection ──")

    net = Network("net2")
    a = Dense(unit=64, useBias=True, dropout=0.0,
              id="a", name="A", role="hidden", inputShape=None, outputShape=None)
    b = Dense(unit=64, useBias=True, dropout=0.0,
              id="b", name="B", role="hidden", inputShape=None, outputShape=None)
    c = Dense(unit=64, useBias=True, dropout=0.0,
              id="c", name="C", role="hidden", inputShape=None, outputShape=None)

    net.addLayer(a)
    net.addLayer(b)
    net.addLayer(c)

    net.addConnect("a", "b")
    net.addConnect("b", "c")
    result = net.addConnect("c", "a")   # this should be rejected — creates a cycle
    assert result == False
    print("✓ Cycle rejected")


def test_remove_layer():
    print("\n── Remove Layer ──")

    net = Network("net3")
    inp = InputLayer(inputShape=(10,))
    d1 = Dense(unit=64, useBias=True, dropout=0.0,
               id="d1", name="D1", role="hidden", inputShape=None, outputShape=None)
    d2 = Dense(unit=32, useBias=True, dropout=0.0,
               id="d2", name="D2", role="hidden", inputShape=None, outputShape=None)

    net.addLayer(inp)
    net.addLayer(d1)
    net.addLayer(d2)
    net.addConnect("input", "d1")
    net.addConnect("d1", "d2")

    assert net.removeLayer("input") == False   # locked
    assert net.removeLayer("d1") == True
    assert "d1" not in net.layers
    assert "d1" not in net._outgoing
    assert "d1" not in net._incoming
    # connections involving d1 should be gone
    assert all(c._fromLayer != "d1" and c._toLayer != "d1" for c in net.connections)
    print("✓ removeLayer / locked protection / adjacency list cleanup")


def test_shape_propagation():
    print("\n── Shape Propagation ──")

    net = Network("net4")
    inp = InputLayer(inputShape=(3, 224, 224))
    conv = Conv2D(filters=32, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                  dropout=0.0, droupoutStyle=None,
                  id="conv1", name="Conv1", role="hidden",
                  inputShape=None, outputShape=None)
    pool = Pooling(poolSize=(2,2), strides=(2,2), padding=(0,0),
                   id="pool1", name="Pool1", role="hidden",
                   inputShape=None, outputShape=None)

    net.addLayer(inp)
    net.addLayer(conv)
    net.addLayer(pool)
    net.addConnect("input", "conv1")
    net.addConnect("conv1", "pool1")

    net.propagate_shapes()

    assert conv._outputShape == (32, 224, 224)
    assert pool._outputShape == (32, 112, 112)
    print("✓ Shape propagation through Conv2D → Pooling")


def test_serialization():
    print("\n── Serialization ──")

    # Dataset
    ds = Dataset(id="ds1", name="CIFAR10", inputShape=(3, 32, 32), numClasses=10)
    assert ds.validate() == True
    restored = Dataset.fromDict(ds.toDict())
    assert restored._id == ds._id
    assert restored._inputShape == ds._inputShape
    print("✓ Dataset toDict / fromDict")

    # Project
    project = Project(name="TestProject", path="/tmp/test_project")
    project.dataset = ds
    project.trainingLogs = {"loss": [0.9, 0.7], "accuracy": [0.6, 0.8]}

    d = project.toDict()
    print(d)
    assert d["name"] == "TestProject"
    assert d["dataset"]["id"] == "ds1"

    print("✓ Project toDict")


if __name__ == "__main__":
    test_layers()
    test_network_basic()
    test_network_cycle()
    test_remove_layer()
    test_shape_propagation()
    test_serialization()
    print("\n✅ All tests passed")