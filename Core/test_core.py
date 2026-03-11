"""
test_new_additions.py
Tests for everything added in this pass:
  - Flatten layer
  - RecurrentType enum on Recurrent
  - PoolingType enum on Pooling
  - DropoutStyle enum on Conv2D
  - All three serialization round-trips (toDict / fromDict)
  - Flatten as the bridge between Conv2D and Dense
"""

import sys, traceback
from layer import (
    InputLayer, Flatten, Dense, Conv2D, Recurrent, Pooling, Activation,
    ActivationType, RecurrentType, PoolingType, DropoutStyle,
)
from network import Network

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
        print(f"  ✗ {label}  ← FAILED (wrong exception: {type(e).__name__}: {e})")
        failed += 1


# ── Flatten ───────────────────────────────────────────────────────────────────

def test_flatten_layer():
    section("Flatten — getOutputShape")
    f = Flatten(id="f1", name="F1", role="hidden")

    check("shapes start as None",           f._inputShape is None and f._outputShape is None)
    check("_isInput is False",              f._isInput == False)
    check("_isLocked is False",             f._isLocked == False)

    check("3D (3,4,4) → (48,)",            f.getOutputShape((3, 4, 4)) == (48,))
    check("3D (16,7,7) → (784,)",          f.getOutputShape((16, 7, 7)) == (784,))
    check("2D (32,10) → (320,)",           f.getOutputShape((32, 10)) == (320,))
    check("1D already flat passthrough",   f.getOutputShape((128,)) == (128,))
    check("1D (1,) passthrough",           f.getOutputShape((1,)) == (1,))


def test_flatten_serialization():
    section("Flatten — toDict / fromDict round-trip")
    net = Network("net_flat")
    inp  = InputLayer((3, 8, 8))
    flat = Flatten(id="flat1", name="Flatten1", role="hidden")
    d1   = Dense(unit=64, useBias=True, dropout=0.0, id="d1", name="D1", role="hidden")

    net.addLayer(inp); net.addLayer(flat); net.addLayer(d1)
    net.addConnect("input", "flat1")
    net.addConnect("flat1", "d1")
    net.propagate_shapes()

    restored = Network.fromDict(net.toDict())

    flat_r = restored.layers["flat1"]
    check("type is Flatten",               type(flat_r).__name__ == "Flatten")
    check("outputShape preserved",         flat_r._outputShape == (192,))   # 3*8*8
    check("inputShape preserved",          flat_r._inputShape  == (3, 8, 8))
    check("d1 inputShape is flat",         restored.layers["d1"]._inputShape == (192,))
    check("d1 outputShape correct",        restored.layers["d1"]._outputShape == (64,))


def test_flatten_bridges_conv_to_dense():
    section("Flatten — bridges Conv2D → Dense in a real network")
    net = Network("cnn")
    inp  = InputLayer((1, 28, 28))
    conv = Conv2D(filters=8, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                  dropout=0.0, dropoutStyle=None, id="conv1", name="C1", role="hidden")
    flat = Flatten(id="flat1", name="Flat", role="hidden")
    d1   = Dense(unit=10, useBias=True, dropout=0.0, id="d1", name="D1", role="output")

    net.addLayer(inp); net.addLayer(conv); net.addLayer(flat); net.addLayer(d1)

    check("input→conv accepted",            net.addConnect("input", "conv1") == True)
    check("conv→flat accepted",             net.addConnect("conv1", "flat1") == True)
    check("flat→dense accepted",            net.addConnect("flat1", "d1")    == True)

    # without Flatten, Conv→Dense must still be rejected
    net2 = Network("cnn2")
    inp2  = InputLayer((1, 28, 28))
    conv2 = Conv2D(filters=8, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                   dropout=0.0, dropoutStyle=None, id="conv1", name="C1", role="hidden")
    d2    = Dense(unit=10, useBias=True, dropout=0.0, id="d1", name="D1", role="output")
    net2.addLayer(inp2); net2.addLayer(conv2); net2.addLayer(d2)
    net2.addConnect("input", "conv1")
    check("conv→dense rejected without Flatten", net2.addConnect("conv1", "d1") == False)

    # propagate shapes and verify sizes
    net.propagate_shapes()
    check("conv outputShape",               conv._outputShape == (8, 28, 28))
    check("flat outputShape",               flat._outputShape == (8 * 28 * 28,))
    check("dense inputShape matches flat",  d1._inputShape    == (8 * 28 * 28,))
    check("dense outputShape",              d1._outputShape   == (10,))


# ── RecurrentType ─────────────────────────────────────────────────────────────

def test_recurrent_type_enum():
    section("RecurrentType enum")
    check("LSTM value",  RecurrentType.LSTM.value == "lstm")
    check("GRU value",   RecurrentType.GRU.value  == "gru")
    check("roundtrip",   RecurrentType(RecurrentType.LSTM.value) == RecurrentType.LSTM)


def test_recurrent_default_type():
    section("Recurrent — default type is LSTM")
    r = Recurrent(hiddenSize=64, numLayer=1, dropout=0.0,
                  id="r1", name="R1", role="hidden",
                  bidirection=False, returnSeq=True)
    check("default recurrentType is LSTM", r._recurrentType == RecurrentType.LSTM)


def test_recurrent_type_serialization():
    section("Recurrent — recurrentType toDict / fromDict")
    net = Network("rnn_net")
    inp = InputLayer((10, 32))
    lstm = Recurrent(hiddenSize=64, numLayer=1, dropout=0.0,
                     id="lstm1", name="LSTM1", role="hidden",
                     bidirection=False, returnSeq=False,
                     recurrentType=RecurrentType.LSTM)
    gru  = Recurrent(hiddenSize=32, numLayer=1, dropout=0.0,
                     id="gru1", name="GRU1", role="output",
                     bidirection=False, returnSeq=False,
                     recurrentType=RecurrentType.GRU)

    net.addLayer(inp); net.addLayer(lstm); net.addLayer(gru)
    net.addConnect("input", "lstm1")
    net.addConnect("lstm1", "gru1")
    net.propagate_shapes()

    restored = Network.fromDict(net.toDict())

    lstm_r = restored.layers["lstm1"]
    gru_r  = restored.layers["gru1"]

    check("LSTM type preserved",    lstm_r._recurrentType == RecurrentType.LSTM)
    check("GRU type preserved",     gru_r._recurrentType  == RecurrentType.GRU)
    check("LSTM hiddenSize",        lstm_r._hiddenSize == 64)
    check("GRU hiddenSize",         gru_r._hiddenSize  == 32)

    # old saved data without recurrentType field should default to LSTM
    d = net.toDict()
    del d["layers"]["lstm1"]["recurrentType"]
    restored2 = Network.fromDict(d)
    check("missing recurrentType defaults to LSTM",
          restored2.layers["lstm1"]._recurrentType == RecurrentType.LSTM)


# ── PoolingType ───────────────────────────────────────────────────────────────

def test_pooling_type_enum():
    section("PoolingType enum")
    check("MAX value",     PoolingType.MAX.value     == "max")
    check("AVERAGE value", PoolingType.AVERAGE.value == "average")
    check("roundtrip",     PoolingType(PoolingType.MAX.value) == PoolingType.MAX)


def test_pooling_default_type():
    section("Pooling — default type is MAX")
    p = Pooling(poolSize=(2,2), strides=(2,2), padding=(0,0),
                id="p1", name="P1", role="hidden")
    check("default poolingType is MAX", p._poolingType == PoolingType.MAX)


def test_pooling_type_serialization():
    section("Pooling — poolingType toDict / fromDict")
    net  = Network("pool_net")
    inp  = InputLayer((3, 32, 32))
    pmax = Pooling(poolSize=(2,2), strides=(2,2), padding=(0,0),
                   id="pmax", name="MaxPool", role="hidden",
                   poolingType=PoolingType.MAX)
    pavg = Pooling(poolSize=(2,2), strides=(2,2), padding=(0,0),
                   id="pavg", name="AvgPool", role="hidden",
                   poolingType=PoolingType.AVERAGE)
    flat = Flatten(id="flat1", name="Flat", role="hidden")
    d1   = Dense(unit=10, useBias=True, dropout=0.0, id="d1", name="D1", role="output")

    net.addLayer(inp); net.addLayer(pmax); net.addLayer(pavg)
    net.addLayer(flat); net.addLayer(d1)
    net.addConnect("input", "pmax")
    net.addConnect("pmax",  "pavg")
    net.addConnect("pavg",  "flat1")
    net.addConnect("flat1", "d1")
    net.propagate_shapes()

    restored = Network.fromDict(net.toDict())

    check("MAX type preserved",     restored.layers["pmax"]._poolingType == PoolingType.MAX)
    check("AVERAGE type preserved", restored.layers["pavg"]._poolingType == PoolingType.AVERAGE)

    # old saved data without poolingType should default to MAX
    d = net.toDict()
    del d["layers"]["pmax"]["poolingType"]
    restored2 = Network.fromDict(d)
    check("missing poolingType defaults to MAX",
          restored2.layers["pmax"]._poolingType == PoolingType.MAX)


# ── DropoutStyle ──────────────────────────────────────────────────────────────

def test_dropout_style_enum():
    section("DropoutStyle enum")
    check("STANDARD value", DropoutStyle.STANDARD.value == "standard")
    check("SPATIAL value",  DropoutStyle.SPATIAL.value  == "spatial")
    check("roundtrip",      DropoutStyle("spatial") == DropoutStyle.SPATIAL)


def test_conv2d_dropout_style_serialization():
    section("Conv2D — dropoutStyle toDict / fromDict")
    net = Network("conv_net")
    inp  = InputLayer((3, 32, 32))
    c_std  = Conv2D(filters=16, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                    dropout=0.3, dropoutStyle=DropoutStyle.STANDARD,
                    id="c_std", name="C_std", role="hidden")
    c_spa  = Conv2D(filters=16, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                    dropout=0.3, dropoutStyle=DropoutStyle.SPATIAL,
                    id="c_spa", name="C_spa", role="hidden")
    c_none = Conv2D(filters=16, kernelSize=(3,3), strides=(1,1), padding=(1,1),
                    dropout=0.0, dropoutStyle=None,
                    id="c_none", name="C_none", role="hidden")
    flat = Flatten(id="flat1", name="Flat", role="hidden")
    d1   = Dense(unit=10, useBias=True, dropout=0.0, id="d1", name="D1", role="output")

    net.addLayer(inp);   net.addLayer(c_std); net.addLayer(c_spa)
    net.addLayer(c_none); net.addLayer(flat); net.addLayer(d1)
    net.addConnect("input",  "c_std")
    net.addConnect("c_std",  "c_spa")
    net.addConnect("c_spa",  "c_none")
    net.addConnect("c_none", "flat1")
    net.addConnect("flat1",  "d1")
    net.propagate_shapes()

    restored = Network.fromDict(net.toDict())

    check("STANDARD preserved",  restored.layers["c_std"]._dropoutStyle  == DropoutStyle.STANDARD)
    check("SPATIAL preserved",   restored.layers["c_spa"]._dropoutStyle   == DropoutStyle.SPATIAL)
    check("None preserved",      restored.layers["c_none"]._dropoutStyle  is None)


# ── Full CNN round-trip: Conv → Pool → Flatten → Dense ───────────────────────

def test_full_cnn_roundtrip():
    section("Full CNN pipeline — Conv→Pool→Flatten→Dense round-trip")
    net  = Network("lenet_mini")
    inp  = InputLayer((1, 28, 28))
    c1   = Conv2D(filters=6, kernelSize=(5,5), strides=(1,1), padding=(2,2),
                  dropout=0.0, dropoutStyle=None, id="c1", name="Conv1", role="hidden")
    p1   = Pooling(poolSize=(2,2), strides=(2,2), padding=(0,0),
                   id="p1", name="Pool1", role="hidden", poolingType=PoolingType.MAX)
    flat = Flatten(id="flat1", name="Flatten", role="hidden")
    d1   = Dense(unit=84, useBias=True, dropout=0.0, id="d1", name="FC1", role="hidden")
    d2   = Dense(unit=10, useBias=True, dropout=0.0, id="d2", name="FC2", role="output")

    for layer in [inp, c1, p1, flat, d1, d2]:
        net.addLayer(layer)

    net.addConnect("input", "c1")
    net.addConnect("c1",    "p1")
    net.addConnect("p1",    "flat1")
    net.addConnect("flat1", "d1")
    net.addConnect("d1",    "d2")
    net.propagate_shapes()

    check("c1 outputShape  (6,28,28)",  c1._outputShape   == (6, 28, 28))
    check("p1 outputShape  (6,14,14)",  p1._outputShape   == (6, 14, 14))
    check("flat outputShape (1176,)",   flat._outputShape == (1176,))
    check("d1 outputShape  (84,)",      d1._outputShape   == (84,))
    check("d2 outputShape  (10,)",      d2._outputShape   == (10,))

    restored = Network.fromDict(net.toDict())

    check("restored: layer count",      len(restored.layers) == 6)
    check("restored: connection count", len(restored.connections) == 5)
    check("restored: validates",        restored.validate() == True)
    check("restored: flat outputShape", restored.layers["flat1"]._outputShape == (1176,))
    check("restored: d2 outputShape",   restored.layers["d2"]._outputShape   == (10,))
    check("restored: p1 poolingType",   restored.layers["p1"]._poolingType    == PoolingType.MAX)
    check("restored: c1 type correct",  type(restored.layers["c1"]).__name__  == "Conv2D")
    check("restored: flat type correct",type(restored.layers["flat1"]).__name__== "Flatten")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_flatten_layer,
        test_flatten_serialization,
        test_flatten_bridges_conv_to_dense,
        test_recurrent_type_enum,
        test_recurrent_default_type,
        test_recurrent_type_serialization,
        test_pooling_type_enum,
        test_pooling_default_type,
        test_pooling_type_serialization,
        test_dropout_style_enum,
        test_conv2d_dropout_style_serialization,
        test_full_cnn_roundtrip,
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
        print("  ✅ All new-addition tests passed")
    else:
        print("  ❌ Some tests failed — check output above")
    print(f"{'═' * 54}")
    sys.exit(0 if failed == 0 else 1)
