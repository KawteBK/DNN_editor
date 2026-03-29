"""
run_load_modify_train.py
========================
Script 2 — Load the project saved by run_create_train_save.py,
add a Dropout layer before the output, then continue training for
a few more epochs and save again.

Run AFTER run_create_train_save.py:
    python run_load_modify_train.py
"""

import os
import torch

from Core.layer import Dense, Activation, ActivationType, Flatten
from Core.network import Network
from Core.trainingConfig import TrainingConfig
from Core.networkConfig import NetworkConfig, OptimizerType, LossType, MetricType
from Core.project import Project
from Core.projectManager import ProjectManager

from Infra.trainer import PyTorchTrainer
from Infra.PytorchModelBuilder import PyTorchModelBuilder


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Force CPU
# ─────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print("Device PyTorch will use: cpu\n")

PROJECT_DIR = "./my_project"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load the saved project
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading project from disk")
print("=" * 60)

if not ProjectManager.exists(PROJECT_DIR):
    raise FileNotFoundError(
        f"No saved project found at '{PROJECT_DIR}'.\n"
        "Run run_create_train_save.py first."
    )

project = ProjectManager.load(PROJECT_DIR)

print(f"  Project name : {project.name}")
print(f"  Weights file : {project.weightsPath}")
print(f"  Has logs     : {bool(project.trainingLogs)}")
print(f"  Layer count  : {len(project.network.layers)}")

print("\nPrevious training logs:")
for metric, values in project.trainingLogs.items():
    print(f"  {metric}: {values}")

print("\n✓ Project loaded successfully\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Inspect the loaded network
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — Inspecting the loaded network")
print("=" * 60)

net = project.network
print("Layer order (topological):")
for layer in net.topological_sort():
    print(f"  {layer._id:12s}  type={type(layer).__name__:15s}  out={layer._outputShape}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Modify the network
#     Change: remove the direct "out → softmax" connection, insert a new
#     Dense(64) + ReLU between relu2 and the output, then rewire.
#
#     Before:  relu2 → out(10) → softmax
#     After:   relu2 → fc2(64) → relu3 → out(10) → softmax
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — Modifying the network (adding a hidden layer)")
print("=" * 60)

# Disconnect out from relu2 and softmax from out
net.removeConnect("relu2",    "out")
net.removeConnect("out",      "softmax")

# Add new layers
from Core.layer import Dense, Activation, ActivationType
new_fc  = Dense(unit=64, useBias=True, dropout=0.3,
                id="fc2", name="FC2", role="hidden")
new_relu = Activation(ActivationType.RELU, id="relu3", name="ReLU3")

net.addLayer(new_fc)
net.addLayer(new_relu)

# Rewire: relu2 → fc2 → relu3 → out → softmax
net.addConnect("relu2",   "fc2")
net.addConnect("fc2",     "relu3")
net.addConnect("relu3",   "out")
net.addConnect("out",     "softmax")

# Recalculate shapes for the whole network
net.propagate_shapes()

print("Network after modification:")
for layer in net.topological_sort():
    print(f"  {layer._id:12s}  type={type(layer).__name__:15s}  out={layer._outputShape}")

assert net.validate(), "Modified network failed validation"
print("\n✓ Modified network is valid\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Update training config for the second run
#     (fewer epochs, slightly lower LR — common fine-tuning practice)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4 — Updating training config")
print("=" * 60)

project.trainingConfig = TrainingConfig(
    batchSize=64,
    epochs=3,
    validationSplit=0.1,
    shuffle=True,
    earlyStop=True,        # turn on early stopping this time
)

project.networkConfig = NetworkConfig(
    optimizer=OptimizerType.ADAM,
    loss=LossType.CROSS_ENTROPY,
    metric=[MetricType.ACCURACY],
    learningRate=3e-4,     # lower LR for the second run
)

print(f"  Epochs      : {project.trainingConfig._epochs}")
print(f"  LR          : {project.networkConfig._learningRate}")
print(f"  Early stop  : {project.trainingConfig._earlyStop}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Train again  (note: training from scratch on the new architecture —
#     the old weights.pt was for the old, smaller graph and can't be reused
#     directly without surgery. Loading pretrained weights is left as a
#     future enhancement once the architecture is locked.)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5 — Training modified network  (CPU)")
print("=" * 60)

def on_epoch(epoch, total, train_loss, val_metrics):
    val_acc  = val_metrics.get("accuracy", None)
    val_loss = val_metrics.get("loss", None)
    acc_str  = f"  val_acc={val_acc:.2f}%  val_loss={val_loss:.4f}" if val_acc is not None else ""
    print(f"  Epoch {epoch}/{total}  train_loss={train_loss:.4f}{acc_str}")
if __name__ == '__main__':
 project = PyTorchTrainer.train(project, on_epoch=on_epoch)

 print("\nNew training logs:")
 for metric, values in project.trainingLogs.items():
    print(f"  {metric}: {values}")
 print()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save the updated project  (overwrites project.json + logs, new weights.pt)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6 — Saving updated project")
print("=" * 60)

ProjectManager.save(project)

print(f"Files in {PROJECT_DIR}:")
for f in sorted(os.listdir(PROJECT_DIR)):
    size = os.path.getsize(os.path.join(PROJECT_DIR, f))
    print(f"  {f}  ({size:,} bytes)")

print()
print("✓ Done! The project now has the modified architecture + new weights.")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Quick sanity check — reload and run a forward pass
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 7 — Sanity check: reload and run a forward pass")
print("=" * 60)

final = ProjectManager.load(PROJECT_DIR)
model = PyTorchModelBuilder.build(final.network)
model.eval()

dummy = torch.randn(4, 1, 28, 28)   # batch of 4 fake MNIST images
with torch.no_grad():
    out = model(dummy)

print(f"  Input shape  : {tuple(dummy.shape)}")
print(f"  Output shape : {tuple(out.shape)}")
assert out.shape == (4, 10), f"Unexpected output shape: {out.shape}"
print("\n✓ Forward pass OK — output is (batch=4, classes=10) as expected.")