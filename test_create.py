"""
run_create_train_save.py
========================
Script 1 — Build a CNN from scratch, train it for a few epochs on MNIST,
then save the full project to disk.

Expected output layout after running:
    ./my_project/
        project.json          ← architecture + configs + logs
        training_logs.json    ← metric history (duplicate for easy access)
        weights.pt            ← model weights (written by the trainer)

Run from the repo root (the folder that contains Core/ and Infra/):
    python run_create_train_save.py
"""

import os
import torch

# ── Core
from Core.layer import (
    InputLayer, Conv2D, Activation, ActivationType,
    Pooling, PoolingType, Flatten, Dense, DropoutStyle,
)
from Core.network import Network
from Core.dataset import Dataset, BuiltinDataset
from Core.trainingConfig import TrainingConfig
from Core.networkConfig import NetworkConfig, OptimizerType, LossType, MetricType
from Core.project import Project
from Core.projectManager import ProjectManager

# ── Infra
from Infra.trainer import PyTorchTrainer


  
# ─────────────────────────────────────────────────────────────────────────────
# 0.  Force CPU  (change to "cuda" if you have a GPU)
# ─────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hides all GPUs from PyTorch
# torch.device("cuda"...) inside the trainer will now fall back to CPU cleanly

print("Device PyTorch will use:", "cpu")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build the network
#     Architecture: Input(1,28,28) → Conv → ReLU → Pool → Flatten → Dense → Softmax
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Building the network")
print("=" * 60)

net = Network("mnist-cnn-v1")

net.addLayer(InputLayer(inputShape=(1, 28, 28)))

net.addLayer(Conv2D(
    filters=16, kernelSize=(3, 3), strides=(1, 1), padding=(1, 1),
    dropout=0.0, dropoutStyle=None,
    id="conv1", name="Conv1", role="hidden"
))
net.addLayer(Activation(ActivationType.RELU,  id="relu1", name="ReLU1"))

net.addLayer(Pooling(
    poolSize=(2, 2), strides=(2, 2), padding=(0, 0),
    id="pool1", name="Pool1", role="hidden",
    poolingType=PoolingType.MAX
))

net.addLayer(Flatten(id="flat1", name="Flatten"))

net.addLayer(Dense(unit=128, useBias=True, dropout=0.0,
                   id="fc1", name="FC1", role="hidden"))
net.addLayer(Activation(ActivationType.RELU, id="relu2", name="ReLU2"))

net.addLayer(Dense(unit=10, useBias=True, dropout=0.0,
                   id="out", name="Output", role="output"))
net.addLayer(Activation(ActivationType.SOFTMAX, id="softmax", name="Softmax"))

# Wire layersjjk
net.addConnect("input",  "conv1")
net.addConnect("conv1",  "relu1")
net.addConnect("relu1",  "pool1")
net.addConnect("pool1",  "flat1")
net.addConnect("flat1",  "fc1")
net.addConnect("fc1",    "relu2")
net.addConnect("relu2",  "out")
net.addConnect("out",    "softmax")

net.propagate_shapes()

print("Network summary:")
for k, v in net.generateSummary().items():
    print(f"  {k}: {v}")

print("\nLayer shapes after propagation:")
for lid, layer in net.layers.items():
    print(f"  {lid:12s}  in={str(layer._inputShape):20s}  out={layer._outputShape}")

assert net.validate(), "Network failed validation — check connections/cycles"
print("\n✓ Network is valid\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Assemble the project
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — Assembling the project")
print("=" * 60)

PROJECT_DIR = "./my_project"

project = Project(name="MNIST CNN Demo", path=PROJECT_DIR)
project.network = net

project.dataset = Dataset.fromBuiltin(BuiltinDataset.MNIST)

project.trainingConfig = TrainingConfig(
    batchSize=64,
    epochs=3,               # keep it short for a demo
    validationSplit=0.1,    # 10% of training data used for validation
    shuffle=True,
    earlyStop=False,
)

project.networkConfig = NetworkConfig(
    optimizer=OptimizerType.ADAM,
    loss=LossType.CROSS_ENTROPY,
    metric=[MetricType.ACCURACY],
    learningRate=1e-3,
)

print(f"  Project name : {project.name}")
print(f"  Save path    : {os.path.abspath(PROJECT_DIR)}")
print(f"  Dataset      : {project.dataset._name}")
print(f"  Epochs       : {project.trainingConfig._epochs}")
print(f"  Batch size   : {project.trainingConfig._batchSize}")
print(f"  Optimizer    : {project.networkConfig._optimizer.value}")
print(f"  LR           : {project.networkConfig._learningRate}")
print()

assert project.isReadyToTrain(), "Project not ready — missing network/dataset/config"
print("✓ Project is ready to train\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Train
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — Training  (CPU, this may take a minute or two)")
print("=" * 60)

def on_epoch(epoch, total, train_loss, val_metrics):
    val_acc  = val_metrics.get("accuracy", None)
    val_loss = val_metrics.get("loss", None)
    acc_str  = f"  val_acc={val_acc:.2f}%  val_loss={val_loss:.4f}" if val_acc is not None else ""
    print(f"  Epoch {epoch}/{total}  train_loss={train_loss:.4f}{acc_str}")
if __name__ == "__main__":

  project = PyTorchTrainer.train(project, on_epoch=on_epoch)

  print("\nTraining logs snapshot:")
  for metric, values in project.trainingLogs.items():
    print(f"  {metric}: {values}")
  print()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Save
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4 — Saving project to disk")
print("=" * 60)

ProjectManager.save(project)

saved_files = os.listdir(PROJECT_DIR)
print(f"Files in {PROJECT_DIR}:")
for f in sorted(saved_files):
    size = os.path.getsize(os.path.join(PROJECT_DIR, f))
    print(f"  {f}  ({size:,} bytes)")

print()
print("✓ Done! Run run_load_modify_train.py next to reload and continue training.")