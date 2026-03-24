import os
import torch
import torch.nn as nn
import torch.optim as optim

from project import Project
from networkConfig import OptimizerType, LossType, MetricType
from pytorch_model_builder import PyTorchModelBuilder
from pytorch_datasets import PyTorchDatasets


class PyTorchTrainer:

    # ── public ────────────────────────────────────────────────────────────────

    @staticmethod
    def train(project: Project, on_epoch=None) -> Project:
        """
        Full training loop. Takes a Project, returns the same Project
        with trainingLogs and weightsPath populated.

        on_epoch is an optional callback for UI progress:
            on_epoch(epoch: int, total: int, train_loss: float, val_metrics: dict)
        Pass None if you don't need it (e.g. in tests or CLI usage).
        """
        if not project.isReadyToTrain():
            raise ValueError(
                "Project is not ready to train. "
                "Make sure network, dataset, trainingConfig and networkConfig are all set."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── build model ───────────────────────────────────────────────────────
        model = PyTorchModelBuilder.build(project.network)
        model = model.to(device)

        # ── data ──────────────────────────────────────────────────────────────
        train_loader, val_loader = PyTorchDatasets.get_dataloaders(
            project.dataset, project.trainingConfig
        )

        # ── optimizer + loss ──────────────────────────────────────────────────
        optimizer = PyTorchTrainer._build_optimizer(model, project.networkConfig)
        criterion = PyTorchTrainer._build_loss(project.networkConfig)

        # ── training state ────────────────────────────────────────────────────
        tc            = project.trainingConfig
        nc            = project.networkConfig
        epochs        = tc._epochs
        early_stop    = tc._earlyStop
        best_val_loss = float("inf")
        patience      = 5       # could be exposed in TrainingConfig later
        patience_counter = 0
        best_weights  = None    # state_dict snapshot for best epoch

        # ── history dict — one list per metric, grown each epoch ──────────────
        history = {"train_loss": []}
        if val_loader:
            history["val_loss"] = []
            for m in nc._metric:
                history[f"val_{m.value}"] = []

        # ── epoch loop ────────────────────────────────────────────────────────
        for epoch in range(epochs):

            train_loss = PyTorchTrainer._train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            history["train_loss"].append(round(train_loss, 6))

            val_metrics = {}
            if val_loader:
                val_metrics = PyTorchTrainer._evaluate(
                    model, val_loader, criterion, device, nc._metric
                )
                history["val_loss"].append(round(val_metrics.get("loss", 0.0), 6))
                for m in nc._metric:
                    key = f"val_{m.value}"
                    history[key].append(round(val_metrics.get(m.value, 0.0), 6))

            # ── optional UI callback ──────────────────────────────────────────
            if on_epoch is not None:
                on_epoch(epoch + 1, epochs, train_loss, val_metrics)

            # ── early stopping ────────────────────────────────────────────────
            if early_stop and val_loader:
                current_val_loss = val_metrics.get("loss", float("inf"))
                if current_val_loss < best_val_loss:
                    best_val_loss    = current_val_loss
                    patience_counter = 0
                    # snapshot the best weights — deepcopy so later epochs don't overwrite it
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                        break

        # ── restore best weights if early stopping was used ───────────────────
        if best_weights is not None:
            model.load_state_dict(best_weights)

        # ── save weights to disk ──────────────────────────────────────────────
        weights_path = os.path.join(project.path, Project.WEIGHTS_FILENAME)
        torch.save(model.state_dict(), weights_path)
        project.setWeightsPath(weights_path)

        # ── write history back into project ───────────────────────────────────
        project.updateLogs(history)

        return project

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _train_one_epoch(model, loader, optimizer, criterion, device) -> float:
        """
        One full pass over the training DataLoader.

        Sets model.train() so dropout and batchnorm behave correctly.
        Returns average loss over the epoch (total loss / number of samples).
        """
        model.train()
        running_loss = 0.0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)   # un-average the batch loss

        return running_loss / len(loader.dataset)

    @staticmethod
    def _evaluate(model, loader, criterion, device, metrics: list) -> dict:
        """
        Evaluation pass — no gradients, model in eval mode.

        Returns a dict keyed by metric name, e.g.:
            {"loss": 0.41, "accuracy": 88.3, "f1_score": 0.871}

        Accuracy is always computed (it's essentially free).
        F1, precision, recall use sklearn and require collecting
        all predictions — only done when those metrics are requested.
        """
        model.eval()
        running_loss = 0.0
        correct      = 0
        total        = 0
        all_preds    = []
        all_labels   = []

        needs_sklearn = any(
            m in metrics
            for m in [MetricType.F1_SCORE, MetricType.PRECISION, MetricType.RECALL]
        )

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs  = model(inputs)
                loss     = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted  = outputs.max(1)
                total        += labels.size(0)
                correct      += predicted.eq(labels).sum().item()

                if needs_sklearn:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        results = {"loss": running_loss / total}

        if MetricType.ACCURACY in metrics or True:   # accuracy always included
            results["accuracy"] = 100.0 * correct / total

        if needs_sklearn:
            from sklearn.metrics import f1_score, precision_score, recall_score
            if MetricType.F1_SCORE in metrics:
                results["f1_score"] = f1_score(
                    all_labels, all_preds, average="macro", zero_division=0
                )
            if MetricType.PRECISION in metrics:
                results["precision"] = precision_score(
                    all_labels, all_preds, average="macro", zero_division=0
                )
            if MetricType.RECALL in metrics:
                results["recall"] = recall_score(
                    all_labels, all_preds, average="macro", zero_division=0
                )

        return results

    @staticmethod
    def _build_optimizer(model: nn.Module, networkConfig) -> optim.Optimizer:
        """
        Maps OptimizerType enum → the correct torch optimizer.

        SGD gets momentum=0.9 as a sensible default.
        Learning rate comes from networkConfig._learningRate.
        """
        lr  = networkConfig._learningRate
        opt = networkConfig._optimizer

        if opt == OptimizerType.ADAM:
            return optim.Adam(model.parameters(), lr=lr)
        elif opt == OptimizerType.SGD:
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif opt == OptimizerType.RMSPROP:
            return optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

    @staticmethod
    def _build_loss(networkConfig) -> nn.Module:
        """
        Maps LossType enum → the correct torch loss function.

        BCE uses BCEWithLogitsLoss (numerically stabler than BCE + Sigmoid).
        """
        loss = networkConfig._loss

        if loss == LossType.CROSS_ENTROPY:
            return nn.CrossEntropyLoss()
        elif loss == LossType.MSE:
            return nn.MSELoss()
        elif loss == LossType.BCE:
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss}")