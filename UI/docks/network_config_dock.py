from PyQt6.QtWidgets import (
    QDockWidget, QScrollArea, QTabBar, QVBoxLayout, QWidget, QFormLayout, QLabel,
    QLineEdit, QComboBox, QCheckBox
)

from Core.networkConfig import OptimizerType, LossType, MetricType
from PyQt6.QtCore import Qt


# Maps UI display strings ↔ Core enums
_OPTIMIZERS = {"Adam": OptimizerType.ADAM, "SGD": OptimizerType.SGD, "RMSprop": OptimizerType.RMSPROP}
_LOSSES     = {"Cross Entropy": LossType.CROSS_ENTROPY, "MSE": LossType.MSE, "BCE": LossType.BCE}
_METRICS    = {"Accuracy": MetricType.ACCURACY, "F1 Score": MetricType.F1_SCORE,
               "Precision": MetricType.PRECISION, "Recall": MetricType.RECALL}


class NetworkConfigDock(QDockWidget):
    # CHANGED: accepts project_vm. Previously the fields were display-only
    # and never wrote back to any Core object.
    def __init__(self, project_vm):
        super().__init__("Training Config")
        self.project_vm = project_vm   # ADDED

        widget = QWidget()
        layout = QFormLayout()
    


        self.epochs     = QLineEdit("10")
        self.batch_size = QLineEdit("32")
        self.lr         = QLineEdit("0.001")
        self.val_split  = QLineEdit("0.2")

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(list(_OPTIMIZERS.keys()))

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(list(_LOSSES.keys()))

        # ADDED: metric checkboxes — the old dock had no metrics field at all.
        # Accuracy is checked by default.
        self.metric_checks: dict[str, QCheckBox] = {}
        for m_name in _METRICS:
            cb = QCheckBox()
            cb.setChecked(m_name == "Accuracy")
            self.metric_checks[m_name] = cb

        # ADDED: early stop + shuffle toggles — were missing from the old dock
        self.early_stop = QCheckBox()
        self.early_stop.setChecked(True)
        self.shuffle = QCheckBox()
        self.shuffle.setChecked(True)

        layout.addRow(QLabel("Epochs:"),           self.epochs)
        layout.addRow(QLabel("Batch size:"),        self.batch_size)
        layout.addRow(QLabel("Learning rate:"),     self.lr)
        layout.addRow(QLabel("Validation split:"),  self.val_split)
        layout.addRow(QLabel("Optimizer:"),         self.optimizer_combo)
        layout.addRow(QLabel("Loss:"),              self.loss_combo)
        for m_name, cb in self.metric_checks.items():
            layout.addRow(QLabel(f"Metric — {m_name}:"), cb)
        layout.addRow(QLabel("Early stop:"),        self.early_stop)
        layout.addRow(QLabel("Shuffle:"),           self.shuffle)

        widget.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        self.setWidget(scroll)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)


        # ADDED: connect every field so project_vm stays in sync as the user types
        self.epochs.editingFinished.connect(self._push)
        self.batch_size.editingFinished.connect(self._push)
        self.lr.editingFinished.connect(self._push)
        self.val_split.editingFinished.connect(self._push)
        self.optimizer_combo.currentIndexChanged.connect(self._push)
        self.loss_combo.currentIndexChanged.connect(self._push)
        for cb in self.metric_checks.values():
            cb.checkStateChanged.connect(self._push)
        self.early_stop.checkStateChanged.connect(self._push)
        self.shuffle.checkStateChanged.connect(self._push)

    # ── push all fields to ProjectVM in one call ──────────────────────────────

    def _push(self) -> None:
        # ADDED: reads all current field values and calls the two ProjectVM
        # setters. Both training config and network config are updated together
        # so they're always consistent.
        try:
            epochs    = int(self.epochs.text())
            batch     = int(self.batch_size.text())
            lr        = float(self.lr.text())
            val_split = float(self.val_split.text())
        except ValueError:
            return   # don't push if the user is mid-typing an invalid value

        optimizer = _OPTIMIZERS[self.optimizer_combo.currentText()]
        loss      = _LOSSES[self.loss_combo.currentText()]
        metrics   = [_METRICS[name] for name, cb in self.metric_checks.items()
                     if cb.isChecked()]
        if not metrics:
            metrics = [MetricType.ACCURACY]   # always keep at least one metric

        self.project_vm.set_training_config(
            batch_size=batch,
            epochs=epochs,
            val_split=val_split,
            shuffle=self.shuffle.isChecked(),
            early_stop=self.early_stop.isChecked(),
        )
        self.project_vm.set_network_config(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            learning_rate=lr,
        )

    # ADDED: called by MainWindow._on_project_changed() after load.
    # Repopulates every field from the loaded project so the dock
    # reflects the saved config rather than the defaults.
    def load_from_project(self, project) -> None:
        tc = project.trainingConfig
        nc = project.networkConfig
        if tc is None or nc is None:
            return

        # Block all signals while populating to avoid triggering _push
        # (which would write partial data back before all fields are set)
        widgets = [self.epochs, self.batch_size, self.lr, self.val_split,
                   self.optimizer_combo, self.loss_combo, self.early_stop, self.shuffle,
                   *self.metric_checks.values()]
        for w in widgets:
            w.blockSignals(True)

        self.epochs.setText(str(tc._epochs))
        self.batch_size.setText(str(tc._batchSize))
        self.lr.setText(str(nc._learningRate))
        self.val_split.setText(str(tc._validationSplit))
        self.early_stop.setChecked(tc._earlyStop)
        self.shuffle.setChecked(tc._shuffle)

        # Set optimizer combo
        for display, enum_val in _OPTIMIZERS.items():
            if enum_val == nc._optimizer:
                self.optimizer_combo.setCurrentText(display)
                break

        # Set loss combo
        for display, enum_val in _LOSSES.items():
            if enum_val == nc._loss:
                self.loss_combo.setCurrentText(display)
                break

        # Set metric checkboxes
        for m_name, cb in self.metric_checks.items():
            cb.setChecked(_METRICS[m_name] in nc._metric)

        for w in widgets:
            w.blockSignals(False)