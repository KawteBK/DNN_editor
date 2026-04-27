from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton
)
from matplotlib.pylab import info

from Core.dataset import BuiltinDataset


# Metadata for display — mirrors what's in Core but kept here to avoid
# importing the whole dataset module just for label strings
_DISPLAY_INFO = {
    "MNIST":         ("MNIST",        "10",   "(1, 28, 28)",  "60,000",    "10,000", BuiltinDataset.MNIST),
    "Fashion-MNIST": ("Fashion-MNIST","10",   "(1, 28, 28)",  "60,000",    "10,000", BuiltinDataset.FASHION_MNIST),
    "KMNIST":        ("KMNIST",       "10",   "(1, 28, 28)",  "60,000",    "10,000", BuiltinDataset.KMNIST),
    "CIFAR-10":      ("CIFAR-10",     "10",   "(3, 32, 32)",  "50,000",    "10,000", BuiltinDataset.CIFAR10),
    "CIFAR-100":     ("CIFAR-100",    "100",  "(3, 32, 32)",  "50,000",    "10,000", BuiltinDataset.CIFAR100),
    "SVHN":          ("SVHN",         "10",   "(3, 32, 32)",  "73,257",    "26,032", BuiltinDataset.SVHN),
    "STL-10":        ("STL-10",       "10",   "(3, 96, 96)",  "5,000",     "8,000",  BuiltinDataset.STL10),
}


class DatasetDock(QDockWidget):
    # CHANGED: accepts project_vm. Previously only updated its own labels
    # and never told Core which dataset was selected.
    def __init__(self, project_vm):
        super().__init__("Dataset")
        self.project_vm = project_vm   # ADDED

        widget = QWidget()
        layout = QVBoxLayout()

        # CHANGED: combo now lists all 7 built-in datasets (was only 4, and
        # included ImageNet which isn't in Core at all)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(list(_DISPLAY_INFO.keys()))
        layout.addWidget(self.dataset_combo)

        self.info_name    = QLabel()
        self.info_classes = QLabel()
        self.info_input   = QLabel()
        self.info_train   = QLabel()
        self.info_test    = QLabel()
        self.info_task    = QLabel("Task: Classification")

        for lbl in [self.info_name, self.info_classes, self.info_input,
                    self.info_train, self.info_test, self.info_task]:
            layout.addWidget(lbl)

        self.preview_btn = QPushButton("Preview")
        layout.addWidget(self.preview_btn)

        layout.addStretch()
        widget.setLayout(layout)
        self.setWidget(widget)

        # CHANGED: on change, updates labels AND tells project_vm which dataset
        # was picked so project.dataset gets set properly
        self.dataset_combo.currentTextChanged.connect(self._on_selection_changed)

        # Populate labels for the default selection
        self._update_labels(self.dataset_combo.currentText())

    # ── signal handler ────────────────────────────────────────────────────────

    def _on_selection_changed(self, display_name: str) -> None:
        self._update_labels(display_name)
        info = _DISPLAY_INFO.get(display_name)
        if info:
            self.project_vm.set_dataset(info[5])
            #  find the main window and update train button
            main_window = self.window()
            if hasattr(main_window, '_update_train_button_state'):
                main_window._update_train_button_state()


    def _update_labels(self, display_name: str) -> None:
        info = _DISPLAY_INFO.get(display_name)
        if not info:
            return
        name, classes, shape, train, test, _ = info
        self.info_name.setText(f"Name: {name}")
        self.info_classes.setText(f"Classes: {classes}")
        self.info_input.setText(f"Input shape: {shape}")
        self.info_train.setText(f"Train samples: {train}")
        self.info_test.setText(f"Test samples: {test}")

    # ADDED: called by MainWindow._on_project_changed() after a project is loaded.
    # Repopulates the combo and labels from the loaded project's dataset so
    # the dock reflects the saved state rather than the default selection.
    def load_from_project(self, project) -> None:
        if project.dataset is None:
            return
        # Find the display name that matches the loaded builtin type
        for display_name, info in _DISPLAY_INFO.items():
            if info[5] == project.dataset._builtinType:
                # Block signals while setting so we don't re-trigger set_dataset
                self.dataset_combo.blockSignals(True)
                self.dataset_combo.setCurrentText(display_name)
                self.dataset_combo.blockSignals(False)
                self._update_labels(display_name)
                break