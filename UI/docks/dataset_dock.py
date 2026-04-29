from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QDialog, QGridLayout, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage
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

        # ADDED: wire up the preview button
        self.preview_btn.clicked.connect(self._on_preview_clicked)

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

    # ── dataset preview ───────────────────────────────────────────────────────

    def _on_preview_clicked(self) -> None:
        """
        Opens a dialog showing a 4×4 grid of sample images from the selected
        dataset. Uses torchvision to download/load the data on demand — the
        same library the trainer already depends on — so no extra deps needed.
        """
        display_name = self.dataset_combo.currentText()
        info = _DISPLAY_INFO.get(display_name)
        if info is None:
            return

        builtin_type = info[5]

        # Show a loading cursor while we fetch samples
        from PyQt6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            samples = self._load_preview_samples(builtin_type, n=16)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Preview failed",
                                f"Could not load dataset samples:\n{exc}")
            return
        QApplication.restoreOverrideCursor()

        dialog = QDialog(self.window())
        dialog.setWindowTitle(f"Preview — {display_name}")
        dialog.setMinimumSize(520, 520)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(6)

        cols = 4
        for idx, (img_array, label_text) in enumerate(samples):
            row, col = divmod(idx, cols)

            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)
            cell_layout.setSpacing(2)

            # Convert numpy array → QPixmap
            pixmap = self._array_to_pixmap(img_array)
            img_lbl = QLabel()
            img_lbl.setPixmap(pixmap.scaled(
                96, 96,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            ))
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            cap_lbl = QLabel(label_text)
            cap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap_lbl.setStyleSheet("font-size: 10px; color: #cccccc;")

            cell_layout.addWidget(img_lbl)
            cell_layout.addWidget(cap_lbl)
            grid.addWidget(cell, row, col)

        scroll.setWidget(container)
        dlg_layout = QVBoxLayout(dialog)
        dlg_layout.addWidget(scroll)
        dialog.exec()

    # ── helpers ───────────────────────────────────────────────────────────────

    # Maps BuiltinDataset enum → (torchvision class name, class labels list or None)
    _TV_MAP = {
        BuiltinDataset.MNIST:         ("MNIST",        None),
        BuiltinDataset.FASHION_MNIST: ("FashionMNIST", [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]),
        BuiltinDataset.KMNIST:        ("KMNIST",       None),
        BuiltinDataset.CIFAR10:       ("CIFAR10",      [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"]),
        BuiltinDataset.CIFAR100:      ("CIFAR100",     None),
        BuiltinDataset.SVHN:          ("SVHN",         None),
        BuiltinDataset.STL10:         ("STL10",        [
            "airplane", "bird", "car", "cat", "deer",
            "dog", "horse", "monkey", "ship", "truck"]),
    }

    def _load_preview_samples(self, builtin_type: BuiltinDataset, n: int):
        """
        Returns a list of (numpy_array, label_str) tuples.
        Downloads the dataset to the system temp dir if not already cached.
        """
        import tempfile, numpy as np
        import torchvision.datasets as tvd
        import torchvision.transforms as T

        tv_name, class_names = self._TV_MAP.get(builtin_type, (None, None))
        if tv_name is None:
            raise ValueError(f"No torchvision mapping for {builtin_type}")

        cache_dir = tempfile.gettempdir()
        transform = T.ToTensor()

        cls = getattr(tvd, tv_name)
        # SVHN and STL10 use 'split' instead of 'train'
        if tv_name in ("SVHN", "STL10"):
            ds = cls(root=cache_dir, split="train", download=True, transform=transform)
        else:
            ds = cls(root=cache_dir, train=True, download=True, transform=transform)

        samples = []
        for i in range(min(n, len(ds))):
            img_tensor, label_idx = ds[i]
            # img_tensor shape: (C, H, W) in [0, 1]
            arr = (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]   # grayscale → 2-D for QImage

            if class_names:
                label_str = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
            else:
                label_str = str(label_idx)

            samples.append((arr, label_str))

        return samples

    def _array_to_pixmap(self, arr) -> QPixmap:
        """Converts a uint8 numpy array (H,W) or (H,W,3) to QPixmap."""
        import numpy as np
        if arr.ndim == 2:
            # Grayscale
            h, w = arr.shape
            arr_c = np.ascontiguousarray(arr)
            qimg = QImage(arr_c.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, c = arr.shape
            arr_c = np.ascontiguousarray(arr)
            if c == 3:
                qimg = QImage(arr_c.data, w, h, w * 3, QImage.Format.Format_RGB888)
            else:
                qimg = QImage(arr_c.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimg.copy())   # .copy() detaches from numpy buffer

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