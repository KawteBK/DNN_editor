import time

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QFileDialog, QProgressBar, QFrame,
                              QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

# ── Class-name lookup (mirrors results_dock._CLASS_NAMES) ────────────────────
_CLASS_NAMES = {
    "mnist":         [str(i) for i in range(10)],
    "fashion_mnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
    "kmnist":        ["O", "Ki", "Su", "Tsu", "Na", "Ha", "Ma", "Ya", "Re", "Wo"],
    "cifar10":       ["Plane", "Car", "Bird", "Cat", "Deer",
                      "Dog", "Frog", "Horse", "Ship", "Truck"],
    "cifar100":      [f"C{i}" for i in range(100)],
    "svhn":          [str(i) for i in range(10)],
    "stl10":         ["Plane", "Bird", "Car", "Cat", "Deer",
                      "Dog", "Horse", "Monkey", "Ship", "Truck"],
}


class ImageDropZone(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setText("⬆️  Drop an image here or click to browse\nPNG, JPG — resized automatically")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaaaaa;
                border-radius: 10px;
                padding: 10px;
                color: #888888;
                font-size: 11px;
            }
        """)
        self.setFixedHeight(70)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.parent().load_image(urls[0].toLocalFile())


class OngletInference(QWidget):
    def __init__(self, project_vm=None):
        super().__init__()
        self._project_vm = project_vm   # injected by MainWindow after construction
        self._image_path = None
        self._history: list[str] = []

        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ── LEFT PANEL ────────────────────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        lbl_input = QLabel("INPUT")
        lbl_input.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        left_layout.addWidget(lbl_input)

        # mode buttons
        mode_row = QHBoxLayout()
        self.btn_single = QPushButton("Single image")
        self.btn_batch  = QPushButton("Batch folder")
        self.btn_single.setCheckable(True)
        self.btn_single.setChecked(True)
        self.btn_batch.setCheckable(True)
        self.btn_single.setStyleSheet("""
            QPushButton {
                padding: 8px;
                border-radius: 8px;
                background: #1592A2;
                color: white;
            }
            QPushButton:hover { background: #0e6b77; }
        """)
        self.btn_batch.setStyleSheet("padding: 8px; border-radius: 8px;")
        self.btn_single.clicked.connect(lambda: self._set_mode("single"))
        self.btn_batch.clicked.connect(lambda: self._set_mode("batch"))
        mode_row.addWidget(self.btn_single)
        mode_row.addWidget(self.btn_batch)
        left_layout.addLayout(mode_row)

        # drop zone
        self.drop_zone = ImageDropZone(self)
        self.drop_zone.mousePressEvent = lambda e: self.browse_image()
        left_layout.addWidget(self.drop_zone)

        # image preview
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setFixedHeight(70)
        self.image_preview.setStyleSheet("border: 1px solid #ccc; border-radius: 8px;")
        self.image_preview.setText("preview")
        left_layout.addWidget(self.image_preview)
        left_layout.addSpacing(10)

        # ── normalization checkbox ─────────────────────────────────────────────
        self.chk_normalize = QCheckBox("Apply dataset normalization")
        self.chk_normalize.setChecked(True)
        self.chk_normalize.setStyleSheet("""
            QCheckBox {
                font-size: 11px;
                color: #cccccc;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border-radius: 3px;
                border: 1px solid #888888;
                background: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background: #1592A2;
                border: 1px solid #1592A2;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #1592A2;
            }
        """)
        self.chk_normalize.stateChanged.connect(self._update_normalization_status)
        left_layout.addWidget(self.chk_normalize)

        # status label — tells user exactly what the backend will do
        self.lbl_norm_status = QLabel()
        self.lbl_norm_status.setWordWrap(True)
        self.lbl_norm_status.setStyleSheet("font-size: 10px; color: #888888;")
        left_layout.addWidget(self.lbl_norm_status)
        self._update_normalization_status()   # set initial text

        left_layout.addSpacing(6)

        # run button
        self.btn_run = QPushButton("Run Inference")
        self.btn_run.setStyleSheet("""
            QPushButton {
                padding: 10px;
                border-radius: 8px;
                background: #1592A2;
                color: white;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background: #0e6b77; }
            QPushButton:disabled { background: #555555; color: #888888; }
        """)
        self.btn_run.clicked.connect(self.run_inference)
        left_layout.addWidget(self.btn_run)

        left_layout.addStretch()
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(320)
        main_layout.addWidget(left_widget)

        # ── RIGHT PANEL ───────────────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)

        lbl_result = QLabel("RESULT")
        lbl_result.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        right_layout.addWidget(lbl_result)

        self.lbl_prediction = QLabel("-- No prediction yet --")
        self.lbl_prediction.setStyleSheet("""
            background: #1592A2;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: bold;
            color: white;
        """)
        self.lbl_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.lbl_prediction)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self.stat_class      = self._make_stat_card("--",   "class index", "Predicted")
        self.stat_confidence = self._make_stat_card("--%",  "top-1",       "Confidence")
        self.stat_time       = self._make_stat_card("--ms", "wall clock",  "Inference time")
        stats_row.addWidget(self.stat_class)
        stats_row.addWidget(self.stat_confidence)
        stats_row.addWidget(self.stat_time)
        right_layout.addLayout(stats_row)

        lbl_top5 = QLabel("Top-5 probabilities")
        lbl_top5.setStyleSheet("font-size: 12px; color: #888; margin-top: 6px;")
        right_layout.addWidget(lbl_top5)

        self._prob_bars: list[tuple] = []
        for _ in range(5):
            row = QHBoxLayout()
            name_lbl = QLabel("--")
            name_lbl.setFixedWidth(100)
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setStyleSheet(
                "QProgressBar::chunk { background: #1592A2; border-radius: 4px; }"
            )
            pct_lbl = QLabel("0%")
            pct_lbl.setFixedWidth(36)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(name_lbl)
            row.addWidget(bar)
            row.addWidget(pct_lbl)
            self._prob_bars.append((name_lbl, bar, pct_lbl))
            right_layout.addLayout(row)

        lbl_history = QLabel("HISTORY")
        lbl_history.setStyleSheet(
            "font-weight: bold; font-size: 10px; color: #ffffff; margin-top: 8px;"
        )
        right_layout.addWidget(lbl_history)

        self.lbl_history = QLabel("No predictions yet")
        self.lbl_history.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_history.setStyleSheet("""
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            color: #ffffff;
            font-size: 11px;
        """)
        right_layout.addWidget(self.lbl_history)

        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)

    # ── public API ────────────────────────────────────────────────────────────

    def set_project_vm(self, project_vm) -> None:
        """Called by MainWindow after construction to inject the ProjectVM."""
        self._project_vm = project_vm
        self._update_normalization_status()

    # ── normalization status ──────────────────────────────────────────────────

    def _update_normalization_status(self):
        """
        Updates the label below the checkbox to describe exactly what the
        backend transform pipeline will do when Run Inference is clicked.
        """
        if not self.chk_normalize.isChecked():
            self.lbl_norm_status.setText(
                "Raw pixel values will be passed to the model (ToTensor only, "
                "range [0, 1]). Only use this if the model was trained without "
                "normalization."
            )
            return

        project = self._project_vm.project if self._project_vm else None
        if project is None or project.dataset is None:
            self.lbl_norm_status.setText(
                "Normalization enabled — stats will be applied once a dataset "
                "is selected."
            )
            return

        from Core.dataset import DatasetSource
        if (project.dataset._source != DatasetSource.BUILTIN
                or project.dataset._builtinType is None):
            self.lbl_norm_status.setText(
                "Normalization enabled — no built-in stats for custom datasets; "
                "normalization will be skipped at inference time."
            )
            return

        try:
            from Infra.pytorchDataset import _NORMALIZATION_STATS
            mean, std = _NORMALIZATION_STATS.get(
                project.dataset._builtinType, ((0.5,), (0.5,))
            )
            mean_str = ", ".join(f"{v:.4f}" for v in mean)
            std_str  = ", ".join(f"{v:.4f}" for v in std)
            self.lbl_norm_status.setText(
                f"Will apply {project.dataset._name} stats:\n"
                f"mean=({mean_str})  std=({std_str})"
            )
        except Exception:
            self.lbl_norm_status.setText(
                "Normalization enabled — dataset stats will be loaded at inference time."
            )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_stat_card(self, value: str, subtitle: str, title: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("border: 1px solid #ddd; border-radius: 8px; padding: 6px;")
        card.setMinimumWidth(120)
        layout = QVBoxLayout()
        layout.setSpacing(2)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 11px; color: #888; border: none;")
        lbl_value = QLabel(value)
        lbl_value.setStyleSheet("font-size: 10px; font-weight: bold; border: none;")
        lbl_sub = QLabel(subtitle)
        lbl_sub.setStyleSheet("font-size: 10px; color: #888; border: none;")
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        layout.addWidget(lbl_sub)
        card.setLayout(layout)
        card.label_valeur = lbl_value   # keep attribute name for MainWindow compat
        return card

    def _set_mode(self, mode: str):
        active = """
            QPushButton {
                padding: 8px; border-radius: 8px;
                background: #1592A2; color: white;
            }
            QPushButton:hover { background: #0e6b77; }
        """
        inactive = "padding: 8px; border-radius: 8px;"
        if mode == "single":
            self.btn_single.setStyleSheet(active)
            self.btn_batch.setStyleSheet(inactive)
        else:
            self.btn_batch.setStyleSheet(active)
            self.btn_single.setStyleSheet(inactive)

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        self._image_path = path
        pixmap = QPixmap(path).scaled(150, 100, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_preview.setPixmap(pixmap)

    # ── core: real inference ──────────────────────────────────────────────────

    def run_inference(self):
        # ── guard: image selected?
        if not self._image_path:
            self.lbl_prediction.setText("⚠️ Please select an image first.")
            return

        # ── guard: project loaded?
        project = self._project_vm.project if self._project_vm else None
        if project is None:
            self.lbl_prediction.setText("⚠️ No project loaded.")
            return
        if project.network is None:
            self.lbl_prediction.setText("⚠️ Project has no network.")
            return

        # ── guard: weights on disk?
        if not project.weightsPath:
            import os
            from Core.project import Project as _Proj
            candidate = os.path.join(project.path, _Proj.WEIGHTS_FILENAME)
            if os.path.isfile(candidate):
                project.setWeightsPath(candidate)
            else:
                self.lbl_prediction.setText(
                    "⚠️ Model not trained yet — run training first."
                )
                return

        # ── build transform pipeline
        try:
            from torchvision import transforms
            from PIL import Image as PILImage
        except ImportError as e:
            self.lbl_prediction.setText(f"⚠️ Missing dependency: {e}")
            return

        transform_steps = [transforms.ToTensor()]

        if self.chk_normalize.isChecked() and project.dataset is not None:
            from Core.dataset import DatasetSource
            from Infra.pytorchDataset import _NORMALIZATION_STATS
            if (project.dataset._source == DatasetSource.BUILTIN
                    and project.dataset._builtinType is not None):
                mean, std = _NORMALIZATION_STATS.get(
                    project.dataset._builtinType, ((0.5,), (0.5,))
                )
                transform_steps.append(transforms.Normalize(mean=mean, std=std))

        # resize to the network's expected spatial dimensions
        input_shape = None
        input_layer = project.network.layers.get("input")
        if input_layer is not None:
            input_shape = input_layer._outputShape   # e.g. (1, 28, 28) or (3, 32, 32)
            if input_shape and len(input_shape) == 3:
                _, H, W = input_shape
                transform_steps.insert(0, transforms.Resize((H, W)))

        # convert to grayscale if model expects single-channel input
        if input_shape and len(input_shape) == 3 and input_shape[0] == 1:
            transform_steps.insert(0, transforms.Grayscale(num_output_channels=1))

        transform = transforms.Compose(transform_steps)

        # ── load + transform image
        try:
            img = PILImage.open(self._image_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)   # (1, C, H, W)
        except Exception as e:
            self.lbl_prediction.setText(f"⚠️ Image load error: {e}")
            return

        # ── run inference
        try:
            from Infra.trainer import PyTorchTrainer
            t0 = time.perf_counter()
            results = PyTorchTrainer.predict(project, tensor)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            self.lbl_prediction.setText(f"⚠️ Inference error: {e}")
            return

        # ── unpack
        predicted_idx = results["classes"][0]
        probabilities = results["probabilities"][0]
        confidence    = results["confidence"][0]

        dataset_key = (
            project.dataset._builtinType.value
            if project.dataset and project.dataset._builtinType
            else None
        )
        class_names = list(_CLASS_NAMES.get(dataset_key, [
            str(i) for i in range(len(probabilities))
        ]))
        while len(class_names) < len(probabilities):
            class_names.append(str(len(class_names)))
        class_names = class_names[:len(probabilities)]

        predicted_name = class_names[predicted_idx]

        # ── update result banner
        self.lbl_prediction.setText(
            f"Class {predicted_idx} — {predicted_name}   {confidence * 100:.1f}%"
        )

        # ── stat boxes
        self.stat_class.label_valeur.setText(str(predicted_idx))
        self.stat_confidence.label_valeur.setText(f"{confidence * 100:.1f}%")
        self.stat_time.label_valeur.setText(f"{elapsed_ms:.0f}ms")

        # ── top-5 bars
        top5 = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)[:5]
        for i, (name_lbl, bar, pct_lbl) in enumerate(self._prob_bars):
            if i < len(top5):
                idx, prob = top5[i]
                pct = prob * 100
                name_lbl.setText(class_names[idx])
                bar.setValue(int(pct))
                pct_lbl.setText(f"{pct:.1f}%")
            else:
                name_lbl.setText("--")
                bar.setValue(0)
                pct_lbl.setText("0%")

        # ── history
        self._history.append(f"{predicted_name} — {confidence * 100:.1f}%")
        self.lbl_history.setText("\n".join(self._history[-5:]))