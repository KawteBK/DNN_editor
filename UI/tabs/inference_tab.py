import time

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QFileDialog, QProgressBar, QFrame,
                              QCheckBox)
from PyQt6.QtCore import Qt, QByteArray
from PyQt6.QtGui import QPixmap, QPainter
from PyQt6.QtSvg import QSvgRenderer


# ── Icon helpers ──────────────────────────────────────────────────────────────

def _make_svg_pixmap(svg: str, size: int = 14) -> QPixmap:
    """Render a tiny inline SVG string to a QPixmap."""
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    return pm


def _icon_check(size: int = 14, color: str = "#1592A2") -> QPixmap:
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>
      <circle cx='8' cy='8' r='7.5' fill='{color}'/>
      <polyline points='4,8.5 7,11.5 12,5' fill='none'
                stroke='white' stroke-width='1.8'
                stroke-linecap='round' stroke-linejoin='round'/>
    </svg>"""
    return _make_svg_pixmap(svg, size)


def _icon_cross(size: int = 14, color: str = "#E67E22") -> QPixmap:
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>
      <circle cx='8' cy='8' r='7.5' fill='{color}'/>
      <line x1='5' y1='5' x2='11' y2='11'
            stroke='white' stroke-width='1.8' stroke-linecap='round'/>
      <line x1='11' y1='5' x2='5' y2='11'
            stroke='white' stroke-width='1.8' stroke-linecap='round'/>
    </svg>"""
    return _make_svg_pixmap(svg, size)


def _icon_warning(size: int = 14, color: str = "#E67E22") -> QPixmap:
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>
      <polygon points='8,1.5 15,14.5 1,14.5' fill='{color}'/>
      <text x='8' y='12.5' text-anchor='middle'
            font-size='8' font-weight='bold'
            fill='white' font-family='sans-serif'>!</text>
    </svg>"""
    return _make_svg_pixmap(svg, size)


def _icon_upload(size: int = 20, color: str = "#888888") -> QPixmap:
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>
      <line x1='8' y1='1' x2='8' y2='11'
            stroke='{color}' stroke-width='2' stroke-linecap='round'/>
      <polyline points='4,5 8,1 12,5' fill='none'
                stroke='{color}' stroke-width='2'
                stroke-linecap='round' stroke-linejoin='round'/>
      <line x1='2' y1='15' x2='14' y2='15'
            stroke='{color}' stroke-width='2' stroke-linecap='round'/>
    </svg>"""
    return _make_svg_pixmap(svg, size)


def _pixmap_label(pixmap: QPixmap) -> QLabel:
    lbl = QLabel()
    lbl.setPixmap(pixmap)
    lbl.setFixedSize(pixmap.width(), pixmap.height())
    lbl.setStyleSheet("border:none; background:transparent;")
    return lbl

# ── Class-name lookup ────────────────────────────────────────────────────────
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

_BG      = "#2b2b2b"
_SURFACE = "#3c3f41"
_TEAL    = "#1592A2"
_ORANGE  = "#E67E22"
_GREEN   = "#2ECC71"
_RED     = "#E74C3C"
_DIM     = "#888888"
_TEXT    = "#cccccc"
_BORDER  = "#555555"


class ImageDropZone(QFrame):
    def __init__(self, tab, parent=None):
        super().__init__(parent)
        self._tab = tab
        self.setAcceptDrops(True)
        self.setFixedHeight(70)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaaaaa;
                border-radius: 10px;
            }
        """)
        inner = QVBoxLayout(self)
        inner.setContentsMargins(8, 6, 8, 6)
        inner.setSpacing(2)
        inner.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_lbl = _pixmap_label(_icon_upload(20, "#888888"))
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        txt = QLabel("Drop an image here or click to browse\nPNG, JPG — resized automatically")
        txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        txt.setStyleSheet("border:none; color:#888888; font-size:11px;")
        inner.addWidget(txt)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self._tab.load_image(urls[0].toLocalFile())


class OngletInference(QWidget):
    def __init__(self, project_vm=None):
        super().__init__()
        self._project_vm = project_vm
        self._image_path = None
        self._history: list[str] = []

        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self._build_left_panel())
        main_layout.addWidget(self._build_right_panel())
        self.setLayout(main_layout)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # ── INPUT header
        lbl = QLabel("INPUT")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        layout.addWidget(lbl)

        # ── mode buttons
        mode_row = QHBoxLayout()
        self.btn_single = QPushButton("Single image")
        #self.btn_batch  = QPushButton("Batch folder")
        self.btn_single.setCheckable(True)
        self.btn_single.setChecked(True)
        #self.btn_batch.setCheckable(True)
        _active_style = """
            QPushButton { padding:8px; border-radius:8px;
                          background:#1592A2; color:white; }
            QPushButton:hover { background:#0e6b77; }
        """
        self.btn_single.setStyleSheet(_active_style)
        #self.btn_batch.setStyleSheet("padding:8px; border-radius:8px;")
        self.btn_single.clicked.connect(lambda: self._set_mode("single"))
        #self.btn_batch.clicked.connect(lambda: self._set_mode("batch"))
        mode_row.addWidget(self.btn_single)
        #mode_row.addWidget(self.btn_batch)
        layout.addLayout(mode_row)

        # ── drop zone + preview
        self.drop_zone = ImageDropZone(self, self)
        self.drop_zone.mousePressEvent = lambda e: self.browse_image()
        layout.addWidget(self.drop_zone)

        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setFixedHeight(150)
        self.image_preview.setStyleSheet("border:1px solid #ccc; border-radius:8px;")
        self.image_preview.setText("preview")
        layout.addWidget(self.image_preview)
        layout.addSpacing(12)

        # ── PREPROCESSING section header
        lbl_pre = QLabel("PREPROCESSING")
        lbl_pre.setStyleSheet("font-weight:bold; font-size:11px; color:#888888;"
                              " letter-spacing:1px;")
        layout.addWidget(lbl_pre)

        # ── Normalization — locked ON, teal info box
        norm_card = QFrame()
        norm_card.setStyleSheet(
            f"background:{_SURFACE}; border-radius:6px; border:1px solid {_TEAL};"
        )
        norm_card_layout = QVBoxLayout(norm_card)
        norm_card_layout.setContentsMargins(10, 8, 10, 8)
        norm_card_layout.setSpacing(4)

        lbl_norm_title = QLabel("Normalization on")
        lbl_norm_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_norm_title.setStyleSheet(
            f"color:#ffffff; font-size:11px; font-weight:bold; border:none;"
        )
        norm_card_layout.addWidget(lbl_norm_title)

        layout.addWidget(norm_card)

        layout.addSpacing(8)

        #  Experiment mode toggle
        exp_card = QFrame()
        exp_card.setStyleSheet(
            f"background:{_SURFACE}; border-radius:6px; border:1px solid {_BORDER};"
        )
        exp_layout = QVBoxLayout(exp_card)
        exp_layout.setContentsMargins(10, 8, 10, 8)
        exp_layout.setSpacing(6)

        exp_top = QHBoxLayout()
        exp_top.setSpacing(6)
        exp_icon = _pixmap_label(_icon_warning(14, _ORANGE))
        exp_top.addWidget(exp_icon)
        self.chk_experiment = QCheckBox("Experiment mode")
        self.chk_experiment.setChecked(False)
        self.chk_experiment.setStyleSheet(f"""
            QCheckBox {{
                font-size: 11px;
                color: {_ORANGE};
                font-weight: bold;
                spacing: 6px;
                border: none;
            }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border-radius: 3px;
                border: 1px solid {_ORANGE};
                background: {_BG};
            }}
            QCheckBox::indicator:checked {{
                background: {_ORANGE};
                border: 1px solid {_ORANGE};
            }}
        """)
        exp_top.addWidget(self.chk_experiment)
        exp_top.addStretch()
        exp_layout.addLayout(exp_top)

        lbl_exp_desc = QLabel(
            "Disable normalization : runs a second pass with raw pixel "
            "values so you can compare it against the normalized result."
        )
        lbl_exp_desc.setWordWrap(True)
        lbl_exp_desc.setStyleSheet(f"color:{_DIM}; font-size:10px; border:none;")
        exp_layout.addWidget(lbl_exp_desc)

        # warning banner — hidden until experiment mode is enabled
        self._warn_row = QHBoxLayout()
        self._warn_row.setSpacing(6)
        warn_icon = _pixmap_label(_icon_warning(14, _RED))
        self._warn_row.addWidget(warn_icon, alignment=Qt.AlignmentFlag.AlignTop)
        self.lbl_exp_warning = QLabel(
            "This model was trained on normalized data. The raw prediction "
            "below will likely be wrong — that's the point."
        )
        self.lbl_exp_warning.setWordWrap(True)
        self.lbl_exp_warning.setStyleSheet(
            f"color:{_RED}; font-size:10px; font-weight:bold; border:none;"
        )
        self._warn_row.addWidget(self.lbl_exp_warning)
        self._warn_container = QWidget()
        self._warn_container.setLayout(self._warn_row)
        self._warn_container.setVisible(False)
        exp_layout.addWidget(self._warn_container)

        self.chk_experiment.stateChanged.connect(self._on_experiment_toggled)
        layout.addWidget(exp_card)

        layout.addSpacing(8)

        # ── Run button
        self.btn_run = QPushButton("Run Inference")
        self.btn_run.setStyleSheet(f"""
            QPushButton {{
                padding: 10px; border-radius: 8px;
                background: {_TEAL}; color: white;
                font-size: 13px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #0e6b77; }}
            QPushButton:disabled {{ background: #555555; color: #888888; }}
        """)
        self.btn_run.clicked.connect(self.run_inference)
        layout.addWidget(self.btn_run)

        layout.addStretch()
        widget.setLayout(layout)
        widget.setMaximumWidth(320)
        return widget

    def _build_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)

        lbl_result = QLabel("RESULT")
        lbl_result.setStyleSheet(
            "font-weight:bold; font-size:13px; color:#ffffff;"
        )
        layout.addWidget(lbl_result)

        # ── two result columns side by side
        cols = QHBoxLayout()
        cols.setSpacing(12)

        self._norm_col  = self._build_result_column(
            title="With normalization",
            title_color=_TEAL,
            tag="norm",
            icon_pm=_icon_check(14, _TEAL),
        )
        self._raw_col = self._build_result_column(
            title="Without normalization",
            title_color=_ORANGE,
            tag="raw",
            icon_pm=_icon_cross(14, _ORANGE),
        )

        cols.addWidget(self._norm_col["widget"])
        cols.addWidget(self._raw_col["widget"])
        layout.addLayout(cols)

        # raw column hidden until experiment mode is on
        self._raw_col["widget"].setVisible(False)

        # ── history
        lbl_history = QLabel("HISTORY")
        lbl_history.setStyleSheet(
            f"font-weight:bold; font-size:10px; color:#ffffff; margin-top:8px;"
        )
        layout.addWidget(lbl_history)

        self.lbl_history = QLabel("No predictions yet")
        self.lbl_history.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_history.setStyleSheet("""
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            color: #ffffff;
            font-size: 11px;
        """)
        layout.addWidget(self.lbl_history)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _build_result_column(self, title: str, title_color: str,
                              tag: str, icon_pm: QPixmap = None) -> dict:
        """
        Builds one result column (normalized or raw) and returns a dict
        with references to every widget that needs updating after inference.
        """
        col = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        title_row.setSpacing(5)
        if icon_pm is not None:
            title_row.addWidget(_pixmap_label(icon_pm))
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(
            f"font-size:11px; font-weight:bold; color:{title_color};"
        )
        title_row.addWidget(lbl_title)
        title_row.addStretch()
        layout.addLayout(title_row)

        lbl_pred = QLabel("-- --")
        lbl_pred.setStyleSheet(f"""
            background: {_SURFACE};
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 13px;
            font-weight: bold;
            color: {title_color};
            border: 1px solid {title_color};
        """)
        lbl_pred.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_pred.setWordWrap(True)
        layout.addWidget(lbl_pred)

        # stat cards row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(6)
        stat_class = self._make_stat_card("--",   "class index", "Predicted")
        stat_conf  = self._make_stat_card("--%",  "top-1",       "Confidence")
        stat_time  = self._make_stat_card("--ms", "wall clock",  "Time")
        stats_row.addWidget(stat_class)
        stats_row.addWidget(stat_conf)
        stats_row.addWidget(stat_time)
        layout.addLayout(stats_row)

        # top-5 bars
        lbl_top5 = QLabel("Top-5 probabilities")
        lbl_top5.setStyleSheet(f"font-size:11px; color:{_DIM}; margin-top:4px;")
        layout.addWidget(lbl_top5)

        prob_bars = []
        for _ in range(5):
            row = QHBoxLayout()
            row.setSpacing(4)
            name_lbl = QLabel("--")
            name_lbl.setMinimumWidth(55)
            name_lbl.setMaximumWidth(80)
            name_lbl.setStyleSheet("font-size:10px;")
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(7)
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background:{title_color};"
                f" border-radius:3px; }}"
            )
            pct_lbl = QLabel("0%")
            pct_lbl.setFixedWidth(42)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct_lbl.setStyleSheet("font-size:10px;")
            row.addWidget(name_lbl)
            row.addWidget(bar, 1)
            row.addWidget(pct_lbl)
            prob_bars.append((name_lbl, bar, pct_lbl))
            layout.addLayout(row)

        layout.addStretch()
        col.setLayout(layout)

        return {
            "widget":     col,
            "lbl_pred":   lbl_pred,
            "stat_class": stat_class,
            "stat_conf":  stat_conf,
            "stat_time":  stat_time,
            "prob_bars":  prob_bars,
        }

    # ── public API ────────────────────────────────────────────────────────────

    def set_project_vm(self, project_vm) -> None:
        self._project_vm = project_vm
        self._refresh_norm_stats_label()

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_experiment_toggled(self):
        enabled = self.chk_experiment.isChecked()
        self._warn_container.setVisible(enabled)
        self._raw_col["widget"].setVisible(enabled)

    # ── normalization stats label ─────────────────────────────────────────────

    def _refresh_norm_stats_label(self):
        pass  # norm stats label removed — card just shows "Normalization on"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_stat_card(self, value: str, subtitle: str, title: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f"border:1px solid {_BORDER}; border-radius:6px; padding:4px;"
        )
        card.setMinimumWidth(80)
        layout = QVBoxLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(6, 4, 6, 4)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"font-size:10px; color:{_DIM}; border:none;")
        lbl_value = QLabel(value)
        lbl_value.setStyleSheet(
            f"font-size:11px; font-weight:bold; border:none; color:{_TEXT};"
        )
        lbl_sub = QLabel(subtitle)
        lbl_sub.setStyleSheet(f"font-size:9px; color:{_DIM}; border:none;")
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        layout.addWidget(lbl_sub)
        card.setLayout(layout)
        card.label_valeur = lbl_value
        return card

    def _set_mode(self, mode: str):
        active = """
            QPushButton { padding:8px; border-radius:8px;
                          background:#1592A2; color:white; }
            QPushButton:hover { background:#0e6b77; }
        """
        inactive = "padding:8px; border-radius:8px;"
        self.btn_single.setStyleSheet(active   if mode == "single" else inactive)
        #self.btn_batch.setStyleSheet (inactive if mode == "single" else active)

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        self._image_path = path
        pixmap = QPixmap(path).scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_preview.setPixmap(pixmap)

    # ── transform builder ─────────────────────────────────────────────────────

    def _build_transform(self, project, normalize: bool):
        """
        Returns a torchvision Compose pipeline.
        normalize=True  → the same pipeline used during training (correct).
        normalize=False → ToTensor only, no Normalize (experiment / demonstration).
        """
        from torchvision import transforms

        steps = [transforms.ToTensor()]

        if normalize and project.dataset is not None:
            from Core.dataset import DatasetSource
            from Infra.pytorchDataset import _NORMALIZATION_STATS
            if (project.dataset._source == DatasetSource.BUILTIN
                    and project.dataset._builtinType is not None):
                mean, std = _NORMALIZATION_STATS.get(
                    project.dataset._builtinType, ((0.5,), (0.5,))
                )
                steps.append(transforms.Normalize(mean=mean, std=std))

        input_shape = None
        input_layer = project.network.layers.get("input")
        if input_layer is not None:
            input_shape = input_layer._outputShape
            if input_shape and len(input_shape) == 3:
                _, H, W = input_shape
                steps.insert(0, transforms.Resize((H, W)))

        if input_shape and len(input_shape) == 3 and input_shape[0] == 1:
            steps.insert(0, transforms.Grayscale(num_output_channels=1))

        return transforms.Compose(steps)

    def _run_single_pass(self, project, normalize: bool):
        """
        Runs one forward pass and returns (results_dict, elapsed_ms).
        Raises on any error so the caller can display it cleanly.
        """
        from PIL import Image as PILImage
        from Infra.trainer import PyTorchTrainer

        transform = self._build_transform(project, normalize=normalize)
        img = PILImage.open(self._image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        t0 = time.perf_counter()
        results = PyTorchTrainer.predict(project, tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return results, elapsed_ms

    def _fill_column(self, col: dict, results: dict,
                     elapsed_ms: float, class_names: list):
        """Writes one set of inference results into a result column."""
        idx        = results["classes"][0]
        probs      = results["probabilities"][0]
        confidence = results["confidence"][0]

        name = class_names[idx] if idx < len(class_names) else str(idx)
        col["lbl_pred"].setText(f"Class {idx} — {name}   {confidence * 100:.1f}%")
        col["stat_class"].label_valeur.setText(str(idx))
        col["stat_conf"].label_valeur.setText(f"{confidence * 100:.1f}%")
        col["stat_time"].label_valeur.setText(f"{elapsed_ms:.0f}ms")

        top5 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
        for i, (name_lbl, bar, pct_lbl) in enumerate(col["prob_bars"]):
            if i < len(top5):
                ci, prob = top5[i]
                pct = prob * 100
                name_lbl.setText(
                    class_names[ci] if ci < len(class_names) else str(ci)
                )
                bar.setValue(int(pct))
                pct_lbl.setText(f"{pct:.1f}%")
            else:
                name_lbl.setText("--")
                bar.setValue(0)
                pct_lbl.setText("0%")

    # ── core: real inference ──────────────────────────────────────────────────

    def run_inference(self):
        # ── guards
        if not self._image_path:
            self._norm_col["lbl_pred"].setText("[!] Please select an image first.")
            return

        project = self._project_vm.project if self._project_vm else None
        if project is None:
            self._norm_col["lbl_pred"].setText("[!] No project loaded.")
            return
        if project.network is None:
            self._norm_col["lbl_pred"].setText("[!] Project has no network.")
            return

        if not project.weightsPath:
            import os
            from Core.project import Project as _Proj
            candidate = os.path.join(project.path, _Proj.WEIGHTS_FILENAME)
            if os.path.isfile(candidate):
                project.setWeightsPath(candidate)
            else:
                self._norm_col["lbl_pred"].setText(
                    "[!] Model not trained yet — run training first."
                )
                return

        try:
            from torchvision import transforms   # noqa — just verify import
            from PIL import Image                # noqa
        except ImportError as e:
            self._norm_col["lbl_pred"].setText(f"[!] Missing dependency: {e}")
            return

        # ── resolve class names once
        dataset_key = (
            project.dataset._builtinType.value
            if project.dataset and project.dataset._builtinType
            else None
        )
        num_classes = project.dataset._numClasses if project.dataset else 10
        class_names = list(_CLASS_NAMES.get(dataset_key,
                                             [str(i) for i in range(num_classes)]))
        while len(class_names) < num_classes:
            class_names.append(str(len(class_names)))

        # ── normalized pass (always runs)
        try:
            norm_results, norm_ms = self._run_single_pass(project, normalize=True)
        except Exception as e:
            self._norm_col["lbl_pred"].setText(f"[!] Inference error: {e}")
            return

        self._fill_column(self._norm_col, norm_results, norm_ms, class_names)

        # ── raw pass (only in experiment mode)
        if self.chk_experiment.isChecked():
            try:
                raw_results, raw_ms = self._run_single_pass(project, normalize=False)
                self._fill_column(self._raw_col, raw_results, raw_ms, class_names)
            except Exception as e:
                self._raw_col["lbl_pred"].setText(f"[!] Raw inference error: {e}")

        # ── history (normalized result only)
        predicted_idx  = norm_results["classes"][0]
        confidence     = norm_results["confidence"][0]
        predicted_name = class_names[predicted_idx] if predicted_idx < len(class_names) \
                         else str(predicted_idx)
        self._history.append(f"{predicted_name} — {confidence * 100:.1f}%")
        self.lbl_history.setText("\n".join(self._history[-5:]))

        # keep norm stats label fresh
        self._refresh_norm_stats_label()