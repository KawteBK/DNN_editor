from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGridLayout, QFrame, QSizePolicy,
)
from PyQt6.QtGui import QFont
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np


# Class name tables — used to label confusion matrix axes.
# Keyed by BuiltinDataset.name (the enum's .name attribute, e.g. "FASHION_MNIST").
_CLASS_NAMES = {
    "MNIST":         [str(i) for i in range(10)],
    "FASHION_MNIST": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
    "KMNIST":        ["O", "Ki", "Su", "Tsu", "Na", "Ha", "Ma", "Ya", "Re", "Wo"],
    "CIFAR10":       ["Plane", "Car", "Bird", "Cat", "Deer",
                      "Dog", "Frog", "Horse", "Ship", "Truck"],
    "CIFAR100":      [f"C{i}" for i in range(100)],
    "SVHN":          [str(i) for i in range(10)],
    "STL10":         ["Plane", "Bird", "Car", "Cat", "Deer",
                      "Dog", "Horse", "Monkey", "Ship", "Truck"],
}


def _separator() -> QFrame:
    """Thin horizontal rule — visual boundary between dock sections."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #444;")
    return line


class ResultsDock(QDockWidget):
    def __init__(self):
        super().__init__("Results")

        # ── root scroll so the dock can hold all three sections ───────────────
        # CHANGED: wrapped everything in a QVBoxLayout on a plain QWidget.
        # Previously there was no scroll and the dock would clip content when
        # the confusion matrix section was added.
        root_widget = QWidget()
        layout = QVBoxLayout(root_widget)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)
        self.setWidget(root_widget)

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 1 — Live training monitor (unchanged, updates every epoch)
        # ═══════════════════════════════════════════════════════════════════════
        monitor_label = QLabel("Training monitor")
        monitor_label.setFont(self._bold_font(10))
        layout.addWidget(monitor_label)

        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(4)

        self.loss_fig    = Figure(figsize=(4, 2.5))
        self.loss_canvas = FigureCanvasQTAgg(self.loss_fig)
        self.loss_ax     = self.loss_fig.add_subplot(111)
        self.loss_ax.set_title("Loss")
        charts_layout.addWidget(self.loss_canvas)

        self.acc_fig    = Figure(figsize=(4, 2.5))
        self.acc_canvas = FigureCanvasQTAgg(self.acc_fig)
        self.acc_ax     = self.acc_fig.add_subplot(111)
        self.acc_ax.set_title("Accuracy")
        charts_layout.addWidget(self.acc_canvas)

        layout.addLayout(charts_layout)

        # Live stat row (epoch counter + last val metrics)
        live_row = QGridLayout()
        self.epoch_label   = QLabel("Epoch: --")
        self.live_val_loss = QLabel("Val Loss: --")
        self.live_val_acc  = QLabel("Val Acc: --")
        self.live_val_loss.setFont(self._bold_font())
        self.live_val_acc.setFont(self._bold_font())
        live_row.addWidget(self.epoch_label,   0, 0)
        live_row.addWidget(self.live_val_loss, 0, 1)
        live_row.addWidget(self.live_val_acc,  0, 2)
        layout.addLayout(live_row)

        layout.addWidget(_separator())

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 2 — NEW: Final evaluation (populated ONCE after training ends)
        # These labels show test-set metrics — the true generalisation estimate.
        # They are intentionally separate from the live charts above.
        # ═══════════════════════════════════════════════════════════════════════
        eval_label = QLabel("Final evaluation  (held-out test set)")
        eval_label.setFont(self._bold_font(10))
        layout.addWidget(eval_label)

        eval_grid = QGridLayout()
        eval_grid.setHorizontalSpacing(16)

        self.test_loss_label = QLabel("Test loss:")
        self.test_loss_val   = QLabel("—")
        self.test_acc_label  = QLabel("Test accuracy:")
        self.test_acc_val    = QLabel("—")
        self.test_f1_label   = QLabel("Test F1:")
        self.test_f1_val     = QLabel("—")

        # Style the value labels so they stand out
        for lbl in [self.test_loss_val, self.test_acc_val, self.test_f1_val]:
            lbl.setFont(self._bold_font())
            lbl.setStyleSheet("color: #2ECC71;")

        eval_grid.addWidget(self.test_loss_label, 0, 0)
        eval_grid.addWidget(self.test_loss_val,   0, 1)
        eval_grid.addWidget(self.test_acc_label,  1, 0)
        eval_grid.addWidget(self.test_acc_val,    1, 1)
        eval_grid.addWidget(self.test_f1_label,   2, 0)
        eval_grid.addWidget(self.test_f1_val,     2, 1)
        layout.addLayout(eval_grid)

        # Hint shown before training is ever run
        self._eval_hint = QLabel("Run training to see test-set evaluation.")
        self._eval_hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._eval_hint)

        layout.addWidget(_separator())

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 3 — NEW: Confusion matrix (matplotlib imshow)
        # Shown after training completes. Stays blank until then.
        # ═══════════════════════════════════════════════════════════════════════
        cm_label = QLabel("Confusion matrix  (test set)")
        cm_label.setFont(self._bold_font(10))
        layout.addWidget(cm_label)

        # Square figure — size grows with number of classes but we cap the display
        self.cm_fig    = Figure(figsize=(4.5, 4.5))
        self.cm_canvas = FigureCanvasQTAgg(self.cm_fig)
        self.cm_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.cm_canvas.setMinimumHeight(200)
        layout.addWidget(self.cm_canvas)

        self._cm_hint = QLabel("Confusion matrix appears here after training.")
        self._cm_hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._cm_hint)

        layout.addStretch()

        # ── internal history lists ────────────────────────────────────────────
        self._train_losses: list[float] = []
        self._val_losses:   list[float] = []
        self._val_accs:     list[float] = []

        # Remember which dataset is active so we can label the confusion matrix
        self._dataset_key: str | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def set_dataset_key(self, key: str | None) -> None:
        """
        NEW — called by MainWindow._on_project_changed() so the confusion
        matrix can label its axes with real class names instead of integers.
        key is the BuiltinDataset enum's .name, e.g. "FASHION_MNIST".
        """
        self._dataset_key = key

    def on_epoch(self, epoch: int, total: int, train_loss: float, val_metrics: dict) -> None:
        """
        Called by MainWindow every epoch via TrainingVM.epoch_progress signal.
        Updates the live monitor section only — does NOT touch the evaluation section.
        Unchanged in behaviour from original, labels renamed for clarity.
        """
        self._train_losses.append(train_loss)
        val_loss = val_metrics.get("loss", None)
        val_acc  = val_metrics.get("accuracy", None)

        if val_loss is not None:
            self._val_losses.append(val_loss)
        if val_acc is not None:
            self._val_accs.append(val_acc)

        self.epoch_label.setText(f"Epoch: {epoch}/{total}")

        # Redraw loss chart
        self.loss_ax.clear()
        self.loss_ax.set_title("Loss")
        self.loss_ax.plot(self._train_losses, label="train", color="#1592A2")
        if self._val_losses:
            self.loss_ax.plot(self._val_losses, label="val", color="#E67E22")
        self.loss_ax.legend(fontsize=7)
        self.loss_ax.tick_params(labelsize=7)
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()

        # Redraw accuracy chart
        self.acc_ax.clear()
        self.acc_ax.set_title("Accuracy")
        if self._val_accs:
            self.acc_ax.plot(self._val_accs, label="val acc", color="#2ECC71")
            self.acc_ax.legend(fontsize=7)
        self.acc_ax.tick_params(labelsize=7)
        self.acc_fig.tight_layout()
        self.acc_canvas.draw()

        if val_loss is not None:
            self.live_val_loss.setText(f"Val Loss: {val_loss:.4f}")
        if val_acc is not None:
            self.live_val_acc.setText(f"Val Acc: {val_acc:.1f}%")

    def on_training_finished(self, history: dict) -> None:
        """
        NEW — called by MainWindow._on_training_finished() after training ends.
        Reads the "test_*" keys and "confusion_matrix" from the history dict
        written by trainer.py and populates the evaluation and confusion matrix
        sections.

        WHY a separate method instead of folding this into on_epoch:
        The test pass runs exactly once, after all epochs complete. on_epoch
        is called once per epoch during training. Mixing them would require
        special-casing the last epoch, which is fragile — especially with early
        stopping where "last epoch" isn't known in advance.
        """
        # ── Section 2: scalar test metrics ────────────────────────────────────
        test_loss = history.get("test_loss")
        test_acc  = history.get("test_accuracy")
        test_f1   = history.get("test_f1_score")

        if test_loss is not None:
            self._eval_hint.setVisible(False)
            self.test_loss_val.setText(f"{test_loss:.4f}")
        if test_acc is not None:
            self.test_acc_val.setText(f"{test_acc:.1f}%")
        if test_f1 is not None:
            self.test_f1_val.setText(f"{test_f1:.3f}")
        else:
            self.test_f1_val.setText("—  (not in metrics)")

        # ── Section 3: confusion matrix ───────────────────────────────────────
        cm = history.get("confusion_matrix")
        if cm:
            self._draw_confusion_matrix(cm)

    def clear(self) -> None:
        """
        Called by MainWindow._on_project_changed() when a new project loads.
        Resets all three sections to their blank state.
        CHANGED: also clears the evaluation and confusion matrix sections.
        """
        self._train_losses.clear()
        self._val_losses.clear()
        self._val_accs.clear()

        # Clear live monitor charts
        for ax, canvas, title in [
            (self.loss_ax, self.loss_canvas, "Loss"),
            (self.acc_ax,  self.acc_canvas,  "Accuracy"),
        ]:
            ax.clear()
            ax.set_title(title)
            canvas.draw()

        self.epoch_label.setText("Epoch: --")
        self.live_val_loss.setText("Val Loss: --")
        self.live_val_acc.setText("Val Acc: --")

        # Reset final evaluation labels
        self.test_loss_val.setText("—")
        self.test_acc_val.setText("—")
        self.test_f1_val.setText("—")
        self._eval_hint.setVisible(True)

        # Clear confusion matrix
        self.cm_fig.clear()
        self.cm_canvas.draw()
        self._cm_hint.setVisible(True)

    # ── private ───────────────────────────────────────────────────────────────

    def _draw_confusion_matrix(self, cm: list) -> None:
        """
        NEW — renders the confusion matrix using matplotlib imshow.

        cm is a nested Python list (C×C) as produced by
        sklearn.metrics.confusion_matrix and stored in history["confusion_matrix"].

        WHY imshow instead of a table widget:
          - Color intensity immediately shows where the errors concentrate.
          - Scales to any number of classes without layout changes.
          - Consistent with the matplotlib stack already used for the charts.

        Normalises each row to [0, 1] so that per-class accuracy differences
        don't drown out small classes — a class with 100 samples and 90 correct
        shows the same shade as a class with 10,000 samples and 9,000 correct.
        """
        self._cm_hint.setVisible(False)
        cm_array = np.array(cm, dtype=float)

        # Row-normalise: divide each row by its sum so values are 0..1
        row_sums = cm_array.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # avoid divide-by-zero for empty classes
        cm_norm = cm_array / row_sums

        n_classes = cm_array.shape[0]
        class_names = _CLASS_NAMES.get(self._dataset_key, [str(i) for i in range(n_classes)])
        # Trim or pad label list to match actual matrix size
        class_names = class_names[:n_classes]
        while len(class_names) < n_classes:
            class_names.append(str(len(class_names)))

        self.cm_fig.clear()
        ax = self.cm_fig.add_subplot(111)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        self.cm_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Axis labels — skip individual tick labels for large matrices (>20 classes)
        if n_classes <= 20:
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title("Confusion matrix (row-normalised)", fontsize=9)

        # Annotate each cell with its raw count — only for small matrices
        if n_classes <= 15:
            thresh = cm_norm.max() / 2.0
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(
                        j, i, int(cm_array[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > thresh else "black",
                    )

        self.cm_fig.tight_layout()
        self.cm_canvas.draw()

    @staticmethod
    def _bold_font(size: int = 9) -> QFont:
        f = QFont()
        f.setBold(True)
        f.setPointSize(size)
        return f
