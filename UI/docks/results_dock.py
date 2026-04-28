from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSizePolicy, QTabWidget,
)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np


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

_BG      = "#2b2b2b"
_SURFACE = "#3c3f41"
_BORDER  = "#555555"
_TEXT    = "#cccccc"
_DIM     = "#888888"
_TEAL    = "#1592A2"
_ORANGE  = "#E67E22"
_GREEN   = "#2ECC71"




def _make_stat_card(label: str, value: str = "—", value_color: str = _TEXT):
    """Small surface card with a dim label + big bold value. Returns (widget, value_label)."""
    card = QWidget()
    card.setStyleSheet(f"background:{_SURFACE}; border-radius:6px;")
    vl = QVBoxLayout(card)
    vl.setContentsMargins(10, 8, 10, 8)
    vl.setSpacing(2)
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color:{_DIM}; font-size:11px; background:transparent;")
    val = QLabel(value)
    val.setStyleSheet(
        f"color:{value_color}; font-size:17px; font-weight:bold; background:transparent;"
    )
    vl.addWidget(lbl)
    vl.addWidget(val)
    return card, val


def _styled_fig(figsize=(5, 2.6)):
    fig = Figure(figsize=figsize, facecolor=_BG)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setStyleSheet(f"background:{_BG};")
    return fig, canvas


def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(_BG)
    ax.set_title(title, color=_TEXT, fontsize=9, pad=4)
    ax.tick_params(colors=_DIM, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)


_TAB_STYLE = f"""
    QTabWidget::pane {{ border:none; background:{_BG}; }}
    QTabBar::tab {{
        background:{_SURFACE}; color:{_DIM};
        padding:5px 14px; border:none; font-size:12px;
    }}
    QTabBar::tab:selected {{
        background:{_BG}; color:{_TEXT};
        border-bottom:2px solid {_TEAL};
    }}
    QTabBar::tab:hover {{ color:{_TEXT}; }}
"""


class ResultsDock(QDockWidget):
    """
    Training monitoring panel — two tabs:

      Monitor    — live epoch stats + loss/accuracy charts (updates each epoch)
      Evaluation — held-out test-set scalar metrics + confusion matrix (set once after training)

    The dock auto-switches tabs:
      • Training starts  → jump to Monitor
      • Training ends    → jump to Evaluation
    """

    def __init__(self):
        super().__init__("Training Monitor")

        self._dataset_key: str | None = None
        self._train_losses: list[float] = []
        self._val_losses:   list[float] = []
        self._val_accs:     list[float] = []

        outer = QWidget()
        outer.setStyleSheet(f"background:{_BG}; color:{_TEXT};")
        root = QVBoxLayout(outer)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setWidget(outer)

        self._tabs = QTabWidget()
      
        self._tabs.setStyleSheet(_TAB_STYLE)
        root.addWidget(self._tabs)
        self._tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self._tabs.addTab(self._build_monitor_tab(),    "Monitor")
        self._tabs.addTab(self._build_evaluation_tab(), "Evaluation")

        self._tabs.setStyleSheet("""
    QTabBar::tab {
        font-size: 13px;
        font-weight: bold;
        padding: 8px 16px;
        min-width: 80px;
    }
    QTabBar::tab:selected {
        background-color: #3c3f41;
    }
""")

#         ("""
# QTabBar::tab {
#     font-weight: bold;
# }
# """)

    # ── tab builders ──────────────────────────────────────────────────────────

    def _build_monitor_tab(self) -> QWidget:
        tab = QWidget()
        tab.setStyleSheet(f"background:{_BG};")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # stat cards
        row = QHBoxLayout()
        row.setSpacing(6)
        self._card_epoch,      self._val_epoch      = _make_stat_card("Epoch",      "— / —")
        self._card_train_loss, self._val_train_loss  = _make_stat_card("Train loss", "—")
        self._card_val_loss,   self._val_val_loss    = _make_stat_card("Val loss",   "—")
        self._card_val_acc,    self._val_val_acc     = _make_stat_card("Val acc",    "—", _GREEN)
        for c in [self._card_epoch, self._card_train_loss,
                  self._card_val_loss, self._card_val_acc]:
            row.addWidget(c)
        layout.addLayout(row)

        # charts
        charts = QHBoxLayout()
        charts.setSpacing(6)

        self.loss_fig, self.loss_canvas = _styled_fig((4, 2.4))
        self.loss_ax = self.loss_fig.add_subplot(111)
        _style_ax(self.loss_ax, "Loss")
        self.loss_fig.tight_layout(pad=0.8)
        self.loss_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.acc_fig, self.acc_canvas = _styled_fig((4, 2.4))
        self.acc_ax = self.acc_fig.add_subplot(111)
        _style_ax(self.acc_ax, "Accuracy (%)")
        self.acc_fig.tight_layout(pad=0.8)
        self.acc_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        charts.addWidget(self.loss_canvas)
        charts.addWidget(self.acc_canvas)
        layout.addLayout(charts)

        self._monitor_hint = QLabel("Charts will update live once training starts.")
        self._monitor_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._monitor_hint.setStyleSheet(f"color:{_DIM}; font-size:11px;")
        layout.addWidget(self._monitor_hint)
        layout.addStretch()
        return tab

    def _build_evaluation_tab(self) -> QWidget:
        """
        Layout:
          title
          ── 3-column row ──────────────────────────────────────────────────
          col 0 (metrics_col): Test loss / Test accuracy / F1  — vertically stacked cards
          col 1 (pr_col):      Precision / Recall              — vertically stacked cards
          col 2 (cm_col):      Confusion matrix canvas         — expands to fill height
          ──────────────────────────────────────────────────────────────────
          hint label (hidden once data arrives)
        """
        tab = QWidget()
        tab.setStyleSheet(f"background:{_BG};")
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(8, 10, 8, 8)
        outer.setSpacing(6)

        title = QLabel("Final evaluation — held-out test set")
        title.setStyleSheet(
            f"color:{_DIM}; font-size:11px; font-weight:bold; letter-spacing:1px;"
        )
        outer.addWidget(title)

        # ── 3-column body ─────────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(8)

        # col 0 — Test loss / accuracy / F1  (vertically stacked)
        metrics_col = QVBoxLayout()
        metrics_col.setSpacing(6)
        self._eval_card_loss, self._eval_val_loss = _make_stat_card("Test loss",     "—", _ORANGE)
        self._eval_card_acc,  self._eval_val_acc  = _make_stat_card("Test accuracy", "—", _GREEN)
        self._eval_card_f1,   self._eval_val_f1   = _make_stat_card("F1 score",      "—", _TEAL)
        metrics_col.addWidget(self._eval_card_loss)
        metrics_col.addWidget(self._eval_card_acc)
        metrics_col.addWidget(self._eval_card_f1)
        metrics_col.addStretch()

        # col 1 — Precision / Recall  (vertically stacked)
        pr_col = QVBoxLayout()
        pr_col.setSpacing(6)
        self._eval_card_prec, self._eval_val_prec = _make_stat_card("Precision", "—")
        self._eval_card_rec,  self._eval_val_rec  = _make_stat_card("Recall",    "—")
        pr_col.addWidget(self._eval_card_prec)
        pr_col.addWidget(self._eval_card_rec)
        pr_col.addStretch()

        # col 2 — Confusion matrix
        cm_col = QVBoxLayout()
        cm_col.setSpacing(4)
        cm_title = QLabel("Confusion matrix (row-normalised)")
        cm_title.setStyleSheet(f"color:{_DIM}; font-size:10px; font-weight:bold;")
        cm_col.addWidget(cm_title)

        self.cm_fig, self.cm_canvas = _styled_fig((4.5, 3.5))
        self.cm_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cm_canvas.setMinimumHeight(160)
        cm_col.addWidget(self.cm_canvas)

        self._cm_hint = QLabel("Confusion matrix appears here after training.")
        self._cm_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cm_hint.setWordWrap(True)
        self._cm_hint.setStyleSheet(
            f"color:{_DIM}; font-size:11px; padding:12px;"
            f"background:{_SURFACE}; border-radius:6px;"
        )
        cm_col.addWidget(self._cm_hint)

        # assemble columns — give the CM column more stretch
        body.addLayout(metrics_col, 1)
        body.addLayout(pr_col, 1)
        body.addLayout(cm_col, 3)
        outer.addLayout(body, 1)

        # hint shown before first training run
        self._eval_hint = QLabel(
            "Train the model to see test-set evaluation.\n"
            "These metrics are computed once on data the model never saw during training."
        )
        self._eval_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._eval_hint.setWordWrap(True)
        self._eval_hint.setStyleSheet(
            f"color:{_DIM}; font-size:11px; padding:16px;"
            f"background:{_SURFACE}; border-radius:6px;"
        )
        outer.addWidget(self._eval_hint)
        return tab

    # ── public API ────────────────────────────────────────────────────────────

    def set_dataset_key(self, key: str | None) -> None:
        self._dataset_key = key

    def on_epoch(self, epoch: int, total: int, train_loss: float, val_metrics: dict) -> None:
        if epoch == 1:
            self._tabs.setCurrentIndex(0)
            self._monitor_hint.setVisible(False)

        self._train_losses.append(train_loss)
        val_loss = val_metrics.get("loss")
        val_acc  = val_metrics.get("accuracy")

        if val_loss is not None:
            self._val_losses.append(val_loss)
        if val_acc is not None:
            self._val_accs.append(val_acc)

        # stat cards
        self._val_epoch.setText(f"{epoch} / {total}")
        self._val_train_loss.setText(f"{train_loss:.4f}")
        if val_loss is not None:
            self._val_val_loss.setText(f"{val_loss:.4f}")
        if val_acc is not None:
            self._val_val_acc.setText(f"{val_acc:.1f}%")

        # loss chart
        self.loss_ax.clear()
        _style_ax(self.loss_ax, "Loss")
        self.loss_ax.plot(self._train_losses, label="train", color=_TEAL, linewidth=1.5)
        if self._val_losses:
            self.loss_ax.plot(self._val_losses, label="val", color=_ORANGE, linewidth=1.5)
        self.loss_ax.legend(
            fontsize=7, facecolor=_SURFACE, edgecolor=_BORDER, labelcolor=_TEXT)
        self.loss_fig.tight_layout(pad=0.8)
        self.loss_canvas.draw()

        # accuracy chart
        self.acc_ax.clear()
        _style_ax(self.acc_ax, "Accuracy (%)")
        if self._val_accs:
            self.acc_ax.plot(self._val_accs, label="val", color=_GREEN, linewidth=1.5)
            self.acc_ax.legend(
                fontsize=7, facecolor=_SURFACE, edgecolor=_BORDER, labelcolor=_TEXT)
        self.acc_fig.tight_layout(pad=0.8)
        self.acc_canvas.draw()

    def on_training_finished(self, history: dict) -> None:
        test_loss = history.get("test_loss")
        test_acc  = history.get("test_accuracy")
        test_f1   = history.get("test_f1_score")
        test_prec = history.get("test_precision")
        test_rec  = history.get("test_recall")

        if test_loss is not None:
            self._eval_hint.setVisible(False)
            self._eval_val_loss.setText(f"{test_loss:.4f}")
        if test_acc is not None:
            self._eval_val_acc.setText(f"{test_acc:.1f}%")
        if test_f1 is not None:
            self._eval_val_f1.setText(f"{test_f1:.3f}")
        else:
            self._eval_val_f1.setText("— n/a")
        if test_prec is not None:
            self._eval_val_prec.setText(f"{test_prec:.3f}")
        if test_rec is not None:
            self._eval_val_rec.setText(f"{test_rec:.3f}")

        cm = history.get("confusion_matrix")
        if cm:
            self._draw_confusion_matrix(cm)

        if test_loss is not None:
            self._tabs.setCurrentIndex(1)   # jump to Evaluation tab

    def clear(self) -> None:
        self._train_losses.clear()
        self._val_losses.clear()
        self._val_accs.clear()

        for ax, fig, canvas, title in [
            (self.loss_ax, self.loss_fig, self.loss_canvas, "Loss"),
            (self.acc_ax,  self.acc_fig,  self.acc_canvas,  "Accuracy (%)"),
        ]:
            ax.clear()
            _style_ax(ax, title)
            fig.tight_layout(pad=0.8)
            canvas.draw()

        self._val_epoch.setText("— / —")
        self._val_train_loss.setText("—")
        self._val_val_loss.setText("—")
        self._val_val_acc.setText("—")
        self._monitor_hint.setVisible(True)

        self._eval_val_loss.setText("—")
        self._eval_val_acc.setText("—")
        self._eval_val_f1.setText("—")
        self._eval_val_prec.setText("—")
        self._eval_val_rec.setText("—")
        self._eval_hint.setVisible(True)

        self.cm_fig.clear()
        self.cm_canvas.draw()
        self._cm_hint.setVisible(True)

        self._tabs.setCurrentIndex(0)

    # ── private ───────────────────────────────────────────────────────────────

    def _draw_confusion_matrix(self, cm: list) -> None:
        self._cm_hint.setVisible(False)
        cm_array = np.array(cm, dtype=float)
        row_sums = cm_array.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_array / row_sums

        n_classes   = cm_array.shape[0]
        names       = _CLASS_NAMES.get(self._dataset_key, [str(i) for i in range(n_classes)])
        names       = (names + [str(i) for i in range(n_classes)])[:n_classes]

        self.cm_fig.clear()
        self.cm_fig.patch.set_facecolor(_BG)
        ax = self.cm_fig.add_subplot(111)
        ax.set_facecolor(_BG)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        cb = self.cm_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=_DIM, labelsize=7)

        if n_classes <= 20:
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7, color=_TEXT)
            ax.set_yticklabels(names, fontsize=7, color=_TEXT)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel("Predicted", fontsize=8, color=_DIM)
        ax.set_ylabel("True",      fontsize=8, color=_DIM)
        ax.set_title("Row-normalised  (diagonal = correct)", fontsize=8, color=_TEXT, pad=6)
        ax.tick_params(colors=_DIM)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)

        if n_classes <= 15:
            thresh = cm_norm.max() / 2.0
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(
                        j, i, int(cm_array[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > thresh else "black",
                    )

        self.cm_fig.tight_layout(pad=0.8)
        self.cm_canvas.draw()