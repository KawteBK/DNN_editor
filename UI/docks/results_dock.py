from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
)
from PyQt6.QtGui import QFont
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ResultsDock(QDockWidget):
    def __init__(self):
        super().__init__("Results")

        widget = QWidget()
        layout = QVBoxLayout()

        charts_layout = QHBoxLayout()

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

        results_layout = QGridLayout()
        self.test_loss        = QLabel("Val Loss: --")
        self.test_acc         = QLabel("Val Accuracy: --")
        self.epoch_label      = QLabel("Epoch: --")   # ADDED: live epoch counter
        self.total_params     = QLabel("Total Params: --")
        self.trainable_params = QLabel("Trainable: --")

        font = QFont()
        font.setBold(True)
        self.test_loss.setFont(font)
        self.test_acc.setFont(font)

        results_layout.addWidget(QLabel("Training Results"), 0, 0)
        results_layout.addWidget(self.epoch_label,      0, 1)   # ADDED
        results_layout.addWidget(self.test_loss,        1, 0)
        results_layout.addWidget(self.test_acc,         2, 0)
        results_layout.addWidget(self.total_params,     1, 1)
        results_layout.addWidget(self.trainable_params, 2, 1)

        layout.addLayout(results_layout)
        widget.setLayout(layout)
        self.setWidget(widget)

        # ADDED: internal history lists so the charts can draw a full curve
        # (not just the last point) as epochs accumulate
        self._train_losses: list[float] = []
        self._val_losses:   list[float] = []
        self._val_accs:     list[float] = []

    # ADDED: called by MainWindow when TrainingVM emits epoch_progress.
    # Updates the live charts and the stat labels after every epoch.
    def on_epoch(self, epoch: int, total: int, train_loss: float, val_metrics: dict) -> None:
        self._train_losses.append(train_loss)
        val_loss = val_metrics.get("loss", None)
        val_acc  = val_metrics.get("accuracy", None)

        if val_loss is not None:
            self._val_losses.append(val_loss)
        if val_acc is not None:
            self._val_accs.append(val_acc)

        # Update epoch counter label
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

        # Update stat labels
        if val_loss is not None:
            self.test_loss.setText(f"Val Loss: {val_loss:.4f}")
        if val_acc is not None:
            self.test_acc.setText(f"Val Accuracy: {val_acc:.1f}%")

    # ADDED: called by MainWindow._on_project_changed() when a new project loads.
    # Wipes the charts and labels so stale results from a previous session
    # don't sit there confusingly while a new project is being built.
    def clear(self) -> None:
        self._train_losses.clear()
        self._val_losses.clear()
        self._val_accs.clear()

        for ax, canvas, title in [
            (self.loss_ax, self.loss_canvas, "Loss"),
            (self.acc_ax,  self.acc_canvas,  "Accuracy"),
        ]:
            ax.clear()
            ax.set_title(title)
            canvas.draw()

        self.test_loss.setText("Val Loss: --")
        self.test_acc.setText("Val Accuracy: --")
        self.epoch_label.setText("Epoch: --")