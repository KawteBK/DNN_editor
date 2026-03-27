from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QGridLayout)
from PyQt6.QtCore import Qt
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

        # Courbes
        charts_layout = QHBoxLayout()

        # Courbe Loss
        self.loss_fig = Figure(figsize=(4, 2.5))
        self.loss_canvas = FigureCanvasQTAgg(self.loss_fig)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_title("Loss Curve (Train + Validation)")
        charts_layout.addWidget(self.loss_canvas)

        # Courbe Accuracy
        self.acc_fig = Figure(figsize=(4, 2.5))
        self.acc_canvas = FigureCanvasQTAgg(self.acc_fig)
        self.acc_ax = self.acc_fig.add_subplot(111)
        self.acc_ax.set_title("Metric Curve (train + val)")
        charts_layout.addWidget(self.acc_canvas)

        layout.addLayout(charts_layout)

        # Résultats finaux
        results_layout = QGridLayout()

        self.test_loss = QLabel("Test Loss : --")
        self.test_acc = QLabel("Test Accuracy : --")
        self.total_params = QLabel("Total Params : --")
        self.trainable_params = QLabel("Trainable : --")

        font = QFont()
        font.setBold(True)
        self.test_loss.setFont(font)
        self.test_acc.setFont(font)

        results_layout.addWidget(QLabel("Final Test Result"), 0, 0)
        results_layout.addWidget(self.test_loss, 1, 0)
        results_layout.addWidget(self.test_acc, 2, 0)
        results_layout.addWidget(self.total_params, 1, 1)
        results_layout.addWidget(self.trainable_params, 2, 1)

        layout.addLayout(results_layout)
        widget.setLayout(layout)
        self.setWidget(widget)
    