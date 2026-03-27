from PyQt6.QtWidgets import (QDockWidget, QWidget, QFormLayout,QLabel, QLineEdit, QCheckBox)

class LayerConfigDock(QDockWidget):
    def __init__(self):
        super().__init__("Layer Config")

        widget = QWidget()
        layout = QFormLayout()

        # Layer Name
        self.layer_name = QLineEdit()
        self.layer_name.setPlaceholderText("ex: conv1")
        layout.addRow(QLabel("Layer Name :"), self.layer_name)

        # Dropout
        self.dropout = QLineEdit()
        self.dropout.setPlaceholderText("ex: 0.5")
        layout.addRow(QLabel("Dropout neurons :"), self.dropout)

        # Input Size
        self.input_size = QLineEdit()
        self.input_size.setPlaceholderText("ex: 128")
        layout.addRow(QLabel("Input Size :"), self.input_size)

        # Use Bias
        self.use_bias = QCheckBox()
        self.use_bias.setChecked(True)
        layout.addRow(QLabel("Use Bias :"), self.use_bias)

        widget.setLayout(layout)
        self.setWidget(widget)