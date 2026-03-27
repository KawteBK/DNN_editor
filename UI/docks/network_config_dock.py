from PyQt6.QtWidgets import (QDockWidget, QWidget, QFormLayout,
                              QLabel, QLineEdit, QComboBox)

class NetworkConfigDock(QDockWidget):
    def __init__(self):
        super().__init__("Network Config")

        widget = QWidget()
        layout = QFormLayout()

        # Epochs
        self.epochs = QLineEdit()
        self.epochs.setPlaceholderText("ex: 100")
        layout.addRow(QLabel("Epochs :"), self.epochs)

        # Batch Size
        self.batch_size = QLineEdit()
        self.batch_size.setPlaceholderText("ex: 32")
        layout.addRow(QLabel("Batch Size :"), self.batch_size)

        # Learning Rate
        self.lr = QLineEdit()
        self.lr.setPlaceholderText("ex: 0.001")
        layout.addRow(QLabel("Learning Rate :"), self.lr)

        # Optimizer
        self.optimizer = QComboBox()
        self.optimizer.addItems(["Adam", "SGD", "RMSprop"])
        layout.addRow(QLabel("Optimizer :"), self.optimizer)

        # Loss Function
        self.loss_fn = QComboBox()
        self.loss_fn.addItems(["CrossEntropy", "MSE", "BCE"])
        layout.addRow(QLabel("Loss Function :"), self.loss_fn)

        # Validation Split
        self.val_split = QLineEdit()
        self.val_split.setPlaceholderText("ex: 0.2")
        layout.addRow(QLabel("Validation split :"), self.val_split)

        # Device
        self.device = QComboBox()
        self.device.addItems(["CPU", "GPU (CUDA)"])
        layout.addRow(QLabel("Device :"), self.device)

        widget.setLayout(layout)
        self.setWidget(widget)