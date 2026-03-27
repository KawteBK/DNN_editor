from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QComboBox, QPushButton)
from PyQt6.QtCore import Qt

class DatasetDock(QDockWidget):
    def __init__(self):
        super().__init__("Dataset")
        
        widget = QWidget()
        layout = QVBoxLayout()

        # Sélection dataset
        dataset_layout = QHBoxLayout()
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["MNIST", "CIFAR-10", "CIFAR-100", "ImageNet"])
        dataset_layout.addWidget(self.dataset_combo)
        layout.addLayout(dataset_layout)

        # Infos
        self.info_name = QLabel("Name : MNIST")
        self.info_type = QLabel("Type : Image")
        self.info_classes = QLabel("Number of classes : 10")
        self.info_input = QLabel("Input Shape : (1, 28, 28)")
        self.info_train = QLabel("Training Samples : 60,000")
        self.info_test = QLabel("Test Samples : 10,000")
        self.info_task = QLabel("Task type : Classification")

        for label in [self.info_name, self.info_type, self.info_classes,
                      self.info_input, self.info_train, self.info_test, self.info_task]:
            layout.addWidget(label)

        # Bouton Preview
        self.preview_btn = QPushButton("Preview")
        layout.addWidget(self.preview_btn)

        layout.addStretch()
        widget.setLayout(layout)
        self.setWidget(widget)

        # MAJ les info quand on change de dataset
        self.dataset_combo.currentTextChanged.connect(self.update_info)

    def update_info(self, dataset):
        infos = {
            "MNIST":    ("MNIST", "Image", "10", "(1, 28, 28)", "60,000", "10,000", "Classification"),
            "CIFAR-10": ("CIFAR-10", "Image", "10", "(3, 32, 32)", "50,000", "10,000", "Classification"),
            "CIFAR-100":("CIFAR-100", "Image", "100", "(3, 32, 32)", "50,000", "10,000", "Classification"),
            "ImageNet": ("ImageNet", "Image", "1000", "(3, 224, 224)", "1,200,000", "50,000", "Classification"),
        }
        i = infos[dataset]
        self.info_name.setText(f"Name : {i[0]}")
        self.info_type.setText(f"Type : {i[1]}")
        self.info_classes.setText(f"Number of classes : {i[2]}")
        self.info_input.setText(f"Input Shape : {i[3]}")
        self.info_train.setText(f"Training Samples : {i[4]}")
        self.info_test.setText(f"Test Samples : {i[5]}")
        self.info_task.setText(f"Task type : {i[6]}")