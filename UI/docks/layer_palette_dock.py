from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt

class LayerPaletteDock(QDockWidget):
    def __init__(self, canvas):
        super().__init__("Layer Palette")
        self.canvas = canvas

        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel("Ajouter une couche :")
        layout.addWidget(label)

        layers = ["Dense", "Convolutionnel", "Pooling", "Recurrent", "Sigmoid"]
        for layer in layers:
            btn = QPushButton(layer)
            btn.clicked.connect(lambda checked, l=layer: self.add_layer(l))
            layout.addWidget(btn)

        layout.addStretch()
        widget.setLayout(layout)
        self.setWidget(widget)

    def add_layer(self, title):
        self.canvas.add_node(title, 200, 200)