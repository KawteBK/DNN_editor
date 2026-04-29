from PyQt6.QtWidgets import QDockWidget, QScrollArea, QWidget, QVBoxLayout, QPushButton, QLabel

#from UI.canvas.node_item import LAYER_COLORS

LAYER_COLORS = {
    "InputLayer":  ("#24A35983", "#1A7A43"),   # green   always the root
    "Dense":       ("#1592A283", "#105860"),   # teal
    "Conv2D":      ("#7F77DD83", "#4A4490"),   # purple
    "Pooling":     ("#E67E2283", "#9A5515"),   # amber
    "Recurrent":   ("#D85A3083", "#8C3515"),   # coral
    "Flatten":     ("#B2992883", "#906E00"),   # yellowish brown 5ra vibes
    "Activation":  ("#D4537E83", "#8C2E4F"),   # pink
    "Merge":       ("#956AA683", "#6C3483"),   # purpley-pink, twilight sparkle vibes
}


class LayerPaletteDock(QDockWidget):
    # CHANGED: accepts network_vm instead of canvas
    # Previously called canvas.add_node() directly, now calls network_vm.add_layer()
    # so the Core model is updated before the canvas draws anything
    def __init__(self, network_vm):
        super().__init__("Layer Palette")
        self.network_vm = network_vm   # ADDED

        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Add a layer:"))

        # CHANGED: layer names now match Core layer type strings exactly.
        # "Convolutionnel" -> "Conv2D", "Sigmoid" removed (it's an Activation type,
        # not a layer — Activation with type=SIGMOID is handled in config dock).
        # Flatten and Activation added since they exist in Core but were missing.
        layers = ["Dense", "Conv2D", "Pooling", "Recurrent", "Flatten", "Merge", "Activation"]

        for layer_type in layers:
            btn = QPushButton(layer_type)

            base_color, dark_color = LAYER_COLORS[layer_type]

            btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {base_color};
            color: #f7f5ff;
            border: 1px solid {dark_color};
            padding: 4px;
            border-radius: 6px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {dark_color};
        }}
        QPushButton:pressed {{
            background-color: {dark_color};
            padding-left: 8px;
            padding-top: 8px;
        }}
    """)

            btn.clicked.connect(lambda checked, t=layer_type: self.network_vm.add_layer(t))
            layout.addWidget(btn)

        layout.addStretch()
        widget.setLayout(layout)
        self.setWidget(widget)

        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.setWidget(scroll)