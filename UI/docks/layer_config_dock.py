from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QFormLayout, QLabel,
    QLineEdit, QCheckBox, QComboBox, QScrollArea, QVBoxLayout
)
from PyQt6.QtCore import QTimer, Qt

from Core.layer import (
    Dense, Conv2D, MergeMode, Pooling, Recurrent, Flatten, Activation, InputLayer,
    ActivationType, PoolingType, RecurrentType, DropoutStyle,Merge
)


class LayerConfigDock(QDockWidget):
    # CHANGED: accepts network_vm. Previously had a fixed form that showed the
    # same fields regardless of which layer was selected. Now the form rebuilds
    # itself dynamically based on the selected layer's type.
    def __init__(self, network_vm):
        super().__init__("Layer Config")
        self.network_vm = network_vm   # ADDED
        self._current_layer_id: str | None = None
        self._suppress_signals = False   # ADDED: prevents feedback loops while populating

        # ADDED: scroll area so the form doesn't get clipped on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            



        self._form_widget = QWidget()
        self._layout = QFormLayout()
        self._form_widget.setLayout(self._layout)
        scroll.setWidget(self._form_widget)
        self.setWidget(scroll)

        # ADDED: connect to NetworkVM so the form rebuilds when a node is clicked
        self.network_vm.layer_selected.connect(self._on_layer_selected)

        self._show_placeholder()

    # ── NetworkVM signal handler ──────────────────────────────────────────────

    def _on_layer_selected(self, layer) -> None:
        # ADDED: called when the user clicks a node on the canvas.
        # Clears the old form and builds a new one specific to the layer type.
        self._current_layer_id = layer._id if layer else None
        self._clear_form()

        if layer is None:
            self._show_placeholder()
            return 

        if isinstance(layer, InputLayer):
            self._build_input_form(layer)
        elif isinstance(layer, Dense):
            self._build_dense_form(layer)
        elif isinstance(layer, Conv2D):
            self._build_conv2d_form(layer)
        elif isinstance(layer, Pooling):
            self._build_pooling_form(layer)
        elif isinstance(layer, Recurrent):
            self._build_recurrent_form(layer)
        elif isinstance(layer, Flatten):
            self._build_flatten_form(layer)
        elif isinstance(layer, Activation):
            self._build_activation_form(layer)
        elif isinstance(layer, Merge):
            self._build_merge_form(layer)

    # ── form builders — one per layer type
    def _build_input_form(self, layer) -> None:
        # InputLayer is locked — show info only, no editable fields
        self._suppress_signals=True
        self._layout.addRow(QLabel("Type:"), QLabel("Input Layer"))
        self._layout.addRow(QLabel("Shape:"), QLabel(str(layer._outputShape)))
        self._layout.addRow(QLabel(""), QLabel("(locked — set by dataset)"))
        self._suppress_signals=False

    def _build_dense_form(self, layer: Dense) -> None:
        self._suppress_signals=True
        self._add_name_row(layer)

        units = self._make_int_field(layer._unit)
        units.editingFinished.connect(
            lambda: self._push({"unit": self._safe_int(units.text(), layer._unit)})
        )
        self._layout.addRow(QLabel("Units:"), units)

        bias = QCheckBox()
        bias.setChecked(layer._useBias)
        bias.checkStateChanged.connect(
            lambda: self._push({"useBias": bias.isChecked()})
        )
        self._layout.addRow(QLabel("Use bias:"), bias)

        dropout = self._make_float_field(layer._dropout)
        dropout.editingFinished.connect(
            lambda: self._push({"dropout": self._safe_float(dropout.text(), layer._dropout)})
        )
        self._layout.addRow(QLabel("Dropout:"), dropout)

        self._add_shape_row(layer)
        self._suppress_signals=False

    def _build_conv2d_form(self, layer: Conv2D) -> None:
        self._suppress_signals=True
        self._add_name_row(layer)

        filters = self._make_int_field(layer._filters)
        filters.editingFinished.connect(
            lambda: self._push({"filters": self._safe_int(filters.text(), layer._filters)})
        )
        self._layout.addRow(QLabel("Filters:"), filters)

        kernel = self._make_tuple_field(layer._kernelSize)
        kernel.editingFinished.connect(
            lambda: self._push({"kernelSize": self._safe_tuple(kernel.text(), layer._kernelSize)})
        )
        self._layout.addRow(QLabel("Kernel size:"), kernel)

        stride = self._make_tuple_field(layer._strides)
        stride.editingFinished.connect(
            lambda: self._push({"strides": self._safe_tuple(stride.text(), layer._strides)})
        )
        self._layout.addRow(QLabel("Strides:"), stride)

        padding = self._make_tuple_field(layer._padding)
        padding.editingFinished.connect(
            lambda: self._push({"padding": self._safe_tuple(padding.text(), layer._padding)})
        )
        self._layout.addRow(QLabel("Padding:"), padding)

        dropout = self._make_float_field(layer._dropout)
        dropout.editingFinished.connect(
            lambda: self._push({"dropout": self._safe_float(dropout.text(), layer._dropout)})
        )
        self._layout.addRow(QLabel("Dropout:"), dropout)

        # ADDED: DropoutStyle combo — only relevant for Conv2D
        ds_combo = QComboBox()
        ds_combo.addItems(["None", "Standard", "Spatial"])
        current = {None: 0, DropoutStyle.STANDARD: 1, DropoutStyle.SPATIAL: 2}
        ds_combo.setCurrentIndex(current.get(layer._dropoutStyle, 0))
        ds_combo.currentIndexChanged.connect(
            lambda i: self._push({"dropoutStyle": [None, DropoutStyle.STANDARD, DropoutStyle.SPATIAL][i]})
        )
        self._layout.addRow(QLabel("Dropout style:"), ds_combo)

        self._add_shape_row(layer)
        self._suppress_signals=False

    def _build_pooling_form(self, layer: Pooling) -> None:
        self._suppress_signals=True
        self._add_name_row(layer)

        pool_size = self._make_tuple_field(layer._poolSize)
        pool_size.editingFinished.connect(
            lambda: self._push({"poolSize": self._safe_tuple(pool_size.text(), layer._poolSize)})
        )
        self._layout.addRow(QLabel("Pool size:"), pool_size)

        stride = self._make_tuple_field(layer._strides)
        stride.editingFinished.connect(
            lambda: self._push({"strides": self._safe_tuple(stride.text(), layer._strides)})
        )
        self._layout.addRow(QLabel("Strides:"), stride)

        padding = self._make_tuple_field(layer._padding)
        padding.editingFinished.connect(
            lambda: self._push({"padding": self._safe_tuple(padding.text(), layer._padding)})
        )
        self._layout.addRow(QLabel("Padding:"), padding)

        pt_combo = QComboBox()
        pt_combo.addItems(["Max", "Average"])
        pt_combo.setCurrentIndex(0 if layer._poolingType == PoolingType.MAX else 1)
        pt_combo.currentIndexChanged.connect(
            lambda i: self._push({"poolingType": [PoolingType.MAX, PoolingType.AVERAGE][i]})
        )
        self._layout.addRow(QLabel("Pool type:"), pt_combo)

        self._add_shape_row(layer)
        self._suppress_signals=False

    def _build_recurrent_form(self, layer: Recurrent) -> None:
        self._suppress_signals=True
        self._add_name_row(layer)

        hidden = self._make_int_field(layer._hiddenSize)
        hidden.editingFinished.connect(
            lambda: self._push({"hiddenSize": self._safe_int(hidden.text(), layer._hiddenSize)})
        )
        self._layout.addRow(QLabel("Hidden size:"), hidden)

        num_layers = self._make_int_field(layer._numLayer)
        num_layers.editingFinished.connect(
            lambda: self._push({"numLayer": self._safe_int(num_layers.text(), layer._numLayer)})
        )
        self._layout.addRow(QLabel("Num layers:"), num_layers)

        dropout = self._make_float_field(layer._dropout)
        dropout.editingFinished.connect(
            lambda: self._push({"dropout": self._safe_float(dropout.text(), layer._dropout)})
        )
        self._layout.addRow(QLabel("Dropout:"), dropout)

        rt_combo = QComboBox()
        rt_combo.addItems(["LSTM", "GRU"])
        rt_combo.setCurrentIndex(0 if layer._recurrentType == RecurrentType.LSTM else 1)
        rt_combo.currentIndexChanged.connect(
            lambda i: self._push({"recurrentType": [RecurrentType.LSTM, RecurrentType.GRU][i]})
        )
        self._layout.addRow(QLabel("Type:"), rt_combo)

        bidir = QCheckBox()
        bidir.setChecked(layer._bidirection)
        bidir.checkStateChanged.connect(
            lambda: self._push({"bidirection": bidir.isChecked()})
        )
        self._layout.addRow(QLabel("Bidirectional:"), bidir)

        ret_seq = QCheckBox()
        ret_seq.setChecked(layer._returnSeq)
        ret_seq.checkStateChanged.connect(
            lambda: self._push({"returnSeq": ret_seq.isChecked()})
        )
        self._layout.addRow(QLabel("Return sequence:"), ret_seq)

        self._add_shape_row(layer)
        self._suppress_signals=False

    def _build_flatten_form(self, layer: Flatten) -> None:
        self._suppress_signals=True

        self._add_name_row(layer)
        self._layout.addRow(QLabel(""), QLabel("No parameters — flattens input"))
        self._add_shape_row(layer)
        self._suppress_signals=False

    def _build_activation_form(self, layer: Activation) -> None:
        self._suppress_signals=True
        self._add_name_row(layer)

        at_combo = QComboBox()
        options = [a for a in ActivationType]
        at_combo.addItems([a.value for a in options])
        current_idx = options.index(layer._activationType) if layer._activationType in options else 0
        at_combo.setCurrentIndex(current_idx)
        at_combo.currentIndexChanged.connect(
            lambda i: self._push({"activationType": options[i]})
        )
        self._layout.addRow(QLabel("Activation:"), at_combo)

        self._add_shape_row(layer)
        self._suppress_signals=False


    def _build_merge_form(self, layer):
        self._suppress_signals=True
        self._add_name_row(layer)
    
        # Merge mode combo box
        mode_combo = QComboBox()
        mode_combo.addItems(["Add (element-wise sum)", "Concatenate (along last dim)"])
        mode_combo.setCurrentIndex(0 if layer._mergeMode == MergeMode.ADD else 1)
        mode_combo.currentIndexChanged.connect(
            lambda i: self._push({"mergeMode": [MergeMode.ADD, MergeMode.CONCATENATE][i]})
        )
        self._layout.addRow(QLabel("Merge mode:"), mode_combo)
        self._layout.addRow(QLabel(""), QLabel("Add: all input shapes must match exactly"))
        self._layout.addRow(QLabel(""), QLabel("Concatenate: all dims except last must match"))
    
        self._add_shape_row(layer)
        self._suppress_signals=False

    # ── shared row helpers ────────────────────────────────────────────────────

    def _add_name_row(self, layer) -> None:
            self._layout.addRow(QLabel("Type:"), QLabel(type(layer).__name__))
            self._layout.addRow(QLabel("ID:"), QLabel(layer._id))
            # Fix alignment - make labels right-aligned
            name_label = QLabel("Name:")
            name_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            name_field = QLineEdit(layer._name)
            name_field.editingFinished.connect(
            lambda: self._push({"name": name_field.text()})
            )
            self._layout.addRow(name_label, name_field)

    def _add_shape_row(self, layer) -> None:
        # ADDED: shows the computed output shape as read-only info
        shape_str = str(layer._outputShape) if layer._outputShape else "not yet computed"
        self._layout.addRow(QLabel("Output shape:"), QLabel(shape_str))

    # ── push changes to NetworkVM ─────────────────────────────────────────────

    def _push(self, params: dict) -> None:
        # ADDED: sends field changes to NetworkVM, which updates the Core layer
        # and re-propagates shapes. The canvas then refreshes its shape labels
        # via the network_changed signal.
        if self._current_layer_id is None:
            return
        self.network_vm.update_layer_params(self._current_layer_id, params)

    # ── form lifecycle ────────────────────────────────────────────────────────

    def _clear_form(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_placeholder(self) -> None:
        self._layout.addRow(QLabel("Select a layer on the canvas\nto edit its parameters."))

    # ── field factories ───────────────────────────────────────────────────────

    def _make_int_field(self, value) -> QLineEdit:
        f  = QLineEdit(str(value))
        f.setMaximumWidth(150)  # Limit width
        return f

    def _make_float_field(self, value) -> QLineEdit:
        f = QLineEdit(str(value))
        f.setMaximumWidth(150)
        return f

    def _make_tuple_field(self, value) -> QLineEdit:
        # Displays tuples as "3, 3" for easy editing
        f = QLineEdit(", ".join(str(v) for v in value))
        f.setMaximumWidth(150)
        return f

    # ── safe parsers — never crash on bad user input ──────────────────────────

    def _safe_int(self, text: str, fallback: int) -> int:
        try:
            return int(text.strip())
        except ValueError:
            return fallback

    def _safe_float(self, text: str, fallback: float) -> float:
        try:
            return float(text.strip())
        except ValueError:
            return fallback

    def _safe_tuple(self, text: str, fallback: tuple) -> tuple:
        try:
            return tuple(int(v.strip()) for v in text.split(","))
        except (ValueError, AttributeError):
            return fallback
        


        # Adding  visual feedback for config changes 

    def _flash_field(self, field: QWidget):
        """Temporarily highlight a field to show value was applied."""
        original_bg = field.styleSheet()
        field.setStyleSheet("background-color: #279AAE; border: 1px solid #B3F4FF;")
        QTimer.singleShot(200, lambda: field.setStyleSheet(original_bg))

    def _push(self, params: dict) -> None:
        if self._current_layer_id is None:
            return
    
        #Find which widget was changed and flash it
        sender = self.sender()
        if sender and isinstance(sender, QLineEdit):
            self._flash_field(sender)
    
        self.network_vm.update_layer_params(self._current_layer_id, params)