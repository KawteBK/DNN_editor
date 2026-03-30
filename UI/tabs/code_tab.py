from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QApplication


class CodeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.code_editor = QTextEdit()
        self.code_editor.setReadOnly(True)
        self.code_editor.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4;"
            "font-family: 'Courier New', monospace; font-size: 13px;"
        )
        # CHANGED: placeholder message in English to match the rest of the UI
        self.code_editor.setPlaceholderText(
            "# Generated PyTorch script will appear here.\n"
            "# Build a valid network and select a dataset to generate code."
        )

        btn_copy = QPushButton("Copy code")
        btn_copy.clicked.connect(self._copy_code)

        layout.addWidget(self.code_editor)
        layout.addWidget(btn_copy)
        self.setLayout(layout)

    # ADDED: called by MainWindow._refresh_code_tab() whenever network_changed fires.
    # Pushes the generated script string into the editor.
    def set_code(self, code: str) -> None:
        self.code_editor.setPlainText(code)

    def _copy_code(self) -> None:
        clipboard = QApplication.clipboard()
        clipboard.setText(self.code_editor.toPlainText())