from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QApplication, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import QTimer


# Shared button base style — both buttons use this so they look consistent
_BTN_BASE = (
    "QPushButton {"
    "  padding: 6px 18px;"
    "  border-radius: 4px;"
    "  font-size: 12px;"
    "  border: 1px solid #444;"
    "}"
)
_BTN_NORMAL_COPY   = _BTN_BASE + "QPushButton { background:#2d2d2d; color:#d4d4d4; }"
_BTN_FLASHED_COPY  = _BTN_BASE + "QPushButton { background:#1a6b35; color:#ffffff; border-color:#22c55e; }"
_BTN_EXPORT = _BTN_BASE + """
QPushButton {
  background:#1e3a5f;
  color:#93c5fd;
  border: 1px solid #3b82f6;
}
QPushButton:pressed {
  background:#162d4a;
  padding-left: 20px;
  padding-top: 8px;
}
"""

class CodeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── code editor ───────────────────────────────────────────────────────
        self.code_editor = QTextEdit()
        self.code_editor.setReadOnly(True)
        self.code_editor.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4;"
            "font-family: 'Courier New', monospace; font-size: 13px;"
            "border: none;"
        )
        self.code_editor.setPlaceholderText(
            "# Generated PyTorch script will appear here.\n"
            "# Build a valid network and select a dataset to generate code."
        )

        # ── button row ────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(8, 6, 8, 8)
        btn_row.setSpacing(10)

        self.btn_copy = QPushButton("Copy code")
        self.btn_copy.setStyleSheet(_BTN_NORMAL_COPY)
        self.btn_copy.setMinimumWidth(160)
        self.btn_copy.clicked.connect(self._copy_code)

        self.btn_export = QPushButton("⬇  Export .py")
        self.btn_export.setStyleSheet(_BTN_EXPORT)
        self.btn_export.setMinimumWidth(160)
        self.btn_export.setToolTip("Save the generated script to a file")
        self.btn_export.clicked.connect(self._export_code)

        btn_row.addStretch()
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_export)
        btn_row.addStretch()

        layout.addWidget(self.code_editor, stretch=1)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    # ── public API ────────────────────────────────────────────────────────────

    def set_code(self, code: str) -> None:
        """Called by MainWindow._refresh_code_tab() whenever the network changes."""
        self.code_editor.setPlainText(code)

    def clear(self) -> None:
        """Clears the editor and resets the placeholder."""
        self.code_editor.clear()
        self.code_editor.setPlaceholderText(
            "# No project loaded.\n"
            "# Use File → New Project or File → Open to get started."
        )

    # ── button handlers ───────────────────────────────────────────────────────

    def _copy_code(self) -> None:
        code = self.code_editor.toPlainText()
        if not code:
            return
        QApplication.clipboard().setText(code)

        # Green flash for 1500 ms, then revert
        self.btn_copy.setText("✓ Copied!")
        self.btn_copy.setStyleSheet(_BTN_FLASHED_COPY)
        QTimer.singleShot(1500, self._reset_copy_button)

    def _reset_copy_button(self) -> None:
        self.btn_copy.setText("Copy code")
        self.btn_copy.setStyleSheet(_BTN_NORMAL_COPY)

    def _export_code(self) -> None:
        """Open a save-file dialog and write the generated script there."""
        code = self.code_editor.toPlainText()
        if not code:
            QMessageBox.information(
                self, "Nothing to export",
                "Generate code first by building a valid network and selecting a dataset."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PyTorch script",
            "model.py",           # default filename
            "Python files (*.py);;All files (*)",
        )
        if not path:
            return   # user cancelled

        # Ensure the file ends with .py
        if not path.endswith(".py"):
            path += ".py"

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))