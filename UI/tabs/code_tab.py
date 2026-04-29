from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QApplication, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import QTimer, QRegularExpression
from PyQt6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont


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

def _fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    """Helper — builds a QTextCharFormat from a hex color string."""
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    if bold:
        fmt.setFontWeight(QFont.Weight.Bold)
    if italic:
        fmt.setFontItalic(True)
    return fmt


class PythonHighlighter(QSyntaxHighlighter):
    """
    VS Code Dark+ inspired syntax highlighter for generated PyTorch code.
    Uses only QRegularExpression — zero extra dependencies.
    """

    # ── token palette (VS Code Dark+ colours) ────────────────────────────────
    _KEYWORD   = _fmt("#569cd6", bold=True)   # blue  — def, class, import …
    _BUILTIN   = _fmt("#4ec9b0")              # teal  — torch, nn, self …
    _DECORATOR = _fmt("#dcdcaa")              # yellow — @torch.no_grad etc.
    _STRING    = _fmt("#ce9178")              # orange — "…" / '…' / """…"""
    _NUMBER    = _fmt("#b5cea8")              # green  — 0.001, 128, 1e-4
    _COMMENT   = _fmt("#6a9955", italic=True) # green italic — # …
    _SELF      = _fmt("#9cdcfe")              # light blue — self
    _CLASSNAME = _fmt("#4ec9b0", bold=True)   # teal bold — class Foo
    _FUNCNAME  = _fmt("#dcdcaa")              # yellow — def forward

    def __init__(self, document):
        super().__init__(document)

        kw = [
            "False", "None", "True", "and", "as", "assert", "async", "await",
            "break", "class", "continue", "def", "del", "elif", "else",
            "except", "finally", "for", "from", "global", "if", "import",
            "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
            "return", "try", "while", "with", "yield",
        ]

        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = []

        # Decorators  @something
        self._rules.append((QRegularExpression(r"@[\w\.]+"), self._DECORATOR))

        # class Name
        self._rules.append((QRegularExpression(r"\bclass\s+(\w+)"),  self._KEYWORD))
        self._rules.append((QRegularExpression(r"(?<=class\s)\w+"),  self._CLASSNAME))

        # def name
        self._rules.append((QRegularExpression(r"\bdef\s+(\w+)"),    self._KEYWORD))
        self._rules.append((QRegularExpression(r"(?<=def\s)\w+"),    self._FUNCNAME))

        # keywords
        kw_pattern = r"\b(" + "|".join(kw) + r")\b"
        self._rules.append((QRegularExpression(kw_pattern), self._KEYWORD))

        # self / cls
        self._rules.append((QRegularExpression(r"\b(self|cls)\b"), self._SELF))

        # torch / nn / F  (common PyTorch namespaces)
        self._rules.append((QRegularExpression(r"\b(torch|nn|F|optim|DataLoader|Dataset)\b"), self._BUILTIN))

        # numbers  (int, float, scientific, hex)
        self._rules.append((QRegularExpression(
            r"\b(0x[0-9A-Fa-f]+|[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?)\b"
        ), self._NUMBER))

        # single-line strings  '…'  "…"
        self._rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), self._STRING))
        self._rules.append((QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), self._STRING))

        # comments  # …
        self._rules.append((QRegularExpression(r"#[^\n]*"), self._COMMENT))

        # ── multi-line string state (triple-quoted) ───────────────────────────
        self._tri_single = QRegularExpression(r"'''")
        self._tri_double = QRegularExpression(r'"""')
        # state codes: 0 = normal, 1 = inside ''', 2 = inside """
        self._STATE_TRIPLE_SINGLE = 1
        self._STATE_TRIPLE_DOUBLE = 2

    def highlightBlock(self, text: str) -> None:  # type: ignore[override]
        # ── single-line rules ─────────────────────────────────────────────────
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)

        # ── triple-quoted strings (multi-line) ────────────────────────────────
        self.setCurrentBlockState(0)

        for delimiter, state in (
            (self._tri_double, self._STATE_TRIPLE_DOUBLE),
            (self._tri_single, self._STATE_TRIPLE_SINGLE),
        ):
            start = 0
            if self.previousBlockState() != state:
                m = delimiter.match(text, start)
                start = m.capturedStart() if m.hasMatch() else -1

            while start >= 0:
                m = delimiter.match(text, start + 3)
                if m.hasMatch():
                    end = m.capturedEnd()
                    self.setFormat(start, end - start, self._STRING)
                    start_next = delimiter.match(text, end)
                    start = start_next.capturedStart() if start_next.hasMatch() else -1
                else:
                    self.setCurrentBlockState(state)
                    self.setFormat(start, len(text) - start, self._STRING)
                    break


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

        # ── syntax highlighting ───────────────────────────────────────────────
        self._highlighter = PythonHighlighter(self.code_editor.document())

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