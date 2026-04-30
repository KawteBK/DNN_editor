from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPen
import sys

# Modifier key label — ⌘ on macOS, Ctrl on Windows/Linux
_MOD   = "⌘"    if sys.platform == "darwin" else "Ctrl"
_SHIFT = "⇧"    if sys.platform == "darwin" else "Shift"


class _LinkButton(QPushButton):
    """
    A button that looks like a VS Code sidebar link —
    no border, no background, just coloured text with an arrow prefix
    and a subtle shortcut hint aligned to the right.
    """

    def __init__(self, arrow: str, text: str, shortcut: str = "", parent=None):
        # Build label: arrow + text padded, then shortcut hint right-aligned via spaces
        self._label_text = text
        self._shortcut   = shortcut
        display = f"  {arrow}  {text}"
        super().__init__(display, parent)
        self._base_color   = "#4FC3F7"
        self._hover_color  = "#81D4FA"
        self._hint_color   = "#3d5060"
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setFont(QFont("Consolas", 13))
        self._apply_style(self._base_color)
        if shortcut:
            self.setToolTip(shortcut)

    def _apply_style(self, color: str) -> None:
        self.setStyleSheet(f"""
            QPushButton {{
                color: {color};
                background: transparent;
                border: none;
                text-align: left;
                padding: 4px 0px;
                letter-spacing: 0.5px;
            }}
            QPushButton:pressed {{
                color: #B3E5FC;
            }}
        """)

    def paintEvent(self, event):
        """Draw the button text + a right-aligned shortcut hint."""
        super().paintEvent(event)
        if not self._shortcut:
            return
        from PyQt6.QtGui import QPainter, QColor, QFont
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        hint_font = QFont("Consolas", 10)
        painter.setFont(hint_font)
        painter.setPen(QColor(self._hint_color if not self.underMouse() else "#4a6880"))
        # Draw shortcut hint flush-right with a small right margin
        rect = self.rect().adjusted(0, 0, -8, 0)
        painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                         self._shortcut)
        painter.end()

    def enterEvent(self, event):
        self._apply_style(self._hover_color)
        self._hint_color = "#4a6880"
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_style(self._base_color)
        self._hint_color = "#3d5060"
        self.update()
        super().leaveEvent(event)


class WelcomeScreen(QWidget):
    """
    Full-window welcome screen shown when no project is loaded.

    Emits:
        new_project_requested  — user clicked "New Project"
        open_project_requested — user clicked "Open Existing Project"
    """

    new_project_requested  = pyqtSignal()
    open_project_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # ── vertical centering spacer ──────────────────────────────────────
        root.addStretch(2)

        centre = QHBoxLayout()
        centre.addStretch(1)
        centre.addWidget(self._build_card())
        centre.addStretch(1)
        root.addLayout(centre)

        root.addStretch(3)

        # ── bottom hint ────────────────────────────────────────────────────
        hint = QLabel(
            f"New  {_MOD}+N   ·   Open  {_MOD}+O   ·   "
            f"Save  {_MOD}+S   ·   Save As  {_MOD}+{_SHIFT}+S"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #555a63; font-size: 11px; font-family: Consolas;")
        root.addWidget(hint)
        root.addSpacing(18)

    def _build_card(self) -> QWidget:
        card = QWidget()
        card.setFixedWidth(420) # OG GRAY: #2b2d30
        card.setStyleSheet("""
            QWidget {
                background-color: #2b2d30;  
                border: 1px solid #3c3f41;
                border-radius: 6px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(0)

        # ── logo / app name ────────────────────────────────────────────────
        logo = QLabel("⬡") # a hexagonal shape to evoke the idea of a neural network node, and also because it looks cool, buuut i want to later change it to a custom SVG icon that i design myself, maybe with some circuitry patterns inside the hexagon to make it more unique and thematic
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("""
            color: #4FC3F7;
            font-size: 36px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(logo)
        layout.addSpacing(10)

        title = QLabel("NeuroCraft")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Consolas", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #d4d4d4; background: transparent; border: none;")
        layout.addWidget(title)

        subtitle = QLabel("Design · Train · Deploy")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            color: #6e7681;
            font-size: 11px;
            font-family: Consolas;
            letter-spacing: 3px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(subtitle)

        layout.addSpacing(32)

        # ── divider ────────────────────────────────────────────────────────
        divider = QLabel()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #3c3f41; border: none;")
        layout.addWidget(divider)

        layout.addSpacing(24)

        # ── section label ──────────────────────────────────────────────────
        start_label = QLabel("Start")
        start_label.setStyleSheet("""
            color: #9da5b4;
            font-size: 11px;
            font-family: Consolas;
            letter-spacing: 2px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(start_label)
        layout.addSpacing(10)

        # ── link buttons ───────────────────────────────────────────────────
        btn_new = _LinkButton("$(file-add)", "New Project", f"{_MOD}+N", self)
        btn_new.clicked.connect(self.new_project_requested)
        layout.addWidget(btn_new)

        layout.addSpacing(6)

        btn_open = _LinkButton("$(folder-opened)", "Open Existing Project", f"{_MOD}+O", self)
        btn_open.clicked.connect(self.open_project_requested)
        layout.addWidget(btn_open)

        layout.addSpacing(28)

        # ── bottom tip ─────────────────────────────────────────────────────
        tip = QLabel("Tip: right-click on the canvas to connect layers")
        tip.setWordWrap(True)
        tip.setStyleSheet("""
            color: #4a5060;
            font-size: 10px;
            font-family: Consolas;
            background: transparent;
            border: none;
        """)
        layout.addWidget(tip)

        return card

    # ── background ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        """Draw a subtle dot-grid on the dark background — VS Code welcome vibe."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1f22"))

        # dot grid
        dot_color = QColor("#2a2c30")
        painter.setPen(QPen(dot_color, 1))
        spacing = 28
        for x in range(0, self.width(), spacing):
            for y in range(0, self.height(), spacing):
                painter.drawPoint(x, y)

        painter.end()