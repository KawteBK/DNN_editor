from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient


class _LinkButton(QPushButton):
    """
    A button that looks like a VS Code sidebar link —
    no border, no background, just coloured text with an arrow prefix
    and a subtle underline that slides in on hover.
    """

    def __init__(self, arrow: str, text: str, parent=None):
        super().__init__(f"  {arrow}  {text}", parent)
        self._base_color   = "#4FC3F7"   # VS Code light-blue link colour
        self._hover_color  = "#81D4FA"
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setFont(QFont("Consolas", 13))
        self._apply_style(self._base_color)

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

    def enterEvent(self, event):
        self._apply_style(self._hover_color)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_style(self._base_color)
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
        hint = QLabel("File → New Project  ·  File → Open")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #555a63; font-size: 11px; font-family: Consolas;")
        root.addWidget(hint)
        root.addSpacing(18)

    def _build_card(self) -> QWidget:
        card = QWidget()
        card.setFixedWidth(420)
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
        logo = QLabel("⬡")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("""
            color: #4FC3F7;
            font-size: 36px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(logo)
        layout.addSpacing(10)

        title = QLabel("DNN Editor")
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
        btn_new = _LinkButton("$(file-add)", "New Project", self)
        btn_new.clicked.connect(self.new_project_requested)
        layout.addWidget(btn_new)

        layout.addSpacing(6)

        btn_open = _LinkButton("$(folder-opened)", "Open Existing Project", self)
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