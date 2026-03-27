from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QApplication

class CodeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.code_editor = QTextEdit()
        self.code_editor.setReadOnly(True)
        self.code_editor.setStyleSheet(
            "background-color: #1e1e1e; color: #ffffff; font-family: Courier New; font-size: 13px;"
        )
        self.code_editor.setText("# Le code PyTorch sera généré ici automatiquement :)")

        btn_copy = QPushButton("📋 Copier le code")
        btn_copy.setStyleSheet("padding: 5px 15px;")
        btn_copy.clicked.connect(self.copy_code)

        layout.addWidget(self.code_editor)
        layout.addWidget(btn_copy)
        self.setLayout(layout)

    def copy_code(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.code_editor.toPlainText())
