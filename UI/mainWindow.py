import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QToolBar, QPushButton, QTabWidget, QLabel
from PyQt6.QtCore import Qt
from tabs.network_tab import NetworkTab
from tabs.code_tab import CodeTab
from docks.layer_palette_dock import LayerPaletteDock
from docks.layer_config_dock import LayerConfigDock
from docks.network_config_dock import NetworkConfigDock
from docks.dataset_dock import DatasetDock
from docks.results_dock import ResultsDock

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNN Editor")
        self.setMinimumSize(1200, 800)
        self.showMaximized()

        # menu d'en haut
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New")
        file_menu.addAction("Open")
        file_menu.addAction("Save")
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo")
        edit_menu.addAction("Redo")
        menubar.addMenu("Window")
        train_menu = menubar.addMenu("Train")
        train_menu.addAction("Start Training")
        menubar.addMenu("Stop")
        menubar.addMenu("About")
        menubar.addMenu("Setting")

        # barre btn
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        toolbar.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        btn_stop = QPushButton("⏹ Stop")
        btn_train = QPushButton("▶ Train")
        btn_delete = QPushButton("🗑️")
        btn_add = QPushButton("➕")

        btn_train.setStyleSheet("background-color: #28a745; color: white; padding: 5px 15px;")
        btn_stop.setStyleSheet("background-color: #dc3545; color: white; padding: 5px 15px;")
        btn_add.setStyleSheet("padding: 5px 10px;")
        btn_delete.setStyleSheet("padding: 5px 10px;")

        toolbar.addWidget(btn_stop)
        toolbar.addWidget(btn_train)
        toolbar.addSeparator()
        toolbar.addWidget(btn_delete)
        toolbar.addWidget(btn_add)
        toolbar.addSeparator()
        self.training_time_label = QLabel("⏱️ Training Time : --")
        toolbar.addWidget(self.training_time_label)

        # onglets
        tabs = QTabWidget()
        self.canvas = NetworkTab()
        tabs.addTab(self.canvas, "Network")
        self.code_tab = CodeTab()
        tabs.addTab(self.code_tab, "Code")
        self.setCentralWidget(tabs)

        # panneaux
        self.palette_dock = LayerPaletteDock(self.canvas)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.palette_dock)
        self.config_dock = LayerConfigDock()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.config_dock)
        self.dataset_dock = DatasetDock()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dataset_dock)
        self.network_config_dock = NetworkConfigDock()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.network_config_dock)
        self.results_dock = ResultsDock()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.results_dock)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
