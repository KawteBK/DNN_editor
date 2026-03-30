import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QPushButton,
    QTabWidget, QLabel, QFileDialog, QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt, QTimer

from tabs.network_tab import NetworkTab
from tabs.code_tab import CodeTab
from docks.layer_palette_dock import LayerPaletteDock
from docks.layer_config_dock import LayerConfigDock
from docks.network_config_dock import NetworkConfigDock
from docks.dataset_dock import DatasetDock
from docks.results_dock import ResultsDock

from ViewModel.project_vm import ProjectVM
from ViewModel.network_vm import NetworkVM
from ViewModel.training_vm import TrainingVM


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNN Editor")
        self.setMinimumSize(1200, 800)
        self.showMaximized()

        # ── ViewModels ────────────────────────────────────────────────────────
        self.project_vm  = ProjectVM(self)
        self.network_vm  = NetworkVM(self)
        self.training_vm = TrainingVM(self)

        # ── menu bar ──────────────────────────────────────────────────────────
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("New",  self._on_new)
        file_menu.addAction("Open", self._on_open)
        file_menu.addAction("Save", self._on_save)

        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo")   # stubs — wired later
        edit_menu.addAction("Redo")

        menubar.addMenu("Window")

        train_menu = menubar.addMenu("Train")
        train_menu.addAction("Start Training", self._on_train)

        menubar.addMenu("About")
        menubar.addMenu("Settings")

        # ── toolbar ───────────────────────────────────────────────────────────
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        toolbar.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        self.btn_stop  = QPushButton("Stop")
        self.btn_train = QPushButton("Train")
        btn_delete     = QPushButton("Delete")
        btn_add        = QPushButton("Add")

        self.btn_train.setStyleSheet(
            "background-color: #28a745; color: white; padding: 5px 15px;"
        )
        self.btn_stop.setStyleSheet(
            "background-color: #dc3545; color: white; padding: 5px 15px;"
        )
        self.btn_stop.setEnabled(False)   # only active while training

        self.btn_train.clicked.connect(self._on_train)
        self.btn_stop.clicked.connect(self._on_stop)

        toolbar.addWidget(self.btn_stop)
        toolbar.addWidget(self.btn_train)
        toolbar.addSeparator()
        toolbar.addWidget(btn_delete)
        toolbar.addWidget(btn_add)
        toolbar.addSeparator()

        self.training_time_label = QLabel("Training Time: --")
        toolbar.addWidget(self.training_time_label)

        # Timer to update the training time label every second
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_training_time)

        # Status bar for non-blocking operation feedback
        self.statusBar().showMessage("Ready")

        # ── tabs ──────────────────────────────────────────────────────────────
        tabs = QTabWidget()
        self.canvas   = NetworkTab(self.network_vm)
        self.code_tab = CodeTab()
        tabs.addTab(self.canvas, "Network")
        tabs.addTab(self.code_tab, "Code")
        self.setCentralWidget(tabs)

        # ── docks ─────────────────────────────────────────────────────────────
        self.palette_dock       = LayerPaletteDock(self.network_vm)
        self.config_dock        = LayerConfigDock(self.network_vm)
        self.dataset_dock       = DatasetDock(self.project_vm)
        self.network_config_dock = NetworkConfigDock(self.project_vm)
        self.results_dock       = ResultsDock()

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,   self.palette_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,   self.config_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,  self.dataset_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,  self.network_config_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.results_dock)

        # ── cross-VM wiring ───────────────────────────────────────────────────
        # When a project loads, update the NetworkVM's network reference
        self.project_vm.project_changed.connect(self._on_project_changed)

        # Surface errors to the user
        self.project_vm.error_occurred.connect(self._show_error)
        self.network_vm.operation_failed.connect(
            lambda msg: self.statusBar().showMessage(msg, 4000)
        )

        # Training lifecycle → UI state
        self.training_vm.training_started.connect(self._on_training_started)
        self.training_vm.training_stopped.connect(self._on_training_stopped)
        self.training_vm.training_finished.connect(self._on_training_finished)
        self.training_vm.training_failed.connect(self._show_error)
        self.training_vm.epoch_progress.connect(self.results_dock.on_epoch)

        # Network changes → regenerate code tab
        self.network_vm.network_changed.connect(self._refresh_code_tab)

        # ── start with a blank project ────────────────────────────────────────
        self.project_vm.new_project(name="untitled", path=os.path.expanduser("~/untitled_dnn"))

    # ── menu / toolbar handlers ───────────────────────────────────────────────

    def _on_new(self):
        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok or not name.strip():
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose project folder")
        if not folder:
            return
        path = os.path.join(folder, name.strip())
        self.project_vm.new_project(name=name.strip(), path=path)

    def _on_open(self):
        folder = QFileDialog.getExistingDirectory(self, "Open project folder")
        if folder:
            self.project_vm.load(folder)

    def _on_save(self):
        self.project_vm.save()
        self.statusBar().showMessage("Project saved.", 3000)

    def _on_train(self):
        if self.project_vm.project is None:
            self._show_error("No project loaded.")
            return
        self.training_vm.start(self.project_vm.project)

    def _on_stop(self):
        self.training_vm.stop()

    # ── project change ────────────────────────────────────────────────────────

    def _on_project_changed(self, project):
        """
        Called whenever a new project is loaded or created.
        Hands the new network to NetworkVM, which then emits
        layer_added / connection_added so the canvas rebuilds itself.
        """
        self.setWindowTitle(f"DNN Editor — {project.name}")
        self.network_vm.set_network(project.network)
        self.dataset_dock.load_from_project(project)
        self.network_config_dock.load_from_project(project)
        self.results_dock.clear()

    # ── training lifecycle ────────────────────────────────────────────────────

    def _on_training_started(self):
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._timer.start()
        self.statusBar().showMessage("Training…")

    def _on_training_stopped(self):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._timer.stop()

    def _on_training_finished(self, project):
        # Write the updated project (logs + weightsPath) back and save
        self.project_vm._project = project
        self.project_vm.save()
        elapsed = self.training_vm.elapsed_seconds()
        self.statusBar().showMessage(
            f"Training complete — {elapsed:.0f}s", 5000
        )

    def _update_training_time(self):
        elapsed = self.training_vm.elapsed_seconds()
        mins, secs = divmod(int(elapsed), 60)
        self.training_time_label.setText(f"Training Time: {mins:02d}:{secs:02d}")

    # ── code tab ─────────────────────────────────────────────────────────────

    def _refresh_code_tab(self):
        """Regenerates the PyTorch script whenever the network changes."""
        project = self.project_vm.project
        if project is None or project.network is None:
            return
        if not project.network.validate():
            return   # don't generate code for an incomplete network
        try:
            from PyTorch.codeGenerator import PyTorchCodeGenerator
            # Code generator needs dataset + configs — skip if not set yet
            if project.dataset is None or project.networkConfig is None:
                return
            code = PyTorchCodeGenerator.generate(project)
            self.code_tab.set_code(code)
        except Exception:
            pass   # silent — code tab stays as-is if generation fails

    # ── error display ─────────────────────────────────────────────────────────

    def _show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())