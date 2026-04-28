import sys

import os
try:
    import winsound # Windows only
except:
    pass

#this to fix all the bullshit about module notfound when importing from another folder that  is not on the same level as the file that imports it
sys.path.insert(0, os.path.dirname(__file__))                      # -> UI/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))     # -> ai_editor/

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QPushButton,
    QTabWidget, QLabel, QFileDialog, QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt, QTimer
from tabs.network_tab import NetworkTab
from tabs.code_tab import CodeTab
from tabs.inference_tab import OngletInference
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

        self._file_menu = menubar.addMenu("File")
        self._file_menu.addAction("New project", self._on_new)
        self._file_menu.addAction("Open",        self._on_open)
        self._file_menu.addAction("Save",        self._on_save)

        self._edit_menu = menubar.addMenu("Edit")
        self._edit_menu.addAction("Undo")   # stubs — wired later
        self._edit_menu.addAction("Redo")

        window_menu = menubar.addMenu("Window")

        self._train_menu = menubar.addMenu("Train")
        self._train_menu.addAction("Start Training", self._on_train)

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

        btn_add.clicked.connect(self._on_add_layer)
        btn_delete.clicked.connect(self._on_delete_selected)

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
        self.inference_tab = OngletInference(self.project_vm)
        tabs.addTab(self.canvas, "Network")
        tabs.addTab(self.code_tab, "Code")
        tabs.addTab(self.inference_tab, "Inference")
        self.setCentralWidget(tabs)


        #style for teh tabs! sorry you were too ugly

        tabs.setStyleSheet("""
    QTabBar::tab {
        font-size: 13px;
        font-weight: bold;
        padding: 8px 16px;
        min-width: 80px;
    }
    QTabBar::tab:selected {
        background-color: #3c3f41;
    }
""")

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

        # setting dock sizes:
        # self.palette_dock.setMinimumWidth(180)
        # # self.palette_dock.setMaximumWidth(220)
        # self.config_dock.setMinimumHeight(300)  # Make taller
        # self.config_dock.setMinimumWidth(250)
        # self.dataset_dock.setMinimumWidth(220)
        # self.network_config_dock.setMinimumWidth(250)

        self.results_dock.setMaximumHeight(300)

        self.resizeDocks([self.results_dock], [200], Qt.Orientation.Vertical)



#the palete dok hide and unhide
        for dock in [self.palette_dock, self.config_dock, self.dataset_dock, 
                self.network_config_dock, self.results_dock]:
            action = window_menu.addAction(dock.windowTitle())
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(lambda checked, d=dock: d.setVisible(checked))
            dock.visibilityChanged.connect(lambda visible, a=action: a.setChecked(visible))

        # ── cross-VM wiring ───────────────────────────────────────────────────


        self.training_vm.training_started.connect(self.canvas.start_training_animation)
        self.training_vm.training_stopped.connect(self.canvas.stop_training_animation)
        #self.project_vm.dataset_changed.connect(self._update_train_button_state) #enable the train button when dataset is set, since it's required to train duh


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
        tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(tabs)

        # ── start with a empty project ────────────────────────────────────────
        self.project_vm._project = None 
        self._update_ui_for_no_project() 
#cuz 
        self.statusBar().showMessage("Ready. Use File → New Project to get started.", 5000)

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


    def _on_add_layer(self):
        """Opens a dialog to choose layer type and adds it to canvas."""
        from PyQt6.QtWidgets import QInputDialog
    
        layer_types = ["Dense", "Conv2D", "Pooling", "Recurrent", "Flatten", "Activation", "Merge"]
        layer_type, ok = QInputDialog.getItem(
            self, "Add Layer", "Select layer type:", layer_types, 0, False
        )
        if ok and layer_type:
            self.network_vm.add_layer(layer_type)


    def _on_delete_selected(self):
        """Deletes selected layer or connection with confirmation."""
        from PyQt6.QtWidgets import QMessageBox
    
        # Get selected items from canvas
        selected_items = self.canvas.scene.selectedItems()
    
        if not selected_items:
            self.statusBar().showMessage("Nothing selected. Click a layer or connection first.", 3000)
            return
    
        # Check if it's a node (layer) or connection
        from canvas.node_item import NodeItem
        from canvas.connection_item import ConnectionItem
    
        for item in selected_items:
            if isinstance(item, NodeItem):
                reply = QMessageBox.question(
                    self, "Delete Layer",
                    f"Delete layer '{item.layer_id}'?\nThis will also remove all its connections.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.network_vm.remove_layer(item.layer_id)
                break
            
            elif isinstance(item, ConnectionItem):
                reply = QMessageBox.question(
                    self, "Delete Connection",
                    "Delete this connection?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    # Need to get from/to ids from connection
                    self.network_vm.remove_connection(
                        item.start_node.layer_id, 
                        item.end_node.layer_id
                    )
                break

    # ── project change ────────────────────────────────────────────────────────

    def _on_project_changed(self, project):
        """
        Called whenever a new project is loaded or created.
        Hands the new network to NetworkVM, which then emits
        layer_added / connection_added so the canvas rebuilds itself.
        """
        if project is None:
         self._update_ui_for_no_project()
         return

        self._update_ui_for_project(project) #updating the canvas with the new no canva ui

        self.setWindowTitle(f"DNN Editor — {project.name}")
        self.canvas.clear_canvas()
        self.network_vm.set_network(project.network)
        self.dataset_dock.load_from_project(project)
        self.network_config_dock.load_from_project(project)
        self.results_dock.clear()
        if project.dataset and project.networkConfig:
          self._refresh_code_tab()
        self._update_train_button_state()


    def _on_tab_changed(self, index: int):
        """Refresh code tab when selected."""
        if index == 1:  # Code tab index
            self._refresh_code_tab()


    # ── training lifecycle ────────────────────────────────────────────────────

    def _on_training_started(self):
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._timer.start()
        self.statusBar().showMessage("Training…")

        # DISABLE ALL GRAPH EDITING ACTIONS
        # Palette dock buttons
        self.palette_dock.setEnabled(False)
    
        # Canvas interaction
        self.canvas.setEnabled(False)
    
        # Config docks - disable editing
        self.config_dock.setEnabled(False)
        self.network_config_dock.setEnabled(False)
    
        # Dataset dock - disable changes during training
        self.dataset_dock.setEnabled(False)
    
        # Disable add/delete buttons in toolbar
        for action in self.findChildren(QPushButton):
            if action.text() in ["Add", "Delete"]:
                action.setEnabled(False)
    
        # Disable File menu actions that modify state
        for action in self._file_menu.actions():
            if action.text() in ["New project", "Open"]:
                action.setEnabled(False)


    def _on_training_stopped(self):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._timer.stop()


        # RE-ENABLE ALL UI
        self.palette_dock.setEnabled(True)
        self.canvas.setEnabled(True)
        self.config_dock.setEnabled(True)
        self.network_config_dock.setEnabled(True)
        self.dataset_dock.setEnabled(True)
    
        for action in self.findChildren(QPushButton):
            if action.text() in ["Add", "Delete"]:
                action.setEnabled(True)
    
        for action in self._file_menu.actions():
            if action.text() in ["New project", "Open"]:
                action.setEnabled(True)
    
        self._update_train_button_state()
    #finished the reenabling :P

    def _update_training_time(self):
        elapsed = self.training_vm.elapsed_seconds()
        mins, secs = divmod(int(elapsed), 60)
        self.training_time_label.setText(f"Training Time: {mins:02d}:{secs:02d}")


    def _on_training_finished(self, project):
        #import winsound  # Windows
        # OR for cross-platform:
        # from PyQt6.QtMultimedia import QSound
    
        # Write the updated project back and save
        self.project_vm._project = project
        self.project_vm.save()
        elapsed = self.training_vm.elapsed_seconds()
    
        # Play sound
        try:
            # Windows
            winsound.MessageBeep(winsound.MB_ICONINFORMATION)
            # OR winsound.Beep(1000, 500)  # frequency, duration
        except:
            try:
                # Linux/Mac - simple console beep
                print('\a')
            except:
                pass
    
        # Show notification
        from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
    
        if QSystemTrayIcon.isSystemTrayAvailable():
            tray_icon = QSystemTrayIcon(self)
            tray_icon.show()
            tray_icon.showMessage(
                "Training Complete", 
                f"Finished in {elapsed:.0f} seconds!",
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )
    
        self.statusBar().showMessage(f"Training complete — {elapsed:.0f}s", 5000)

    # ── code tab ─────────────────────────────────────────────────────────────

    def _refresh_code_tab(self):
        """Regenerates the PyTorch script whenever the network changes."""
        project = self.project_vm.project
        if project is None or project.network is None:
            return
        if not project.network.validate():
            return   # don't generate code for an incomplete network
        try:
            from Infra.codeGenerator import PyTorchCodeGenerator
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


    def _update_ui_for_no_project(self):
        """Disables all UI elements that depend on having an active project."""
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(False)

        # Disable menu actions that require an open project
        for menu in [self._file_menu, self._edit_menu, self._train_menu]:
            for action in menu.actions():
                if action.text() not in ["New project", "Open", "Quit", "Exit"]:
                    action.setEnabled(False)

        # Clear the canvas and code tab
        self.canvas.clear_canvas()
        self.code_tab.clear()
        self.code_tab.set_code("")
        self.code_tab.code_editor.setPlaceholderText(
            "# No project loaded.\n"
            "# Use File → New Project or File → Open to get started."
        )

        self.results_dock.clear()
        self.statusBar().showMessage("No project loaded. Create or open a project to begin.")
        self.setWindowTitle("DNN Editor")



    def _update_ui_for_project(self, project):
        """Enables all UI elements when a project is loaded/created."""
        self._update_train_button_state()

        # Re-enable all menu actions
        for menu in [self._file_menu, self._edit_menu, self._train_menu]:
            for action in menu.actions():
                action.setEnabled(True)

        self.setWindowTitle(f"DNN Editor — {project.name}")
        self.statusBar().showMessage(f"Loaded project: {project.name}", 3000)


    def _update_train_button_state(self): 
       """Enable/disable Train button based on project readiness."""
       if self.project_vm.project and self.project_vm.project.isReadyToTrain():
          self.btn_train.setEnabled(True)
          self.btn_train.setToolTip("") 
       else:
        self.btn_train.setEnabled(False)
        # Optionally show tooltip explaining why
        if self.project_vm.project:
            missing = []
            if self.project_vm.project.network is None or not self.project_vm.project.network.validate():
                missing.append("valid network")
            if self.project_vm.project.dataset is None:
                missing.append("dataset selected")
            if self.project_vm.project.trainingConfig is None:
                missing.append("training config")
            if self.project_vm.project.networkConfig is None:
                missing.append("network config")
            if missing:
                self.btn_train.setToolTip(f"Cannot train: missing {', '.join(missing)}")
            else:
                self.btn_train.setToolTip("")
        else:
            self.btn_train.setToolTip("No project loaded")

    #  enhance status message method
    def _show_status(self, message: str, duration_ms: int = 3000):
        """Show status message with optional beep for important messages."""
        self.statusBar().showMessage(message, duration_ms)
        # Optional: beep for important messages
        if "saved" in message.lower() or "complete" in message.lower():
            try:
                winsound.MessageBeep(winsound.MB_ICONINFORMATION)
            except:
                pass

    def closeEvent(self, event):
        """Show confirmation dialog before quitting."""
        if self.training_vm.is_training:
            reply = QMessageBox.question(
                self, "Training in Progress",
                "Training is currently running.\nQuitting will stop training.\nAre you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
    
        if self.project_vm.project:
            reply = QMessageBox.question(
                self, "Exit",
                "Do you want to save before exiting?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._on_save()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
    
        event.accept()

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())