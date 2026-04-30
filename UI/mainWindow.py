import platform
import subprocess

try:
    import winsound  # Windows-only
except ImportError:
    winsound = None



import sys
import os


#this to fix all the bullshit about module notfound when importing from another folder that  is not on the same level as the file that imports it
sys.path.insert(0, os.path.dirname(__file__))                      # -> UI/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))     # -> ai_editor/

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QPushButton,
    QTabWidget, QLabel, QFileDialog, QMessageBox, QInputDialog,
    QStackedWidget,                          # added this for aang
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from tabs.network_tab import NetworkTab
from tabs.code_tab import CodeTab
from tabs.inference_tab import OngletInference
from docks.layer_palette_dock import LayerPaletteDock
from docks.layer_config_dock import LayerConfigDock
from docks.network_config_dock import NetworkConfigDock
from docks.dataset_dock import DatasetDock
from docks.results_dock import ResultsDock
from docks.welcome_screen import WelcomeScreen  # added this for aang

from ViewModel.project_vm import ProjectVM
from ViewModel.network_vm import NetworkVM
from ViewModel.training_vm import TrainingVM


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroCraft")
        self.setMinimumSize(1200, 800)
        self.showMaximized()

        # ── ViewModels ────────────────────────────────────────────────────────
        self.project_vm  = ProjectVM(self)
        self.network_vm  = NetworkVM(self)
        self.training_vm = TrainingVM(self)

        # Tracks whether there are unsaved changes.
        # True  → quit dialog shows "save before exiting?"
        # False → close immediately (nothing to lose)
        self._dirty = False

        # ── menu bar ──────────────────────────────────────────────────────────
        menubar = self.menuBar()

        self._file_menu = menubar.addMenu("File")

        act_new = self._file_menu.addAction("New project", self._on_new)
        act_new.setShortcut(QKeySequence.StandardKey.New)          # Ctrl+N / Cmd+N

        act_open = self._file_menu.addAction("Open", self._on_open)
        act_open.setShortcut(QKeySequence.StandardKey.Open)        # Ctrl+O / Cmd+O

        act_save = self._file_menu.addAction("Save", self._on_save)
        act_save.setShortcut(QKeySequence.StandardKey.Save)        # Ctrl+S / Cmd+S

        act_save_as = self._file_menu.addAction("Save As\u2026", self._on_save_as)
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)   # Ctrl+Shift+S / Cmd+Shift+S

        self._edit_menu = menubar.addMenu("Edit")
       # self._edit_menu.addAction("Undo")   # stubs  wired later 
        #self._edit_menu.addAction("Redo")

        window_menu = menubar.addMenu("Window")

        self._train_menu = menubar.addMenu("Train")
        self._act_train = self._train_menu.addAction("Start Training", self._on_train)
        self._act_train.setShortcut(QKeySequence("Ctrl+T"))   # Ctrl+T / Cmd+T cross-platform

        self._act_stop = self._train_menu.addAction("Stop Training", self._on_stop)
        self._act_stop.setShortcut(QKeySequence("Ctrl+Shift+T"))  # Ctrl+Shift+T to stop
        self._act_stop.setEnabled(False)   # only active while training

        menubar.addMenu("About")
        menubar.addMenu("Settings")


        # sounds are played via _play_sound() — no asset file required
        # ── toolbar ───────────────────────────────────────────────────────────
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        toolbar.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        # ── shared button style (matches Connect button, more rounded) ────────
        # All buttons share this base; each overrides only background/border color.
        _BTN_BASE = (
            "QPushButton {{"
            "  background-color: {bg};"
            "  color: {fg};"
            "  padding: 5px 16px;"
            "  margin: 2px;"
            "  border: 1px solid {border};"
            "  border-radius: 8px;"
            "  font-weight: bold;"
            "  font-size: 13px;"
            "}}"
            "QPushButton:hover {{ background-color: {hover}; }}"
            "QPushButton:disabled {{ background-color: #1e1e1e; color: #555; border-color: #333; }}"
        )

        def _btn_style(bg, fg, border, hover):
            return _BTN_BASE.format(bg=bg, fg=fg, border=border, hover=hover)

        # ── NeuroCraft branding label in toolbar (fix 12) ─────────────────────
        brand_label = QLabel("NeuroCraft")
        brand_label.setStyleSheet(
            "QLabel {"
            "  color: #4FC3F7;"
            "  font-family: Consolas, 'Courier New', monospace;"
            "  font-size: 16px;"
            "  font-weight: bold;"
            "  padding: 0px 12px;"
            "  letter-spacing: 1px;"
            "}"
        )

        # ── buttons ───────────────────────────────────────────────────────────
        self.btn_stop    = QPushButton("  Stop")
        self.btn_train   = QPushButton("  Train")
        self.btn_delete  = QPushButton("  Delete")
        self.btn_add     = QPushButton("  Add")
        self.btn_connect = QPushButton("  Connect")
        self.btn_connect.setCheckable(True) 
         #for the training time label
        self.training_time_label = QLabel("Training Time: --")
        toolbar.addWidget(self.training_time_label)

        # Try to load qtawesome icons; fall back to unicode if not installed
        try:
            import qtawesome as qta
            self.btn_train.setIcon(qta.icon("fa5s.play",        color="#ffffff"))
            self.btn_stop.setIcon( qta.icon("fa5s.stop",        color="#ffffff"))
            self.btn_delete.setIcon(qta.icon("fa5s.trash-alt",  color="#ffffff"))
            self.btn_add.setIcon(  qta.icon("fa5s.plus",        color="#ffffff"))
            self.btn_connect.setIcon(qta.icon("fa5s.link",      color="#7ecfda"))
        except Exception:
            # Fallback: prepend unicode symbols
            self.btn_train.setText("▶  Train")
            self.btn_stop.setText("■  Stop")
            self.btn_delete.setText("X Delete")
            self.btn_add.setText("＋  Add")
            self.btn_connect.setText("⛓  Connect")

        # ── per-button styles — muted to fit the dark editor aesthetic ─────────
        self.btn_train.setStyleSheet(
            _btn_style(bg="#162b1c", fg="#aee1ae", border="#2e603a", hover="#2b4934"))
        self.btn_stop.setStyleSheet(
            _btn_style(bg="#2b1515", fg="#e3a8a8", border="#7B3D2F", hover="#512727"))
        self.btn_delete.setStyleSheet(
            _btn_style(bg="#2b1515", fg="#e3a8a8", border="#7B3D2F", hover="#572727"))
        self.btn_add.setStyleSheet(
            _btn_style(bg="#131e2b", fg="#a5cde1", border="#2f4c71", hover="#512727"))

        self._CONNECT_STYLE_OFF = (
            "QPushButton {"
            "  background-color: #111e22; color: #5a9aa8;"
            "  padding: 5px 10px; margin: 2px; border: 1px solid #1e3d45;"
            "  border-radius: 8px; font-weight: bold; font-size: 13px;"
            "}"
            "QPushButton:hover { background-color: #182a30; }"
            "QPushButton:disabled { background-color: #1e1e1e; color: #555; border-color: #333; }"
        )
        self._CONNECT_STYLE_ON = (
            "QPushButton {"
            "  background-color: #0d4a55; color: #7ecfda;"
            "  padding: 5px 10px; margin: 2px; border: 2px solid #1a7a8a;"
            "  border-radius: 8px; font-weight: bold; font-size: 13px;"
            "}"
            "QPushButton:hover { background-color: #0f5a67; }"
        )
        self.btn_connect.setStyleSheet(self._CONNECT_STYLE_OFF)

        self.btn_connect.setToolTip(
            "Connect mode: click-drag from one node to another to link them.\n"
            "Right-click drag always works too."
        )

        # ── wire signals ──────────────────────────────────────────────────────
        self.btn_add.clicked.connect(self._on_add_layer)
        self.btn_delete.clicked.connect(self._on_delete_selected)
        self.btn_connect.clicked.connect(self._on_toggle_connect_mode)
        self.btn_train.clicked.connect(self._on_train)
        self.btn_stop.clicked.connect(self._on_stop)

        self.btn_stop.setEnabled(False)   # only active while training

        # ── keep references for show/hide ─────────────────────────────────────
        # These are the widgets that should be invisible on the welcome screen.
        # training_time_label is included so it also hides/shows with the rest.
        self._project_toolbar_widgets = [
            self.btn_stop, self.btn_train, self.btn_delete,
            self.btn_add, self.btn_connect, self.training_time_label,
        ]

        # ── add to toolbar (RightToLeft layout: last added = rightmost) ──────
        toolbar.addWidget(self.btn_stop)
        toolbar.addWidget(self.btn_train)
        toolbar.addSeparator()
        toolbar.addWidget(self.btn_delete)
        toolbar.addWidget(self.btn_add)
        toolbar.addWidget(self.btn_connect)
        toolbar.addSeparator()

        

        # Hide project-specific widgets immediately — welcome screen shows first.
        # We do this right here so no show/repaint event can undo the hide
        # before _update_ui_for_no_project() runs at the end of __init__.
        for w in [self.btn_stop, self.btn_train, self.btn_delete,
                  self.btn_add, self.btn_connect, self.training_time_label]:
            w.setVisible(False)

        # Brand label goes on the LEFT — toolbar is RightToLeft so addWidget
        # pushes it to the far left.  We use a spacer widget to pin it.
        from PyQt6.QtWidgets import QWidget as _QWidget
        spacer = _QWidget(); spacer.setSizePolicy(
            spacer.sizePolicy().horizontalPolicy(),  # type: ignore[arg-type]
            spacer.sizePolicy().verticalPolicy(),
        )
        from PyQt6.QtWidgets import QSizePolicy
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(brand_label)

        # Timer to update the training time label every second
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_training_time)

        # Status bar for non-blocking operation feedback
        self.statusBar().showMessage("Ready")

        # ── tabs ──────────────────────────────────────────────────────────────
        # added this for aang — tabs live inside a QStackedWidget so the
        # welcome screen (page 0) can replace them when no project is open
        self._central_stack = QStackedWidget()  # added this for aang

        self._welcome = WelcomeScreen()  # added this for aang
        self._welcome.new_project_requested.connect(self._on_new)   # added this for aang
        self._welcome.open_project_requested.connect(self._on_open) # added this for aang

        self.tabs = QTabWidget()
        self.canvas   = NetworkTab(self.network_vm)
        self.code_tab = CodeTab()
        self.inference_tab = OngletInference(self.project_vm)
        self.tabs.addTab(self.canvas, "Network")
        self.tabs.addTab(self.code_tab, "Code")
        self.tabs.addTab(self.inference_tab, "Inference")

        self._central_stack.addWidget(self._welcome)  # added this for aang — page 0
        self._central_stack.addWidget(self.tabs)      # added this for aang — page 1
        self.setCentralWidget(self._central_stack)


        #style for teh tabs! sorry you were too ugly

       # Tab style — matches ResultsDock's _TAB_STYLE exactly
        self.tabs.setStyleSheet("""
    QTabWidget::pane { border: none; background: #2b2b2b; }
    QTabBar::tab {
        background: #393d3f; color: #888888;
        padding: 8px 18px; border: none;
        font-size: 13px; font-weight: bold;
        min-width: 80px;
    }
    QTabBar::tab:selected {
        background: #2b2b2b; color: #cccccc;
        border-bottom: 2px solid #06b6d4;
    }
    QTabBar::tab:hover { color: #cccccc; }
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
        self.palette_dock.setMinimumWidth(200)
        # # self.palette_dock.setMaximumWidth(220)
        self.config_dock.setMinimumHeight(230)  # Make taller
        self.config_dock.setMinimumWidth(190)
        # self.dataset_dock.setMinimumWidth(220)
        self.network_config_dock.setMinimumWidth(200)

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
        self.training_vm.training_started.connect(self.results_dock.clear)  # clear results on every new run
        self.training_vm.training_stopped.connect(self._on_training_stopped)
        self.training_vm.training_finished.connect(self._on_training_finished)
        self.training_vm.training_failed.connect(self._show_error)
        self.training_vm.epoch_progress.connect(self.results_dock.on_epoch)

        # Network changes → regenerate code tab
        self.network_vm.network_changed.connect(self._refresh_code_tab)
        self.network_vm.network_changed.connect(self._mark_dirty)   # unsaved changes
        self.project_vm.project_changed.connect(self._mark_dirty)   # new/loaded project starts dirty
        self.tabs.currentChanged.connect(self._on_tab_changed)  # added this for aang

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
        self._dirty = False   # project is now in sync with disk
        self.statusBar().showMessage("Project saved.", 3000)

    def _on_save_as(self):                                                      #BOOOOOM
        """Save a copy of the project under a new name and/or location."""      #BOOOOOM
        if self.project_vm.project is None:                                     #BOOOOOM
            self._show_error("No project loaded.")                             #BOOOOOM
            return     
        # Ask for a new name first; pre-fill with the current name              #BOOOOOM
        new_name, ok = QInputDialog.getText(                                    #BOOOOOM
            self, "Save As — Project Name",                                   #BOOOOOM
            "New project name:",                                                 #BOOOOOM
            text=self.project_vm.project.name,                                  #BOOOOOM
        )                                                                        #BOOOOOM
        if not ok or not new_name.strip():                                      #BOOOOOM
            return                                                               #BOOOOOM
 
        # Then ask for the parent folder                                         #BOOOOOM
        folder = QFileDialog.getExistingDirectory(                              #BOOOOOM
            self, "Save As — Choose parent folder"                              #BOOOOOM
        )                                                                       #BOOOOOM
        if not folder:                                                           #BOOOOOM
            return                                                               #BOOOOOM
 
        success = self.project_vm.save_as(folder, new_name)                    #BOOOOOM
        if success:                                                             #BOOOOOM
            self._dirty = False                                                 #BOOOOOM
            self.statusBar().showMessage(                                       #BOOOOOM
                f"Project saved as '{new_name}'.", 3000                        #BOOOOOM
            )

    def _mark_dirty(self):
        """Called whenever the network or project changes — flags unsaved work."""
        if self.project_vm.project is not None:
            self._dirty = True

    def _on_train(self):
        if self.project_vm.project is None:
            self._show_error("No project loaded.")
            return
        self.training_vm.start(self.project_vm.project)

    def _on_stop(self):
        self.training_vm.stop()
        self.statusBar().showMessage("Stopping after current epoch…")


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

        self.setWindowTitle(f"NeuroCraft — {project.name}")
        self.canvas.clear_canvas()
        self.network_vm.set_network(project.network)
        self.dataset_dock.load_from_project(project)
        self.network_config_dock.load_from_project(project)
        self.results_dock.clear()
        self.code_tab.clear()

        # Project just came from disk — nothing is unsaved yet
        self._dirty = False
        print("hello from mainwindow project changed")

        if project.dataset and project.networkConfig:
          self._refresh_code_tab()
        self._update_train_button_state()


    def _on_tab_changed(self, index: int):
        """Refresh code tab when selected."""
        if index == 1:  # Code tab index
            self._refresh_code_tab()
        self.results_dock.setVisible(index != 2) # hide results dock when inference tab is selected, since it won't be relevant and takes up space;WAY TOO MUCH AND TEH INFERENC ELOOKS SO SQUEEZED AND it's becuase of my small screen but still i hate how it looks

    def _on_toggle_connect_mode(self) -> None:                              #BOOOOOM
        """
        Toggles the canvas between normal (scroll/select) and connect mode.
        The button is checkable so it stays visually depressed while active.
        """                                                                  #BOOOOOM
        active = self.btn_connect.isChecked()                               #BOOOOOM
        self.canvas.set_connect_mode(active)                                #BOOOOOM
        if active:                                                           #BOOOOOM
            self.btn_connect.setStyleSheet(self._CONNECT_STYLE_ON)          #BOOOOOM
            self.statusBar().showMessage(                                    #BOOOOOM
                "Connect mode ON — click a source node, then click a target node.  "  #BOOOOOM
                "Esc to cancel selection · click Connect again to exit.", 0  #BOOOOOM
            )                                                                #BOOOOOM
        else:                                                                #BOOOOOM
            self.btn_connect.setStyleSheet(self._CONNECT_STYLE_OFF)         #BOOOOOM
            self.statusBar().showMessage("Connect mode OFF.", 2000)          #BOOOOOM
 
    # ── training lifecycle ────────────────────────────────────────────────────

    def _on_training_started(self):
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._act_train.setEnabled(False)   # shortcut disabled while running
        self._act_stop.setEnabled(True)     # shortcut enabled to allow stopping
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
        self.btn_add.setEnabled(False)
        self.btn_delete.setEnabled(False)
    
        # Disable File menu actions that modify state
        for action in self._file_menu.actions():
            if action.text() in ["New project", "Open"]:
                action.setEnabled(False)


    def _on_training_stopped(self):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._act_stop.setEnabled(False)    # can't stop what isn't running
        self._timer.stop()


        # RE-ENABLE ALL UI
        self.palette_dock.setEnabled(True)
        self.canvas.setEnabled(True)
        self.config_dock.setEnabled(True)
        self.network_config_dock.setEnabled(True)
        self.dataset_dock.setEnabled(True)
    
        self.btn_add.setEnabled(True)
        self.btn_delete.setEnabled(True)
    
        for action in self._file_menu.actions():
            if action.text() in ["New project", "Open"]:
                action.setEnabled(True)
    
        self._update_train_button_state()
    #finished the reenabling :P
        # Only show "stopped" if it wasn't a clean finish, just forced to stop by the impatient user

        if self.statusBar().currentMessage().startswith("Stopping"):  # added this for aang
            self.statusBar().showMessage("Training stopped by user.", 4000)  # added this for aang

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

        #here we populate the result dock:

 
        #first the dataset for the confusion matrix
        if project.dataset and project.dataset._builtinType:
            self.results_dock.set_dataset_key(project.dataset._builtinType.name)
        #training logs for the the rest of scores, precicion ect
        if project.trainingLogs:
            self.results_dock.on_training_finished(project.trainingLogs)
    
        # Play sound + notify
        self._play_sound()
        self._show_notification("Training Complete", f"Finished in {elapsed:.0f} seconds!")
    
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

        # Train/Stop shortcuts always disabled with no project
        self._act_train.setEnabled(False)
        self._act_stop.setEnabled(False)

        # Clear the canvas and code tab
        self.canvas.clear_canvas()
        self.code_tab.clear()
        self.code_tab.code_editor.setPlaceholderText(
            "# No project loaded.\n"
            "# Use File → New Project or File → Open to get started."
        )

        self.results_dock.clear()
        self.statusBar().showMessage("No project loaded. Create or open a project to begin.")
        self.setWindowTitle("NeuroCraft")

        # Flip to welcome screen and hide all docks
        self._central_stack.setCurrentWidget(self._welcome)
        for dock in [self.palette_dock, self.config_dock, self.dataset_dock,
                     self.network_config_dock, self.results_dock]:
            dock.hide()
        # Hide project toolbar buttons on welcome screen
        for w in self._project_toolbar_widgets:
            w.setVisible(False)



    def _update_ui_for_project(self, project):
        """Enables all UI elements when a project is loaded/created."""
        self._update_train_button_state()

        # Re-enable all menu actions
        for menu in [self._file_menu, self._edit_menu, self._train_menu]:
            for action in menu.actions():
                action.setEnabled(True)

        # Stop action only active during training — reset it here
        self._act_stop.setEnabled(False)

        self.setWindowTitle(f"NeuroCraft — {project.name}")
        self.statusBar().showMessage(f"Loaded project: {project.name}", 3000)

        # Flip back to the tabs and restore all docks
        self._central_stack.setCurrentWidget(self.tabs)
        for dock in [self.palette_dock, self.config_dock, self.dataset_dock,
                     self.network_config_dock, self.results_dock]:
            dock.show()
        # Restore project toolbar buttons when project is active
        for w in self._project_toolbar_widgets:
            w.setVisible(True)


    def _update_train_button_state(self): 
       """Enable/disable Train button based on project readiness."""
       if self.project_vm.project and self.project_vm.project.isReadyToTrain():
          self.btn_train.setEnabled(True)
          self._act_train.setEnabled(True)
          self.btn_train.setToolTip("") 
       else:
        self.btn_train.setEnabled(False)
        self._act_train.setEnabled(False)
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

    def _play_sound(self) -> None:
        """Play a completion beep cross-platform (no asset file required)."""
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.Popen(
                    ["afplay", "/System/Library/Sounds/Glass.aiff"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif system == "Windows" and winsound:
                winsound.MessageBeep(winsound.MB_ICONINFORMATION)
            else:
                # Linux: try paplay (PulseAudio) then aplay (ALSA)
                for cmd in [
                    ["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                    ["aplay",  "/usr/share/sounds/alsa/Front_Center.wav"],
                ]:
                    if subprocess.call(
                        ["which", cmd[0]],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    ) == 0:
                        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        break
        except Exception:
            pass  # sound is optional — never crash over it

    def _show_notification(self, title: str, message: str) -> None:
        """Show an OS notification cross-platform."""
        try:
            system = platform.system()
            if system == "Darwin":
                # osascript works on every macOS version without extra deps
                script = (
                    f'display notification "{message}" '
                    f'with title "{title}"'
                )
                subprocess.Popen(
                    ["osascript", "-e", script],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif system == "Windows":
                from PyQt6.QtWidgets import QSystemTrayIcon
                if QSystemTrayIcon.isSystemTrayAvailable():
                    tray_icon = QSystemTrayIcon(self)
                    tray_icon.show()
                    tray_icon.showMessage(
                        title, message,
                        QSystemTrayIcon.MessageIcon.Information, 3000,
                    )
            else:
                # Linux: try notify-send (libnotify)
                if subprocess.call(
                    ["which", "notify-send"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ) == 0:
                    subprocess.Popen(
                        ["notify-send", title, message],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
        except Exception:
            pass  # notification is optional — never crash over it

    #  enhance status message method
    def _show_status(self, message: str, duration_ms: int = 3000):
        """Show status message in the status bar, with a beep for key events."""
        self.statusBar().showMessage(message, duration_ms)
        if "saved" in message.lower() or "complete" in message.lower():
            self._play_sound()

    def closeEvent(self, event):
        """Show confirmation dialog before quitting."""
        # 1. If training is running, always warn regardless of dirty state
        if self.training_vm.is_training:
            reply = QMessageBox.question(
                self, "Training in Progress",
                "Training is currently running.\nQuitting will stop training.\nAre you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        # 2. Only ask about saving if there are actual unsaved changes
        if self._dirty and self.project_vm.project is not None:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes.\nSave before exiting?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._on_save()   # _on_save() also sets _dirty = False
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        event.accept()

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())