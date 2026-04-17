import os
from PyQt6.QtCore import QObject, pyqtSignal

from Core.project import Project
from Core.projectManager import ProjectManager
from Core.network import Network
from Core.dataset import Dataset, BuiltinDataset
from Core.trainingConfig import TrainingConfig
from Core.networkConfig import NetworkConfig, OptimizerType, LossType, MetricType
from Core.layer import InputLayer


class ProjectVM(QObject):
    """
    Owns the single Project instance for the whole application.

    Signals
    -------
    project_changed   — emitted whenever a new project is loaded or created.
                        All other VMs and docks connect to this to refresh
                        their state from the new project.
    error_occurred    — emitted when an operation fails. MainWindow connects
                        this to a QMessageBox so errors surface to the user.
    """

    project_changed = pyqtSignal(object)   # passes the new Project
    error_occurred  = pyqtSignal(str)      # passes an error message string

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Project | None = None

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def project(self) -> Project | None:
        return self._project

    def new_project(self, name: str, path: str) -> None:
        """
        Creates a blank project with an empty network.
        The network has no layers yet — the user builds it on the canvas.
        """
        project = Project(name=name, path=path)
        project.network = Network(id=name)


        # Create default InputLayer with placeholder shape
        # Shape (1, 28, 28) works for MNIST/Fashion-MNIST/KMNIST
        # Will be updated when user selects a different dataset
        default_shape = (1, 28, 28)
        input_layer = InputLayer(inputShape=default_shape)
        project.network.addLayer(input_layer)

        # Sensible defaults so the config docks have something to show bilama
        project.trainingConfig = TrainingConfig(
            batchSize=32,
            epochs=10,
            validationSplit=0.2,
            shuffle=True,
            earlyStop=True,
        )
        project.networkConfig = NetworkConfig(
            optimizer=OptimizerType.ADAM,
            loss=LossType.CROSS_ENTROPY,
            metric=[MetricType.ACCURACY],
            learningRate=0.001,
        )
        # Dataset left as None — user picks it in the dataset dock

        # SAVE IMMEDIATELY to disk
        try:
            os.makedirs(path, exist_ok=True)
            ProjectManager.save(project)  #to actuelly write on the disk
        except Exception as e:
            self.error_occurred.emit(f"Failed to create project: {e}")
            return

        self._project = project
        self.project_changed.emit(self._project)

    def save(self) -> None:
        """Saves the current project to its path. Emits error_occurred on failure."""
        if self._project is None:
            self.error_occurred.emit("No project to save.")
            return
        try:
            ProjectManager.save(self._project)
        except Exception as e:
            self.error_occurred.emit(f"Save failed: {e}")

    def load(self, project_dir: str) -> None:
        """
        Loads a project from a directory. On success emits project_changed.
        On failure emits error_occurred and leaves the current project untouched.
        """
        try:
            project = ProjectManager.load(project_dir)
            self._project = project
            self.project_changed.emit(self._project)
        except FileNotFoundError:
            self.error_occurred.emit(
                f"No project.json found in:\n{project_dir}"
            )
        except Exception as e:
            self.error_occurred.emit(f"Load failed: {e}")

    def set_dataset(self, builtin_type: BuiltinDataset) -> None:
        """Called by DatasetDock when the user picks a dataset."""
        if self._project is None:
            return
        dataset = Dataset.fromBuiltin(builtin_type)
        self._project.dataset = dataset

           # Update InputLayer shape to match the selected dataset
        input_layer = self._project.network.layers.get("input")
        if input_layer and isinstance(input_layer, InputLayer):
            new_shape = dataset._inputShape
            input_layer._outputShape = new_shape
            input_layer._inputShape = new_shape
            
            # Re-propagate shapes through the entire network
            try:
                self._project.network.propagate_shapes()
            except Exception as e:
                self.error_occurred.emit(f"Shape propagation error: {e}")

    def set_training_config(self, batch_size: int, epochs: int,
                             val_split: float, shuffle: bool,
                             early_stop: bool) -> None:
        """Called by NetworkConfigDock when training fields change."""
        if self._project is None:
            return
        self._project.trainingConfig = TrainingConfig(
            batchSize=batch_size,
            epochs=epochs,
            validationSplit=val_split,
            shuffle=shuffle,
            earlyStop=early_stop,
        )

    def set_network_config(self, optimizer: OptimizerType, loss: LossType,
                            metrics: list[MetricType],
                            learning_rate: float) -> None:
        """Called by NetworkConfigDock when optimizer/loss fields change."""
        if self._project is None:
            return
        self._project.networkConfig = NetworkConfig(
            optimizer=optimizer,
            loss=loss,
            metric=metrics,
            learningRate=learning_rate,
        )

    def is_ready_to_train(self) -> bool:
        return self._project is not None and self._project.isReadyToTrain()
