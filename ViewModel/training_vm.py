import time
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from Core.project import Project
from Infra.trainer import PyTorchTrainer


class _TrainingWorker(QObject):
    """
    Runs in a QThread. Never touches any Qt widget directly —
    it only emits signals, which are received by TrainingVM
    on the main thread.
    """

    epoch_done      = pyqtSignal(int, int, float, dict)  # epoch, total, train_loss, val_metrics
    training_done   = pyqtSignal(object)                  # the updated Project
    training_failed = pyqtSignal(str)                     # error message

    def __init__(self, project: Project):
        super().__init__()
        self._project = project
        # threading.Event is thread-safe and doesn't require raising across
        # call boundaries — request_stop() sets it, on_epoch() reads it and
        # returns False to tell the trainer to break the loop cleanly.
        import threading  # added this for aang
        self._stop_event = threading.Event()  # added this for aang

    def request_stop(self) -> None:
        self._stop_event.set()  # added this for aang

    def run(self) -> None:
        # on_epoch is called by PyTorchTrainer after every epoch.
        # Returning False signals "please stop" without raising any exception,
        # so the trainer's own except-block is never triggered by a stop.
        def on_epoch(epoch: int, total: int, train_loss: float, val_metrics: dict) -> bool:  # added this for aang
            if self._stop_event.is_set():  # added this for aang
                return False          # added this for aang — trainer checks this and breaks the loop
            self.epoch_done.emit(epoch, total, train_loss, val_metrics)
            return True               # added this for aang — keep going

        try:
            result = PyTorchTrainer.train(self._project, on_epoch=on_epoch)
            self.training_done.emit(result)
        except Exception as e:
            self.training_failed.emit(str(e))


class TrainingVM(QObject):
    """
    Manages the training lifecycle from the UI's perspective.

    Signals (connect these in MainWindow / ResultsDock)
    -------
    epoch_progress  — (epoch: int, total: int, train_loss: float, val_metrics: dict)
                      Fired after each epoch. ResultsDock uses this to update
                      the live loss/accuracy charts.
    training_finished — (project: Project)
                      Fired when training completes (or is stopped).
                      ProjectVM should call save() after receiving this.
    training_failed — (error: str)
                      Fired on an unhandled exception during training.
    training_started  — ()
                      Fired at the start so the UI can disable the Train button
                      and enable Stop.
    training_stopped  — ()
                      Fired when training ends for any reason (done, stopped,
                      or failed). UI re-enables the Train button.
    """

    epoch_progress    = pyqtSignal(int, int, float, dict)
    training_finished = pyqtSignal(object)
    training_failed   = pyqtSignal(str)
    training_started  = pyqtSignal()
    training_stopped  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _TrainingWorker | None = None
        self._start_time: float = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def is_training(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, project: Project) -> None:
        """
        Launches training in a background QThread.
        Safe to call from MainWindow's Train button handler.
        """
        if self.is_training:
            return   # already running — ignore double-click

        if not project.isReadyToTrain():
            self.training_failed.emit(
                "Project is not ready to train.\n"
                "Make sure you have:\n"
                "  • A valid network (input + output layers)\n"
                "  • A dataset selected\n"
                "  • Training and network config filled in"
            )
            return

        self._thread = QThread()
        self._worker = _TrainingWorker(project)
        self._worker.moveToThread(self._thread)

        # Wire worker signals → TrainingVM signals (crosses thread boundary safely)
        self._worker.epoch_done.connect(self._on_epoch_done)
        self._worker.training_done.connect(self._on_training_done)
        self._worker.training_failed.connect(self._on_training_failed)

        # Start worker when thread starts
        self._thread.started.connect(self._worker.run)

        self._start_time = time.time()
        self._thread.start()
        self.training_started.emit()

    def stop(self) -> None:
        """
        Requests the training loop to stop after the current epoch.
        Non-blocking — the thread finishes on its own and emits training_finished.
        """
        if self._worker is not None:
            self._worker.request_stop()

    # ── private slots (run on main thread via Qt signal dispatch) ─────────────

    def _on_epoch_done(self, epoch: int, total: int,
                       train_loss: float, val_metrics: dict) -> None:
        self.epoch_progress.emit(epoch, total, train_loss, val_metrics)

    def _on_training_done(self, project: Project) -> None:
        self._cleanup_thread()
        self.training_finished.emit(project)
        self.training_stopped.emit()

    def _on_training_failed(self, error: str) -> None:
        self._cleanup_thread()
        self.training_failed.emit(error)
        self.training_stopped.emit()

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    def elapsed_seconds(self) -> float:
        """Returns seconds since training started. Used by the toolbar timer label."""
        if self._start_time == 0.0:
            return 0.0
        return time.time() - self._start_time