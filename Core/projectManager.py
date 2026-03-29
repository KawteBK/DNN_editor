import os
import json

from .project import Project


class ProjectManager:

    # ── public ────────────────────────────────────────────────────────────────

    @staticmethod
    def save(project: Project) -> None:
        """
        Persists all three project files to project.path:
          - project.json        → architecture + configs + training logs
          - training_logs.json  → metric history (duplicate for easy access)
          - weights.pt          → NOT written here; the trainer owns that file.
                                  This method only records the path if already set.

        Call this after PyTorchTrainer.train() has finished so that
        trainingLogs and weightsPath are already populated.
        """
        os.makedirs(project.path, exist_ok=True)

        # ── project.json ──────────────────────────────────────────────────────
        architecture_path = os.path.join(project.path, Project.ARCHITECTURE_FILENAME)
        with open(architecture_path, "w", encoding="utf-8") as f:
            json.dump(project.toDict(), f, indent=2)

        # ── training_logs.json ────────────────────────────────────────────────
        logs = project.getLogsDict()
        if logs:
            logs_path = os.path.join(project.path, Project.LOGS_FILENAME)
            with open(logs_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)

        # now for the script.py to save the whole script to be executed without needing the editor
        SCRIPT_FILENAME = "script.py"
        script_path = os.path.join(project.path, SCRIPT_FILENAME)
        with open(script_path, "w", encoding="utf-8") as f:
            from Infra.codeGenerator import PyTorchCodeGenerator
            code = PyTorchCodeGenerator.generate(project)
            f.write(code)
           


    # ── load ──────────────────────────────────────────────────────────────────

    @staticmethod
    def load(project_dir: str) -> Project:
        """
        Reconstructs a Project from a saved project directory.

        Reads project.json first (architecture + configs).
        Then tries to load training_logs.json if it exists.
        Sets weightsPath if weights.pt is present in the directory.

        Raises FileNotFoundError if project.json is missing.
        """
        architecture_path = os.path.join(project_dir, Project.ARCHITECTURE_FILENAME)

        if not os.path.isfile(architecture_path):
            raise FileNotFoundError(
                f"No project.json found in '{project_dir}'. "
                "Make sure the path points to a saved project folder."
            )

        with open(architecture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        project = Project.fromDict(data)

        # ── restore logs if present ───────────────────────────────────────────
        logs_path = os.path.join(project_dir, Project.LOGS_FILENAME)
        if os.path.isfile(logs_path):
            with open(logs_path, "r", encoding="utf-8") as f:
                project.loadLogs(json.load(f))

        # ── restore weights path if file exists ───────────────────────────────
        weights_path = os.path.join(project_dir, Project.WEIGHTS_FILENAME)
        if os.path.isfile(weights_path):
            project.setWeightsPath(weights_path)

        return project

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def exists(project_dir: str) -> bool:
        """Returns True if the directory contains a valid project.json."""
        return os.path.isfile(
            os.path.join(project_dir, Project.ARCHITECTURE_FILENAME)
        )

    @staticmethod
    def has_weights(project_dir: str) -> bool:
        """Returns True if weights.pt is present in the project directory."""
        return os.path.isfile(
            os.path.join(project_dir, Project.WEIGHTS_FILENAME)
        )

    @staticmethod
    def has_logs(project_dir: str) -> bool:
        """Returns True if training_logs.json is present in the project directory."""
        return os.path.isfile(
            os.path.join(project_dir, Project.LOGS_FILENAME)
        )