from __future__ import annotations
from .network import Network
from .dataset import Dataset
from .trainingConfig import TrainingConfig
from .networkConfig import NetworkConfig


class Project:

    # standard filenames ; infrastructure uses these, doesn't invent its own
    ARCHITECTURE_FILENAME = "project.json"
    WEIGHTS_FILENAME = "weights.pt"
    LOGS_FILENAME = "training_logs.json"

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path                          # the folder where the 3 files live
        self.network: Network | None = None
        self.dataset: Dataset | None = None
        self.trainingConfig: TrainingConfig | None = None
        self.networkConfig: NetworkConfig | None = None
        self.trainingLogs: dict = {}
        self.weightsPath: str | None = None       # set by infrastructure after saving weights

    # first project.json 
    
    def toDict(self) -> dict:
        """Everything that goes into project.json , architecture + all configs."""
        return {
            "name": self.name,
            "path": self.path,
            "network": self._serializeNetwork(),
            "dataset": self._serializeDataset(),
            "trainingConfig": self._serializeTrainingConfig(),
            "networkConfig": self._serializeNetworkConfig(),
            "trainingLogs": self.trainingLogs, 
        }

    @classmethod
    def fromDict(cls, data: dict) -> Project:
        """Reconstruct a Project from a loaded project.json dict."""
        project = cls(name=data["name"], path=data["path"])
        # each of these will call fromDict on the respective class
        # once Network, Dataset, etc. implement it
        project.network = Network.fromDict(data.get("network", {}))
        project.dataset = Dataset.fromDict(data.get("dataset", {}))
        project.trainingConfig = TrainingConfig.fromDict(data.get("trainingConfig", {}))
        project.networkConfig = NetworkConfig.fromDict(data.get("networkConfig", {}))
        project.trainingLogs = data.get("trainingLogs", {})
        return project

    # training_logs.json 

    def getLogsDict(self) -> dict:
        """Returns the logs dict , infrastructure writes this to training_logs.json."""
        return self.trainingLogs

    def loadLogs(self, data: dict) -> None:
        """Called by infrastructure after reading training_logs.json."""
        self.trainingLogs = data

    def updateLogs(self, history: dict) -> None:
        """Called by the trainer at the end of training with the metrics history."""
        if not isinstance(history, dict):
            raise ValueError("history must be a dict")
        self.trainingLogs = history

    #  weights.pt
    def setWeightsPath(self, path: str) -> None:
        """Infrastructure calls this after it saves the weights file."""
        self.weightsPath = path

    def getWeightsPath(self) -> str | None:
        return self.weightsPath

    #  Validation 

    def isReadyToTrain(self) -> bool:
        if self.network is None or not self.network.validate():
            print("Project not ready to train: network is invalid")
            return False
        if self.dataset is None:
            print("Project not ready to train: dataset is not set")
            return False
        if self.trainingConfig is None:
            print("Project not ready to train: training config is not set")
            return False
        if self.networkConfig is None:
            print("Project not ready to train: network config is not set")
            return False
        return True

    def isReadyToLoad(self) -> bool:
        """Checks that the project has the minimum needed to be useful after loading."""
        return self.network is not None and self.network.validate()

    # Private serialization helpers 
    def _serializeNetwork(self) -> dict:
        if self.network is None:
            return {}
        return self.network.toDict()
        


      

    def _serializeTrainingConfig(self) -> dict:
        if self.trainingConfig is None:
            return {}
        return self.trainingConfig.toDict()

    def _serializeNetworkConfig(self) -> dict:
        if self.networkConfig is None:
            return {}
        return self.networkConfig.toDict()
    
    def _serializeDataset(self) -> dict:
        if self.dataset is None:
         return {}
        return self.dataset.toDict()