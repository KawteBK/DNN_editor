class TrainingConfig:

    def __init__(self, batchSize:int, epochs:int, validationSplit:float, shuffle: bool, earlyStop:bool):
        self._batchSize = batchSize
        self._epochs = epochs
        self._validationSplit = validationSplit
        self._shuffle = shuffle
        self._earlyStop = earlyStop


    def toDict(self) -> dict:
        return {
            "batchSize": self._batchSize,
            "epochs": self._epochs,
            "validationSplit": self._validationSplit,
            "shuffle": self._shuffle,
            "earlyStop": self._earlyStop,
        }

    @classmethod
    def fromDict(cls, data: dict) -> "TrainingConfig":
        return cls(
            batchSize=data["batchSize"],
            epochs=data["epochs"],
            validationSplit=data["validationSplit"],
            shuffle=data["shuffle"],
            earlyStop=data["earlyStop"],
        )

