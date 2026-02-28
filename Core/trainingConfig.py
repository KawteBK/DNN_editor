class TrainingConfig:

    def __init__(self, batchSize:int, epochs:int, validationSplit:float, shuffle: bool, earlyStop:bool):
        self._batchSize = batchSize
        self._epochs = epochs
        self._validationSplit = validationSplit
        self._shuffle = shuffle
        self._earlyStop = earlyStop

