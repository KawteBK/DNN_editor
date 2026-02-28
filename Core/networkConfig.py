from enum import Enum

class OptimizerType(Enum):
    SGD="sgd"
    ADAM="adam"
    RMSPROP = "rmsprop"

class LossType(Enum):
    MSE="mse"
    CROSS_ENTROPY="cross_entropy"
    BCE="bce"


class MetricType(Enum):
    ACCURACY="accuracy"
    PRECISION="precision"
    RECALL="recall"
    F1_SCORE="f1_score"


class NetworkConfig:
    def __init__(self, optimizer:OptimizerType, loss:LossType, metric:list[MetricType] , learningRate:float):
        self._optimizer = optimizer
        self._loss = loss
        self._metric = metric
        self._learningRate=learningRate