from dataclasses import dataclass
from abc import ABC


@dataclass
class ModelProps(ABC):
    """model properties dataclass"""

    ensemble_path: str
    state_dict_path: str | None
    preprocessor_path: str | None
    target_preprocessor_path: str | None
    drop_prob: float
    forward_passes: int
    is_classification: bool
    output_size: int
    layers_to_perturb: int | None
    force_cpu: bool


@dataclass
class WineQualityModelProps(ModelProps):
    """wine quality model properties dataclass

    Args:
        input size: int, nr of input features
        last hidden layer: int, size of last hidden layer
        hidden layer 1: int, size of hidden layer 1
        hidden layer 2: int | None, size of hidden layer 2
        hidden layer 3: int | None, size of hidden layer 3
        hidden layer 4: int | None, size of hidden layer 4
        hidden layer 5: int | None, size of hidden layer 5
        hidden layer 6: int | None, size of hidden layer 6
        drop prob: float, dropout or dropconnect probability
        force_cpu: bool, whether to force CPU usage"""

    input_size: int
    last_hidden_layer: int
    hidden_layer_1: int
    hidden_layer_2: int | None
    hidden_layer_3: int | None
    hidden_layer_4: int | None
    hidden_layer_5: int | None
    hidden_layer_6: int | None


@dataclass
class MNISTModelProps(ModelProps):
    """wine quality model properties dataclass

    Args:
        last hidden layer: int, size of last hidden layer
        hidden layer 1: int, size of hidden layer 1
        hidden layer 2: int | None, size of hidden layer 2
        hidden layer 3: int | None, size of hidden layer 3
        hidden layer 4: int | None, size of hidden layer 4
        hidden layer 5: int | None, size of hidden layer 5
        hidden layer 6: int | None, size of hidden layer 6
        drop prob: float, dropout or dropconnect probability
        force_cpu: bool, whether to force CPU usage"""

    last_hidden_layer: int
    hidden_layer_1: int
    hidden_layer_2: int | None
    hidden_layer_3: int | None
    hidden_layer_4: int | None
    hidden_layer_5: int | None
    hidden_layer_6: int | None


@dataclass
class ModelTrainingProps:
    """model training properties class

    Args:
        initialisation_strategy: str | None
        epochs: int
        learn_rate: float
        weight_decay: float
        early_stopping_patience: int | None
        lr_scheduler_factor: float
        momentum: float
        optimizer: str
        loss_function: str
        batch_size: int
    """

    initialisation_strategy: str | None
    batch_size: int

    epochs: int
    learn_rate: float
    weight_decay: float

    early_stopping_patience: int | None
    lr_scheduler_factor: float

    momentum: float
    optimizer: str
    loss_function: str
