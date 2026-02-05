from dataclasses import dataclass
from abc import ABC


@dataclass
class TrainingPipelineProps(ABC):
    name: str


@dataclass
class CalibrationPipelineProps(TrainingPipelineProps):
    k_folds: int
    drop_probabilities: list[float]
