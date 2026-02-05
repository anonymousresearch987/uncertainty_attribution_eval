from abc import ABC, abstractmethod


class Metric(ABC):
    name: str

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_name(self):
        return self.name

    @abstractmethod
    def evaluate_uncertainty_attributions(self, **kwargs):
        pass
