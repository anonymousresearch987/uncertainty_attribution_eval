from abc import ABC, abstractmethod

import torch


class XUQGenerator(ABC):
    @abstractmethod
    def __init__() -> None:
        pass

    @abstractmethod
    def compute_uncertainty_attr(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        pass
