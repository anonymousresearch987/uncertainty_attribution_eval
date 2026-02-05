from abc import ABC, abstractmethod


class Dataset(ABC):
    """abstract dataset class"""

    @abstractmethod
    def create_data(self):
        pass

    @abstractmethod
    def serve_dataset(self, val_split: float, val_set_required: bool):
        pass

    @abstractmethod
    def serve_dataset_as_folds(
        self, k_folds: int, val_split: float, val_set_required: bool, random_state: int
    ) -> dict:
        pass

    @abstractmethod
    def serve_dataset_as_folds_for_calibration(self, val_split: float, k_folds: int) -> dict:
        pass

    @abstractmethod
    def serve_dataset_as_dataloader(self, val_split: float, batch_size: int) -> tuple:
        pass
