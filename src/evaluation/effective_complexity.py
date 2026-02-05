import numpy as np
import torch

from numpy.typing import NDArray
from src.evaluation.metric import Metric


class EffectiveComplexity(Metric):
    """The effective complexity class includes functions to assess the complexity of uncertainty attributions."""

    name = "EffectiveComplexity"

    def __init__(self) -> None:
        """inits EffectiveComplexity"""

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the metric
        """
        return self.name

    def _compute_single_effective_complexity(
        self, uncertainty_attribution: torch.Tensor, eps: float
    ) -> float:
        """compute the effective complexity, i.e. number of features with uncertainty attribution above a threshold eps, of a single uncertainty attribution.

        Args:
            uncertainty_attribution (torch.Tensor): A single uncertainty attribution to be evaluated
            eps (float): threshold for effective complexity

        Returns:
            float: The computed effective complexity of the uncertainty attribution
        """
        sum = abs(uncertainty_attribution.sum())
        if sum == 0:
            return 0
        uncertainty_attribution = uncertainty_attribution / sum
        effective_complexity = (np.abs(uncertainty_attribution) > eps).sum()
        return effective_complexity

    def _compute_effective_complexity(self, uncertainty_attributions: torch.Tensor) -> list[float]:
        """computes the effective complexity of all uncertainty attributions

        Args:
            uncertainty_attributions (torch.Tensor): A tensor containing multiple uncertainty attributions

        Returns:
            list(float): A list of computed effective complexities for each uncertainty attribution
        """
        nr_features = len(uncertainty_attributions[0])
        eps = 1 / nr_features
        effective_complexities = []
        for uncertainty_attribution in uncertainty_attributions:
            effective_complexity = self._compute_single_effective_complexity(
                uncertainty_attribution, eps
            )
            effective_complexities.append(effective_complexity)
        return effective_complexities

    def evaluate_uncertainty_attributions(
        self, uncertainty_attributions: torch.Tensor, **kwargs
    ) -> tuple[dict, NDArray]:
        """Evaluate uncertainty attributions w.r.t. the effective complexity metric

        Args:
            uncertainty_attributions (torch.Tensor): uncertainty attributions to be evaluated

        Returns:
            dict: A dictionary containing the mean and standard deviation of the effective complexities
            np.ndarray: An array of computed effective complexities for each uncertainty attribution in uncertainty attributions
        """
        attr_dict = {}
        effective_complexities = np.array(
            self._compute_effective_complexity(uncertainty_attributions=uncertainty_attributions)
        )
        attr_dict["mean"] = effective_complexities.mean()
        attr_dict["std"] = effective_complexities.std()
        return attr_dict, effective_complexities
