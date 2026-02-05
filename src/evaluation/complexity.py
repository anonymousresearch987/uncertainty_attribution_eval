import torch
import numpy as np

from numpy.typing import NDArray
from scipy.stats import entropy
from tqdm import tqdm
from src.evaluation.metric import Metric


class Complexity(Metric):
    """The complexity class includes functions to assess the compactness of uncertainty attributions.
    Implementation of Complexity metric by Bhatt et al., 2020.

    References:
        Umang Bhatt et al.: "Evaluating and aggregating feature-based model explanations." IJCAI (2020): 3016-3022.
    """

    name = "Complexity"

    def __init__(self) -> None:
        """inits Complexity"""
        pass

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the metric
        """
        return self.name

    def _compute_single_complexity(self, uncertainty_attribution: torch.Tensor) -> float:
        """compute the complexity (entropy) of a single uncertainty attribution.

        Args:
            uncertainty_attribution (torch.Tensor): A single uncertainty attribution to be evaluated

        Returns:
            float: The computed complexity (entropy) of the uncertainty attribution
        """
        attribution_sum = abs(uncertainty_attribution.sum())
        if attribution_sum == 0:
            return 0.0
        uncertainty_attribution = uncertainty_attribution / attribution_sum
        complexity = float(entropy(uncertainty_attribution))
        return complexity

    def _compute_complexity(self, uncertainty_attributions: torch.Tensor) -> list[float]:
        """computes the complexity of all uncertainty attributions

        Args:
            uncertainty_attributions (torch.Tensor): A tensor containing multiple uncertainty attributions

        Returns:
            list(float): A list of computed complexities for each uncertainty attribution
        """
        complexities = []
        for uncertainty_attribution in tqdm(
            uncertainty_attributions, desc="Computing Complexity Metric"
        ):
            complexity = self._compute_single_complexity(uncertainty_attribution)
            complexities.append(complexity)
        return complexities

    def evaluate_uncertainty_attributions(
        self, uncertainty_attributions: torch.Tensor, **kwargs
    ) -> tuple[dict, NDArray]:
        """Evaluate uncertainty attributions w.r.t. the complexity metric

        Args:
            uncertainty_attributions (torch.Tensor): uncertainty attributions to be evaluated

        Returns:
            dict: A dictionary containing the mean and standard deviation of the complexities
            np.ndarray: An array of computed complexities for each uncertainty attribution in uncertainty attributions
        """
        attr_dict = {}
        if uncertainty_attributions.dim() == 4:
            uncertainty_attributions = uncertainty_attributions.reshape(
                uncertainty_attributions.size(0), -1
            )
        elif uncertainty_attributions.dim() == 3:
            uncertainty_attributions = uncertainty_attributions.reshape(1, -1)
        complexities = np.array(
            self._compute_complexity(uncertainty_attributions=uncertainty_attributions)
        )
        attr_dict["mean"] = complexities.mean()
        attr_dict["std"] = complexities.std()
        return attr_dict, complexities
