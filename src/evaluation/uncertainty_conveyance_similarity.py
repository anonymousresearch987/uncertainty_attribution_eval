import torch
import numpy as np
import torch.nn as nn

from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from numpy.typing import NDArray

from src.models.props import ModelProps
from src.evaluation.metric import Metric
from src.explainer.explainer import Explainer
from src.uncertainty_attributions.analytical_xuq import AnalyticalXUQGenerator


class UncertaintyConveyanceSimilarity(Metric):
    """The UncertaintyConveyanceSimilarity class includes functions to evaluate conveyance of uncertainty attributions by comparing them to their analytical approximation."""

    name = "Uncertainty Conveyance Similarity"

    def __init__(
        self,
    ) -> None:
        """inits UncertaintyConveyanceSimilarity"""
        pass

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the metric
        """
        return self.name

    def calculate_spearmanr_correlation(self, array1: NDArray, array2: NDArray) -> list[float]:
        """Calculates spearman rank correlations between two arrays of vectors.

        Args:
            array1 (np.ndarray): contains first set of vectors
            array2 (np.ndarray): contains second set of vectors

        Returns:
            list: returns a list of spearman rank correlations
        """
        if len(array1[0].shape) > 2:
            for a1, a2 in zip(array1, array2):
                array1 = np.array([a.flatten() for a in array1])
                array2 = np.array([a.flatten() for a in array2])
        correlations = []
        for a1, a2 in zip(array1, array2):
            # check if vectors are constant
            const1 = np.max(a1) == np.min(a1)
            const2 = np.max(a2) == np.min(a2)

            # both vectors have non-zero norm -> compute spearmanr
            if const1 or const2:
                # both constant and identical -> perfect correlation
                if const1 and const2 and np.allclose(a1, a2):
                    correlations.append(1.0)
                    print("Both vectors constant and identical, spearmanr defined as 1.0")
                else:
                    # at least one constant but not identical -> defined as 0
                    correlations.append(0.0)
                    print(
                        "At least one vector constant but not identical, spearmanr defined as 0.0"
                    )
            else:
                # neither constant -> compute spearmanr
                res = spearmanr(a1, a2)[0]
                correlations.append(res)
        return correlations

    def calculate_cosine_similarities(self, array1: NDArray, array2: NDArray) -> list[float]:
        """Calculates cosine similarities between two arrays of vectors.

        Args:
            array1 (np.ndarray): contains first set of vectors
            array2 (np.ndarray): contains second set of vectors

        Returns:
            list: returns a list of cosine similarities
        """
        similarities = []
        for a1, a2 in zip(array1, array2):
            a1 = a1.flatten()
            a2 = a2.flatten()
            n1 = np.linalg.norm(a1)
            n2 = np.linalg.norm(a2)

            # both vectors have non-zero norm -> compute cosine similarity
            if (not np.isclose(n1, 0.0)) and (not np.isclose(n2, 0.0)):
                res = 1 - cosine(a1, a2)
                similarities.append(res)
            else:
                # both zero -> identical (defined as max similarity)
                if np.isclose(n1, 0.0) and np.isclose(n2, 0.0):
                    similarities.append(1.0)
                    print("Both vectors zero, cosine similarity defined as 1.0")
                else:
                    # only one zero -> no correlation
                    similarities.append(0.0)
                    print("One vector zero, cosine similarity defined as 0.0")
        return similarities

    def scale_uncertainty_attributions(self, uncertainty_attributions: NDArray) -> NDArray:
        """scales uncertainty attributions by dividing by the sum of values in each vector

        Args:
            uncertainty_attributions (np.ndarray): uncertainty attributions

        Returns:
            np.ndarray: scaled uncertainty attributions
        """
        scaled = np.zeros_like(uncertainty_attributions)
        for i, vec in enumerate(uncertainty_attributions):
            sum = np.sum(vec)
            if sum == 0:
                scaled[i] = vec
            else:
                scaled[i] = vec / sum
        return scaled

    def get_analytical_approximation(
        self,
        base_model: nn.Module,
        X_test: torch.Tensor,
        pred_tests: torch.Tensor,
        explainer: Explainer,
        uq_strategy: str,
        model_props: ModelProps,
    ) -> torch.Tensor:
        """compute analytical approximations of uncertainty attributions

        Args:
            base_model (nn.Module()): model
            X_test (torch.Tensor): test data
            pred_tests (torch.Tensor): test predictions
            explainer (Explainer): explainer instance
            uq_strategy (str): uncertainty quantification strategy
            model_props (ModelProps): model properties

        Returns:
            torch.Tensor: analytical uncertainty attributions
        """
        analytical_xuq = AnalyticalXUQGenerator(
            explainer,
            uq_strategy,
            drop_prob=model_props.drop_prob,
            layers_to_perturb=model_props.layers_to_perturb,
        )
        analy_uncertainty_attributions, _ = analytical_xuq.compute_uncertainty_attr(
            X_test.size(0), X_test, pred_tests, base_model
        )
        return analy_uncertainty_attributions

    def evaluate_uncertainty_attributions(
        self,
        uncertainty_attributions: torch.Tensor,
        unpacked_dataset: tuple,
        pred_tests: torch.Tensor,
        base_model: nn.Module,
        explainer: Explainer,
        uq_strategy: str,
        nr_testsamples: int,
        model_props: ModelProps,
        **kwargs,
    ):
        """Evaluates the uncertainty attributions by comparing them to an analytical approximation. This tests whether the uncertainty from the model reliably propagates to the uncertainty attributions. We compute the cosine similarity and spearman rank correlation. The closer those values to 1, the better

        Args:
            uncertainty_attributions (torch.Tensor): uncertainty attributions to evaluate
            unpacked_dataset (tuple): unpacked dataset containing test data, unpacked_dataset = (X_train, X_test, X_val, y_train, y_test, y_val)
            pred_tests (torch.Tensor): test set predictions
            base_model (nn.Module()): model without uncertainty quantification (dropout, dropconnect) used for computing the analytical approximation
            explainer (Explainer): explainer instance
            uq_strategy (str): uncertainty quantification strategy (dropout, dropconnect)
            nr_testsamples (int): number of test samples to evaluate
            model_props (ModelProps): model properties
        Returns:
            attr_dict: dictionary of aggregated metrics, including the mean and standard deviation of spearman rank correlations and cosine similarities
            (spearmanr_correlations, cosine_similarities): tuple of lists containing spearman rank correlations and cosine similarities for each sample
        """
        X_test = unpacked_dataset[1][:nr_testsamples]
        analyt_uncertainty_attributions = self.get_analytical_approximation(
            base_model, X_test, pred_tests, explainer, uq_strategy, model_props
        )

        scaled_uncertainty_attributions = self.scale_uncertainty_attributions(
            uncertainty_attributions.detach().numpy()
        )
        scaled_analyt_uncertainty_attributions = self.scale_uncertainty_attributions(
            analyt_uncertainty_attributions.detach().numpy()
        )

        spearmanr_correlations = self.calculate_spearmanr_correlation(
            scaled_uncertainty_attributions,
            scaled_analyt_uncertainty_attributions,
        )
        cosine_similarities = self.calculate_cosine_similarities(
            scaled_uncertainty_attributions,
            scaled_analyt_uncertainty_attributions,
        )
        attr_dict = {}
        attr_dict["spearmanr_mean"] = np.array(spearmanr_correlations).mean()
        attr_dict["spearmanr_std"] = np.array(spearmanr_correlations).std()
        attr_dict["cosine_mean"] = np.array(cosine_similarities).mean()
        attr_dict["cosine_std"] = np.array(cosine_similarities).std()
        return attr_dict, (spearmanr_correlations, cosine_similarities)
