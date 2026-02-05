import copy
import torch
import numpy as np
import torch.nn as nn

from numpy.typing import NDArray
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from tqdm import tqdm

from src.evaluation.metric import Metric
from src.uncertainty_attributions.empirical_xuq import EmpiricalXUQGenerator
from src.models.models import NeuralNetworkBase


class Repeatability(Metric):
    """The Repeatability class includes functions to assess the consistency of uncertainty attributions."""

    name: str = "Repeatability"
    nr_samples: int

    def __init__(self, nr_samples=50) -> None:
        """init Repeatability Metric

        Args:
            nr_samples (int, optional): number of samples. Defaults to 50.
        """
        super().__init__()
        self.nr_samples = nr_samples

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the metric
        """
        return self.name

    def calculate_spearman_similarities(self, array1: NDArray, array2: NDArray) -> list[float]:
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
            const1 = np.allclose(a1, a1[0])
            const2 = np.allclose(a2, a2[0])

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

    def set_ensemble_seed(
        self, model: nn.Module, forward_passes: int, seed: int
    ) -> list[nn.Module]:
        """set ensemble seed

        Args:
            model (nn.Module): base model architecture
            forward_passes (int): number of forward passes / ensemble size
            seed (int): seed

        Returns:
            list[nn.Module]: ensemble of models with set seeds
        """
        ensemble = []
        for m in range(forward_passes):
            model.train()
            model_copy = copy.deepcopy(model)
            model_copy.set_internal_seed(m + seed)
            ensemble.append(model_copy)
        self.ensemble = ensemble
        return ensemble

    def similarity_one_seed(
        self,
        uncertainty_attributions: torch.Tensor,
        model: nn.Module,
        X_test: torch.Tensor,
        empirical_xuq_generator: EmpiricalXUQGenerator,
        forward_passes: int,
        pred_tests: torch.Tensor | None,
        seed: int,
    ) -> tuple[list[float], list[float]]:
        """computes the similarity of uncertainty attributions by two different random seeds

        Args:
            uncertainty_attributions (torch.Tensor): uncertainty attributions
            model (nn.Module): model
            X_test (torch.Tensor): test data
            empirical_xuq_generator (EmpiricalXUQGenerator): empirical XUQ generator
            forward_passes (int): number of forward passes
            seed (int): random seed

        Returns:
            tuple[list[float], list[float]]: cosine similarities and spearman similarities
        """
        ensemble = self.set_ensemble_seed(model=model, forward_passes=forward_passes, seed=seed)
        attr, _ = empirical_xuq_generator.compute_uncertainty_attr(
            nr_testsamples=X_test.shape[0], X_test=X_test, ensemble=ensemble, pred_test=pred_tests
        )
        if hasattr(attr, "detach"):
            attr = attr.detach().cpu().numpy()
        else:
            attr = np.asarray(attr)
        cosine_similarities = self.calculate_cosine_similarities(
            self.scale_uncertainty_attributions(uncertainty_attributions.cpu().numpy()),
            self.scale_uncertainty_attributions(attr),
        )
        spearman_similarities = self.calculate_spearman_similarities(
            self.scale_uncertainty_attributions(uncertainty_attributions.cpu().numpy()),
            self.scale_uncertainty_attributions(attr),
        )
        return cosine_similarities, spearman_similarities

    def evaluate_uncertainty_attributions(
        self,
        uncertainty_attributions: torch.Tensor,
        X_test: torch.Tensor,
        empirical_xuq_generator: EmpiricalXUQGenerator,
        uq_model: NeuralNetworkBase,
        forward_passes: int = 50,
        pred_tests: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[dict, tuple[NDArray, NDArray]]:
        """Check if uncertainty attributions are deterministic (i.e., the same across multiple runs).

        Args:
            uncertainty_attributions (torch.Tensor): A tensor including the uncertainty attributions to be evaluated
            X_test (torch.Tensor): Test data
            empirical_xuq_generator (EmpiricalXUQGenerator): empirical uncertainty attribution generator
            uq_model (NeuralNetworkBase): Uncertainty quantification model
            forward_passes (int, optional): Number of forward passes / ensemble size. Defaults to 50.
            pred_tests (torch.Tensor | None, optional): Test set predictions. Required if uq_model

        Returns:
            dict: A dictionary containing the average spearman and cosine similarity across runs
            tuple: A tuple of two np.ndarrays, containing spearman and cosine similarities for each sample
        """
        model = uq_model.model

        forward_passes = len(uq_model.ensemble) if uq_model.ensemble is not None else 0
        similarities_all_cosine = []
        similarities_all_spearman = []
        for i in tqdm(range(self.nr_samples), desc="Computing repeatability"):
            cosine, spearman = self.similarity_one_seed(
                uncertainty_attributions=uncertainty_attributions,
                model=model,
                X_test=X_test,
                empirical_xuq_generator=empirical_xuq_generator,
                forward_passes=forward_passes,
                seed=i,
                pred_tests=pred_tests,
            )
            similarities_all_cosine.append(cosine)
            similarities_all_spearman.append(spearman)
        similarities_all_cosine = np.array(similarities_all_cosine)
        avg_cosine_per_sample = np.mean(similarities_all_cosine, axis=0)
        avg_cosine = np.mean(avg_cosine_per_sample)
        std_cosine = np.std(avg_cosine_per_sample)

        similarities_all_spearman = np.array(similarities_all_spearman)
        avg_spearman_per_sample = np.mean(similarities_all_spearman, axis=0)
        avg_spearman = np.mean(avg_spearman_per_sample)
        std_spearman = np.std(avg_spearman_per_sample)

        attr_dict = {}
        attr_dict["mean_spearman"] = avg_spearman
        attr_dict["std_spearman"] = std_spearman
        attr_dict["mean_cosine"] = avg_cosine
        attr_dict["std_cosine"] = std_cosine

        return attr_dict, (avg_spearman_per_sample, avg_cosine_per_sample)
