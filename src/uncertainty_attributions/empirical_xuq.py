import numpy as np
from src.uncertainty_attributions.xuqgenerator import XUQGenerator
from src.explainer.explainer import Explainer, InputXGradientExplainer
import torch
import torch.nn as nn


class EmpiricalXUQGenerator(XUQGenerator):
    """Class to generate empricial uncertainty attributions based on the method proposed by Bley et al. 2025.


    References:
        Bley, F., Lapuschkin, S., Samek, W., & Montavon, G. (2025). Explaining predictive uncertainty by exposing second-order effects. Pattern Recognition, 160, 111171.
    """

    def __init__(self, explainer: Explainer, uq_strategy: str, mc_passes: int = 50) -> None:
        """inits AnalyticalXUQGenerator Object

        Args:
            explainer (Explainer): explainer instance
            uq_strategy (str): uncertainty quantification strategy, dropout or dropconnect
            mc_passes (int, optional): number of forward passes for MC methods. Defaults to 50.
        """
        self.explainer = explainer
        self.uq_strategy = uq_strategy
        self.mc_passes = mc_passes

    def compute_ensemble_explanations(
        self, X_test: torch.Tensor, pred_test: torch.Tensor | None, ensemble: list[nn.Module]
    ) -> torch.Tensor:
        """function to compute explanations for all models in the ensemble

        Args:
            X_test (torch.Tensor):  dataset for which explanations are computed
            pred_test (torch.Tensor | None): model predictions for X_test if given
            ensemble (list[nn.Module]): list of models in the ensemble

        Raises:
            ValueError: No feature attributions returned by explainer
            Exception: Explainer method not implemented

        Returns:
            torch.Tensor: list of model explanations with shape (nr_models, nr_samples, nr_features)
        """

        shape = X_test.shape
        model_explanations = torch.zeros(size=(len(ensemble), *shape))
        for i, model in enumerate(ensemble):
            model.train()
            if isinstance(self.explainer, Explainer):
                explanation = self.explainer.get_feature_attributions(
                    model=model, inputs=X_test, targets=pred_test
                )
                if explanation is not None:
                    model_explanations[i] = explanation
                else:
                    raise ValueError("Explainer returned None for feature attributions.")
            else:
                raise Exception(f"The explanation method {self.explainer} is not implemented")

        if isinstance(self.explainer, InputXGradientExplainer):
            model_explanations = model_explanations.detach().numpy()
            model_explanations = torch.from_numpy(model_explanations)

        return model_explanations

    def compute_uncertainty_attributions(
        self, nr_testsamples: int, model_explanations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """function to compute uncertainty attributions

        Args:
            nr_testsamples (int):   number of test samples
            model_explanations (torch.Tensor):  model explanations with shape (nr_models, nr_samples, nr_features)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:  uncertainty attributions and mean explanations
        """

        Cov = []
        mean_explanations = []
        for n in range(nr_testsamples):
            explanations_n = model_explanations[:, n, :]
            mean_n = explanations_n.mean(dim=0)
            mean_explanations.append(mean_n)
            if len(explanations_n.shape) <= 2:
                cov_n = torch.cov(explanations_n.T, correction=0)

                Cov.append(cov_n)
            else:
                cov_n = np.cov(explanations_n.reshape((self.mc_passes, -1)).T)
                covariance_diags = np.copy(np.diag(cov_n).reshape(mean_n.shape))
                Cov.append(torch.from_numpy(covariance_diags))

        Covariance_matrix = torch.stack(Cov)
        # as proposed in Bley et al., we can simplify the covariance matrix to the diagonal entries to get the uncertainty attributions
        if len(model_explanations[:, 0, :].shape) <= 2:
            uncertainty_attributions = Covariance_matrix.diagonal(dim1=-2, dim2=-1)
        else:
            uncertainty_attributions = Covariance_matrix
        mean_explanations = torch.Tensor(np.array(mean_explanations))
        return uncertainty_attributions, mean_explanations

    def compute_uncertainty_attr(
        self,
        nr_testsamples: int,
        X_test: torch.Tensor,
        pred_test: torch.Tensor | None,
        ensemble: list[nn.Module],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """computes the uncertainty attributions and mean attributions, i.e. the mean of all feature attributions per sample, of the given test set

        Args:
            nr_testsamples (int): nr of test samples
            X_test (torch.Tensor): test set
            pred_test (torch.Tensor | None): model predictions for X_test if given
            ensemble (list[nn.Module]): ensemble based on uq_strategy

        Returns:
            (uncertainty_attributions, mean_explanations): A tensor of uncertainty attributions of all test samples in X_test and a tensor of mean explanations of all test samples in X_test
        """
        model_explanations = self.compute_ensemble_explanations(X_test, pred_test, ensemble)
        uncertainty_attributions, mean_explanations = self.compute_uncertainty_attributions(
            nr_testsamples=nr_testsamples, model_explanations=model_explanations
        )
        return uncertainty_attributions, mean_explanations
