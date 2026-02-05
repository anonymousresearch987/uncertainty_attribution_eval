from typing import Optional
import numpy as np
from tqdm import tqdm
from src.uncertainty_attributions.jacobian_dropconnect import JacobianGeneratorDropConnect
from src.uncertainty_attributions.jacobian_dropout import JacobianGeneratorDropout
from src.uncertainty_attributions.xuqgenerator import XUQGenerator
from src.explainer.explainer import Explainer

import torch
import torch.nn as nn


class AnalyticalXUQGenerator(XUQGenerator):
    """Class to generate analytical approximations of uncertainty attributions"""

    def __init__(
        self,
        explainer: Explainer,
        uq_strategy: str,
        drop_prob: float = 0.1,
        layers_to_perturb: Optional[int] = None,
    ) -> None:
        """inits AnalyticalXUQGenerator Object

        Args:
            explainer (Explainer): explainer instance
            uq_strategy (str): uncertainty quantification strategy, dropout or dropconnect
            drop_prob (float, optional): dropout/dropconnect probability. Defaults to 0.1.
            layers_to_perturb (int, optional): layers to perturb. Defaults to None.
        """
        self.explainer = explainer
        self.uq_strategy = uq_strategy
        self.drop_prob = drop_prob
        self.layers_to_perturb = layers_to_perturb

    def compute_single_uncertainty_attributions(
        self,
        ex_ind: int,
        x_test: torch.Tensor,
        pred_test: torch.Tensor | None,
        model: nn.Module,
        drop_prob: float = 0.1,
    ):
        """compute uncertainty attributions for individual sample

        Args:
            ex_ind (int): index of test sample
            x_test (torch.Tensor): test set
            pred_test (torch.Tensor | None): test predictions if given
            model (nn.Module): trained model
            drop_prob (float, optional): dropout/dropconnect probability. Defaults to 0.1.

        Returns:
            np.ndarray: uncertainty attribution
        """
        ex_row = x_test[ex_ind]

        target_row = pred_test[ex_ind] if pred_test is not None else None

        if isinstance(ex_row, torch.Tensor):
            feat_vec = ex_row.detach()
        else:
            feat_vec = torch.as_tensor(ex_row, dtype=torch.float32)

        explainer_input = feat_vec.unsqueeze(0) if feat_vec.ndim == 1 else feat_vec

        # generate explanation as reference

        model.eval()
        ex_exp = None
        if isinstance(self.explainer, Explainer):
            ex_exp = self.explainer.get_feature_attributions(model, explainer_input, target_row)

        if self.uq_strategy == "dropout":
            JacobianGenerator = JacobianGeneratorDropout(
                self.explainer, drop_prob=drop_prob, layers_to_perturb=self.layers_to_perturb
            )
        elif self.uq_strategy == "dropconnect":
            JacobianGenerator = JacobianGeneratorDropConnect(
                self.explainer, drop_prob=drop_prob, layers_to_perturb=self.layers_to_perturb
            )
        else:
            raise ValueError(f"Unknown uq_strategy {self.uq_strategy}")
        assert ex_exp is not None, "Explanation must be computed before approximating jacobian."
        jacobian = JacobianGenerator.approximate_jacobian(model, feat_vec, target_row, ex_exp)
        delta = JacobianGenerator.compute_delta(model, feat_vec)

        # jacobian expected shape: (num_nodes_or_weights, num_features)
        if len(x_test.shape) > 2:
            num_features = np.prod(x_test.shape[1:])
            jacobian_size = jacobian.reshape(jacobian.shape[0], -1).shape
        else:
            num_features = x_test.shape[1]
            jacobian_size = jacobian.shape
            if len(jacobian_size) > 2:
                jacobian = jacobian.squeeze()
                jacobian_size = jacobian.shape
        assert jacobian_size[0] == delta.shape[0], "Jacobian columns and delta size must match."
        assert jacobian_size[1] == num_features, "Jacobian rows and feature size must match."

        if len(jacobian.shape) > 2:
            jacobian = jacobian.reshape(jacobian.shape[0], -1)
            Cov = (jacobian.numpy().T * (drop_prob * (1 - drop_prob))) @ delta @ (jacobian.numpy())
            uncertainty_attribution = np.diag(Cov).reshape(ex_row.shape)
        else:
            Cov = (jacobian.numpy().T * (drop_prob * (1 - drop_prob))) @ delta @ (jacobian.numpy())
            uncertainty_attribution = np.diag(Cov)
        return uncertainty_attribution

    def compute_uncertainty_attr(
        self,
        nr_testsamples: int,
        X_test: torch.Tensor,
        pred_test: torch.Tensor | None,
        model: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute uncertainty attributions for whole test set

        Args:
            uq_strategy (str): "dropout" or "dropconnect"
            nr_testsamples (int): number of explanation samples
            x_test (torch.Tensor): test data whole
            pred_test (torch.Tensor): target predictions
            model (nn.Module): trained model

        Returns:
           np.ndarray: uncertainty attributions for each sample in the test set
        """
        uncertainty_attributions = []

        for i in tqdm(range(nr_testsamples), desc="Approximating Jacobians"):
            uncertainty_attribution = self.compute_single_uncertainty_attributions(
                i, X_test, pred_test, model
            )
            uncertainty_attributions.append(uncertainty_attribution)
        uncertainty_attributions = np.array(uncertainty_attributions)
        return torch.Tensor(uncertainty_attributions), torch.Tensor()
