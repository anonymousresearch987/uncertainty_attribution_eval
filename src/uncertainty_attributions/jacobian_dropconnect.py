import copy
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from src.explainer.explainer import Explainer
from src.uncertainty_attributions.jacobian import JacobianGenerator


class JacobianGeneratorDropConnect(JacobianGenerator):
    """Class to generate Jacobian matrices of explanations with respect to model weights (approximating DropConnect)."""

    layers_to_perturb: Optional[int] = None

    def __init__(
        self,
        explainer: Explainer,
        drop_prob: float = 0.1,
        layers_to_perturb: Optional[int] = 1,
    ) -> None:
        """inits JacobianGeneratorDropConnect

        Args:
            explainer (Explainer): explainer instance
            drop_prob (float, optional): dropout/dropconnect probability. Defaults to 0.1.
            layers_to_perturb (Optional[int], optional): number of layers to perturb. Defaults to 1.
        """
        super().__init__()
        self.explainer = explainer
        self.drop_prob = drop_prob
        self.layers_to_perturb = layers_to_perturb

    def get_weights(self, model: nn.Module) -> torch.Tensor:
        """Collect flattened weights from the last `layers_to_perturb` hidden linear layers. If layers_to_perturb is None, collect from all hidden linear layers.

        Args:
            model (nn.Module): PyTorch model

        Returns:
            torch.Tensor: 1D tensor with concatenated flattened weights
        """

        selected_layers = [
            layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)
        ]
        if not selected_layers:
            raise AssertionError("Model must have at least one linear layer.")

        layers_to_perturb = getattr(self, "layers_to_perturb", None)
        if layers_to_perturb is not None:
            if not isinstance(layers_to_perturb, int):
                try:
                    n = int(layers_to_perturb)
                except Exception:
                    raise TypeError(
                        "layers_to_perturb must be an int or convertible to int"
                    )
            else:
                n = layers_to_perturb
            if n <= 0:
                raise ValueError("layers_to_perturb must be >= 1")
            # select last n hidden layers (if n > available, take all hidden layers)
            selected_layers = (
                selected_layers[-n:] if n <= len(selected_layers) else selected_layers
            )
        flat_parts = [layer.weight.detach().flatten() for layer in selected_layers]
        if not flat_parts:
            raise AssertionError("No layers selected to collect weights from.")

        return torch.cat(flat_parts)

    def set_weight_by_index(
        self, model: nn.Module, flat_weights: torch.Tensor
    ) -> nn.Module:
        """
        Set weights of the selected hidden linear layers (last `layers_to_perturb` internal layers
            or all internal layers if layers_to_perturb is None) from a flattened tensor.

        Args:
            model (nn.Module): PyTorch model
            flat_weights (torch.Tensor): flattened weights to set (1D)
        Returns:
            model with modified weights (in-place)
        """
        selected_layers = [
            layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)
        ]
        if not selected_layers:
            raise AssertionError("Model must have at least one linear layer.")
        if len(selected_layers) == 0:
            raise AssertionError(
                "Model has no hidden linear layers to set weights for."
            )
        layers_to_perturb = getattr(self, "layers_to_perturb", None)

        if layers_to_perturb is not None:
            if not isinstance(layers_to_perturb, int):
                try:
                    n = int(layers_to_perturb)
                except Exception:
                    raise TypeError(
                        "layers_to_perturb must be an int or convertible to int"
                    )
            else:
                n = layers_to_perturb
            if n <= 0:
                raise ValueError("layers_to_perturb must be >= 1")
            # select last n hidden layers (if n > available, take all hidden layers)
            selected_layers = (
                selected_layers[-n:] if n <= len(selected_layers) else selected_layers
            )

        total_elems = sum(layer.weight.numel() for layer in selected_layers)
        if flat_weights.numel() != total_elems:
            raise ValueError(
                f"Mismatch in elements: expected {total_elems}, but got {flat_weights.numel()}"
            )
        flat_weights = flat_weights.detach()
        # distribute values into each layer in order (last-hidden-layers order preserved)
        idx = 0
        with torch.no_grad():
            for layer in selected_layers:
                n = layer.weight.numel()
                part = flat_weights[idx : idx + n]
                part_view = (
                    part.view_as(layer.weight)
                    .to(layer.weight.device, dtype=layer.weight.dtype)
                    .clone()
                )
                layer.weight.data.copy_(part_view)
                idx += n

        return model

    def get_perturbed_explanations(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor | None,
        epsilon: float = 1e-1,
    ) -> torch.Tensor:
        """
        Perturb model weights to get explanations to approximate explanation Jacobian.

        Args:
            model: PyTorch model (assumes final layer is nn.Linear)
            ex_row: input sample (torch.Tensor)
            target: target label (torch.Tensor)
            epsilon: perturbation step size
        Returns:
            J: numpy array of shape (#features, #weights)
        """
        with torch.no_grad():
            base_weights = self.get_weights(model).detach().clone()
            N = base_weights.shape[0]

        pert_matrix = None
        for i in range(N):
            perturbed_weights = base_weights.clone()
            perturbed_weights[i] += epsilon
            perturbed_model = copy.deepcopy(model)
            perturbed_model = self.set_weight_by_index(
                perturbed_model, perturbed_weights
            )

            expl_inp = torch.Tensor(input).unsqueeze(0)
            explanations = self.explainer.get_feature_attributions(
                perturbed_model, expl_inp, target
            )

            if isinstance(explanations, torch.Tensor):
                expl_flat = explanations.detach().cpu().numpy().flatten()
            else:
                expl_flat = np.asarray(explanations).flatten()

            if pert_matrix is None:
                pert_matrix = np.zeros((N, expl_flat.shape[0]), dtype=float)

            pert_matrix[i, :] = expl_flat

        return torch.Tensor(pert_matrix)

    def compute_delta(self, model, input):
        """compute delta with repect to nodes for dropout, with respect to weights for dropconnect

        Args:
            model (nn.Module): trained model
            input (torch.Tensor): test data

        Returns:
            np.ndarray: delta matrix
        """
        weights = self.get_weights(model).detach().cpu().numpy()
        delta = np.diag(np.power(weights, 2))
        return delta

    def approximate_jacobian(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor | None,
        explanation: torch.Tensor,
        epsilon: float = 1e-1,
    ) -> torch.Tensor:
        """approximate the jacobian of explanations with respect to model weights

        Args:
            model (nn.Module): trained model
            input (torch.Tensor): test data
            target (torch.Tensor | None): target labels
            explanation (torch.Tensor): explanation
            epsilon (float, optional): perturbation step size. Defaults to 1e-1.

        Returns:
            torch.Tensor: jacobian matrix
        """
        device = next(model.parameters()).device

        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32, device=device)

        perturbated_explanations = self.get_perturbed_explanations(
            model, input, target, epsilon
        )

        # ensure explanation is a torch tensor and flattened
        if isinstance(explanation, np.ndarray):
            expl_t = torch.tensor(
                explanation.flatten(), dtype=perturbated_explanations.dtype
            )
        elif isinstance(explanation, torch.Tensor):
            expl_t = explanation.flatten().to(dtype=perturbated_explanations.dtype)

        jacobian = (perturbated_explanations - expl_t.unsqueeze(0)) / epsilon
        return jacobian
