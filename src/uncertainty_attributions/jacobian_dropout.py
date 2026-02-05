import copy
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from numpy.typing import NDArray
from src.explainer.explainer import Explainer


from src.uncertainty_attributions.jacobian import JacobianGenerator


class JacobianGeneratorDropout(JacobianGenerator):
    """Class to generate Jacobian matrices of explanations with respect to activations (approximating Dropout)."""

    def __init__(
        self,
        explainer: Explainer,
        drop_prob: float = 0.1,
        layers_to_perturb: Optional[int] = None,
    ) -> None:
        """init Jacobian Genertor for Dropout models

        Args:
            explainer (Explainer): explainer model
            drop_prob (float, optional): _description_. Defaults to 0.1.
            layers_to_perturb (Optional[int], optional): number of layers to be perturbed. Defaults to None.
        """
        super().__init__()
        self.explainer = explainer
        self.drop_prob = drop_prob
        self.layers_to_perturb = layers_to_perturb

    def _detect_data_type(self, input: torch.Tensor) -> str:
        """Detect if input is image or tabular data based on shape.

        Args:
            input: torch.Tensor of shape (batch, ...) or (batch, features)

        Returns:
            str: "image" if 4D (batch, channels, height, width) or 3D (batch, height, width),
                 "tabular" if 2D (batch, features)
        """
        if input.ndim == 4:  # (batch, channels, height, width)
            return "image"
        elif input.ndim == 3:  # (batch, height, width)
            return "image"
        elif input.ndim == 2:  # (batch, features)
            return "tabular"
        else:
            raise ValueError(f"Unexpected input ndim={input.ndim}, expected 2, 3, or 4")

    def get_perturbed_explanations(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor | None,
        epsilon: float = 1e-1,
    ) -> torch.Tensor:
        """Perturb model neurons to get explanations to approximate explanation Jacobian

        Args:
            model (nn.Module): trained model
            input (torch.Tensor): data (image or tabular)
            target (torch.Tensor | None): target labels
            epsilon (float): perturbation noise

        Returns:
            torch.Tensor: jacobian_matrix of shape (batch_size, neuron_count, n_features)
        """
        linear_layers = [
            module for module in model.modules() if isinstance(module, torch.nn.Linear)
        ]
        linear_layers = linear_layers[:-1]  # exclude output layer

        layers_to_perturb = getattr(self, "layers_to_perturb", None)
        if layers_to_perturb is not None:
            n = int(layers_to_perturb)
            if n <= 0:
                raise ValueError("layers_to_perturb must be >= 1")
            linear_layers = linear_layers[-n:]

        neuron_count = sum(layer.out_features for layer in linear_layers)

        inp = torch.as_tensor(input)
        if inp.ndim == 1:
            inp = inp.unsqueeze(0)

        data_type = self._detect_data_type(inp)

        single_explanation = self.explainer.get_feature_attributions(model, inp[:1], target)

        jacobian_matrix = np.zeros((neuron_count, *single_explanation.shape))
        neuron_idx = 0

        for layer_id, layer in enumerate(linear_layers):
            out_features = layer.out_features
            for i in range(out_features):
                model_pert = copy.deepcopy(model)
                linear_layers_pert = [
                    mod for mod in model_pert.modules() if isinstance(mod, torch.nn.Linear)
                ]
                target_layer = linear_layers_pert[layer_id]

                with torch.no_grad():
                    pert_bias = target_layer.bias.clone()
                    pert_bias[i] += epsilon
                    target_layer.bias.copy_(pert_bias)

                explanations = self.explainer.get_feature_attributions(model_pert, inp, target)

                # Reshape based on data type
                if data_type == "image":
                    # explanations shape: (batch_size, channels, height, width) or (batch_size, height, width)
                    explanations = explanations
                else:  # tabular
                    # explanations shape: (batch_size, features)
                    if explanations.ndim == 1:
                        explanations = explanations.flatten()
                try:
                    jacobian_matrix[neuron_idx, :] = explanations
                except Exception:
                    explanations = explanations.detach().numpy()
                    explanations = torch.from_numpy(explanations)
                    jacobian_matrix[neuron_idx, :] = explanations
                neuron_idx += 1

        return torch.Tensor(jacobian_matrix)

    def get_activations(self, model: nn.Module, input: torch.Tensor) -> torch.Tensor:
        """Collect all activations of the model per sample excluding the output layer

        Args:
            model (nn.Module): PyTorch model
            input (torch.Tensor): input tensor (image or tabular)

        Returns:
            np.ndarray: flat vector of activations of the hidden layers
        """
        activations = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append(output.detach().flatten())

        # collect layers without output layer
        linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
        internal_layers = linear_layers[:-1]  # exclude output layer

        layers_to_perturb = getattr(self, "layers_to_perturb", None)
        if layers_to_perturb is not None:
            n = int(layers_to_perturb)
            if n <= 0:
                raise ValueError("layers_to_perturb must be >= 1")
            internal_layers = internal_layers[-n:]

        if len(internal_layers) == 0:
            return torch.Tensor([])

        hooks = [layer.register_forward_hook(hook_fn) for layer in internal_layers]
        device = next(model.parameters()).device

        model.eval()
        with torch.no_grad():
            inp = torch.as_tensor(input).to(device)
            if inp.ndim == 1:
                inp = inp.unsqueeze(0)
            _ = model(inp)
        for h in hooks:
            h.remove()

        if len(activations) == 0:
            return torch.Tensor([])

        flat_activations = torch.cat(activations).cpu().numpy()

        return flat_activations

    def compute_delta(self, model: nn.Module, input: torch.Tensor) -> NDArray:
        """Compute delta with respect to nodes for dropout.

        Args:
            model (nn.Module): trained model
            input (torch.Tensor): test data (image or tabular)

        Returns:
            np.ndarray: delta matrix (diagonal)
        """
        activations = self.get_activations(model, input)
        delta = np.diag(np.power(activations, 2))
        return delta

    def approximate_jacobian(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor | None,
        explanation: torch.Tensor,
        epsilon: float = 1e-1,
    ) -> torch.Tensor:
        """Approximate the jacobian of explanations with respect to the activations.

        Args:
            model (nn.Module): trained model
            input (torch.Tensor): test data (image or tabular)
            target (torch.Tensor | None): target labels
            explanation (torch.Tensor): explanation vector or matrix
            epsilon (float): perturbation magnitude

        Returns:
            torch.Tensor: jacobian matrix of shape (batch_size, neuron_count, n_features)
        """
        device = next(model.parameters()).device

        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32, device=device)

        perturbated_explanations = self.get_perturbed_explanations(model, input, target, epsilon)
        if isinstance(explanation, np.ndarray):
            explanation = torch.tensor(explanation, dtype=torch.float32)

        if explanation.ndim == 1:
            explanation = explanation.unsqueeze(0)
        jacobian = (perturbated_explanations - explanation) / epsilon

        return jacobian
