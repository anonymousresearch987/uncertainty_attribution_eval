from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import captum
from captum.attr import IntegratedGradients, InputXGradient, ShapleyValueSampling
from captum.attr._core.lime import get_exp_kernel_similarity_function
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma, Pass
from zennit.types import Linear, Activation, Convolution


class Explainer(ABC):
    """abstract Explainer class"""

    name: str

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_feature_attributions(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor | None
    ) -> torch.Tensor:
        pass


class ShapleyValueSamplingExplainer(Explainer):
    """Shapley Value Sampling Explainer"""

    name: str = "Shapley Value Sampling"

    def __init__(self, X_train: torch.Tensor, nsamples: int = 25):
        """inits ShapleyValueSamplingExplainer

        Args:
            X_train (torch.Tensor): training data used for baseline computation
            nsamples (int, optional): number of samples to use for Shapley value estimation. Defaults to 25.
        """
        self.X_train = X_train
        self.nsamples = nsamples

    def get_name(self):
        """get name

        Returns:
            str: name of the explainer
        """
        return self.name

    def get_feature_attributions(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor | None
    ) -> torch.Tensor:
        """get ShapelyValueSampling feature attributions for the inputs

        Args:
            model (nn.Module()): model used for explanation
            inputs (torch.Tensor): inputs to be explained
            targets (torch.Tensor): target labels

        Returns:
            torch.Tensor: feature attributions
        """
        explainer = ShapleyValueSampling(forward_func=model)
        attr = explainer.attribute(
            inputs=torch.Tensor(inputs),
            target=targets,
            baselines=torch.median(torch.Tensor(self.X_train), dim=0).values.unsqueeze(0),
            n_samples=self.nsamples,
        )
        attr = attr if targets is not None else attr.squeeze(1)
        return attr


class InputXGradientExplainer(Explainer):
    """InputXGradient Explainer"""

    name: str = "InputXGradient"

    def __init__(self):
        """inits InputXGradientExplainer"""
        pass

    def get_name(self):
        """get name

        Returns:
            str: name of the explainer
        """
        return self.name

    def get_feature_attributions(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor | None
    ) -> torch.Tensor:
        """get InputXGradient feature attributions for the inputs

        Args:
            model (nn.Module()): model used for explanation
            inputs (torch.Tensor): inputs to be explained
            targets (torch.Tensor): target labels
        Returns:
            torch.Tensor: feature attributions
        """
        explainer = InputXGradient(forward_func=model)
        attr = explainer.attribute(
            inputs=torch.Tensor(inputs),
            target=targets,
        )
        return torch.from_numpy(attr.detach().numpy())


class IntegratedGradientsExplainer(Explainer):
    """Integrated Gradients Explainer"""

    name: str = "Integrated Gradients"

    def __init__(self, multiply_by_inputs: bool = True):
        """inits IntegratedGradientsExplainer

        Args:
            multiply_by_inputs (bool, optional): whether to multiply attributions by inputs. Defaults to True.
        """
        self.multiply_by_inputs = multiply_by_inputs

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the explainer
        """
        return self.name

    def get_feature_attributions(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor | None
    ) -> torch.Tensor:
        """get Integrated Gradients feature attributions for the input

        Args:
            model (nn.Module()): model used for explanation
            inputs (torch.Tensor): inputs to be explained
            targets (torch.Tensor): target labels

        Returns:
            torch.Tensor: feature attributions
        """
        explainer = IntegratedGradients(
            forward_func=model, multiply_by_inputs=self.multiply_by_inputs
        )
        attr = explainer.attribute(
            inputs=torch.Tensor(inputs),
            baselines=torch.Tensor(inputs) * 0,
            target=targets,
        )
        return attr


class LRPExplainer(Explainer):
    """Layer-wise Relevance Propagation (LRP) Explainer"""

    name: str = "LRP"

    def __init__(
        self, gamma=0.3, cnn: bool = False, upper_relevance: torch.Tensor | None = None
    ) -> None:
        """inits LRPExplainer"""
        self.gamma = gamma
        self.cnn = cnn
        self.upper_relevance = upper_relevance

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the explainer
        """
        return self.name

    def lrp_linear(
        self, model: nn.Module, X: torch.Tensor, targets: torch.Tensor | None, gamma: float = 0.3
    ) -> torch.Tensor:
        """Simple Zennit LRP implementation using gamma for torch/module models.

        Args:
            model (nn.Module): model used for explanation
            X (torch.Tensor): inputs to be explained
            targets (torch.Tensor): target labels
            gamma (float, optional): gamma parameter for LRP-Gamma rule. Defaults to 0.3.

        Robust handling of targets that may be:
          - None
          - scalar (0-dim tensor or int)
          - 1D tensor/list/ndarray with one target per sample
        """
        layer_map = [
            (Activation, Pass()),  # ignore activations
            (Linear, Gamma(gamma=gamma)),  # dense Linear layer
        ]
        composite_gamma = LayerMapComposite(layer_map=layer_map)
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
        input_all = X.clone().detach().requires_grad_(True)
        with composite_gamma.context(model) as modified_model:
            output = modified_model(input_all)
            (attribution,) = torch.autograd.grad(
                output, input_all, grad_outputs=torch.ones_like(output)
            )
        return attribution

    def lrp_cnn(
        self, model: nn.Module, X: torch.Tensor, targets: torch.Tensor | None, gamma: float = 0.3
    ) -> torch.Tensor:
        """Simple Zennit LRP implementation using gamma for torch/module models.

        Args:
            model (nn.Module): model used for explanation
            X (torch.Tensor): inputs to be explained
            targets (torch.Tensor): target labels
            gamma (float, optional): gamma parameter for LRP-Gamma rule. Defaults to 0.3.

        Robust handling of targets that may be:
          - None
          - scalar (0-dim tensor or int)
          - 1D tensor/list/ndarray with one target per sample
        """
        layer_map = [
            (Activation, Pass()),
            (Linear, Gamma(gamma=gamma)),
            (Convolution, Gamma(gamma=gamma)),
        ]
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
        explanations = []
        with torch.no_grad():
            dummy_out = model(X[:1])
        self.n_outputs = int(dummy_out.shape[1]) if dummy_out.ndim > 1 else 1

        for i, sample in enumerate(X):
            if targets is None:
                d = 0
            elif isinstance(targets, torch.Tensor):
                if targets.ndim == 0:
                    d = int(targets.item())
                else:
                    d = int(targets[i].item())

            sample = sample[None, ...]
            input = sample.clone().detach().requires_grad_(True)

            composite_name_map = LayerMapComposite(layer_map=layer_map)
            with composite_name_map.context(model) as modified_model:
                output = modified_model(input)  # shape (1, n_outputs) or (1,)
                # create a mask that selects the d-th output
                if output.ndim == 1:
                    # single-output model
                    upper_relevance = output.clone()
                else:
                    mask = torch.zeros_like(output)
                    mask[0, d] = 1.0
                    if self.upper_relevance is not None:
                        upper_relevance = (self.upper_relevance[i : i + 1] * mask).detach()
                    else:
                        upper_relevance = (output * mask).detach()

                # compute gradient-based relevance
                (attribution,) = torch.autograd.grad(
                    outputs=output,
                    inputs=input,
                    grad_outputs=upper_relevance,
                    create_graph=True,
                    retain_graph=True,
                )
            explanations.append(attribution.detach().cpu().numpy())

        attribution = torch.from_numpy(np.vstack(explanations))
        return attribution

    def get_feature_attributions(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor | None
    ) -> torch.Tensor:
        """get LRP attributions for the input

        Args:
            model (nn.Module()): model used for explanation
            inputs (torch.Tensor): inputs to be explained
            targets (torch.Tensor): target labels
        Returns:
            torch.Tensor: feature attributions
        """
        if self.cnn:
            return self.lrp_cnn(model, inputs, targets, self.gamma)
        else:
            return self.lrp_linear(model, inputs, targets, self.gamma)


class LimeExplainer(Explainer):
    """LIME Explainer"""

    name: str = "LIME"

    def __init__(self, num_samples: int = 25):
        """inits LimeExplainer

        Args:
            num_samples (int): number of perturbed samples for LIME
        """
        self.num_samples = num_samples

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the explainer
        """
        return self.name

    def get_feature_attributions(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor | None
    ) -> torch.Tensor:
        """get LIME feature attributions for the input

        Args:
            model (nn.Module()): model used for explanation
            inputs (torch.Tensor): inputs to be explained"""
        inp = np.asarray(inputs)
        if not np.isfinite(inp).all():
            inp = np.nan_to_num(
                inp,
                nan=0.0,
                posinf=float(np.finfo(np.float64).max),
                neginf=-float(np.finfo(np.float64).max),
            )

        lime = captum.attr.Lime(
            forward_func=model,
            similarity_func=get_exp_kernel_similarity_function("euclidean"),
        )
        attr = (
            lime.attribute(
                inputs=torch.Tensor(inp) + 1e-4,
                n_samples=self.num_samples,
                target=targets,
            )
            .detach()
            .numpy()
        )
        return torch.from_numpy(attr)


class GradientSHAPExplainer(Explainer):
    """Gradient SHAP Explainer"""

    name: str = "Gradient SHAP"

    def __init__(self, nsamples=25):
        """inits GradientSHAPExplainer

        Args:
            baselines (torch.Tensor, optional): baseline inputs for SHAP. Defaults to None.
            stdevs (float, optional): standard deviation of noise for SHAP. Defaults to 0.1.
            nsamples (int, optional): number of samples for SHAP. Defaults to 50.
        """
        self.nsamples = nsamples

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the explainer
        """
        return self.name

    def get_feature_attributions(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor | None
    ) -> torch.Tensor:
        """get Gradient SHAP feature attributions for the input

        Args:
            model (nn.Module()): model used for explanation
            inputs (torch.Tensor): inputs to be explained
            targets (torch.Tensor): target labels"""
        explainer = captum.attr.GradientShap(forward_func=model)
        baselines = torch.zeros_like(torch.Tensor(inputs))
        attr = explainer.attribute(
            inputs=torch.Tensor(inputs),
            baselines=baselines,
            target=targets,
            n_samples=self.nsamples,
        )
        return attr
