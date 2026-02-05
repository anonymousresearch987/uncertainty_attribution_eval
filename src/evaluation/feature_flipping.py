import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Callable, Any
from numpy.typing import NDArray

from src.evaluation.metric import Metric
from scipy.ndimage import gaussian_filter
from src.models.models import NeuralNetworkBase


class FeatureFlipping(Metric):
    """The FeatureFlipping class includes functions to assess the correctness of uncertainty attributions.
    Most parts are based on Bley et al., 2025.

    References:
        Bley, F., Lapuschkin, S., Samek, W., & Montavon, G. (2025). Explaining predictive uncertainty by exposing second-order effects. Pattern Recognition, 160, 111171.

    """

    name = "FeatureFlipping"
    baseline: str

    def __init__(self, baseline: str = "median") -> None:
        """init FeatureFlipping metric

        Args:
            baseline (str, optional): baseline. Defaults to "median". Options are "median", "gaussian_blur", and "kde". gaussian_blur is only for image data, kde is only for tabular data.

        """
        self.baseline = baseline

    def get_name(self):
        """get name

        Returns:
            str: name of the metric
        """
        return self.name

    def sample_masked(
        self,
        test_input: torch.Tensor,
        mask: NDArray,
        n_samples: int,
        X_train: torch.Tensor,
        sigma: float = 0.1,
        eps: float = 1e-6,
    ) -> NDArray:
        """sample masked features based on kde weights

        Args:
            test_input (torch.Tensor): input tensor
            mask (NDArray): boolean mask for features to sample
            n_samples (int): number of samples
            X_train (torch.Tensor): train feature set tensor
            sigma (float, optional): standard deviation of perturbation noise. Defaults to 0.1.
            eps (float, optional): value threshold. Defaults to 1e-6.

        Returns:
            NDArray: sampled inputs
        """
        cond_idx = np.where(~mask)[0]
        sample_idx = np.where(mask)[0]

        if len(sample_idx) == 0:
            return test_input.repeat(n_samples, 1).numpy()
        diffs = X_train[:, cond_idx] - test_input[cond_idx]
        dists = np.linalg.norm(diffs, axis=1)

        weights = np.exp(-(dists**2) / (2 * sigma**2))
        sum_w = np.sum(weights)
        if not np.isfinite(sum_w) or sum_w <= eps:
            weights = np.ones_like(weights, dtype=float) / float(len(weights))
        else:
            weights = weights / float(sum_w)

        idx = np.random.choice(len(X_train), size=n_samples, p=weights)
        X_samples = np.tile(test_input, (n_samples, 1))
        X_samples[:, sample_idx] = X_train[idx][:, sample_idx]
        return X_samples

    def compute_sample_perturbation_curve(
        self,
        obj_fun: Callable,
        sample: torch.Tensor,
        uncertainty_attributions: torch.Tensor,
        baseline: NDArray | None,
        X_train: torch.Tensor | None = None,
    ) -> list:
        """compute perturbation curve for single sample

        Args:
            obj_fun (Callable): function with UQ objective (variance, entropy, etc.)
            sample (torch.Tensor): sample tensor
            uncertainty_attributions (torch.Tensor): uncertainty attribution
            baseline (NDArray | None): median baseline if given
            X_train (torch.Tensor | None): train feature set tensor

        Returns:
            list: list of curves
        """
        torch.manual_seed(42)
        np.random.seed(42)
        perturbation_list = []
        perturbation_list.append(obj_fun(sample).item())
        uncertainty_attributions = uncertainty_attributions.flatten()

        if isinstance(uncertainty_attributions, torch.Tensor):
            uncertainty_attributions = uncertainty_attributions.detach().cpu().numpy()

        # Get indices sorted by attribution importance (descending order)
        order_list = np.argsort(uncertainty_attributions)[::-1]

        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()
        if isinstance(baseline, torch.Tensor):
            baseline = baseline.detach().cpu().numpy()

        if len(sample.shape) == 4:
            n_features = sample.shape[-1] * sample.shape[-2]
        elif len(sample.shape) == 3:
            n_features = sample.shape[-1] * sample.shape[-2]
        else:
            n_features = sample.shape[1]  # for tabular data

        # high dimensional data, we only flip every 5 pixels
        flipping_interval = 1
        if n_features > 30:
            flipping_interval = 5
        flattened_sample = sample.flatten()
        if baseline is not None:
            flattened_baseline = baseline.flatten()
        assert isinstance(flattened_sample, np.ndarray), "sample should be numpy array"
        if baseline is not None:
            assert isinstance(flattened_baseline, np.ndarray), "baseline should be numpy array"
        mask = np.ones(flattened_sample.shape[0], dtype=bool)
        if self.baseline == "gaussian_blur":
            flattened_baseline = gaussian_filter(flattened_sample, sigma=1.0)
        for num, order_ind in enumerate(order_list):
            mask[order_ind] = False
            if (num + 1) % flipping_interval == 0:
                # Create a copy of the sample and set masked positions to baseline
                modified_sample = flattened_sample.copy()
                if self.baseline == "median":
                    modified_sample[~mask] = flattened_baseline[  # type: ignore
                        ~mask
                    ]
                elif self.baseline == "gaussian_blur":
                    modified_sample[~mask] = flattened_baseline[  # type: ignore
                        ~mask
                    ]
                elif self.baseline == "kde":
                    assert X_train is not None, "X_train must be provided for kde baseline"
                    modified_samples = self.sample_masked(
                        flattened_sample,
                        ~mask,
                        n_samples=25,
                        X_train=X_train,
                        sigma=0.1,
                    )
                else:
                    raise ValueError(
                        f"{self.baseline} not defined. Please use 'median' or 'gaussian_blur'"
                    )
                if self.baseline == "kde":
                    pert = obj_fun(torch.from_numpy(modified_samples).float()).mean().item()
                else:
                    pert = obj_fun(
                        torch.from_numpy(modified_sample[None]).float().view(sample.shape)
                    ).item()
                perturbation_list.append(pert)
        return perturbation_list

    def compute_perturbation_curve(
        self,
        obj_fun: Callable,
        flipping_samples: torch.Tensor,
        baseline: NDArray | None,
        uncertainty_attributions: torch.Tensor,
        plot_curve: bool,
        X_train: torch.Tensor | None = None,
    ) -> tuple[NDArray, list, Any | None]:
        """compute perturbation curve for all samples

        Args:
            obj_fun (Callable): uq objective function
            flipping_samples (torch.Tensor): samples for flipping experiment
            baseline (NDArray | None): median baseline
            uncertainty_attributions (torch.Tensor): uncertainty attributions
            plot_curve (bool): curve plot
            X_train (torch.Tensor | None): train feature set tensor

        Returns:
            tuple[NDArray, list, Any|None]: list of all perturbations
        """
        perturbation_list = []
        assert len(uncertainty_attributions) == len(
            flipping_samples
        ), "number of samples and uncertainty attributions must match"
        for i in tqdm(range(len(flipping_samples)), desc="Computing Feature Flipping metric"):
            test_sample = flipping_samples[i]
            uncertainty_attribution = uncertainty_attributions[i]
            perturbation_list.append(
                self.compute_sample_perturbation_curve(
                    obj_fun,
                    test_sample,
                    uncertainty_attribution,
                    baseline=baseline,
                    X_train=X_train,
                )
            )
        mean_perturbation_curve = np.mean(perturbation_list, axis=0)
        if plot_curve:
            fig, ax = plt.subplots()
            ax.plot(mean_perturbation_curve)
            ax.set_xlabel("Index")
            ax.set_ylabel("Mean Perturbation")
            ax.set_title("Mean Perturbation Curve")
            plt.show()
            fig.savefig("flipping_curve.png")
            return mean_perturbation_curve, perturbation_list, fig
        else:
            return mean_perturbation_curve, perturbation_list, None

    def calculate_auc_stds(self, flipping_curves: list) -> tuple[NDArray, float, float]:
        """calculate standard deviation for the AUC values

        Args:
            flipping_curves (list): flipping curves

        Returns:
            tuple[NDArray, float, float]: standard deviation, mean of AUC
        """
        auc_list = []
        for curve in flipping_curves:
            if curve[0] == 0:
                normed_curve = np.array(curve)
            else:
                normed_curve = np.array(curve) / curve[0]
            auc = np.trapz(normed_curve, dx=1) / (len(normed_curve) - 1)
            auc_list.append(auc)
        auc_list = np.array(auc_list)
        auc_mean = float(np.mean(auc_list))
        auc_std = float(np.std(auc_list))
        return auc_list, auc_mean, auc_std

    def pixelflipping(
        self,
        unpacked_dataset: tuple,
        uq_model: NeuralNetworkBase,
        benchmark_explanations: torch.Tensor,
        plot_curve: bool = False,
    ) -> tuple[NDArray, list, Any | None]:
        """main implementation of pixel flipping experiment

        Args:
            unpacked_dataset (tuple): tuple of split dataset
            uq_model (NeuralNetworkBase): trained model
            benchmark_explanations (torch.Tensor): uncertainty attributions
            plot_curve (bool): curve plot flag

        Returns:
            mean_flipping_curves: mean of the flipping curves
            all_flipping_curves: list of all flipping curves
            curve_fig: figure of the flipping curve, if plot_curve is True, else None
        """
        n_samples = benchmark_explanations.shape[0]
        X_flipping, _ = self.add_flipping_dataset(unpacked_dataset, uq_model, n_samples=n_samples)
        if self.baseline == "median":
            baseline = self.calculate_median(unpacked_dataset)
        else:
            baseline = None

        obj_fun = uq_model.epistemic_variance

        mean_flipping_curves, all_flipping_curves, curve_fig = self.compute_perturbation_curve(
            obj_fun,
            X_flipping,
            baseline,
            benchmark_explanations,
            plot_curve=plot_curve,
            X_train=unpacked_dataset[0],
        )
        return mean_flipping_curves, all_flipping_curves, curve_fig

    def add_flipping_dataset(
        self, unpacked_dataset: tuple, uq_model: NeuralNetworkBase, n_samples: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """sort dataset for flipping metric

        Args:
            unpacked_dataset (tuple): tuple of split dataset
            uq_model (NeuralNetworkBase): model
            n_samples (int, optional): number of samples for flipping experiment. Defaults to 100.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: flipping feature and target set
        """
        _, X_test, _, y_test = unpacked_dataset

        vars_test = uq_model.forward_uq(X_test)["variance"].detach().numpy()
        indices = np.argsort(vars_test)[-n_samples:]
        X_flipping, y_flipping = X_test[indices], y_test[indices]
        return X_flipping, y_flipping

    def calculate_median(self, unpacked_dataset: tuple) -> NDArray:
        """calculate median of train tensor

        Args:
            unpacked_dataset (tuple): tuple containing train tensor

        Returns:
            NDArray: median of train tensor
        """
        X_train, _, _, _ = unpacked_dataset

        return np.median(X_train, axis=0)

    def evaluate_uncertainty_attributions(
        self,
        uq_model: NeuralNetworkBase,
        unpacked_dataset: tuple,
        uncertainty_attributions: torch.Tensor,
        plot_curve: bool = False,
        **kwargs,
    ) -> tuple[dict, NDArray]:
        """evaluate uncertainty attributions via pixel flipping
        Args:
            uq_model (NeuralNetworkBase): trained model
            unpacked_dataset (tuple of tensors): whole split dataset, unpacked_dataset = (X_train, X_test, X_val, y_train, y_test, y_val)
            uncertainty_attributions (torch.Tensor): uncertainty attributions
            plot_curve (bool): whether to plot and save the flipping curve to pixelflipping_dict
        Returns:
            (dict, NDArray): pixelflipping metrics dictionary including the mean and std of AUCs, and list of AUCs for all samples
        """
        _, all_flipping_curves, curve_fig = self.pixelflipping(
            unpacked_dataset,
            uq_model,
            uncertainty_attributions,
            plot_curve=plot_curve,
        )
        aucs, auc_mean, auc_std = self.calculate_auc_stds(all_flipping_curves)
        if plot_curve:
            pixelflipping_dict = {
                "auc_stds": auc_std,
                "auc_mean": auc_mean,
                "curve_fig": curve_fig,
            }
        else:
            pixelflipping_dict = {
                "auc_stds": auc_std,
                "auc_mean": auc_mean,
            }

        return pixelflipping_dict, aucs
