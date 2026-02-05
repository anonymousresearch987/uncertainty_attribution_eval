import torch
import numpy as np
from src.evaluation.metric import Metric

from src.utils.xuq_utils import (
    scale_uncertainty_attributions,
)

from src.uncertainty_attributions.empirical_xuq import EmpiricalXUQGenerator
from src.evaluation.ood_experiment.ood_generator import OODGenerator
from src.evaluation.ood_experiment.ood_evaluation import OODEvaluator
from tqdm import tqdm


class RelativeRankImprovement(Metric):
    """The RelativeRankImprovement class includes functions to assess the conveyance of uncertainty attributions."""

    name: str = "Relative Rank Improvement"

    def __init__(
        self, img: bool, ood_strategy="sigma_based", sample_percentage: float = 0.1
    ) -> None:
        """init RelativeRankImprovement metric

        Args:
            img (bool): whether the data is image data
            ood_strategy (str, optional): out of distribution samples generation strategy. Defaults to "sigma_based". Other options: "minmax_based", "extreme_random", "patch_inversion"
            sample_percentage (float, optional): percentage of features/pixels to be perturbed for each sample. Defaults to 0.1.
        """
        self.ood_strategy = ood_strategy
        self.img = img
        self.sample_percentage = sample_percentage
        super().__init__()

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the metric
        """
        return self.name

    def evaluate_uncertainty_attributions(
        self,
        unpacked_dataset,
        ensemble: list[torch.nn.Module],
        uncertainty_attributions: torch.Tensor,
        pred_tests: torch.Tensor,
        empirical_xuq_generator: EmpiricalXUQGenerator,
        **kwargs,
    ):
        """evaluate Uncertainty Attributions using Relative Rank Improvement metric

        Args:
            unpacked_dataset (torch.Tensor): whole split dataset, unpacked_dataset = (X_train, X_test, X_val, y_train, y_test, y_val)
            ensemble (list[torch.nn.Module]): list of trained models
            uncertainty_attributions (torch.Tensor): uncertainty attributions
            pred_tests (torch.Tensor): test set predictions
            empirical_xuq_generator (EmpiricalXUQGenerator): empirical uncertainty attribution generator

        Returns:
            tuple[dict,tuple(list, list)]: a dict containing the computed metrics, and a tuple of lists containing the original ranks and perturbed ranks for all samples and features
        """
        X_train, X_test, _, _ = unpacked_dataset
        feature_names = [f"pixel_{i}" for i in range(len(X_test[0].flatten()))]
        n_features = len(feature_names)
        ood_generator = OODGenerator(X_train, X_test, self.ood_strategy, img=self.img)
        ood_all_samples = ood_generator.generate_ood_dataset(
            sample_percentage=self.sample_percentage
        )
        uncertainty_all_scaled = []
        original_attributions_scaled = []
        for i, sample_dict in enumerate(
            tqdm(ood_all_samples, desc="Computing OOD attributions", unit="sample")
        ):
            sample_uncertainty = {}
            # Get original attributions and scale them
            if self.img:
                original_attributions = np.array(uncertainty_attributions[i]).reshape(
                    uncertainty_attributions[i].shape[0], n_features
                )
            else:
                original_attributions = uncertainty_attributions[i].unsqueeze(0).numpy()
            original_attribution_scaled = scale_uncertainty_attributions(original_attributions)
            original_attributions_scaled.append(original_attribution_scaled)
            # Compute uncertainty attributions for the perturbed samples per feature/patch
            for feature_idx, feature_variants in sample_dict.items():
                if feature_idx == "original":
                    continue
                feature_uncertainty_scaled = {}
                # If patch_inversion, feature_variants is a dict with 'variants' and 'patch_position'
                if (
                    self.img
                    and isinstance(feature_variants, dict)
                    and "variants" in feature_variants
                ):
                    variants = feature_variants["variants"]
                    patch_position = feature_variants.get("patch_position")
                else:
                    variants = feature_variants
                    patch_position = None

                assert isinstance(variants, torch.Tensor), "Variants should be a torch.Tensor"

                ood_attributions, _ = empirical_xuq_generator.compute_uncertainty_attr(
                    nr_testsamples=len(variants),
                    X_test=variants,
                    pred_test=pred_tests[i] if pred_tests is not None else None,
                    ensemble=ensemble,
                )
                ood_attributions = (
                    np.array(ood_attributions).reshape(len(variants), n_features)
                    if self.img
                    else np.array(ood_attributions)
                )
                scaled = scale_uncertainty_attributions(ood_attributions)
                if patch_position is not None:
                    feature_uncertainty_scaled[feature_idx] = {
                        "scaled_attributions": scaled,
                        "patch_position": patch_position,
                    }
                else:
                    feature_uncertainty_scaled[feature_idx] = scaled
                sample_uncertainty[feature_idx] = feature_uncertainty_scaled[feature_idx]
            uncertainty_all_scaled.append(sample_uncertainty)

        ood_evaluator = OODEvaluator(
            n_features,
            uncertainty_all_scaled,
            original_attributions_scaled,
            self.ood_strategy,
            img=self.img,
        )

        metrics_dict, all_values = ood_evaluator.compute_metrics()
        return metrics_dict, all_values
