import torch
import numpy as np
from numpy.typing import NDArray


class OODEvaluator:
    """The OODEvaluator class includes functions to evaluate uncertainty attributions w.r.t. to their ability to detect out-of-distribution/outlier samples.
    Note that the terms outlier and OOD sample are used interchangeably in the context for simplicity."""

    def __init__(
        self,
        n_features: int,
        uncertainty_all_scaled: list[dict],
        original_attributions_scaled: list[NDArray],
        ood_strategy: str,
        img: bool = False,
    ):
        """initializes OODEvaluator Object

        Args:
            n_features (int): number of features
            uncertainty_all_scaled (list of dictionaries): includes the uncertainty attributions of all samples and their perturbations for all features
            original_attributions_scaled (list of np arrays): scaled uncertainty attribution of the original input sample before perturbation
            ood_strategy (str): out of distribution strategy. Currently "sigma_based", "minmax_based", and "extreme_random" and "patch_inversion" are available
            img (bool, optional): whether the data is image data. Defaults to False.

        """
        self.n_features = n_features
        self.uncertainty_all_scaled = uncertainty_all_scaled
        self.original_attributions_scaled = original_attributions_scaled
        self.ood_strategy = ood_strategy
        self.img = img
        if img:
            self.img_height = 28  # default values for MNIST
            self.img_width = 28  # default values for MNIST
            self.img_channels = 1  # default values for MNIST

    def _compute_uncertainty_rankings(
        self, arr: np.ndarray, original_attributions: np.ndarray, target_idx: int
    ) -> tuple[np.ndarray, int]:
        """Compute the uncertainty rankings of the perturbed sample for feature target_idx and the original sample

        Args:
            arr (np array): array of arrays containing the uncertainty attributions of all perturbations for one sample and one feature
            original_attributions (np array): uncertainty attribution of the original sample before perturbation
            target_idx (int): index of the perturbed feature

        Returns:
            (np.ndarray, int): returns an array with the ranks per array in arr and the rank of the original sample
        """
        original_attributions = original_attributions.squeeze()
        original_sorted_idx = np.argsort(-original_attributions)
        original_rank = np.where(original_sorted_idx == target_idx)[0][0] + 1

        ranks = []
        if len(arr.shape) == 1:
            arr = arr[np.newaxis, :]

        for sample in arr:
            sorted_idx = np.argsort(-sample)
            rank = np.where(sorted_idx == target_idx)[0][0] + 1
            ranks.append(rank)
        assert original_rank != 0
        assert rank in ranks and rank != 0
        return np.array(ranks), original_rank

    def _compute_rank_change(self, unc_ranks: np.ndarray, original_rank: int) -> np.ndarray:
        """Computes how the rank of the perturbed feature changed after perturbation

        Args:
            unc_ranks (np.ndarray): uncertainty rank of the considered feature after perturbation
            original_rank (int): uncertainty rank of the considered feature before perturbation

        Returns:
            np.ndarray: uncertainty rank change
        """

        return (original_rank - unc_ranks) / original_rank

    def compute_metrics(self, eps: float = 0.1):
        """
        Compute proportion of unc_rankings that fall into the top eps percent of ranks (called accuracy) and relative uncertainty rank change averaged over all features and samples.

        Args:
            eps (float): fraction (0-1) of top ranks to consider (e.g. 0.1 means top 10%)

        Returns:
            metrics_dict: dict[metric] -> dict containing the computed metrics
            all_values: tuple of arrays (all_ranks, all_rank_changes)
        """

        results_all = self.evaluate_all_samples_features(
            self.uncertainty_all_scaled, self.original_attributions_scaled
        )
        metrics_dict = {}

        # number of top ranks to consider (at least 1) for proportion calculation
        top_k = max(1, int(np.ceil(eps * self.n_features)))
        threshold = top_k - 1  # ranks are 0-based, so threshold inclusive

        all_ranks = []
        all_rank_changes = []
        for sample_res in results_all:
            for f_idx in sample_res.keys():
                ranks = sample_res[f_idx]["unc_rankings"]
                all_ranks.extend(ranks.tolist() if hasattr(ranks, "tolist") else list(ranks))
                rank_change = sample_res[f_idx]["unc_rank_change"]
                all_rank_changes.extend(
                    rank_change.tolist() if hasattr(rank_change, "tolist") else list(rank_change)
                )

        all_ranks = np.array(all_ranks, dtype=int)
        all_rank_changes = np.array(all_rank_changes, dtype=float)
        overall_rank_changes = np.mean(all_rank_changes) if all_rank_changes.size > 0 else 0.0
        overall_prop = np.mean(all_ranks <= threshold) if all_ranks.size > 0 else 0.0
        metrics_dict["accuracy"] = overall_prop
        metrics_dict["avg_unc_rank_change"] = overall_rank_changes

        return metrics_dict, (all_ranks, all_rank_changes)

    def _get_patch_flat_indices(self, arr: np.ndarray | torch.Tensor, patch_position: tuple):
        """
        Output: List of integer indices that contain the patch in `arr.flatten()`.

        Args:
        - arr: np.ndarray or torch.Tensor with Shape (1,H,W) or (1,1,H,W)
        - patch_position: tuple (r0, c0, h, w) or (r0, c0, r1, c1)

        Returns:
        - patch_pos: list of int (patch indices in flattened array)
        - mask: boolean array of shape (arr.flatten().shape[0],) with True for patch pixels
        """
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        C = self.img_channels
        H = self.img_height
        W = self.img_width
        # Support for (r0,c0,h,w) or (r0,c0,r1,c1)
        if len(patch_position) != 4:
            raise ValueError("patch_position must be length-4 tuple")
        r0, c0, a, b = patch_position
        if a >= 0 and b >= 0 and (r0 + a <= H and c0 + b <= W):
            r_start, c_start = int(r0), int(c0)
            r_end, c_end = int(r0 + a), int(c0 + b)
        else:
            r_start, c_start, r_end, c_end = int(r0), int(c0), int(a), int(b)

        # Clip to bounds
        r_start = max(0, min(r_start, H - 1))
        c_start = max(0, min(c_start, W - 1))
        r_end = max(r_start + 1, min(r_end, H))
        c_end = max(c_start + 1, min(c_end, W))

        # Computing Indices
        patch_pos = []
        for ch in range(C):
            base = ch * (H * W)
            for rr in range(r_start, r_end):
                row_base = rr * W
                for cc in range(c_start, c_end):
                    idx = base + row_base + cc
                    patch_pos.append(idx)

        return patch_pos

    def _evaluate_ood_patch(
        self, arr: np.ndarray, original_input_attributions: np.ndarray, patch_position: tuple
    ) -> dict:
        """evaluates one sample regarding the metrics unc_rankings, and unc_rank_change for patch-based perturbations

        Args:
            arr (numpy array): uncertainty attribution
            original_input_attributions (numpy array): uncertainty attributions of the original test sample
            patch_position (tuple): position of the patch (r0, c0, h, w)
        Returns:
            dict: A dictionary containing the metrics for the considered perturbed sample"""

        target_idx_list = self._get_patch_flat_indices(original_input_attributions, patch_position)
        # original_input_attributions = original_input_attributions.squeeze()
        original_input_attributions = original_input_attributions.flatten()
        arr = arr.flatten()
        assert arr.shape == original_input_attributions.shape
        unc_rankings_list = []
        rank_change_list = []
        for target_idx in target_idx_list:
            unc_rankings, original_rank = self._compute_uncertainty_rankings(
                arr, original_input_attributions, target_idx
            )
            rank_change = self._compute_rank_change(unc_rankings, original_rank)
            unc_rankings_list.append(unc_rankings)
            rank_change_list.append(rank_change)

        unc_rankings = np.mean(np.array(unc_rankings_list), axis=0)
        rank_change = np.mean(np.array(rank_change_list), axis=0)

        return {
            "unc_rankings": unc_rankings,
            "unc_rank_change": rank_change,
        }

    def _evaluate_ood(
        self, arr: np.ndarray, original_input_attributions: np.ndarray, target_idx: int
    ) -> dict:
        """evaluates one sample regarding the metrics unc_rankings, and unc_rank_change

        Args:
            arr (numpy array): uncertainty attribution
            original_input_attributions (numpy array): uncertainty attributions of the original test sample
            target_idx (int): index of the considered feature

        Returns:
            dict: A dictionary containing the metrics for the considered perturbed sample
        """
        original_input_attributions = original_input_attributions.squeeze()
        unc_rankings, original_rank = self._compute_uncertainty_rankings(
            arr, original_input_attributions, target_idx
        )
        rank_change = self._compute_rank_change(unc_rankings, original_rank)

        return {
            "unc_rankings": unc_rankings,
            "unc_rank_change": rank_change,
        }

    def evaluate_all_samples_features(
        self, uncertainty_dict_list: list[dict], original_attributions_scaled: list[np.ndarray]
    ) -> list:
        """evaluates all samples and features regarding the metrics unc_rankings, and unc_rank_change

        Args:
            uncertainty_dict_list (list of dictionaries): includes the uncertainty attributions of all samples and their perturbations for all features
            original_attributions_scaled (list of numpy arrays): scaled uncertainty attribution of the original input sample before perturbation

        Returns:
            results_all: A list containing the results (steps, ratios, unc_ranks, unc_rank_change) for all test samples and their perturbations
        """
        results_all = []
        if self.img:
            for i, sample_dict in enumerate(uncertainty_dict_list):
                sample_results = {}
                original_input_attributions = original_attributions_scaled[i]
                for patch_start, patch_dict in sample_dict.items():
                    patch_position = patch_dict["patch_position"]
                    sample_results[patch_start] = self._evaluate_ood_patch(
                        patch_dict["scaled_attributions"],
                        original_input_attributions,
                        patch_position=patch_position,
                    )
                results_all.append(sample_results)

        else:
            for i, sample_dict in enumerate(uncertainty_dict_list):
                sample_results = {}
                original_input_attributions = original_attributions_scaled[i]

                for feature_idx, arr in sample_dict.items():
                    if feature_idx == "original":
                        continue
                    sample_results[feature_idx] = self._evaluate_ood(
                        arr, original_input_attributions, target_idx=feature_idx
                    )

                results_all.append(sample_results)

        return results_all
