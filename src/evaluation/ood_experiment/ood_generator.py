import numpy as np
import torch

from numpy.typing import NDArray


class OODGenerator:
    """The OODGenerator class includes functions to generate out-of-distribution/outlier samples based on different strategies.
    Note that the terms outlier and OOD sample are used interchangeably in the context for simplicity."""

    def __init__(self, X_train: torch.Tensor, X_test: torch.Tensor, ood_strategy: str, img=False):
        """inits OODGenerator
        Args:
            X_train (torch.Tensor): training data
            X_test (torch.Tensor): test data to generate OOD samples for
            ood_strategy (str): out of distribution strategy. Currently "sigma_based", "minmax_based", and "extreme_random" and "patch_inversion" are available
            img (bool): whether the data is image data
        """
        self.X_train = X_train
        self.X_test = X_test
        self.ood_strategy = ood_strategy
        self.img = img

    def _ood_sigma_based(
        self,
        X_train: torch.Tensor,
        x_sample: torch.Tensor,
        feature_idx: int,
        sigma_multipliers: list[int] = [4],
    ) -> list[NDArray]:
        """generate len(sigma_multipliers) perturbations of a specific feature of a sample based on the strategy "sigma_based"

        Args:
            X_train (torch.Tensor): training data
            x_sample (torch.Tensor): one test sample
            feature_idx (int): index of feature to be perturbed
            sigma_multipliers (list[int], optional): factor for OOD generation. The higher the further away the new value from the training distribution. Defaults to [4,5,6].

        Returns:
            samples(list): list of len(sigma_multipliers) tensors. The tensors include one perturbed feature based on the strategy "sigma_based"
        """
        mu = torch.mean(X_train[:, feature_idx]).item()
        std = torch.std(X_train[:, feature_idx]).item()
        samples = []
        for k in sigma_multipliers:
            x_ood = x_sample.clone().detach().numpy()
            x_ood[feature_idx] = mu + k * std
            samples.append(x_ood)
        return samples

    def _ood_minmax_based(
        self, X_train: torch.Tensor, x_sample: torch.Tensor, feature_idx: int, minmax_deltas=[0.2]
    ):
        """generate len(minmax_deltas) perturbations of a specific feature of a sample based on the strategy "minmax_based"

        Args:
            X_train (torch.Tensor): training data
            x_sample (torch.Tensor): one test sample
            feature_idx (int): index of feature to be perturbed
            minmax_deltas (list[float], optional): factor for OOD generation. The higher the further away the new value from the training distribution. Defaults to [0.1,0.2].

        Returns:
            samples(list): list of len(minmax_deltas) tensors. The tensors include one perturbed feature based on the strategy "minmax_based"
        """
        f_min = np.min(X_train[:, feature_idx])
        f_max = np.max(X_train[:, feature_idx])
        samples = []
        for delta in minmax_deltas:
            x_ood = x_sample.clone().detach().numpy()
            x_ood[feature_idx] = f_max + delta * (f_max - f_min)
            samples.append(x_ood)
        return samples

    def _ood_extreme_random(
        self,
        X_train: torch.Tensor,
        x_sample: torch.Tensor,
        feature_idx: int,
        n_samples: int = 3,
        extreme_range_factor: int = 2,
    ):
        """generate n_samples perturbations of a specific feature of a sample based on the strategy "extreme_random"

        Args:
            X_train (torch.Tensor): training data
            x_sample (torch.Tensor): one test sample
            feature_idx (int): index of feature to be perturbed
            n_samples (int, optional): number of perturbed samples. Defaults to 3.
            extreme_range_factor (int, optional): factor for OOD generation. The higher the further away the new value from the training distribution. Defaults to 2.

        Returns:
            samples(list): list of n_samples tensors. The tensors include one perturbed feature based on the strategy "extreme_random"
        """
        f_min = np.min(X_train[:, feature_idx])
        f_max = np.max(X_train[:, feature_idx])
        range_span = f_max - f_min
        samples = []
        for _ in range(n_samples):
            x_ood = x_sample.clone().detach().numpy()
            if np.random.rand() > 0.5:
                x_ood[feature_idx] = f_max + np.random.rand() * range_span * extreme_range_factor
            else:
                x_ood[feature_idx] = f_min - np.random.rand() * range_span * extreme_range_factor
            samples.append(x_ood)
        return samples

    def _generate_patch_inversion(self, x_sample: torch.Tensor, patch_position: tuple):
        """generate ood perturbations for one test image and one patch

        Args:
            x_sample (torch.Tensor): one test image
            patch_position (tuple(int)): position of the patch in the test image

        Returns:
            tensor: image that contains the same information as x_sample except in the patch where it is perturbed
        """
        if isinstance(x_sample, torch.Tensor):
            img = x_sample.detach().cpu().numpy()
        else:
            img = np.asarray(x_sample)

        # Expected shapes: (1, H, W) or (1,1,H,W)
        if img.ndim == 3:
            # (1, H, W) -> treat as (1, 1, H, W) for uniformity
            batch, H, W = img.shape
            img_proc = img.reshape(batch, 1, H, W).copy()
            squeezed_channel = True
        elif img.ndim == 4:
            batch, channels, H, W = img.shape
            img_proc = img.copy()
            squeezed_channel = False
        else:
            raise ValueError("x_sample must have shape (1,H,W) or (1,1,H,W)")

        # Normalize patch_position formats
        # Accept (r, c, h, w) or (r0, c0, r1, c1)
        if len(patch_position) == 4:
            r0, c0, a, b = patch_position
            if a >= 0 and b >= 0 and (r0 + a <= H and c0 + b <= W):
                r_start, c_start = int(r0), int(c0)
                r_end, c_end = int(r0 + a), int(c0 + b)
            else:
                r_start, c_start, r_end, c_end = int(r0), int(c0), int(a), int(b)
        else:
            raise ValueError("patch_position must be a length-4 tuple (r,c,h,w) or (r0,c0,r1,c1)")

        r_start = max(0, min(r_start, H - 1))
        c_start = max(0, min(c_start, W - 1))
        r_end = max(r_start + 1, min(r_end, H))
        c_end = max(c_start + 1, min(c_end, W))

        # Flip pixels inside patch: new_pixel = 255 - old_pixel
        orig_dtype = img_proc.dtype
        img_proc = img_proc.astype(np.float32)

        img_proc[:, :, r_start:r_end, c_start:c_end] = (
            255.0 - img_proc[:, :, r_start:r_end, c_start:c_end]
        )

        if np.issubdtype(orig_dtype, np.integer):
            img_proc = np.clip(img_proc * 255.0, 0, 255).astype(orig_dtype)
        else:
            img_proc = img_proc.astype(orig_dtype)

        if squeezed_channel:
            img_out = img_proc.reshape(batch, H, W)
        else:
            img_out = img_proc

        return torch.from_numpy(img_out)

    def _generate_ood_samples(self, X_train, x_sample, feature_idx=None, patch_position=None):
        """generate ood perturbations for one test sample and features

        Args:
            X_train (torch.Tensor): training data
            x_sample (torch.Tensor): one test sample
            feature_idx (int): index of feature to be perturbed
            patch_position (tuple): position of the patch in the test image
        Returns:
            tensor: returns perturbations of the test sample x_sample based on an ood strategy that modifies only the feature with feature_idx and keeps all remaining features the same
        """
        if self.ood_strategy == "sigma_based":
            assert (
                feature_idx is not None
            ), "Please provide feature_idx for sigma_based OOD strategy."
            samples = self._ood_sigma_based(X_train, x_sample, feature_idx)
        elif self.ood_strategy == "minmax_based":
            assert (
                feature_idx is not None
            ), "Please provide feature_idx for minmax_based OOD strategy."
            samples = self._ood_minmax_based(X_train, x_sample, feature_idx)
        elif self.ood_strategy == "extreme_random":
            assert (
                feature_idx is not None
            ), "Please provide feature_idx for extreme_random OOD strategy."
            samples = self._ood_extreme_random(X_train, x_sample, feature_idx)
        elif self.ood_strategy == "patch_inversion":
            assert (
                patch_position is not None
            ), "Please provide patch_position for patch_inversion OOD strategy."
            samples = self._generate_patch_inversion(x_sample, patch_position)
        else:
            raise Exception(f"The ood strategy {self.ood_strategy} is not implemented")

        samples = torch.from_numpy(np.array(samples))
        return samples

    def generate_ood_dataset(
        self, sample_percentage: float = 1
    ) -> list[dict[str, torch.Tensor | dict]]:
        """generate ood perturbations for the whole testset
        Args:
            X_train (torch.Tensor): training data
            x_sample (torch.Tensor): one test sample
            sample_percentage (float, optional): percentage of features/pixels to be perturbed for each sample. Defaults to 1.

        Returns:
            list of dictionaries: returns perturbations of all test samples x_sample based on an ood strategy, the dictionary includes the original sample and entries for each feature including all perturbations based on the selected strategy
        """
        X_train_flat = self.X_train.flatten(start_dim=1)

        num_pixels = X_train_flat.shape[1]
        num_samples = max(1, int(num_pixels * sample_percentage))
        pixel_indices = np.random.choice(num_pixels, size=num_samples, replace=False)
        ood_all_samples = []
        for sample_tensor in self.X_test:
            sample = sample_tensor
            sample_dict = {}
            sample_dict["original"] = sample_tensor.unsqueeze(0)

            if self.img:
                if self.ood_strategy == "patch_inversion":
                    patch_size = 4  # Example patch size
                    for patch_start in pixel_indices:
                        r = (patch_start // sample.shape[2]) % sample.shape[1]
                        c = patch_start % sample.shape[2]
                        patch_position = (r, c, patch_size, patch_size)
                        patch_variant = self._generate_ood_samples(
                            X_train_flat, sample, patch_position=patch_position
                        )
                        sample_dict[patch_start] = {
                            "variants": patch_variant,
                            "patch_position": patch_position,
                        }
                else:
                    assert (
                        sample_percentage is not None
                    ), "Please provide a sample_percentage for OOD generation on images."
                    sample_flat = sample.reshape(-1)
                    for pixel_idx in pixel_indices:
                        pixel_variants = self._generate_ood_samples(
                            X_train_flat, sample_flat, pixel_idx
                        )
                        pixel_variants = pixel_variants.reshape(
                            pixel_variants.shape[0], *sample.shape
                        )
                        sample_dict[pixel_idx] = pixel_variants
            else:
                for feature_idx in range(self.X_train.shape[1]):
                    feature_variants = self._generate_ood_samples(
                        self.X_train, sample, feature_idx
                    )
                    sample_dict[feature_idx] = feature_variants

            ood_all_samples.append(sample_dict)
        return ood_all_samples
