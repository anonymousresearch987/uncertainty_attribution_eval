from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.models import NeuralNetworkBase
from src.models.props import WineQualityModelProps, ModelTrainingProps, MNISTModelProps


class LinearDropConnect(nn.Linear):
    """Linear Dropconnect layer class"""

    def __init__(self, in_features, out_features, drop_prob, bias=True):
        super(LinearDropConnect, self).__init__(in_features, out_features, bias)
        self.drop_prob: float = drop_prob

    def forward(self, inp):
        """NN forward function

        Args:
            X (torch.Tensor): input to the model
        """
        if self.training:
            # mask over this layer's weights (same shape/device/dtype)
            p_keep = 1.0 - float(self.drop_prob)
            mask = (torch.rand_like(self.weight) < p_keep).to(self.weight.dtype)
            masked_weight = self.weight * mask
            return F.linear(inp, masked_weight, self.bias)
        else:
            return F.linear(inp, self.weight, self.bias)


class MLPWithDropConnect(nn.Module):
    """Neural Network Class with dropconnect"""

    name = "MLPWithDropConnect"

    def __init__(self, model_props: WineQualityModelProps):
        """builds the NN based on model_props. If layers_to_perturb = None, all linear layers have dropconnect enabled, else only the last layers_to_perturb linear layers have dropconnect enabled

        Args:
            model_props (WineQualityModelProps): model properties

        Raises:
            ValueError: raises an error if layers_to_perturb is not None or a positive integer
        """
        super().__init__()  # Correct instantiation
        self.model_props: WineQualityModelProps = model_props
        self.input_size = model_props.input_size
        self.layers_to_perturb = model_props.layers_to_perturb
        self.drop_prob = model_props.drop_prob

        hidden_sizes = []
        first_hidden = getattr(model_props, "hidden_layer_1", None)
        if first_hidden is not None:
            hidden_sizes.append(int(first_hidden))
            for i in range(2, 7):
                layer_size = getattr(model_props, f"hidden_layer_{i}", None)
                if layer_size:
                    hidden_sizes.append(int(layer_size))

        # Determine which layer indices should have dropconnect.
        # We number layers 1..N where last layer (output) has index N.
        total_layers = len(hidden_sizes) + 1
        if self.layers_to_perturb is None:
            perturb_indices = set(range(1, total_layers + 1))
        elif isinstance(self.layers_to_perturb, int) and self.layers_to_perturb > 0:
            start_idx = max(1, total_layers - int(self.layers_to_perturb) + 1)
            perturb_indices = set(range(start_idx, total_layers + 1))
        else:
            raise ValueError("layers_to_perturb must be None or a positive integer")

        # Build hidden layers
        prev_size = self.input_size
        for idx, h_size in enumerate(hidden_sizes, start=1):
            if idx in perturb_indices:
                setattr(self, f"linear{idx}", LinearDropConnect(prev_size, h_size, self.drop_prob))
            else:
                setattr(self, f"linear{idx}", nn.Linear(prev_size, h_size))
            prev_size = h_size

        # Create last/output layer
        if total_layers in perturb_indices:
            self.last_layer = LinearDropConnect(
                prev_size, self.model_props.output_size, self.drop_prob
            )
        else:
            self.last_layer = nn.Linear(prev_size, self.model_props.output_size)

        if self.model_props.force_cpu:
            self.device = "cpu"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        layer_idx = 1
        while hasattr(self, f"linear{layer_idx}"):
            x = F.relu(getattr(self, f"linear{layer_idx}")(x))
            layer_idx += 1
        x = self.last_layer(x)
        return x

    def load_model_from_state_dict(self):
        """Load Model"""
        if self.model_props.state_dict_path:
            if self.model_props.state_dict_path.endswith(".pkl"):
                self.load_state_dict(
                    torch.load(
                        Path(self.model_props.state_dict_path),
                        map_location=torch.device(self.device),
                        weights_only=True,
                    )
                )
        else:
            print("Could not load the state_dict, because the state_dict_path is None!")

    def set_internal_seed(self, seed: int):
        """Set the internal random seed for reproducibility

        Args:
            seed (int): Seed value to set for PyTorch randomness
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class CNNWithDropConnect(nn.Module):
    """Convolutional Neural Network with DropConnect"""

    name = "CNNWithDropConnect"

    def __init__(self, model_props: MNISTModelProps):
        super().__init__()
        self.model_props = model_props
        self.drop_prob = model_props.drop_prob
        self.layers_to_perturb = model_props.layers_to_perturb

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.after_conv_linear = nn.Linear(7 * 7 * 64, model_props.hidden_layer_1)

        hidden_sizes = []
        first_hidden = getattr(model_props, "hidden_layer_2", None)
        if first_hidden is not None:
            hidden_sizes.append(int(first_hidden))
            for i in range(3, 7):
                layer_size = getattr(model_props, f"hidden_layer_{i}", None)
                if layer_size:
                    hidden_sizes.append(int(layer_size))

        total_layers = len(hidden_sizes) + 1
        if self.layers_to_perturb is None:
            perturb_indices = set(range(1, total_layers + 1))
        elif isinstance(self.layers_to_perturb, int) and self.layers_to_perturb > 0:
            start_idx = max(1, total_layers - int(self.layers_to_perturb) + 1)
            perturb_indices = set(range(start_idx, total_layers + 1))
        else:
            raise ValueError("layers_to_perturb must be None or a positive integer")

        # Build hidden layers
        prev_size = model_props.hidden_layer_1
        for idx, h_size in enumerate(hidden_sizes, start=1):
            if idx in perturb_indices:
                setattr(self, f"linear{idx}", LinearDropConnect(prev_size, h_size, self.drop_prob))
            else:
                setattr(self, f"linear{idx}", nn.Linear(prev_size, h_size))
            prev_size = h_size

        # Create last/output layer
        if total_layers in perturb_indices:
            self.last_layer = LinearDropConnect(
                prev_size, self.model_props.output_size, self.drop_prob
            )
        else:
            self.last_layer = nn.Linear(prev_size, self.model_props.output_size)

        if self.model_props.force_cpu:
            self.device = "cpu"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # ensure input has channel dimension (N, C, H, W). Some callers pass (N, H, W).
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.after_conv_linear(x))
        layer_idx = 1
        while hasattr(self, f"linear{layer_idx}"):
            x = F.relu(getattr(self, f"linear{layer_idx}")(x))
            layer_idx += 1
        x = self.last_layer(x)
        return x.to(self.device)

    def load_model_from_state_dict(self):
        """Load Model"""
        if self.model_props.state_dict_path:
            if self.model_props.state_dict_path.endswith(".pkl"):
                self.load_state_dict(
                    torch.load(
                        Path(self.model_props.state_dict_path),
                        map_location=torch.device(self.device),
                        weights_only=True,
                    )
                )
        else:
            print("Could not load the state_dict, because the state_dict_path is None!")

    def set_internal_seed(self, seed: int):
        """Set the internal random seed for reproducibility

        Args:
            seed (int): Seed value to set for PyTorch randomness
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class UQMCDropconnectRegressor(NeuralNetworkBase):
    """Uncertainty Quantification Model with MC Dropconnect"""

    drop_strategy = "dropconnect"
    model_type = "regression"

    def __init__(
        self,
        model_props: WineQualityModelProps,
        training_props: ModelTrainingProps,
        model_trained: bool = False,
    ) -> None:
        """initializes UQMCDropConnectRegressor.

        Args:
            model_props (WineQualityModelProps): model properties
            training_props (ModelTrainingProps): training properties
            model_trained (bool, optional): If model_trained is True, the ensemble is loaded from the previously saved pickle file.
        """
        super().__init__(
            model=MLPWithDropConnect(model_props),
            model_props=model_props,
            training_props=training_props,
        )

        if model_trained:
            self.model, self.ensemble = self.load_ensemble_from_pickle_with_filepath(
                model_props.ensemble_path
            )


class UQMCDropconnectCNNClassifier(NeuralNetworkBase):
    """Uncertainty Quantification CNN Classifier with MC Dropconnect"""

    drop_strategy = "dropconnect"
    model_type = "classification"

    def __init__(
        self,
        model_props: MNISTModelProps,
        training_props: ModelTrainingProps,
        model_trained: bool = False,
    ) -> None:
        """initializes UQMCDropConnectCNNClassifier.

        Args:
            model_props (MNISTModelProps): model properties
            training_props (ModelTrainingProps): training properties
            model_trained (bool, optional): If model_trained is True, the ensemble is loaded from the previously saved pickle file.
        """
        super().__init__(
            model=CNNWithDropConnect(model_props),
            model_props=model_props,
            training_props=training_props,
        )

        if model_trained:
            self.model, self.ensemble = self.load_ensemble_from_pickle_with_filepath(
                model_props.ensemble_path
            )

    def forward_uq(self, x):
        """forwards an input x through the model with dropconnect enabled. specific for classification

        Args:
            x (torch.Tensor): input tensor to the model

        Raises:
            RuntimeError: ensemble is empty

        Returns:
            dict: dictionary containing mean, variance of the UQ enabled model predictions
        """
        soft_max_probs = []
        if self.ensemble is None or len(self.ensemble) == 0:
            self.get_ensemble(save_model=False, ensemble_path="")
            if self.ensemble is None:
                raise RuntimeError("Ensemble is empty.")
        for member in self.ensemble:
            member.train()
            with torch.no_grad():
                device = next(member.parameters()).device
                inp = x.to(device)
                output = member(inp)
                soft_max_probs.append(F.softmax(output, dim=1))

        soft_max_probs_tensor = torch.stack(soft_max_probs, dim=0)
        mean_softmax = torch.mean(soft_max_probs_tensor, dim=0)
        predicted_classes = torch.argmax(mean_softmax, dim=1)
        variance_prediction = torch.var(soft_max_probs_tensor, dim=0, unbiased=False)

        predicted_class_idx = []
        for idx in predicted_classes:
            predicted_class_idx.append(idx)

        return {
            "mean": torch.stack(
                [
                    mean_softmax[row_idx, predicted_class_id]
                    for row_idx, predicted_class_id in enumerate(predicted_class_idx)
                ]
            ),
            "variance": torch.stack(
                [
                    variance_prediction[row_idx, predicted_class_id]
                    for row_idx, predicted_class_id in enumerate(predicted_class_idx)
                ]
            ),
            "predicted_classes": predicted_classes,
        }

    def epistemic_variance(self, x):
        """computes the uncertainty of a model prediction in terms of the variance of the ensemble predictions

        Args:
            x (torch.Tensor): input tensor to the model

        Returns:
            torch.Tensor: variance of the model predictions
        """
        return self.forward_uq(x)["variance"]
