from pathlib import Path

import torch

import torch.nn as nn
import torch.nn.functional as F
from src.models.models import NeuralNetworkBase
from src.models.props import WineQualityModelProps, ModelTrainingProps, MNISTModelProps


class MLPWithDropout(nn.Module):
    """Neural Network Class with special Dropout Config"""

    name = "MLPWithDropout"

    def __init__(self, model_props: WineQualityModelProps):
        """init MLPWithDropout

        Args:
            model_props (WineQualityModelProps): model properties
        """
        super().__init__()
        self.model_props: WineQualityModelProps = model_props
        self.input_size = self.model_props.input_size
        self.layers_to_perturb = model_props.layers_to_perturb
        self.dropout = nn.Dropout(p=self.model_props.drop_prob)

        self.linear1 = nn.Linear(self.input_size, self.model_props.hidden_layer_1)
        # Build additional hidden layers (hidden_layer_2 .. hidden_layer_6) if present in model_props
        prev_size = self.model_props.hidden_layer_1
        for i in range(2, 7):
            layer_attr = f"hidden_layer_{i}"
            layer_size = getattr(self.model_props, layer_attr, None)
            if layer_size:
                setattr(self, f"linear{i}", nn.Linear(prev_size, layer_size))
                prev_size = layer_size

        self.model_props.last_hidden_layer = prev_size
        self.last_layer = nn.Linear(
            self.model_props.last_hidden_layer, self.model_props.output_size
        )
        # get number of hidden layers
        hidden_count = 0
        while hasattr(self, f"linear{hidden_count + 1}"):
            hidden_count += 1

        total_layers = hidden_count + 1
        self.total_layers = total_layers
        # determine which layers (1..total_layers, where total_layers is the output) get dropout based on layers_to_perturb
        if self.layers_to_perturb is None:
            perturb_indices = set(range(1, total_layers + 1))
        elif isinstance(self.layers_to_perturb, int) and self.layers_to_perturb > 0:
            start_idx = max(1, total_layers - int(self.layers_to_perturb) + 1)
            perturb_indices = set(range(start_idx, total_layers + 1))
        else:
            perturb_indices = set()

        self.perturb_indices = perturb_indices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """NN forward function

        Args:
            X (torch.Tensor): input to the model
        """
        layer_idx = 1
        while hasattr(self, f"linear{layer_idx}"):
            x = F.relu(getattr(self, f"linear{layer_idx}")(x))
            # Apply dropout only during training
            if self.training and layer_idx in self.perturb_indices:
                x = self.dropout(x)
            layer_idx += 1

        # apply dropout before last layer if the output layer is in perturb_indices (only during training)
        if self.training and self.total_layers in self.perturb_indices:
            x = self.dropout(x)

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


class CNNWithDropout(nn.Module):
    """Convolutional Neural Network with Dropout"""

    name = "CNNWithDropout"

    def __init__(self, model_props: MNISTModelProps):
        super().__init__()
        self.model_props = model_props
        self.drop_prob = model_props.drop_prob
        self.layers_to_perturb = model_props.layers_to_perturb

        self.dropout = nn.Dropout(p=self.model_props.drop_prob)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(7 * 7 * 64, model_props.hidden_layer_1)

        prev_size = self.model_props.hidden_layer_1
        for i in range(2, 7):
            layer_attr = f"hidden_layer_{i}"
            layer_size = getattr(self.model_props, layer_attr, None)
            if layer_size:
                setattr(self, f"linear{i}", nn.Linear(prev_size, layer_size))
                prev_size = layer_size

        self.model_props.last_hidden_layer = prev_size
        self.last_layer = nn.Linear(
            self.model_props.last_hidden_layer, self.model_props.output_size
        )
        # get number of hidden layers
        hidden_count = 0
        while hasattr(self, f"linear{hidden_count + 1}"):
            hidden_count += 1

        total_layers = hidden_count + 1
        self.total_layers = total_layers
        # determine which layers (1..total_layers, where total_layers is the output) get dropout based on layers_to_perturb
        if self.layers_to_perturb is None:
            perturb_indices = set(range(1, total_layers + 1))
        elif isinstance(self.layers_to_perturb, int) and self.layers_to_perturb > 0:
            start_idx = max(1, total_layers - int(self.layers_to_perturb) + 1)
            perturb_indices = set(range(start_idx, total_layers + 1))
        else:
            perturb_indices = set()

        self.perturb_indices = perturb_indices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """CNN forward function

        Args:
            X (torch.Tensor): input to the model
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        if x.size(1) != self.linear1.in_features:
            x = x.view(x.size(0), -1)
        layer_idx = 1
        while hasattr(self, f"linear{layer_idx}"):
            x = F.relu(getattr(self, f"linear{layer_idx}")(x))
            # Apply dropout only during training
            if self.training and layer_idx in self.perturb_indices:
                x = self.dropout(x)
            layer_idx += 1

        # apply dropout before last layer if the output layer is in perturb_indices (only during training)
        if self.training and self.total_layers in self.perturb_indices:
            x = self.dropout(x)

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


class UQMCDropoutRegressor(NeuralNetworkBase):
    """Uncertainty Quantification Model with MC Dropout"""

    drop_strategy = "dropout"
    model_type = "regression"

    def __init__(
        self,
        model_props: WineQualityModelProps,
        training_props: ModelTrainingProps,
        model_trained: bool = False,
    ) -> None:
        """initializes UQMCDropoutRegressor.

        Args:
            model_props (WineQualityModelProps): model properties
            training_props (ModelTrainingProps): training properties
            model_trained (bool, optional): If model_trained is True, the ensemble is loaded from the previously saved pickle file.
        """
        super().__init__(
            model=MLPWithDropout(model_props),
            model_props=model_props,
            training_props=training_props,
        )
        if model_trained:
            self.model, self.ensemble = self.load_ensemble_from_pickle_with_filepath(
                model_props.ensemble_path
            )


class UQMCDropoutCNNClassifier(NeuralNetworkBase):
    """Uncertainty Quantification CNN Classifier with MC Dropout"""

    drop_strategy = "dropout"
    model_type = "classification"

    def __init__(
        self,
        model_props: MNISTModelProps,
        training_props: ModelTrainingProps,
        model_trained: bool = False,
    ) -> None:
        """initializes UQMCDropoutCNNClassifier.

        Args:
            model_props (MNISTModelProps): mnist model properties
            training_props (ModelTrainingProps): training properties
            model_trained (bool, optional): If model_trained is True, the ensemble is loaded from the previously saved pickle file.
        """
        super().__init__(
            model=CNNWithDropout(model_props),
            model_props=model_props,
            training_props=training_props,
        )

        if model_trained:
            self.model, self.ensemble = self.load_ensemble_from_pickle_with_filepath(
                model_props.ensemble_path
            )

    def forward_uq(self, x):
        """forwards an input x through the model with dropout enabled (classification)
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
                input = x.to(device)
                output = member(input)
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
