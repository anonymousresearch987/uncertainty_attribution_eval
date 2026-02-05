import os
from typing import Optional, List
from pathlib import Path

import statistics
import copy
import pickle
import torch
import torch.nn as nn
import properscoring as ps

from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import sklearn.metrics as sklm
from src.models.props import ModelProps, ModelTrainingProps
from src.utils.xuq_utils import gaussian_nll_loss, custom_one_hot
from sklearn.metrics import brier_score_loss


class MakeNNDataset(Dataset):
    """make dataset

    Args:
        Dataset (_type_): torch Dataset
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetworkBase(nn.Module):
    """Base class for Neural Network Models with training and evaluation methods"""

    def __init__(
        self,
        model: nn.Module,
        model_props: ModelProps,
        training_props: ModelTrainingProps,
    ) -> None:
        """initialize NeuronalNetworkBase

        Args:
            model (nn.Module): Neural Network Model
            model_props (WineQualityModelProps): model properties
            training_props (ModelTrainingProps): training properties
        """
        super().__init__()
        self.model = model
        self.model_props = model_props
        self.training_props = training_props
        self.ensemble: Optional[List[nn.Module]] = None
        self.state_dict_path: Optional[str] = self.model_props.state_dict_path
        self.early_stopping_patience = self.training_props.early_stopping_patience
        self.lr_scheduler_factor = self.training_props.lr_scheduler_factor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.training_props.loss_function == "MSE":
            self.loss_function = nn.MSELoss()
        if self.training_props.loss_function == "CEL":
            self.loss_function = nn.CrossEntropyLoss()

        if self.training_props.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(  # type: ignore
                self.parameters(),
                lr=self.training_props.learn_rate,
                weight_decay=self.training_props.weight_decay,
            )
        if self.training_props.optimizer == "Adagrad":
            self.optimizer = torch.optim.Adagrad(  # type: ignore
                self.parameters(),
                lr=self.training_props.learn_rate,
                weight_decay=self.training_props.weight_decay,
            )
        elif self.training_props.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(  # type: ignore
                self.parameters(),
                lr=self.training_props.learn_rate,
                momentum=self.training_props.momentum,
            )
        elif self.training_props.optimizer == "RMSProp":
            self.optimizer = torch.optim.RMSprop(  # type: ignore
                self.parameters(),
                lr=self.training_props.learn_rate,
                weight_decay=self.training_props.weight_decay,
                momentum=self.training_props.momentum,
            )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """Trains the Neural Network with the train data. The val data is validating the
        Model while it is trained.

        Args:
            train_loader (DataLoader): train data loader
            val_loader (DataLoader): val data loader

        Returns:
            train_loss: the loss on the train set while model was trained
            val_loss: the loss on the validation set while model was trained
        """
        print("Train model with: %s", str(self.device))
        torch.manual_seed(42)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", factor=self.lr_scheduler_factor
        )
        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        early_stopping_counter = 0
        for epoch in range(0, self.training_props.epochs):
            print("Epochs: %s", str(epoch))
            losses = []
            count = 0
            self.model.train()
            for x, y in iter(train_loader):
                count += 1
                x, y = self.bring_vector_to_device(x, y)
                pred_y: torch.Tensor = self.model(x)
                self.optimizer.zero_grad()
                loss = self.loss_function(pred_y.squeeze(), y.squeeze())
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

            train_loss = statistics.mean(losses)
            if epoch % 10 == 0:
                val_loss, _, _ = self.compute_loss_and_collect_predictions(val_loader)
                print(f"Train Loss: {train_loss} - Validation Loss {val_loss}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if self.early_stopping_patience:
                scheduler.step(val_loss)
                if round(val_loss, 4) < round(best_val_loss, 4):
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.early_stopping_patience:
                        print("Early stopping at epoch %s ", epoch + 1)
                        break
        return (
            train_loss,
            val_loss,
        )

    def score(self, test_loader):
        """Evaluate model on Test Set
        Args:
            test_X (pd.DataFrame): Test data
            test_y (pd.DataFrame): Test labels

        Returns:
            test_loss: test loss
            predictions: model predictions on test set
            ground_truths: ground truths of test set
        """
        self.model.eval()
        with torch.no_grad():
            test_loss, predictions, ground_truths = self.compute_loss_and_collect_predictions(
                test_loader
            )
            print("Final Test Loss: %s", str(test_loss))
        return test_loss, predictions, ground_truths

    def fit_and_evaluate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        save_model: bool = True,
        calibration_metrics: bool = True,
    ):
        """fits and evaluates the model

        Args:
            train_loader (DataLoader): train data loader
            val_loader (DataLoader): val data loader
            test_loader (DataLoader): test data loader
            save_model (bool, optional): Whether to save the model and ensemble after training. Defaults to True.
            calibration_metrics (bool, optional): Whether to compute calibration metrics during evaluation. Defaults to True.

        Returns:
            evaluation_metrics: evaluation metrics of the UQ enabled model
        """

        self.fit(
            train_loader,
            val_loader,
        )
        print("start testing model")
        self.score(test_loader)
        print("finished testing model")
        current_run_dir = "/workspaces/expainable-uncertainty-quantification/results/train_models"
        if not os.path.exists(current_run_dir):
            os.makedirs(current_run_dir)
        model_path = current_run_dir + "/" + str(self.model.name) + "_model.pkl"
        ensemble_path = current_run_dir + "/" + str(self.model.name) + "_ensemble.pkl"
        self.ensemble = self.get_ensemble(save_model=save_model, ensemble_path=ensemble_path)
        if save_model:
            self.save_model(Path(model_path))
        evaluation_metrics = self.evaluate_uq(test_loader, calibration_metrics=calibration_metrics)
        return evaluation_metrics

    def fit_and_calibrate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        drop_prob: float,
    ) -> dict[str, float]:
        """fit and calibrate drop rate for uq

        Args:
            train_loader (DataLoader): train dataset
            val_loader (DataLoader): val dataset
            test_loader (DataLoader): test dataset
            drop_prob (float): drop probabilities

        Returns:
            dict[str, float]: results dictionary
        """
        self.model_props.drop_prob = drop_prob
        self.fit(
            train_loader,
            val_loader,
        )
        evaluation_metrics = self.evaluate_uq(val_loader, calibration_metrics=True)
        self.score(test_loader)
        return evaluation_metrics

    def enable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def evaluate_uq(self, test_loader: DataLoader, calibration_metrics: bool = True):
        """evaluate the UQ model predictions. If calibration_metrics is True, NLL and CRPS are reported in addition to MSE, RMSE, R2, and MAPE for regression tasks and CEL, Accuracy, Recall, Precision, and F1-Score for classification tasks.

        Args:
            test_loader (DataLoader): test data
            calibration_metrics: if uq calibration metrics should be computed

        Returns:
            dict: dictionary containing evaluation metrics: mse_loss, rmse_loss, r2_score, and mape_loss, and if calibration_metrics is True, nll_loss and crps for regression tasks; cel_loss, accuracy, recall, precision, and f1_score, and if calibration_metrics is True, nll_loss and brier_score for classification tasks
        """
        results = {}
        mse_loss = 0.0
        rmse_loss = 0.0
        r2_score = 0.0
        mape_loss = 0.0

        cel_loss = 0.0
        accuracy = 0.0
        recall = 0.0
        precision = 0.0
        f1_score = 0.0

        nll_loss = 0.0
        crps = 0.0
        crps_ens = 0.0
        brier = 0.0
        if self.ensemble is None or len(self.ensemble) == 0:
            self.ensemble = self.get_ensemble(save_model=False, ensemble_path="")
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)

                # Initialize empty predictions
                output_size = self.model_props.output_size
                y_ = torch.zeros(
                    size=(
                        self.model_props.forward_passes,
                        batch_size,
                        output_size,
                    ),
                    device=self.device,
                )

                if self.ensemble and len(self.ensemble) > 0:
                    for i in range(len(self.ensemble)):
                        self.ensemble[i].train()  # enable dropout/dropconnect
                        y_[i] = self.ensemble[i](data)

                mean_y = y_.mean(dim=0)

                if self.model_props.is_classification:
                    cel_loss += F.cross_entropy(mean_y, target.long()).item()
                    _, predicted = torch.max(mean_y.data, 1)
                    accuracy += (predicted == target).sum().item() / batch_size
                    recall += sklm.recall_score(
                        y_true=target.cpu(), y_pred=predicted.cpu(), average="macro"
                    )
                    precision += sklm.precision_score(
                        y_true=target.cpu(), y_pred=predicted.cpu(), average="macro"
                    )
                    f1_score += sklm.f1_score(
                        y_true=target.cpu(), y_pred=predicted.cpu(), average="macro"
                    )
                    if calibration_metrics:
                        # NLL calculation
                        one_hot_targets = custom_one_hot(
                            target.cpu(), length=self.model_props.output_size
                        )
                        probs = F.softmax(mean_y, dim=1).cpu().numpy()
                        nll_loss += -np.mean(
                            np.sum(one_hot_targets * np.log(probs + 1e-15), axis=1)
                        )
                        # CRPS calculation
                        brier_scores = []
                        for class_idx in range(self.model_props.output_size):
                            true_binary = (target.cpu().numpy() == class_idx).astype(int)
                            prob_pos = probs[:, class_idx]
                            brier_score = brier_score_loss(true_binary, prob_pos)
                            brier_scores.append(brier_score)
                        brier += np.mean(brier_scores)

                else:
                    mse_loss += F.mse_loss(mean_y.squeeze(), target.squeeze()).item()
                    rmse_loss += sklm.root_mean_squared_error(
                        y_pred=mean_y.squeeze(), y_true=target.squeeze()
                    )
                    r2_score += sklm.r2_score(y_true=target.squeeze(), y_pred=mean_y.squeeze())
                    mape_loss += sklm.mean_absolute_percentage_error(
                        y_true=target.squeeze(), y_pred=mean_y.squeeze()
                    )
                    crps_ens += np.array(
                        [
                            ps.crps_ensemble(obs.squeeze(), mean_y[i].squeeze())
                            for i, obs in enumerate(target)
                        ],
                        dtype=np.float32,
                    ).mean()
                    if calibration_metrics:
                        nll_loss += gaussian_nll_loss(
                            mean_y.mean().numpy(), mean_y.std().numpy(), target
                        ).item()
                        crps += ps.crps_gaussian(target, mean_y.mean(), mean_y.std()).mean()  # type: ignore

        if not self.model_props.is_classification:
            results["mse_loss"] = mse_loss / len(test_loader)
            results["rmse_loss"] = rmse_loss / len(test_loader)
            results["r2_score"] = r2_score / len(test_loader)
            results["mape_loss"] = mape_loss / len(test_loader)
            if calibration_metrics:
                results["nll_loss"] = nll_loss / len(test_loader)
                results["crps"] = crps / len(test_loader)
                results["crps_ens"] = crps_ens / len(test_loader)
        else:
            results["cel_loss"] = cel_loss / len(test_loader)
            results["accuracy"] = accuracy / len(test_loader)
            results["recall"] = recall / len(test_loader)
            results["precision"] = precision / len(test_loader)
            results["f1_score"] = f1_score / len(test_loader)
            if calibration_metrics:
                results["nll_loss"] = nll_loss / len(test_loader)

                results["brier_score"] = brier / len(test_loader)
        return results

    def predict(self, data, target) -> tuple:
        """Get predictions and real values

        Args:
            data (array like): input data
            target (array like): target data

        Returns:
            tuple: predictions and ground truths as pandas DataFrames
        """
        val_dataset = MakeNNDataset(data, target)
        val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
        pred_y, real_y = [], []
        with torch.no_grad():
            self.model.eval()
            for x, y in iter(val_loader):
                x, y = self.bring_vector_to_device(x, y)

                output = self.model(x)
                pred_y.extend(output.tolist())
                real_y.extend(y.tolist())
        return pd.DataFrame(pred_y), pd.DataFrame(real_y)

    def compute_loss_and_collect_predictions(
        self, data_loader: DataLoader
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute MSE and collect predictions for visualization.

        Args:
            data_loader (DataLoader): DataLoader

        Returns:
            tuple[float, NDArray, NDArray]: loss, predictions and real values
        """
        losses = []
        predictions = []
        ground_truths = []
        self.model.eval()
        for x, y in iter(data_loader):
            x, y = self.bring_vector_to_device(x, y)

            pred_y = self.model(x)
            loss = self.loss_function(pred_y.squeeze(), y.squeeze())
            losses.append(loss.item())

            # Collect predictions and ground truth
            predictions.extend(pred_y.squeeze().cpu().tolist())
            ground_truths.extend(y.squeeze().cpu().tolist())

        loss = statistics.mean(losses)
        predictions = np.array(predictions).reshape(-1, 1)
        ground_truths = np.array(ground_truths).reshape(-1, 1)
        return loss, predictions, ground_truths

    def validate(self, data, target) -> tuple:
        """Model validation"""
        data = torch.Tensor(data).to(torch.float32)
        target = torch.Tensor(target.to_numpy()).to(torch.float32)  # type:ignore
        return self.predict(data, target)

    def bring_vector_to_device(self, x: torch.Tensor, y: torch.Tensor):
        """bring vector to device"""
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def reset_model_weights(self):
        """Reset the Model weights so that it can be retrained."""
        print("reset model weights")
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def load_model_from_state_dict(self):
        """Load Model"""
        if self.state_dict_path:
            if self.state_dict_path.endswith(".pkl"):
                self.load_state_dict(
                    torch.load(
                        Path(self.state_dict_path),
                        map_location=torch.device(self.device),
                        weights_only=True,
                    )
                )
        else:
            print("Could not load the state_dict, because the state_dict_path is None!")

    def save_model(self, path: Path) -> None:
        """Save model"""
        print(f"Save model to path: {str(path)}")

        return torch.save(self.model.state_dict(), path)

    def get_ensemble(self, save_model: bool, ensemble_path: str | None) -> list[nn.Module]:
        """method to get the ensemble of trained models

        Args:
            save_model (bool): save model as pickle or not
            ensemble_path (str): where the base_model and ensemble pickle files would be saved

        Returns:
            list[nn.Module]: the ensemble
        """
        ensemble = []
        for m in range(self.model_props.forward_passes):
            self.model.train()
            model_copy = copy.deepcopy(self.model)
            model_copy.set_internal_seed(m)
            ensemble.append(model_copy)
        self.ensemble = ensemble

        if save_model and ensemble_path:
            self.save_ensemble_as_pickle(ensemble_path)
        return ensemble

    def save_ensemble_as_pickle(self, filepath):
        """Save list of PyTorch model state dicts as a pickle file.

        Args:
            filepath (str): Path where the pickle file will be saved.
        """
        if self.ensemble is None:
            raise ValueError("Ensemble is None, cannot save.")
        else:
            model_state_dicts = [model.state_dict() for model in self.ensemble]
        model_state_dicts.append(self.model.state_dict())
        print(f"saving model to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(model_state_dicts, f)

    def load_ensemble_from_pickle(self):
        """Load list of models from pickle file and re-instantiate them

        Returns:
            list: List of re-instantiated models with their states loaded.
        """
        filepath = Path(self.model_props.ensemble_path)
        with open(filepath, "rb") as f:
            model_state_dicts = pickle.load(f)
        ensemble = []
        for state_dict in model_state_dicts:
            model_copy = copy.deepcopy(self.model)
            model_copy.load_state_dict(state_dict)
            model_copy.to(next(self.model.parameters()).device)
            model_copy.eval()
            ensemble.append(model_copy)
        base_model = ensemble[-1]
        return base_model, ensemble[:-1]

    def load_ensemble_from_pickle_with_filepath(self, filepath: str | Path):
        """Load list of models from pickle file and re-instantiate them.

        Args:
            filepath (str|Path): Path to the pickle file containing model state dicts.
        Returns:
            list: List of re-instantiated models with their states loaded.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Ensemble file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_state_dicts = pickle.load(f)

        if not model_state_dicts:
            raise ValueError("Loaded pickle contains no state dicts.")

        ensemble = []
        for state_dict in model_state_dicts:
            # create a fresh copy of the estimator and load the saved weights into it
            model_copy = copy.deepcopy(self.model)
            model_copy.load_state_dict(state_dict)
            # place on same device as the original estimator and set eval mode
            model_copy.to(next(self.model.parameters()).device)
            model_copy.eval()
            ensemble.append(model_copy)

        base_model = ensemble[-1]
        return base_model, ensemble[:-1]

    def forward_uq(self, x):
        """forwards an input x through the model with dropout/dropconnect enabled.

        Args:
            x (torch.Tensor): input tensor to the model

        Raises:
            RuntimeError: ensemble is empty

        Returns:
            dict: dictionary containing mean, variance of the UQ enabled model predictions
        """
        predictions = []
        if self.ensemble is None or len(self.ensemble) == 0:
            self.get_ensemble(save_model=False, ensemble_path="")
            if self.ensemble is None:
                raise RuntimeError("Ensemble is empty.")
        for member in self.ensemble:
            member.train()
            with torch.no_grad():
                device = next(member.parameters()).device
                input = x.to(device)
                predictions.append(member(input))

        predictions_tensor = torch.stack(predictions)

        mean_prediction = torch.mean(predictions_tensor, dim=0)
        variance_prediction = torch.var(predictions_tensor, dim=0, unbiased=False)

        return {
            "mean": mean_prediction,
            "variance": variance_prediction,
        }

    def epistemic_variance(self, x):
        """computes the uncertainty of a model prediction in terms of the variance of ensemble predictions

        Args:
            x (torch.Tensor): input tensor to the model

        Returns:
            torch.Tensor: variance of the model predictions
        """
        return self.forward_uq(x)["variance"]
