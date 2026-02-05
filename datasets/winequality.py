import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from datasets import Dataset
from src.utils.xuq_utils import numpy_to_torch
from sklearn.model_selection import KFold
import torch


class WineQualityDataset(Dataset):
    """
    Most parts are based on  https://github.com/florianbley/XAI-2ndOrderUncertainty/tree/master."""

    name = "Wine Quality Dataset"

    def check_if_dataset_exists(self, dataset_path):
        """Checks if the data has already been loaded.

        Args:
            dataset_path (str): path to the dataset

        Returns:
            bool: True if dataset exists, False otherwise
        """
        data_root = os.getcwd()
        for data_object in [
            "df_train_input",
            "df_test_input",
            "df_train_output",
            "df_test_output",
        ]:
            object_exists = os.path.exists(
                os.path.join(data_root, dataset_path + "/" + data_object)
            )
            if not object_exists:
                return False
        return True

    def get_rawdata_paths(self) -> tuple[str, str]:
        """get the paths of the raw wine quality csv files

        Returns:
            Tuple[str, str]: paths to the raw red and white wine dataset files
        """
        data_path = "Wine Quality"
        raw_data_path_red = data_path + "/winequality-red_komma.csv"
        raw_data_path_white = data_path + "/winequality-white_komma.csv"
        root_dir = os.getcwd()
        if "datasets" not in root_dir:
            raw_data_path_red = "datasets/" + raw_data_path_red
            raw_data_path_white = "datasets/" + raw_data_path_white
        return raw_data_path_red, raw_data_path_white

    def get_dataset_path(self) -> str:
        """get the Wine Quality data path

        Returns:
            str: data path
        """
        data_path = "Wine Quality"
        data_root = os.getcwd()
        if "datasets" not in data_root:
            data_path = "datasets/" + data_path
        return data_path

    def create_data(
        self, dataset_path_red: str, dataset_path_white: str
    ) -> pd.DataFrame:
        """Load dataset from raw files

        Args:
            dataset_path_red (str): path to the red wine dataset file
            dataset_path_white (str): path to the white wine dataset file

        Returns:
            pandas.DataFrame: concatenated dataframe of red and white wine datasets
        """
        np.random.seed(0)
        df_red = pd.read_csv(dataset_path_red, delimiter=";", decimal=".")
        df_white = pd.read_csv(dataset_path_white, delimiter=";", decimal=".")
        df = pd.concat([df_red, df_white])
        return df

    def create_data_one_split(
        self, df: pd.DataFrame, test_split: float, dataset_path: str
    ) -> tuple:
        """Split data into train and test

        Args:
            df (pandas.DataFrame): dataset as pandas DataFrame
            dataset_path (str): path to save the dataset

        Returns:
            tuple (X_train, X_test, y_train, y_test): containing train and test splits including labels as numpy arrays
        """
        shuffle_int = np.random.permutation(range(len(df)))
        df = df.iloc[shuffle_int]

        len_test_set = int(test_split * len(df))

        df_train = df.iloc[:-len_test_set]
        scaler = StandardScaler()
        scaler.fit(df_train)
        df[df.columns] = scaler.transform(df[df.columns])

        input_cols = [col for col in df.columns if col != "quality"]
        df_input = df[input_cols]
        df_target = df["quality"]

        df_X_train, df_X_test = df_input[:-len_test_set], df_input[-len_test_set:]
        df_y_train, df_y_test = df_target[:-len_test_set], df_target[-len_test_set:]

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        df_X_train.to_pickle(dataset_path + "/df_train_input")
        df_y_train.to_pickle(dataset_path + "/df_train_output")
        df_X_test.to_pickle(dataset_path + "/df_test_input")
        df_y_test.to_pickle(dataset_path + "/df_test_output")
        return df_X_train.values, df_X_test.values, df_y_train.values, df_y_test.values

    def load_data(self, dataset_path):
        X_train = pd.read_pickle(dataset_path + "/df_train_input").values
        y_train = pd.read_pickle(dataset_path + "/df_train_output").values
        X_test = pd.read_pickle(dataset_path + "/df_test_input").values
        y_test = pd.read_pickle(dataset_path + "/df_test_output").values
        return X_train, X_test, y_train, y_test

    def serve_dataset(
        self, val_split: float = 0.1, test_split: float = 0.2, val_set_required=True
    ) -> tuple:
        """serves the dataset as train set, test set, (and val set, if val_set_required = True)

        Args:
            val_set_required (bool, optional): Whether to include a validation set. Defaults to False.

        Returns:
            tuple of numpy.ndarray: returns X_train, X_test, (X_val), y_train, y_test, (y_val)
        """
        if val_set_required:
            assert (
                val_split is not None and 0 < val_split < 1
            ), "val_split must be between 0 and 1."
        dataset_path = self.get_dataset_path()
        raw_data_path_red, raw_data_path_white = self.get_rawdata_paths()
        dataset_exists = self.check_if_dataset_exists(dataset_path)
        if dataset_exists:
            X_train, X_test, y_train, y_test = self.load_data(dataset_path)

        else:
            df = self.create_data(raw_data_path_red, raw_data_path_white)
            (X_train, X_test, y_train, y_test) = self.create_data_one_split(
                df, test_split, dataset_path
            )

        y_train = np.asarray(y_train).reshape(-1, 1)
        y_test = np.asarray(y_test).reshape(-1, 1)

        if val_set_required:
            n_train = int(X_train.shape[0] * (1 - val_split))
            X_train, X_val = X_train[:n_train], X_train[n_train:]
            y_train, y_val = y_train[:n_train], y_train[n_train:]
            return X_train, X_test, X_val, y_train, y_test, y_val
        else:
            return X_train, X_test, [], y_train, y_test, []

    def get_feature_names(self) -> list[str]:
        """get feature names of the wine quality dataset"""
        raw_data_path_red, raw_data_path_white = self.get_rawdata_paths()
        df_red = pd.read_csv(raw_data_path_red, delimiter=";", decimal=".")
        return list(df_red.columns[:-1])

    def serve_dataset_as_dataloader(self, batch_size: int = 32):
        """load winequality data set DataLoader Objects

        Args:
            batch_size (int, optional): training batch size. Defaults to 32.

        Returns:
            train_loader, val_loader, test_loader: DataLoader
        """
        # Assuming numpy_to_torch converts numpy arrays to PyTorch tensors
        X_train, X_test, X_val, y_train, y_test, y_val = numpy_to_torch(
            self.serve_dataset(val_set_required=True)
        )
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def serve_dataset_as_folds_for_calibration(
        self, k_folds: int = 5, val_split: float = 0.1, random_state: int = 42
    ) -> dict:
        """
        Split the provided dataframe into k folds and apply a StandardScaler fitted
        on each fold's training portion. Returns a dict including the train, val and test sets as torch.Tensors.

        Validation set is 10% of the fold's training data as default.
        Args:
            k_folds (int, optional): number of folds for cross-validation. Defaults to 5.
            val_split (float, optional): fraction of training data to use for validation. Defaults to 0.1.
            random_state (int, optional): random state for reproducibility. Defaults to 42.
        Returns:
            folds_dict: dict
                {'fold_1': {'X_train': Tensor, 'X_val': Tensor, 'X_test': Tensor,
                            'y_train': Tensor, 'y_val': Tensor, 'y_test': Tensor}, ...}
        """
        raw_data_path_red, raw_data_path_white = self.get_rawdata_paths()

        df = self.create_data(raw_data_path_red, raw_data_path_white)
        rng = np.random.RandomState(random_state)
        shuffle_int = rng.permutation(len(df))
        df_shuf = df.iloc[shuffle_int].reset_index(drop=True)

        input_cols = [col for col in df_shuf.columns if col != "quality"]
        df_input = df_shuf[input_cols].values
        df_target = df_shuf["quality"].values

        kf = KFold(n_splits=k_folds, shuffle=False)

        folds_dict = {}
        fold_number = 1
        for train_idx, test_idx in kf.split(df_input):
            scaler = StandardScaler().fit(df_input[train_idx])
            X_train_np_full = scaler.transform(df_input[train_idx])
            X_test_np = scaler.transform(df_input[test_idx])

            y_train_np_full = np.asarray(df_target[train_idx]).reshape(-1, 1)
            y_test_np = np.asarray(df_target[test_idx]).reshape(-1, 1)

            n_train = int(X_train_np_full.shape[0] * (1 - val_split))
            X_train_np = X_train_np_full[:n_train]
            X_val_np = X_train_np_full[n_train:]
            y_train_np = y_train_np_full[:n_train]
            y_val_np = y_train_np_full[n_train:]

            X_train = torch.from_numpy(X_train_np.astype(np.float32))
            X_test = torch.from_numpy(X_test_np.astype(np.float32))
            y_train = torch.from_numpy(y_train_np.astype(np.float32))
            y_test = torch.from_numpy(y_test_np.astype(np.float32))
            X_val = (
                torch.from_numpy(X_val_np.astype(np.float32))
                if X_val_np.size
                else torch.empty((0, X_train.shape[1]), dtype=torch.float32)
            )
            y_val = (
                torch.from_numpy(y_val_np.astype(np.float32))
                if y_val_np.size
                else torch.empty((0, 1), dtype=torch.float32)
            )

            folds_dict[f"fold_{fold_number}"] = {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            }
            fold_number += 1

        return folds_dict

    def serve_dataset_as_folds(
        self,
        k_folds: int = 5,
        val_set_required: bool = True,
        val_split: float = 0.1,
        random_state: int = 42,
    ) -> dict:
        """
        Create k folds and fit a StandardScaler on each fold's training portion.
        Like serve_dataset_as_folds, each fold can include a validation split (default: 10% of the fold train).
        Returns dict: {'fold_1': {'X_train': Tensor, 'X_val': Tensor|None, 'X_test': Tensor, 'y_train': Tensor,
                                'y_val': Tensor|None, 'y_test': Tensor}, ...}
        """

        if val_set_required:
            assert (
                val_split is not None and 0 < val_split < 1
            ), "val_split must be between 0 and 1."

        raw_data_path_red, raw_data_path_white = self.get_rawdata_paths()

        df = self.create_data(raw_data_path_red, raw_data_path_white)

        rng = np.random.RandomState(random_state)
        shuffle_int = rng.permutation(len(df))
        df_shuf = df.iloc[shuffle_int].reset_index(drop=True)

        input_cols = [col for col in df_shuf.columns if col != "quality"]
        df_input = df_shuf[input_cols].values
        df_target = df_shuf["quality"].values

        kf = KFold(n_splits=k_folds, shuffle=False)

        folds_dict = {}
        fold_number = 1
        for train_idx, test_idx in kf.split(df_input):
            scaler = StandardScaler().fit(df_input[train_idx])
            X_train_np_full = scaler.transform(df_input[train_idx])
            X_test_np = scaler.transform(df_input[test_idx])

            y_train_np_full = np.asarray(df_target[train_idx]).reshape(-1, 1)
            y_test_np = np.asarray(df_target[test_idx]).reshape(-1, 1)

            if val_set_required:
                n_train = int(X_train_np_full.shape[0] * (1 - val_split))
                X_train_np = X_train_np_full[:n_train]
                X_val_np = X_train_np_full[n_train:]
                y_train_np = y_train_np_full[:n_train]
                y_val_np = y_train_np_full[n_train:]
            else:
                X_train_np = X_train_np_full
                X_val_np = None
                y_train_np = y_train_np_full
                y_val_np = None

            X_train = torch.from_numpy(X_train_np.astype(np.float32))
            X_test = torch.from_numpy(X_test_np.astype(np.float32))
            y_train = torch.from_numpy(y_train_np.astype(np.float32))
            y_test = torch.from_numpy(y_test_np.astype(np.float32))
            X_val = (
                torch.from_numpy(X_val_np.astype(np.float32))
                if X_val_np is not None
                else None
            )
            y_val = (
                torch.from_numpy(y_val_np.astype(np.float32))
                if y_val_np is not None
                else None
            )

            folds_dict[f"fold_{fold_number}"] = {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            }
            fold_number += 1

        return folds_dict
