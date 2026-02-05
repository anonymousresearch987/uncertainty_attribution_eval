import time
import matplotlib.pyplot as plt
import numpy as np
import statistics

from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset
from src.models.models import NeuralNetworkBase
from src.pipelines.train_pipelines import CalibrationPipelineProps


def plot_calibration_results(
    metrics_dict, drop_probabilities, best_indices, model_type, output_path
):
    """
    Plot calibration results for both classification and regression models.

    Args:
        metrics_dict: Dictionary containing metric arrays
        drop_probabilities: List of drop probability values
        best_indices: Dictionary with best indices for each metric
        model_type: "classification" or "regression"
        output_path: Path to save the figure
    """
    if model_type == "classification":
        fig, ax = plt.subplots(1, 3, figsize=(25, 10))

        nll_all = metrics_dict["nll_all"]
        brier_all = metrics_dict["brier_all"]
        cel_all = metrics_dict["cel_all"]

        for key in nll_all.keys():
            ax[0].plot(drop_probabilities, nll_all[key], "-o", label=key)
        ax[0].legend()
        ax[0].set_ylabel("Negative Log Likelihood")

        for key in brier_all.keys():
            ax[1].plot(drop_probabilities, brier_all[key], "-o", label=key)
        ax[1].legend()
        ax[1].set_ylabel("brier score")

        for key in cel_all.keys():
            ax[2].plot(drop_probabilities, cel_all[key], "-o", label=key)
        ax[2].legend()
        ax[2].set_ylabel("CEL score")

        nll_best = best_indices["nll"]
        brier_best = best_indices["brier"]
        cel_best = best_indices["cel"]

        print(
            f"Best p (mode over best p's for each fold):\n"
            f"  Best drop rate {drop_probabilities[nll_best]} w.r.t. NLL\n"
            f"  Best drop rate {drop_probabilities[brier_best]} w.r.t. Brier\n"
            f"  Best drop rate {drop_probabilities[cel_best]} w.r.t CEL\n"
        )

    elif model_type == "regression":
        fig, ax = plt.subplots(1, 4, figsize=(25, 10))

        nll_all = metrics_dict["nll_all"]
        crps_all = metrics_dict["crps_all"]
        mse_all = metrics_dict["mse_all"]
        crps_ens_all = metrics_dict["crps_ens_all"]

        for key in nll_all.keys():
            ax[0].plot(drop_probabilities, nll_all[key], "-o", label=key)
        ax[0].legend()
        ax[0].set_ylabel("Negative Log Likelihood")

        for key in crps_all.keys():
            ax[1].plot(drop_probabilities, crps_all[key], "-o", label=key)
        ax[1].legend()
        ax[1].set_ylabel("CRPS gaussian score")

        for key in mse_all.keys():
            ax[2].plot(drop_probabilities, mse_all[key], "-o", label=key)
        ax[2].legend()
        ax[2].set_ylabel("MSE score")

        for key in crps_ens_all.keys():
            ax[3].plot(drop_probabilities, crps_ens_all[key], "-o", label=key)
        ax[3].legend()
        ax[3].set_ylabel("CRPS ensemble score")

        nll_best = best_indices["nll"]
        crps_best = best_indices["crps"]
        mse_best = best_indices["mse"]
        crps_ens_best = best_indices["crps_ens"]

        print(
            f"Best p (mode over best p's for each fold):\n"
            f"  Best drop rate {drop_probabilities[nll_best]} w.r.t. NLL\n"
            f"  Best drop rate {drop_probabilities[crps_best]} w.r.t. Gaussian CRPS\n"
            f"  Best drop rate {drop_probabilities[crps_ens_best]} w.r.t. Ensemble CRPS\n"
            f"  Best drop rate {drop_probabilities[mse_best]} w.r.t MSE\n"
        )

    fig.savefig(output_path)


class CalibrationPipline:
    def pipeline(
        self, dataset: Dataset, model: NeuralNetworkBase, pipeline_props: CalibrationPipelineProps
    ):
        drop_probabilities = pipeline_props.drop_probabilities
        folds_dict = dataset.serve_dataset_as_folds_for_calibration(
            val_split=0.1, k_folds=pipeline_props.k_folds
        )

        if model.model_type == "classification":
            nll_all = {}
            cel_all = {}
            brier_all = {}

            nll_argmin = []
            cel_argmin = []
            brier_argmin = []

            time_fold = time.time()
            for key in folds_dict.keys():
                nll = []
                brier = []
                cel = []
                if key == "test_set":
                    continue
                train_dataset = TensorDataset(
                    folds_dict[key]["X_train"], folds_dict[key]["y_train"]
                )
                val_dataset = TensorDataset(folds_dict[key]["X_val"], folds_dict[key]["y_val"])
                train_loader = DataLoader(
                    train_dataset, batch_size=model.training_props.batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=model.training_props.batch_size, shuffle=False
                )
                test_dataset = TensorDataset(folds_dict[key]["X_test"], folds_dict[key]["y_test"])

                test_loader = DataLoader(
                    test_dataset, batch_size=model.training_props.batch_size, shuffle=False
                )

                for drop_prob in drop_probabilities:
                    metrics = model.fit_and_calibrate(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        drop_prob=drop_prob,
                    )

                    nll.append(metrics["nll_loss"])
                    brier.append(metrics["brier_score"])
                    cel.append(metrics["cel_loss"])
                nll_all[key] = nll
                brier_all[key] = brier
                cel_all[key] = cel

                nll_argmin.append(np.argmin(np.array(nll) / len(test_loader)))
                brier_argmin.append(np.argmin(np.array(brier) / len(test_loader)))
                cel_argmin.append(np.argmin(np.array(cel) / len(test_loader)))
                print(
                    "Finished Fold:", key + "time:", str((time.time() - time_fold) / 60), "minutes"
                )
                time_fold = time.time()

            nll_best, brier_best, cel_best = (
                statistics.mode(nll_argmin),
                statistics.mode(brier_argmin),
                statistics.mode(cel_argmin),
            )

            metrics_dict = {
                "nll_all": nll_all,
                "brier_all": brier_all,
                "cel_all": cel_all,
            }
            best_indices = {
                "nll": nll_best,
                "brier": brier_best,
                "cel": cel_best,
            }

            output_path = (
                "/workspaces/expainable-uncertainty-quantification/results/calibration_mnist.png"
            )
            plot_calibration_results(
                metrics_dict, drop_probabilities, best_indices, "classification", output_path
            )

        elif model.model_type == "regression":
            crps_all = {}
            nll_all = {}
            mse_all = {}
            crps_ens_all = {}

            nll_argmin = []
            crps_argmin = []
            mse_argmin = []
            crps_ens_argmin = []
            for key in folds_dict.keys():
                nll = []
                crps = []
                mse = []
                crps_ens = []
                if key == "test_set":
                    continue
                train_dataset = TensorDataset(
                    folds_dict[key]["X_train"], folds_dict[key]["y_train"]
                )
                val_dataset = TensorDataset(folds_dict[key]["X_val"], folds_dict[key]["y_val"])
                test_dataset = TensorDataset(folds_dict[key]["X_test"], folds_dict[key]["y_test"])

                test_loader = DataLoader(
                    test_dataset, batch_size=model.training_props.batch_size, shuffle=False
                )

                train_loader = DataLoader(
                    train_dataset, batch_size=model.training_props.batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=model.training_props.batch_size, shuffle=False
                )

                for drop_prob in drop_probabilities:
                    metrics = model.fit_and_calibrate(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        drop_prob=drop_prob,
                    )
                    nll.append(metrics["nll_loss"])
                    crps.append(metrics["crps"])
                    mse.append(metrics["mse_loss"])
                    crps_ens.append(metrics["crps_ens"])
                nll_all[key] = nll
                crps_all[key] = crps
                mse_all[key] = mse
                crps_ens_all[key] = crps_ens

                nll_argmin.append(np.argmin(np.array(nll) / len(test_loader)))
                crps_argmin.append(np.argmin(np.array(crps) / len(test_loader)))
                mse_argmin.append(np.argmin(np.array(mse) / len(test_loader)))
                crps_ens_argmin.append(np.argmin(np.array(crps_ens) / len(test_loader)))

            nll_best, crps_best, mse_best, crps_ens_best = (
                statistics.mode(nll_argmin),
                statistics.mode(crps_argmin),
                statistics.mode(mse_argmin),
                statistics.mode(crps_ens_argmin),
            )

            metrics_dict = {
                "nll_all": nll_all,
                "crps_all": crps_all,
                "mse_all": mse_all,
                "crps_ens_all": crps_ens_all,
            }
            best_indices = {
                "nll": nll_best,
                "crps": crps_best,
                "mse": mse_best,
                "crps_ens": crps_ens_best,
            }

            output_path = (
                "/workspaces/expainable-uncertainty-quantification/results/calibration.png"
            )
            plot_calibration_results(
                metrics_dict, drop_probabilities, best_indices, "regression", output_path
            )
