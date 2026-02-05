import time
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from src.pipelines.train_pipelines.calibration import CalibrationPipelineProps, CalibrationPipline
from datasets import mnist
from src.models.models_dropconnect import UQMCDropconnectCNNClassifier

from src.models.models_dropout import UQMCDropoutCNNClassifier
from src.models.props import MNISTModelProps, ModelTrainingProps

warnings.filterwarnings(
    "ignore",
    category=UndefinedMetricWarning,
)

if __name__ == "__main__":
    uq_strategy = "dropout"  # "dropout" or "dropconnect"
    k_folds = 5
    start_overall = time.time()

    model_props = MNISTModelProps(
        ensemble_path="",
        state_dict_path=None,
        preprocessor_path=None,
        target_preprocessor_path=None,
        output_size=10,
        hidden_layer_1=50,
        hidden_layer_2=50,
        hidden_layer_3=None,
        hidden_layer_4=None,
        hidden_layer_5=None,
        hidden_layer_6=None,
        drop_prob=0.1,
        forward_passes=50,
        last_hidden_layer=50,
        layers_to_perturb=None,
        force_cpu=True,
        is_classification=True,
    )

    training_props = ModelTrainingProps(
        initialisation_strategy=None,
        epochs=10,
        learn_rate=0.001,
        optimizer="Adam",
        loss_function="CEL",
        weight_decay=0.0,
        early_stopping_patience=None,
        lr_scheduler_factor=0.0,
        momentum=0.0,
        batch_size=32,
    )
    if uq_strategy == "dropout":
        model = UQMCDropoutCNNClassifier(model_props=model_props, training_props=training_props)
    elif uq_strategy == "dropconnect":
        model = UQMCDropconnectCNNClassifier(
            model_props=model_props, training_props=training_props
        )
    calibration_props = CalibrationPipelineProps(
        name="MNIST UQ Calibration",
        k_folds=k_folds,
        drop_probabilities=[0.1, 0.2, 0.3, 0.4, 0.5],
    )

    calibration_pipeline = CalibrationPipline()
    calibration_pipeline.pipeline(
        dataset=mnist.MNISTDataset(),
        model=model,
        pipeline_props=calibration_props,
    )

    end_overall = time.time()
    print(f" Overall time to calibrate p: {str((end_overall - start_overall) / 60)} minutes")
