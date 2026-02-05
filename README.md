# No Single Metric Tells the Whole Story: A Multi-Dimensional Evaluation Framework for Uncertainty Attributions

This repository contains code in relation to the paper ***No Single Metric Tells the Whole Story: A Multi-Dimensional Evaluation Framework for Uncertainty Attributions*** currently under review at XAI26.

## Setup

All dependencies are included in the `pyproject.toml` and can be installed using poetry.

1. install poetry `pip install poetry`
2. install dependencies `poetry install`
3. activate virtual environment `poetry shell`

## How to run 

The code to execute the experiments is contained in `evaluate_mnist.py` and `evaluate_winequality.py`, to evaluate uncertainty attributions for MNIST and Winequality datasets, respectively. 

Both scripts train a neural network and generate and evaluate uncertainty attributions based on
    - the specified `uq_strategy` ("dropout"/"dropconnect") and dropout/dropconnect probability
    - all selected feature attribution methods (explainers), and 
    - all specified metrics (complexity, feature flipping, uncertainty conveyance similarity, relative input stability, repeatability, relative rank improvement).  
The evaluation results are saved as pickle and text files. 

Additionally, the dropout/dropconnect probability can be tuned for MNIST and Winequality in `calibrate_p_for_uq_mnist.py` and `calibrate_p_for_uq_winequality.py`, respectively.

The implementations of all metrics from the proposed evaluation framework can be found in src/evaluation.

## Data
The Wine Quality data can be downloaded from https://archive.ics.uci.edu/dataset/186/wine+quality (red and wine). It should be stored in the folder `datasets/Wine Quality` as `winequality-red_komma.csv` and `winequality-white_komma.csv`. MNIST (http://yann.lecun.com/exdb/mnist/) data is downloaded automatically from torchvision, if `download = True` in `create_data()` in datasets/mnist.py.



