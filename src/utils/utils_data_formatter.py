from datetime import datetime
import os
from pprint import pformat
from matplotlib.figure import Figure
import numpy as np
from pandas import DataFrame
import torch


def flatten_dict(d, parent_key="", sep=" - "):
    flat_dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat_dict[new_key] = v
    return flat_dict


def save_figure_to_image(figure: Figure, directory: str, name_tag: str) -> None:
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{directory}/{name_tag}_{timestamp}.png"

    figure.savefig(filename)


def save_dataframe_to_csv(df: DataFrame, directory: str, name_tag: str) -> None:
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{directory}/{name_tag}_{timestamp}.csv"

    df.to_csv(filename)


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    return obj


def save_dict_to_text(dict_obj, save_dir, file_name):
    text_save_path = os.path.join(save_dir, file_name + ".txt")
    with open(text_save_path, "w") as f:
        f.write(pformat(_to_serializable(dict_obj)))
