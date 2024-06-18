from pathlib import Path

import pandas as pd
from sdv.datasets.local import load_csvs

from machine_learning.caret_model import CaretModel
from train import train
from utils.config import Config
from utils.logging import log, LogLevel


def experiment(config: str):
    _config = Config(config)

    verbose = _config.get_nested("verbose")
    seed = _config.get_nested("seed", default=42)

    pipeline = train(config)

    original_data = pipeline.data
    synthetic_data = load_synthetic_data(_config)
    target_name = _config.get_nested("files", "datasets", "target")

    experiment_dir = _config.get_nested("files", "figures", "directory")

    run_model_experiment("lr", original_data, target_name, experiment_dir, appendix="baseline")
    run_model_experiment("lr", original_data, target_name, experiment_dir, appendix="ctgan", test_data=synthetic_data)

    run_model_experiment("dt", original_data, target_name, experiment_dir, appendix="baseline")
    run_model_experiment("dt", original_data, target_name, experiment_dir, appendix="ctgan", test_data=synthetic_data)

    run_model_experiment("xgboost", original_data, target_name, experiment_dir, appendix="baseline")
    run_model_experiment("xgboost", original_data, target_name, experiment_dir, appendix="ctgan", test_data=synthetic_data)


def load_synthetic_data(config: Config):
    file_dir = config.get_nested("files", "output", "directory")
    file_name = config.get_nested("files", "output", "name")
    file_path = f"{file_dir}/{file_name}.csv"

    if not Path(file_path).exists():
        log(f"Synthetic data could not be found at \"{file_path}\".", level=LogLevel.ERROR)
        return

    datasets = load_csvs(
        folder_name=file_dir,
        read_csv_parameters={
            "skipinitialspace": True,
        }
    )

    return datasets[file_name]


def run_model_experiment(model:str, data: pd.DataFrame, target_class: str, experiment_dir: str, test_data: pd.DataFrame = None, appendix: str = ""):
    log(f"Running Logistic Regression model on data with target class \"{target_class}\".", level=LogLevel.INFO)
    lr_model = CaretModel(data, target_class, model, test_data=test_data, appendix=appendix)

    log("Saving plots...", level=LogLevel.INFO)
    lr_model.save(experiment_dir)
    log("Plots saved.", level=LogLevel.SUCCESS)
