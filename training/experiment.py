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

    experiment_only = _config.get_nested("experiment_only", default=False)
    use_original_as_test = _config.get_nested("use_original_as_test", default=False)

    if not experiment_only:
        pipeline = train(config)

        original_data = pipeline.data
    else:
        dataset_dir = _config.get_nested("files", "datasets", "directory")
        dataset_name = _config.get_nested("files", "datasets", "name")
        dataset_path = f"{dataset_dir}/{dataset_name}.csv"

        if not Path(dataset_path).exists():
            log(f"Dataset path {dataset_path} does not exist", LogLevel.ERROR)
            return

        original_data = pd.read_csv(dataset_path)

    if not use_original_as_test:
        synthetic_data = load_synthetic_data(_config)
    else:
        synthetic_data = None
    target_name = _config.get_nested("files", "datasets", "target")

    experiment_dir = _config.get_nested("files", "figures", "directory")

    # run_model_experiment("lr", original_data, target_name, experiment_dir, appendix="original", test_data=original_data)
    run_model_experiment("lr", original_data, target_name, experiment_dir, appendix="ctgan",
                         test_data=synthetic_data, seed=seed)

    # run_model_experiment("dt", original_data, target_name, experiment_dir, appendix="original", test_data=original_data)
    run_model_experiment("dt", original_data, target_name, experiment_dir, appendix="ctgan",
                         test_data=synthetic_data, seed=seed)

    # run_model_experiment("xgboost", original_data, target_name, experiment_dir, appendix="original",
    #                      test_data=original_data)
    run_model_experiment("xgboost", original_data, target_name, experiment_dir, appendix="ctgan",
                         test_data=synthetic_data, seed=seed)


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


def run_model_experiment(model: str, data: pd.DataFrame, target_class: str, experiment_dir: str, test_data: pd.DataFrame = None, appendix: str = "", seed: int = 42):
    log(f"Running Logistic Regression model on data with target class \"{target_class}\".", level=LogLevel.INFO)
    lr_model = CaretModel(data, target_class, model, test_data=test_data, appendix=appendix, random_state=seed)

    log("Saving plots...", level=LogLevel.INFO)
    lr_model.save(experiment_dir)
    log("Plots saved.", level=LogLevel.SUCCESS)
