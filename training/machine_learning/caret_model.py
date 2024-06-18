from pathlib import Path

import pandas as pd
from pycaret.classification import ClassificationExperiment


class CaretModel:
    def __init__(self, data: pd.DataFrame, target_class: str, model: str, random_state: int = 42,
                 test_data: pd.DataFrame = None, appendix: str = ""):
        self.experiment = ClassificationExperiment()
        self.data = data
        self.target_class = target_class
        self.model_name = model
        self.experiment_appendix = appendix

        self.experiment.setup(self.data, target=self.target_class, session_id=random_state, test_data=test_data,
                              index=False)
        self.model = self.experiment.create_model(model)

    def save(self, experiment_dir: str):
        model_experiment_dir = f"{experiment_dir}/{self.model_name}_{self.experiment_appendix}"
        Path(model_experiment_dir).mkdir(parents=True, exist_ok=True)
        self.experiment.plot_model(self.model, plot="auc", save=model_experiment_dir)
        self.experiment.plot_model(self.model, plot="confusion_matrix", save=model_experiment_dir)
        self.experiment.plot_model(self.model, plot="ks", save=model_experiment_dir)
        self.experiment.plot_model(self.model, plot="pr", save=model_experiment_dir)
        self.experiment.plot_model(self.model, plot="error", save=model_experiment_dir)
        self.experiment.get_leaderboard().to_csv(f"{model_experiment_dir}/summary.csv", index=False)
