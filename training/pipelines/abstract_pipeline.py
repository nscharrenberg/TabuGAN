from abc import ABC
from pathlib import Path

from sdv.datasets.local import load_csvs

from utils.config import Config
from utils.logging import log, LogLevel


class AbstractPipeline(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.verbose = config.get_nested("verbose", default=True)
        self.dataset_dir = config.get_nested("files", "datasets", "directory")
        self.dataset_name = config.get_nested("files", "datasets", "name")
        self.output_dir = config.get_nested("files", "output", "directory")
        self.output_name = config.get_nested("files", "output", "name")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.output_path = f"{self.output_dir}/{self.output_name}.csv"
        self.model_dir = config.get_nested("files", "model", "directory")

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.model_name = config.get_nested("files", "model", "name")
        self.model_load = config.get_nested("model", "load", default=False)
        self.model_train = config.get_nested("model", "train", default=True)
        self.model_save = config.get_nested("model", "save", default=True)
        self.model_path = f"{self.model_dir}/{self.model_name}.pkl"
        self.figure_dir = config.get_nested("files", "figures", "directory")
        self.loss_plot_file = config.get_nested("files", "figures", "loss_plot", default="loss_plot.jpeg")

        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

        self.loss_plot_path = f"{self.figure_dir}/{self.loss_plot_file}"

        self.sample_size = config.get_nested("sampling", "size", default=1000)

        log(f"Loading Dataset at \"{self.dataset_dir}/{self.dataset_name}\".", level=LogLevel.INFO,
            verbose=self.verbose)

        datasets = load_csvs(
            folder_name=self.dataset_dir,
            read_csv_parameters={
                "skipinitialspace": True,
                "encoding": self.config.get_nested("files", "datasets", "encoding", default="utf-8"),
            }
        )

        if self.dataset_name not in datasets:
            log(f"Dataset {self.dataset_name} not found in \"{self.dataset_dir}\".", level=LogLevel.ERROR)
            return

        self.data = datasets[self.dataset_name]

    def execute(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def sample(self):
        pass

    def get_loss_plot(self):
        pass