from abc import ABC, abstractmethod

import pandas as pd


class BaseAnomalyModel(ABC):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int = 42):
        self.train_df, self.test_df = train_df, test_df
        self.random_state = random_state
        self.model = None

    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def predict(self, X):
        ...
