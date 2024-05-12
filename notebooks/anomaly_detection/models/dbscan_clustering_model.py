import numpy as np
import pandas as pd

from notebooks.anomaly_detection.models.base_anomaly_model import BaseAnomalyModel
from sklearn.cluster import DBSCAN


class DBSCANClusteringModel(BaseAnomalyModel):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, epsilon: float = 0.1,
                 min_samples: int = 5, metric: str = 'euclidean', random_state: int = 42):
        super().__init__(train_df, test_df, random_state)
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.metric = metric
        self.model = DBSCAN(eps=self.epsilon, min_samples=self.min_samples, metric=metric)

    def fit(self) -> None:
        raise Exception("Unsupervised Clustering method does not support fit method")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.fit_predict(df)
