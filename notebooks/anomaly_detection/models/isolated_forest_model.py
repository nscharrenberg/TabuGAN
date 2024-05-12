import numpy as np
import pandas as pd

from notebooks.anomaly_detection.models.base_anomaly_model import BaseAnomalyModel
from sklearn.ensemble import IsolationForest


class IsolatedForestModel(BaseAnomalyModel):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, n_estimators: int = 100,
                 contamination: float|str = 'auto', random_state: int = 42):
        super().__init__(train_df, test_df, random_state)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.model = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination,
                                     random_state=random_state)

    def fit(self) -> None:
        self.model.fit(self.train_df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df)
