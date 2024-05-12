import numpy as np
import pandas as pd
from keras import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense

from notebooks.anomaly_detection.models.base_anomaly_model import BaseAnomalyModel


class AutoencoderModel(BaseAnomalyModel):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, epochs: int = 50, batch_size: int = 256, shuffle: bool = True, random_state: int = 42):
        super().__init__(train_df, test_df, random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = self.__build_model()

    def __build_model(self) -> Model:
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(self.train_df)

        train_X, _ = train_test_split(data_normalized, test_size=0)

        input_dim = train_X.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(64, activation='relu')(input_layer)
        encoder = Dense(input_dim, activation='sigmoid')(encoder)
        model = Model(inputs=input_layer, outputs=encoder)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit(self) -> None:
        self.model.fit(self.train_df, self.train_df, epochs=self.epochs, batch_size=self.batch_size, shuffle=self)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df)
