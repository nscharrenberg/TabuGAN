from enum import Enum


class ModelType(Enum):
    CTGAN = "ctgan"
    TRANSFORMER_GAN = "transformer"
    VAE_GAN = "vae"
