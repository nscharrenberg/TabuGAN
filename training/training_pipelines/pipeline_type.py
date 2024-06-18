from enum import Enum


class PipelineType(Enum):
    CTGAN = "ctgan"
    TRANSFORMER_GAN = "transformer"
    VAE_GAN = "vae"
