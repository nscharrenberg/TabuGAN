from training_pipelines.ctgan_pipeline import CTGANTrainingPipeline
from training_pipelines.attention_ctgan_pipeline import AttentionCTGANPipeline
from training_pipelines.pipeline_type import PipelineType
from utils.config import Config
from utils.logging import LogLevel, log


def train(config: str):
    _config = Config(config)

    verbose = _config.get_nested("verbose")

    selected_model = _config.get_nested("model", "name")

    log(f"Attempting to load: {selected_model}", LogLevel.INFO, verbose=verbose)

    if selected_model.lower() == PipelineType.CTGAN.value.lower():
        pipeline = CTGANTrainingPipeline(_config)
    elif selected_model.lower() == PipelineType.TRANSFORMER_GAN.value.lower():
        pipeline = AttentionCTGANPipeline(_config)
    elif selected_model.lower() == PipelineType.VAE_GAN.value.lower():
        log(f"APipeline for model \"{selected_model}\" not supported.", LogLevel.ERROR, verbose=verbose)
        return
    else:
        log(f"APipeline for model \"{selected_model}\" not supported.", LogLevel.ERROR, verbose=verbose)
        return

    pipeline.execute()

    return pipeline
