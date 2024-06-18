import typer

from pipelines.ctgan_pipeline import CTGANPipeline
from pipelines.attention_ctgan_pipeline import AttentionCTGANPipeline
from pipelines.pipeline_type import PipelineType
from utils.config import Config
from utils.logging import log, LogLevel

cli = typer.Typer()


@cli.command("execute")
def cli_run(config: str = typer.Option("../configs/baseline.json", help="Path to the config file")):
    _config = Config(config)

    verbose = _config.get_nested("verbose")

    selected_model = _config.get_nested("model", "name")

    log(f"Attempting to load: {selected_model}", LogLevel.INFO, verbose=verbose)

    if selected_model.lower() == PipelineType.CTGAN.value.lower():
        pipeline = CTGANPipeline(_config)
    elif selected_model.lower() == PipelineType.TRANSFORMER_GAN.value.lower():
        pipeline = AttentionCTGANPipeline(_config)
    elif selected_model.lower() == PipelineType.VAE_GAN.value.lower():
        log(f"APipeline for model \"{selected_model}\" not supported.", LogLevel.ERROR, verbose=verbose)
        return
    else:
        log(f"APipeline for model \"{selected_model}\" not supported.", LogLevel.ERROR, verbose=verbose)
        return

    pipeline.execute()


@cli.command("version")
def cli_version():
    log("0.0.1", LogLevel.INFO)


if __name__ == "__main__":
    cli()
