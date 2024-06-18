import typer

from models.model_types import ModelType
from utils.config import Config
from utils.logging import log, LogLevel

cli = typer.Typer()


@cli.command("execute")
def cli_run(config: str = typer.Option("../configs/baseline.json", help="Path to the config file")):
    _config = Config(config)

    verbose = _config.get_nested("verbose")

    selected_model = _config.get_nested("model", "name")

    log(f"Attempting to load: {selected_model}", LogLevel.INFO, verbose=verbose)

    if selected_model.lower() == ModelType.CTGAN.value.lower():
        from models.CTGAN import CTGAN
        model = CTGAN(_config)
    else:
        raise ValueError(f"Model {selected_model} not supported")

    model.execute()


@cli.command("version")
def cli_version():
    log("CTGAN Baseline", LogLevel.INFO)


if __name__ == "__main__":
    cli()
