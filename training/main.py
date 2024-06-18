import typer

from experiment import experiment
from train import train
from utils.logging import log, LogLevel

cli = typer.Typer()


@cli.command("execute")
def cli_run(config: str = typer.Option("../config/ctgan/train.json", help="Path to the config file")):
    log("Starting Training Procedure.", LogLevel.INFO)
    train(config)


@cli.command("experiment")
def cli_experiment(config: str = typer.Option("../config/ctgan/experiment.json", help="Path to the config file")):
    log("Starting Experiment Procedure.", LogLevel.INFO)
    experiment(config)


@cli.command("version")
def cli_version():
    log("0.0.1", LogLevel.INFO)


if __name__ == "__main__":
    cli()
