from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

from models.AbstractGAN import AbstractGAN
from utils.config import Config

from utils.logging import log, LogLevel


class CTGAN(AbstractGAN):
    def __init__(self, config: Config):
        super().__init__(config)

        log(f"Retrieving Metadata.", level=LogLevel.INFO, verbose=self.verbose)
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.data)

        log(f"Setting Up CTGAN Synthesizer.", level=LogLevel.INFO, verbose=self.verbose)

        self.synthesizer = CTGANSynthesizer(
            self.metadata,
            enforce_rounding=False,
            epochs=self.config.get_nested("gan", "epochs", default=300),
            verbose=self.verbose,
            cuda=self.config.get_nested("gan", "cuda", default=True),
            batch_size=self.config.get_nested("gan", "batch_size", default=500),
            discriminator_dim=self.config.get_nested("gan", "discriminator_dim", default=(256, 256)),
            discriminator_lr=self.config.get_nested("gan", "discriminator_lr", default=2e-4),
            discriminator_decay=self.config.get_nested("gan", "discriminator_decay", default=1e-6),
            discriminator_steps=self.config.get_nested("gan", "discriminator_steps", default=1),
            generator_dim=self.config.get_nested("gan", "generator_dim", default=(256, 256)),
            generator_lr=self.config.get_nested("gan", "generator_lr", default=2e-4),
            generator_decay=self.config.get_nested("gan", "generator_decay", default=1e-6),
            embedding_dim=self.config.get_nested("gan", "embedding_dim", default=128),
            log_frequency=self.config.get_nested("gan", "log_frequency", default=True),
            pac=self.config.get_nested("gan", "pac", default=10),
        )

        log(f"CTGAN is ready to train.", level=LogLevel.SUCCESS, verbose=self.verbose)

    def execute(self):
        self.load()
        self.train()
        self.get_loss_plot()
        self.save()
        self.sample()

    def train(self):
        if not self.model_train:
            return

        log("Starting Training.", level=LogLevel.INFO, verbose=self.verbose)
        self.synthesizer.fit(self.data)
        log("Finished Training.", level=LogLevel.SUCCESS, verbose=self.verbose)

    def save(self):
        if not self.model_save:
            return

        log(f"Saving model to \"{self.model_path}\".", level=LogLevel.INFO, verbose=self.verbose)
        self.synthesizer.save(
            filepath=self.model_path
        )
        log(f"Successfully saved model.", level=LogLevel.SUCCESS, verbose=self.verbose)

    def load(self):

        if not self.model_load:
            return

        log(f"Loading existing model from \"{self.model_path}\".", level=LogLevel.INFO, verbose=self.verbose)

        self.synthesizer.load(
            filepath=self.model_path
        )

        log(f"Successfully loaded existing model.", level=LogLevel.SUCCESS, verbose=self.verbose)

    def get_loss_plot(self):
        log(f"Generating loss plot.", level=LogLevel.INFO,
            verbose=self.verbose)

        fig = self.synthesizer.get_loss_values_plot()

        log(f"saving loss plot it to \"{self.loss_plot_path}\".", level=LogLevel.INFO,
            verbose=self.verbose)
        fig.write_image(self.loss_plot_path, engine="kaleido")
        log(f"Successfully saved the loss plot.", level=LogLevel.SUCCESS, verbose=self.verbose)

    def sample(self):
        log(f"Generating Synthetic Data to \"{self.output_path}\".", level=LogLevel.INFO, verbose=self.verbose)
        synthetic_data = self.synthesizer.sample(self.sample_size)
        synthetic_data.to_csv(self.output_path, index=False)
        log(f"Successfully saved the synthetic data.", level=LogLevel.SUCCESS, verbose=self.verbose)


