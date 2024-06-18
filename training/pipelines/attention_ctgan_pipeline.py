from sdv.metadata import SingleTableMetadata

from pipelines.abstract_pipeline import AbstractPipeline
from synthesizers.AttentionCTGAN import AttentionCTGANSynthesizer
from utils.config import Config

from utils.logging import log, LogLevel


class AttentionCTGANPipeline(AbstractPipeline):
    def __init__(self, config: Config):
        super().__init__(config)

        log(f"Retrieving Metadata.", level=LogLevel.INFO, verbose=self.verbose)
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.data)

        log(f"Setting Up Attention CTGAN Synthesizer.", level=LogLevel.INFO, verbose=self.verbose)

        self.synthesizer = AttentionCTGANSynthesizer(
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
            enable_generator_attention=self.config.get_nested("gan", "enable_attention", default=False),
            embedding_dim=self.config.get_nested("gan", "embedding_dim", default=128),
            log_frequency=self.config.get_nested("gan", "log_frequency", default=True),
            pac=self.config.get_nested("gan", "pac", default=10),
            vocabulary_length=self.config.get_nested("transformer","vocabulary_length", default=21979),
            context_window=self.config.get_nested("transformer","context_window", default=38),
            transformer_embedding_length=self.config.get_nested("transformer","embedding_dim", default=992),
            num_heads=self.config.get_nested("transformer","num_heads", default=31),
            transformer_blocks=self.config.get_nested("transformer","transformer_blocks", default=2),
            transformer_model_path = self.config.get_nested("transformer","model_path", default="transformer_model.pth"),
            conditioning_augmentation_dim = self.config.get_nested("transformer","conditioning_augmentation_dim", default=32),
            conditioning_augmentation_lr = self.config.get_nested("transformer","conditioning_augmentation_lr", default=1e-3)
        )

        log(f"Attention CTGAN is ready to train.", level=LogLevel.SUCCESS, verbose=self.verbose)

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


