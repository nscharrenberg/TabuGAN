# Tabular GAN Training Pipeline

This application is a training pipeline for a GAN model that generates tabular data. The pipeline is implemented using Synthetic Data Vault with the following models:
- CTGAN

The pipeline consists of the following components for each model:
- The Training Pipeline in `pipelines`
- The synthesizers (necessary for SDV) in `synthesizers`
- The GAN models in `models`
- The Pipeline configurations in `config`


## How to Run the Training Pipeline
Run the following command to train the model:
```bash
python main.py execute --config config/ctgan/train.json
```

## How to Run the Experiments Pipeline
Run the following command to run the experiments:
```bash
python main.py experiment --config config/ctgan/experiment.json
```
where `config/ctgan/train.json` is the path to the configuration file for the model you want to train.

**Note**: If you already have generated data, and only want to test that, make sure to set `experiment_only` to `true`.

## How to Extend CTGAN with your own model
1. Copy and paste the ctgan directory in the models directory and rename it to your model name e.g. `transformerGAN`.
2. Also make sure to rename the `ctgan.py` file to `transformerGAN.py`.
3. Make the necessary changes to the model file to implement your model.
4. Copy and paste the `CTGAN.py` file in the `synthesizers` directory and rename it to your model name e.g. `transformerGAN.py`.
5. Make the necessary changes to the synthesizer file to implement your model e.g. change the model on line 290.
6. Copy and paste the `ctgan_pipeline.py` file in the `pipelines` directory and rename it to your model name e.g. `transformerGAN_pipeline.py`.
7. Make the necessary changes to the pipeline file to implement your model e.g. change the synthesizer on line 20.
8. Copy and paste the ctgan directory in the config directory and rename it to your model name e.g. `transformerGAN`.
9. Make the necessary changes to the configuration file such as ensuring that the `models.name` is changed to your model name.
10. In `main.py`make sure to add the pipeline model.

If you're adding a completely new model, you might have to add it to `pipeline_type.py` in `pipelines` and add it to the if-statement in the `main.py` file.


## To Do
1. Allow experiments to skip the GAN loading and training, and only utilizing the original data and synthetic data.
2. Perform Statistical Tests using sdv metrics.
3. Add integration for transformer GAN
4. Add integration for VAE GAN.