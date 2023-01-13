from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import LightningModule
from ray import tune
from functools import partial

if __name__ == '__main__':
    callback = TuneReportCallback(
        {
            "loss": "val_loss",
            "mean_accuracy": "val_accuracy"
        },
        on="validation_end")

    class LightningMNISTClassifier:

        def __init__(self, config):
            super(LightningMNISTClassifier, self).__init__()
            self.layer_1_size = config["layer_1_size"]
            self.layer_2_size = config["layer_2_size"]
            self.lr = config["lr"]
            self.batch_size = config["batch_size"]

    config = {
    "layer_1_size": tune.choice([32, 64, 128]),
    "layer_2_size": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128])
    }

    def train_tune(config, epochs=10, gpus=0):
        model = LightningMNISTClassifier(config)
        trainer = pl.Trainer(
            max_epochs=epochs,
            gpus=gpus,
            progress_bar_refresh_rate=0,
            callbacks=[callback])
        trainer.fit(model)

    # tune.run(
    # partial(train_tune, epochs=10, gpus=0),
    # config=config,
    # num_samples=10)


