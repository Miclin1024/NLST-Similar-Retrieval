import attrs
import torch
import warnings
from data import *
from params import ModelParams
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Optional, Callable
from pytorch_lightning.utilities import AttributeDict


class LitMoCo(pl.LightningModule):
    model: torch.nn.Module
    database: DatasetBase
    hparams: AttributeDict
    embedding_dim: Optional[int]

    def __init__(self, hparams: Union[ModelParams, dict, None] = None, **kwargs):
        super(LitMoCo, self).__init__()

        if hparams is None:
            hparams = ModelParams(**kwargs)
        elif isinstance(hparams, dict):
            hparams = ModelParams(**hparams, **kwargs)

        if isinstance(self.hparams, AttributeDict):
            self.hparams.update(AttributeDict(attrs.asdict(hparams)))
        else:
            self.hparams = AttributeDict(attrs.asdict(hparams))

        # Check for configuration issues
        if (
            hparams.gather_keys_for_queue
            and not hparams.shuffle_batch_norm
            and not hparams.encoder_arch.startswith("ws_")
        ):
            warnings.warn(
                "Configuration suspicious: gather_keys_for_queue without shuffle_batch_norm or weight standardization"
            )

        some_negative_examples = hparams.use_negative_examples_from_batch or hparams.use_negative_examples_from_queue
        if hparams.loss_type == "ce" and not some_negative_examples:
            warnings.warn("Configuration suspicious: cross entropy loss without negative examples")


if __name__ == '__main__':
    pass
