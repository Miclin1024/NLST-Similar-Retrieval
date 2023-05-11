import os
import attrs
import torch
import pickle
import hashlib
from definitions import *
from data.reader import *
import pydicom
from abc import abstractmethod
import pytorch_lightning as pl
from lightning.utils import MLP
from lightning.params import ModelParams
from typing import Union, TypeVar, Type, TextIO, ClassVar

TEvaluator = TypeVar("TEvaluator", bound="Evaluator")
model_logdir = os.path.join(LOG_DIR, "models")

pydicom.dcmread()

@attrs.define()
class Evaluator:
    experiment_name: str = attrs.field()
    batch_size: int = attrs.field()
    encoder: Union[pl.LightningModule, torch.nn.Module] = attrs.field()
    reader: NLSTDataReader = attrs.field(default=env_reader)

    # Class variables
    _cache_model_state: ClassVar[str] = ""
    _cache_embeddings: ClassVar[dict[SeriesID, Tensor]] = {}

    @property
    def metadata(self) -> pd.DataFrame:
        return self.reader.metadata

    @property
    def manifest(self) -> pd.DataFrame:
        return self.reader.manifest

    @classmethod
    def from_pl_checkpoint(cls: Type[TEvaluator], hparams: ModelParams,
                           experiment: str, epoch: int) -> TEvaluator:
        prediction_layer = MLP(
            hparams.embedding_dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            hparams.prediction_mlp_layers,
            normalization=MLP.get_normalization(hparams)
        ).to("cuda")
        model = hparams.encoder.to("cuda")
        checkpoint_path = os.path.join(model_logdir, experiment,
                                       f"epoch{epoch}.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        prediction_layer.load_state_dict(
            checkpoint["prediction_model_state_dict"])

        encoder = torch.nn.Sequential(model, prediction_layer)
        return cls(experiment, hparams.eval_batch_size or hparams.batch_size, encoder)

    def _get_patient_ids(self, series_ids: list[SeriesID]) -> np.ndarray:
        return np.array([
            self.reader.read_patient_id(sid) for sid in series_ids
        ])

    def _get_embeddings(self, series_ids: list[SeriesID]) -> torch.Tensor:
        """
        Return the normalized embeddings for the series id list. The input is
        batched and pushed through the encoder stored in the evaluator.
        """

        hasher = hashlib.sha512()
        state_dict = pickle.dumps(self.encoder.state_dict())
        hasher.update(state_dict)
        model_state = hasher.hexdigest()

        if Evaluator._cache_model_state != model_state:
            print(f"Different encoder model state detected, {len(Evaluator._cache_embeddings)} cache entries cleared.")
            Evaluator._cache_embeddings.clear()

        n = len(series_ids)
        cache_count = 0
        embeddings_output: list[torch.Tensor] = [torch.zeros(0, 0).to("cuda")] * n

        print("Encoding raw data for evaluation...")
        for batch_num in range(n // self.batch_size + 1):
            idx_start = batch_num * self.batch_size
            idx_end = min(idx_start + self.batch_size, n)
            if idx_start == idx_end:
                continue

            input_images = []
            using_cache: [bool] = [False] * (idx_end - idx_start)
            for i in range(idx_start, idx_end):
                series_id = series_ids[i]
                if series_id in Evaluator._cache_embeddings.keys():
                    using_cache[i - idx_start] = True
                else:
                    image, metadata = self.reader.read_series(series_ids[i])
                    input_images.append(image.data)
            effective_batch_size = len(input_images)

            if input_images:
                with torch.no_grad():
                    input_batch = torch.stack(input_images, dim=0).to("cuda").to(torch.float)
                    if isinstance(self.encoder, torch.nn.Module):
                        read_emb = self.encoder(input_batch)
                    elif isinstance(self.encoder, pl.LightningModule):
                        read_emb = self.encoder.model(input_batch)
                        read_emb = self.encoder.prediction_model(read_emb)
                    else:
                        raise ValueError(f"Unrecognized encoder, must be either a lighting module or a torch module.")
                    read_emb = read_emb.view(effective_batch_size, -1)
                    read_emb = torch.nn.functional.normalize(read_emb, dim=1)

            read_emb_idx = 0
            for i in range(idx_start, idx_end):
                series_id = series_ids[i]
                if using_cache[i - idx_start]:
                    embeddings_output[i] = Evaluator._cache_embeddings[series_id]
                    cache_count += 1
                else:
                    entry = torch.unsqueeze(read_emb[read_emb_idx], dim=0)
                    read_emb_idx += 1
                    Evaluator._cache_embeddings[series_id] = entry
                    embeddings_output[i] = entry

        print(f"{len(embeddings_output)} embeddings retrieved for model state {model_state[:5]}, "
              f"{cache_count} cache hits.")
        Evaluator._cache_model_state = model_state
        result = torch.cat(embeddings_output, dim=0)
        return result

    def score(self, series_ids: list[SeriesID], log_file: Optional[TextIO] = None) -> dict:
        print(f"Evaluating using {len(series_ids)} series")
        return {}
