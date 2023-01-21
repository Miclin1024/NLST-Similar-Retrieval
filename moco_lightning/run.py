import torch
from attr import evolve
import pytorch_lightning as pl
import torchvision.models.video
from moco_lightning.builder import LitMoCo
from moco_lightning.params import ModelParams
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == '__main__':
    encoder = torchvision.models.video.r3d_18(
        weights=torchvision.models.video.R3D_18_Weights.DEFAULT
    ).to("cuda")

    batch_size = 8
    input_tensor = torch.randn(batch_size, 1, 128, 128, 128).to("cuda")
    output_tensor = encoder(input_tensor)

    mlp_embedding_dim = output_tensor.shape[1]
    mlp_output_dim = int(mlp_embedding_dim)
    mlp_hidden_dim = int(mlp_embedding_dim)

    if torch.has_mps:
        base_config = ModelParams(
            encoder=encoder,
            embedding_dim=mlp_embedding_dim,
            dim=mlp_output_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            lr=0.08,
            batch_size=batch_size,
            gather_keys_for_queue=False,
            loss_type="ip",
            use_both_augmentations_as_queries=True,
            mlp_normalization="bn",
            prediction_mlp_layers=2,
            projection_mlp_layers=2,
            m=0.996,
        )
    else:
        base_config = ModelParams(
            encoder=encoder,
            embedding_dim=mlp_embedding_dim,
            dim=mlp_output_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            lr=0.001,
            batch_size=batch_size,
            gather_keys_for_queue=False,
            loss_type="ip",
            use_both_augmentations_as_queries=True,
            use_negative_examples_from_queue=True,
            mlp_normalization="bn",
            prediction_mlp_layers=2,
            projection_mlp_layers=2,
            m=0.996,
        )

    method = LitMoCo(base_config)
    logger = TensorBoardLogger("logs/tensor_board", name="base")
    if torch.has_mps:
        trainer = pl.Trainer(logger=logger,
                             accelerator="cpu",
                             log_every_n_steps=1,
                             max_epochs=100,
                             )
    else:
        trainer = pl.Trainer(logger=logger,
                             accelerator="gpu",
                             precision=16,
                             devices=2,
                             # strategy="ddp",
                             log_every_n_steps=5,
                             max_epochs=20,
                             )
    trainer.fit(method)

