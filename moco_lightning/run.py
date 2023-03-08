import comet_ml
import logging
import os
import torch
import dotenv
import encoders.resnet
from definitions import *
import pytorch_lightning as pl
import torchvision.models.video
from moco_lightning.builder import LitMoCo
from moco_lightning.params import ModelParams
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers

COMET_LOG = True

if __name__ == '__main__':
    dotenv.load_dotenv()

    encoder_name = "r3d_18"
    print(f"Initializing with encoder {encoder_name}...")

    # Hyperparams
    batch_size = 8
    eval_batch_size = None
    lr = 0.001
    m = 0.996
    K = 2048
    optimizer_name = "sgd"

    if encoder_name == "r3d_18":
        batch_size = 8
        eval_batch_size = 16
        lr = 0.001
        K = 1024
        optimizer_name = "adam"
        encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3),
            torchvision.models.video.r3d_18(
                weights=None
            )
        )
    elif encoder_name == "slow_r50":
        batch_size = 8
        encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=1),
            torch.hub.load("facebookresearch/pytorchvideo", encoder_name, pretrained=True)
        )
    elif encoder_name == "mvit_v2_s":
        batch_size = 4
        K = 4096
        eval_batch_size = 3
        lr = 0.0001
        m = 0.997
        optimizer_name = "adam"
        encoder = create_multiscale_vision_transformers(
            spatial_size=128,
            temporal_size=128,
            input_channels=1,
            embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
            atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
            pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
            pool_kv_stride_adaptive=[1, 8, 8],
            pool_kvq_kernel=[3, 3, 3],
            head_num_classes=512,
        )
    else:
        raise ValueError(f"Unknown encoder {encoder_name}")

    encoder = encoder.to("cuda")

    input_tensor = torch.randn(batch_size, 1, 128, 128, 128).to("cuda")
    print(f"Verifying architecture with input shape {input_tensor.shape}...")
    output_tensor = encoder(input_tensor)
    print(f"Output shape is {output_tensor.shape}")

    mlp_embedding_dim = output_tensor.shape[1]
    mlp_output_dim = int(mlp_embedding_dim)
    mlp_hidden_dim = int(mlp_embedding_dim)

    logger = [
        # TensorBoardLogger("logs/tensor_board", name="base")
    ]

    dotenv_path = os.path.join(ROOT_DIR, ".env")

    with open(dotenv_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    env_dict = {line.split("=")[0]: line.split("=")[1] for line in lines}

    # Increment the value of the specified key
    experiment_version = int(env_dict["COMET_EXPERIMENT_VERSION"])
    env_dict["COMET_EXPERIMENT_VERSION"] = str(experiment_version + 1)
    experiment_name = f"{encoder_name}_moco_v{experiment_version}"

    # Write the updated dictionary back to the .env file
    with open(dotenv_path, "w") as file:
        for key, value in env_dict.items():
            file.write(f"{key}={value}\n")

    if COMET_LOG:
        comet_api_key = os.environ.get("COMET_API_KEY")
        comet_project_name = os.environ.get("COMET_PROJECT_NAME")

        comet_logger = CometLogger(
            api_key=comet_api_key,
            rest_api_key=comet_api_key,
            project_name=comet_project_name,
            experiment_name=experiment_name,
            save_dir="logs/comet",
        )
        logger.append(comet_logger)

    base_config = ModelParams(
        encoder_name=encoder_name,
        encoder=encoder,
        embedding_dim=mlp_embedding_dim,
        dim=mlp_output_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_normalization="bn",
        lr=lr,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        m=m,
        K=K,
        optimizer_name=optimizer_name,
        max_epochs=100,
        prediction_mlp_layers=2,
    )

    method = LitMoCo(experiment_name, base_config)

    torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(logger=logger,
                         accelerator="gpu",
                         precision=16,
                         devices=1,
                         log_every_n_steps=5,
                         max_epochs=base_config.max_epochs,
                         benchmark=True,
                         )
    trainer.fit(method)
