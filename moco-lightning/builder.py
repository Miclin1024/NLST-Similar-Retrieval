import os
import copy
import attrs
import torch
import warnings
from data import *
from .utils import *
from attr import evolve
from functools import partial
from encoders.resnet import *
from .params import ModelParams
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Optional, Callable
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.loggers import TensorBoardLogger


class LitMoCo(pl.LightningModule):
    model: torch.nn.Module
    manager: DatasetManager
    hparams: AttributeDict
    embedding_dim: Optional[int]
    lr_scheduler: Any

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
        ):
            warnings.warn(
                "Configuration suspicious: gather_keys_for_queue without shuffle_batch_norm or weight standardization"
            )

        some_negative_examples = hparams.use_negative_examples_from_batch or hparams.use_negative_examples_from_queue
        if hparams.loss_type == "ce" and not some_negative_examples:
            warnings.warn("Configuration suspicious: cross entropy loss without negative examples")

        self.model = hparams.encoder

        self.manager = DatasetManager(
            int(os.environ.get("MANIFEST_ID")),
            ds_split=[.7, .2, .1]
        )

        # if hparams.use_lagging_model:
        #     # "key" function (no grad)
        #     self.lagging_model = copy.deepcopy(self.model)
        #     for param in self.lagging_model.parameters():
        #         param.requires_grad = False
        # else:
        #     self.lagging_model = None

        self.projection_model = MLP(
            hparams.embedding_dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.projection_mlp_layers,
            normalization=MLP.get_normalization(hparams),
        )

        self.prediction_model = MLP(
            hparams.dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.prediction_mlp_layers,
            normalization=MLP.get_normalization(hparams, prediction=True),
        )

        if hparams.use_lagging_model:
            #  "key" function (no grad)
            self.lagging_projection_model = copy.deepcopy(self.projection_model)
            for param in self.lagging_projection_model.parameters():
                param.requires_grad = False
        else:
            self.lagging_projection_model = None

        if hparams.use_lagging_model:
            #  "key" function (no grad)
            self.lagging_projection_model = copy.deepcopy(self.projection_model)
            for param in self.lagging_projection_model.parameters():
                param.requires_grad = False
        else:
            self.lagging_projection_model = None

        # this classifier is used to compute representation quality each epoch
        # TODO: maybe add a regression model to identify Patient Weight
        # TODO: use logistic regression to identify Emphysema
        self.sklearn_classifier = LogisticRegression(max_iter=100, solver="liblinear")

        if hparams.use_negative_examples_from_queue:
            # create the queue
            self.register_buffer("queue", torch.randn(hparams.dim, hparams.K))
            self.queue = torch.nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.queue = None

    def _get_embeddings(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # bsz = batch size, n_split = , nc = number of channels, nh = number of height, nw = number of weight
        bsz, n_split, nc, nd, nh, nw = x.shape
        assert n_split == 2, "second dimension should be the split image -- dims should be N2CHW"
        im_q = x[:, 0].contiguous()
        im_k = x[:, 1].contiguous()

        # compute query features

        emb_q = self.model(im_q)
        emb_q = emb_q.view(self.hparams.batch_size, self.hparams.embedding_dim)
        q_projection = self.projection_model(emb_q)
        q = self.prediction_model(q_projection)  # queries: NxC

        if self.hparams.use_lagging_model:
            # compute key features
            with torch.no_grad():  # no gradient to keys
                # if self.hparams.shuffle_batch_norm:
                #     im_k, idx_unshuffle = utils.BatchShuffleDDP.shuffle(im_k)
                k = self.lagging_projection_model(self.lagging_model(im_k))  # keys: NxC
                # if self.hparams.shuffle_batch_norm:
                #     k = utils.BatchShuffleDDP.unshuffle(k, idx_unshuffle)
        else:
            emb_k = self.model(im_k)
            emb_k = emb_k.view(self.hparams.batch_size, self.hparams.embedding_dim)
            k_projection = self.projection_model(emb_k)
            k = self.prediction_model(k_projection)  # queries: NxC

        if self.hparams.use_unit_sphere_projection:
            q = torch.nn.functional.normalize(q, dim=1)
            k = torch.nn.functional.normalize(k, dim=1)

        return emb_q, q, k

    def _get_contrastive_predictions(self, q, k):
        if self.hparams.use_negative_examples_from_batch:
            logits = torch.mm(q, k.T)
            labels = torch.arange(0, q.shape[0], dtype=torch.long).to(logits.device)
            return logits, labels

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        if self.hparams.use_negative_examples_from_queue:
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1)
        else:
            logits = l_pos

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        return logits, labels

    def _get_pos_neg_ip(self, emb_q, k):
        with torch.no_grad():
            z = self.projection_model(emb_q)
            z = torch.nn.functional.normalize(z, dim=1)
            ip = torch.mm(z, k.T)
            eye = torch.eye(z.shape[0]).to(z.device)
            pos_ip = (ip * eye).sum() / z.shape[0]
            neg_ip = (ip * (1 - eye)).sum() / (z.shape[0] * (z.shape[0] - 1))

        return pos_ip, neg_ip

    def _get_contrastive_loss(self, logits, labels):
        if self.hparams.loss_type == "ce":
            if self.hparams.use_eqco_margin:
                if self.hparams.use_negative_examples_from_batch:
                    neg_factor = self.hparams.eqco_alpha / self.hparams.batch_size
                elif self.hparams.use_negative_examples_from_queue:
                    neg_factor = self.hparams.eqco_alpha / self.hparams.K
                else:
                    raise Exception("Must have negative examples for ce loss")

                predictions = log_softmax_with_factors(logits / self.hparams.T, neg_factor=neg_factor)
                return F.nll_loss(predictions, labels)

            return F.cross_entropy(logits / self.hparams.T, labels)

        new_labels = torch.zeros_like(logits)
        new_labels.scatter_(1, labels.unsqueeze(1), 1)
        if self.hparams.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits / self.hparams.T, new_labels) * logits.shape[1]

        if self.hparams.loss_type == "ip":
            # inner product
            # negative sign for label=1 (maximize ip), positive sign for label=0 (minimize ip)
            inner_product = (1 - new_labels * 2) * logits
            return torch.mean((inner_product + 1).sum(dim=-1))

        raise NotImplementedError(f"Loss function {self.hparams.loss_type} not implemented")

    def _get_vicreg_loss(self, z_a, z_b, batch_idx):
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2

        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.hparams.variance_loss_epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.hparams.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.hparams.invariance_loss_weight
        weighted_var = loss_var * self.hparams.variance_loss_weight
        weighted_cov = loss_cov * self.hparams.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return {
            "loss": loss,
            "loss_invariance": weighted_inv,
            "loss_variance": weighted_var,
            "loss_covariance": weighted_cov,
        }

    def forward(self, x) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        all_params = list(self.model.parameters())
        x, class_labels = batch  # batch is a tuple, we just want the image

        emb_q, q, k = self._get_embeddings(x)
        pos_ip, neg_ip = self._get_pos_neg_ip(emb_q, k)

        logits, labels = self._get_contrastive_predictions(q, k)
        if self.hparams.use_vicreg_loss:
            losses = self._get_vicreg_loss(q, k, batch_idx)
            contrastive_loss = losses["loss"]
        else:
            losses = {}
            contrastive_loss = self._get_contrastive_loss(logits, labels)

            if self.hparams.use_both_augmentations_as_queries:
                x_flip = torch.flip(x, dims=[1])
                emb_q2, q2, k2 = self._get_embeddings(x_flip)
                logits2, labels2 = self._get_contrastive_predictions(q2, k2)

                pos_ip2, neg_ip2 = self._get_pos_neg_ip(emb_q2, k2)
                pos_ip = (pos_ip + pos_ip2) / 2
                neg_ip = (neg_ip + neg_ip2) / 2
                contrastive_loss += self._get_contrastive_loss(logits2, labels2)

        contrastive_loss = contrastive_loss.mean() * self.hparams.loss_constant_factor

        log_data = {
            "step_train_loss": contrastive_loss,
            "step_pos_cos": pos_ip,
            "step_neg_cos": neg_ip,
            **losses,
        }

        with torch.no_grad():
            self._momentum_update_key_encoder()

        some_negative_examples = (
                self.hparams.use_negative_examples_from_batch or self.hparams.use_negative_examples_from_queue
        )
        if some_negative_examples:
            acc1, acc5 = calculate_accuracy(logits, labels, topk=(1, 5))
            log_data.update({"step_train_acc1": acc1, "step_train_acc5": acc5})

        # dequeue and enqueue
        if self.hparams.use_negative_examples_from_queue:
            self._dequeue_and_enqueue(k)

        self.log_dict(log_data)
        return {"loss": contrastive_loss}

    def validation_step(self, batch, batch_idx):
        x, class_labels = batch
        with torch.no_grad():
            emb = self.model(x)
            emb = emb.view(self.hparams.batch_size, self.hparams.embedding_dim)

        return {"emb": emb, "labels": class_labels}

    def validation_epoch_end(self, outputs):
        embeddings = torch.cat([x["emb"] for x in outputs]).cpu().detach().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().detach().numpy()
        num_split_linear = embeddings.shape[0] // 2
        self.sklearn_classifier.fit(embeddings[:num_split_linear], labels[:num_split_linear])
        train_accuracy = self.sklearn_classifier.score(embeddings[:num_split_linear], labels[:num_split_linear]) * 100
        valid_accuracy = self.sklearn_classifier.score(embeddings[num_split_linear:], labels[num_split_linear:]) * 100

        log_data = {
            "epoch": self.current_epoch,
            "train_class_acc": train_accuracy,
            "valid_class_acc": valid_accuracy,
            "T": self._get_temp(),
            "m": self._get_m(),
        }
        print(f"Epoch {self.current_epoch} accuracy: train: {train_accuracy:.1f}%, validation: {valid_accuracy:.1f}%")
        self.log_dict(log_data)

    def configure_optimizers(self):
        # exclude bias and batch norm from LARS and weight decay
        regular_parameters = []
        regular_parameter_names = []
        excluded_parameters = []
        excluded_parameter_names = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad is False:
                continue
            if any(x in name for x in self.hparams.exclude_matching_parameters_from_lars):
                excluded_parameters.append(parameter)
                excluded_parameter_names.append(name)
            else:
                regular_parameters.append(parameter)
                regular_parameter_names.append(name)

        param_groups = [
            {"params": regular_parameters, "names": regular_parameter_names, "use_lars": True},
            {
                "params": excluded_parameters,
                "names": excluded_parameter_names,
                "use_lars": False,
                "weight_decay": 0,
            },
        ]
        if self.hparams.optimizer_name == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError(f"No such optimizer {self.hparams.optimizer_name}")

        encoding_optimizer = optimizer(
            param_groups,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            encoding_optimizer,
            self.hparams.max_epochs,
            eta_min=self.hparams.final_lr_schedule_value,
        )
        return [encoding_optimizer], [self.lr_scheduler]

    def _get_m(self):
        if self.hparams.use_momentum_schedule is False:
            return self.hparams.m
        return 1 - (1 - self.hparams.m) * (math.cos(math.pi * self.current_epoch / self.hparams.max_epochs) + 1) / 2

    def _get_temp(self):
        return self.hparams.T

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        if not self.hparams.use_lagging_model:
            return
        m = self._get_m()
        for param_q, param_k in zip(self.model.parameters(), self.lagging_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
        for param_q, param_k in zip(self.projection_model.parameters(), self.lagging_projection_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.hparams.gather_keys_for_queue:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.K  # move pointer

        self.queue_ptr[0] = ptr

    def train_dataloader(self):
        return DataLoader(
            self.manager.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.manager.validation_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            persistent_workers=True,
        )


class MLP(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, num_layers, normalization=None
    ):
        super().__init__()
        assert num_layers >= 0, "negative layers?!?"
        if normalization is not None:
            assert callable(normalization), "normalization must be callable"

        if num_layers == 0:
            self.net = torch.nn.Identity()
            return

        if num_layers == 1:
            self.net = torch.nn.Linear(input_dim, output_dim)
            return

        layers = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            if normalization is not None:
                layers.append(normalization())
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def get_normalization(hparams: ModelParams, prediction=False):
        normalization_str = hparams.mlp_normalization
        if prediction and hparams.prediction_mlp_normalization != "same":
            normalization_str = hparams.prediction_mlp_normalization

        if normalization_str is None:
            return None
        elif normalization_str == "bn":
            return partial(torch.nn.BatchNorm1d, num_features=hparams.mlp_hidden_dim)
        elif normalization_str == "ln":
            return partial(torch.nn.LayerNorm, normalized_shape=[hparams.mlp_hidden_dim])
        elif normalization_str == "gn":
            return partial(torch.nn.GroupNorm, num_channels=hparams.mlp_hidden_dim, num_groups=32)
        else:
            raise NotImplementedError(f"mlp normalization {normalization_str} not implemented")


if __name__ == '__main__':
    encoder = resnet34(in_channels=1)
    base_config = ModelParams(
        encoder=encoder,
        embedding_dim=encoder.blocks[-1].blocks[-1].expanded_channels * 512,
        lr=0.08,
        batch_size=8,
        gather_keys_for_queue=False,
        loss_type="ip",
        use_negative_examples_from_queue=False,
        use_both_augmentations_as_queries=True,
        mlp_normalization="bn",
        prediction_mlp_layers=2,
        projection_mlp_layers=2,
        m=0.996,
    )
    configs = {
        "base": base_config,
        "pred_only": evolve(base_config, mlp_normalization=None, prediction_mlp_normalization="bn"),
        "proj_only": evolve(base_config, mlp_normalization="bn", prediction_mlp_normalization=None),
        "no_norm": evolve(base_config, mlp_normalization=None),
        "layer_norm": evolve(base_config, mlp_normalization="ln"),
        "xent": evolve(
            base_config, use_negative_examples_from_queue=True, loss_type="ce", mlp_normalization=None, lr=0.02
        ),
    }
    for seed in range(3):
        for name, config in configs.items():
            method = LitMoCo(config)
            logger = TensorBoardLogger("tb_logs", name=f"{name}_{seed}")
            if torch.has_mps:
                trainer = pl.Trainer(logger=logger,
                                     accelerator="cpu",
                                     log_every_n_steps=1,
                                     max_epochs=100,
                )
            else:
                trainer = pl.Trainer(logger=logger,
                                     accelerator="gpu",
                                     log_every_n_steps=1,
                                     max_epochs=100,
                )
            trainer.fit(method)
