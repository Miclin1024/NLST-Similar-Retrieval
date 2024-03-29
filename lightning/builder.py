import os
import copy
import torch
import warnings

import torchvision.models.video

from data import *
from definitions import *
from evaluations import *
import torch.nn.functional
from typing import Optional
from functools import partial
import pytorch_lightning as pl
from lightning.utils import *
from lightning.lars import LARS
from torch.utils.data import DataLoader
from lightning.params import ModelParams
from pytorch_lightning.utilities import AttributeDict


MODEL_SAVE_PATH = os.path.join(LOG_DIR, "models")


class LitMoCo(pl.LightningModule):
    experiment_name: str
    model: torch.nn.Module
    manager: DatasetManager
    hparams: AttributeDict
    test_mode: bool = False
    embedding_dim: Optional[int]
    evaluators: dict[str, Evaluator]
    lr_scheduler: Any

    def __init__(self, experiment_name: str, hparams: Union[ModelParams, dict, None] = None, **kwargs):
        super(LitMoCo, self).__init__()

        self.experiment_name = experiment_name

        if hparams is None:
            hparams = ModelParams(**kwargs)
        elif isinstance(hparams, dict):
            hparams = ModelParams(**hparams, **kwargs)

        if isinstance(self.hparams, AttributeDict):
            self.hparams.update(AttributeDict(attrs.asdict(hparams)))
        else:
            self.hparams = AttributeDict(attrs.asdict(hparams))

        # Check for configuration issues
        if hparams.gather_keys_for_queue and not hparams.shuffle_batch_norm:
            warnings.warn(
                "Configuration suspicious: gather_keys_for_queue without shuffle_batch_norm or weight standardization"
            )

        some_negative_examples = hparams.use_negative_examples_from_batch or hparams.use_negative_examples_from_queue
        if hparams.loss_type == "ce" and not some_negative_examples:
            warnings.warn("Configuration suspicious: cross entropy loss without negative examples")

        self.model = hparams.encoder

        self.manager = DatasetManager(hparams, ds_split=[.75, .25, 0], default_access_mode="cached")
        _reader = self.manager.reader
        experiment_regression_evaluator = partial(
            RegressionEvaluator, experiment_name, hparams.eval_batch_size or hparams.batch_size, self
        )
        experiment_classification_evaluator = partial(
            ClassificationEvaluator, experiment_name, hparams.eval_batch_size or hparams.batch_size, self
        )
        self.evaluators = {
            "sp": SamePatientEvaluator(experiment_name, hparams.eval_batch_size or hparams.batch_size, self),
            "linear/gender": experiment_classification_evaluator(target_key="gender"),
            "linear/cigsmok": experiment_classification_evaluator(target_key="cigsmok"),
            "linear/diag/fibr": experiment_classification_evaluator(target_key="diagfibr"),
            "linear/diag/heart": experiment_classification_evaluator(target_key="diaghear"),
            "linear/diag/hype": experiment_classification_evaluator(target_key="diaghype"),
            "linear/diag/emph": experiment_classification_evaluator(target_key="diagemph"),
            "linear/diag/diab": experiment_classification_evaluator(target_key="diagdiab"),
            "linear/diag/chro": experiment_classification_evaluator(target_key="diagchro"),
            "linear/invaslc": experiment_classification_evaluator(target_key="invaslc"),
            "linear/icd_topo": experiment_classification_evaluator(target_key="confirmed_icd_topog1", ignore_nan=False),
            "linear/weight": experiment_regression_evaluator(target_key="weight"),
            "linear/height": experiment_regression_evaluator(target_key="height"),
            "linear/age": experiment_regression_evaluator(target_key="age"),
        }

        if hparams.use_lagging_model:
            # "key" function (no grad)
            print(f"Creating momentum key encoder...")
            self.lagging_model = copy.deepcopy(self.model)
            for param in self.lagging_model.parameters():
                param.requires_grad = False
        else:
            self.lagging_model = None

        print(f"Building projection MLP layer...")
        self.projection_model = MLP(
            hparams.embedding_dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.projection_mlp_layers,
            normalization=MLP.get_normalization(hparams),
        )

        print(f"Building prediction MLP layer...")
        self.prediction_model = MLP(
            hparams.dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.prediction_mlp_layers,
            normalization=MLP.get_normalization(hparams, prediction=True),
        )

        if hparams.use_lagging_model:
            #  "key" function (no grad)
            print(f"Creating momentum projection layer...")
            self.lagging_projection_model = copy.deepcopy(self.projection_model)
            for param in self.lagging_projection_model.parameters():
                param.requires_grad = False
        else:
            self.lagging_projection_model = None

        print("Registering buffer...")
        if hparams.use_negative_examples_from_queue:
            # create the queue
            self.register_buffer("queue", torch.randn(hparams.dim, hparams.K))
            self.queue = torch.nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.queue = None

        print(f"Saving hyperparameters...")
        self.save_hyperparameters(attrs.asdict(hparams), ignore="encoder")
        print(f"Builder initialization complete")

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
        q_projection = self.projection_model(emb_q)
        q = self.prediction_model(q_projection)  # queries: NxC

        if self.hparams.use_lagging_model:
            # compute key features
            with torch.no_grad():  # no gradient to keys
                if self.hparams.shuffle_batch_norm:
                    im_k, idx_unshuffle = BatchShuffleDDP.shuffle(im_k)
                k = self.lagging_projection_model(self.lagging_model(im_k))  # keys: NxC
                if self.hparams.shuffle_batch_norm:
                    k = BatchShuffleDDP.unshuffle(k, idx_unshuffle)
        else:
            emb_k = self.model(im_k)
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
                return torch.nn.functional.nll_loss(predictions, labels)
       
            return torch.nn.functional.cross_entropy(logits / self.hparams.T, labels)

        new_labels = torch.zeros_like(logits)
        new_labels.scatter_(1, labels.unsqueeze(1), 1)
        if self.hparams.loss_type == "bce":
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits / self.hparams.T, new_labels) * logits.shape[1]

        if self.hparams.loss_type == "ip":
            # inner product
            # negative sign for label=1 (maximize ip), positive sign for label=0 (minimize ip)
            inner_product = (1 - new_labels * 2) * logits
            return torch.mean((inner_product + 1).sum(dim=-1))

        raise NotImplementedError(f"Loss function {self.hparams.loss_type} not implemented")

    def forward(self, x) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, _ = batch  # batch is a tuple, we just want the image

        emb_q, q, k = self._get_embeddings(x)
        pos_ip, neg_ip = self._get_pos_neg_ip(emb_q, k)
        logits, labels = self._get_contrastive_predictions(q, k)

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
            "train/step_train_loss": contrastive_loss.item(),
            "train/step_pos_cos": pos_ip.item(),
            "train/step_neg_cos": neg_ip.item(),
        }

        with torch.no_grad():
            self._momentum_update_key_encoder()

        # dequeue and enqueue
        if self.hparams.use_negative_examples_from_queue:
            self._dequeue_and_enqueue(k)

        self.log_dict(log_data)
        return {"loss": contrastive_loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch  # batch is a tuple, we just want the image

        with torch.no_grad():
            emb_q, q, k = self._get_embeddings(x)
            pos_ip, neg_ip = self._get_pos_neg_ip(emb_q, k)
            logits, labels = self._get_contrastive_predictions(q, k)

            contrastive_loss = self._get_contrastive_loss(logits, labels)
            contrastive_loss = contrastive_loss.mean() * self.hparams.loss_constant_factor

            log_data = {
                "val/step_train_loss": contrastive_loss.item(),
                "val/step_pos_cos": pos_ip.item(),
                "val/step_neg_cos": neg_ip.item(),
            }

            self.log_dict(log_data)

    def validation_epoch_end(self, outputs):
        for key, evaluator in self.evaluators.items():
            results: dict = evaluator.score(self.manager.validation_ds.effective_series_list)
            self.log_dict({
                f"val/{key}/{metric_key}": value for metric_key, value in results.items()
            })

        self._create_model_checkpoint()

    def _create_model_checkpoint(self):
        experiment_folder = os.path.join(MODEL_SAVE_PATH, self.experiment_name)
        os.makedirs(experiment_folder, exist_ok=True)
        model_path = os.path.join(experiment_folder, f"epoch{self.current_epoch}.pt")

        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "projection_model_state_dict": self.projection_model.state_dict(),
            "prediction_model_state_dict": self.prediction_model.state_dict(),
        }, model_path)
        print(f"Checkpoint saved to {model_path}")

    def configure_optimizers(self):
        # exclude bias and batch norm from LARS and weight decay
        print("Configuring optimizers...")
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
            optimizer = partial(torch.optim.SGD, momentum=self.hparams.momentum)
        elif self.hparams.optimizer_name == "lars":
            optimizer = partial(LARS, momentum=self.hparams.momentum, warmup_epochs=self.hparams.lars_warmup_epochs, eta=self.hparams.lars_eta)
        elif self.hparams.optimizer_name == "adam":
            optimizer = partial(torch.optim.Adam, betas=(0.9, 0.999))
        else:
            raise NotImplementedError(f"No such optimizer {self.hparams.optimizer_name}")

        encoding_optimizer = optimizer(
            param_groups,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            encoding_optimizer,
            self.hparams.max_epochs,
            eta_min=self.hparams.final_lr_schedule_value,
        )
        print(f"Done. Optimizer is {self.hparams.optimizer_name}")
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
        loader = DataLoader(
            self.manager.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            shuffle=True,
            persistent_workers=True,
        )
        size = len(self.manager.train_series)
        print(f"Train data loader with size = {size} initialized")
        self.save_hyperparameters({"train_size": size})
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.manager.validation_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            persistent_workers=True,
        )
        size = len(self.manager.val_series)
        print(f"Validation data loader with size = {size} initialized")
        self.save_hyperparameters({"validation_size": size})
        return loader
