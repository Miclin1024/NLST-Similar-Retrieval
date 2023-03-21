import torch
from functools import partial
import torch.distributed as dist
from lightning.params import ModelParams


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


def log_softmax_with_factors(logits: torch.Tensor, log_factor: float = 1, neg_factor: float = 1) -> torch.Tensor:
    exp_sum_neg_logits = torch.exp(logits).sum(dim=-1, keepdim=True) - torch.exp(logits)
    softmax_result = logits - log_factor * torch.log(torch.exp(logits) + neg_factor * exp_sum_neg_logits)
    return softmax_result


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class BatchShuffleDDP:
    @staticmethod
    @torch.no_grad()
    def shuffle(x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(x.device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @staticmethod
    @torch.no_grad()
    def unshuffle(x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
