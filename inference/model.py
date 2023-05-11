import attrs
import data.reader
from definitions import *
import torchvision.models
from typing import Type, TypeVar
from inference.query import _InferenceQueryMixin
from inference.index import _IndexBuilderMixin


TInferenceAdapter = TypeVar("TInferenceAdapter", bound="InferenceAdapter")
model_logdir = os.path.join(LOG_DIR, "models")


@attrs.define(init=False)
class InferenceAdapter(_InferenceQueryMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pl_checkpoint(cls: Type[TInferenceAdapter], experiment: str, epoch: int) -> TInferenceAdapter:
        model = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3),
            torchvision.models.video.r3d_18(weights=None)
        ).to("cuda")
        checkpoint_path = os.path.join(model_logdir, experiment, f"epoch{epoch}.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        layers = list(model.children())
        channel_expander = layers[0]
        layers = list(layers[1].children())

        conv = torch.nn.Sequential(channel_expander, *layers[:5])

        return cls(f"{experiment}_e{epoch}", conv)


if __name__ == '__main__':
    adapter = InferenceAdapter.from_pl_checkpoint("r3d_18_moco_v119", 70)
    # adapter.load_index()
    query_path = "/mnt/sohn2020/NLST_032022/manifest_10-1632962895431/NLST/" \
                 "213611/01-02-2000-NA-NLST-ACRIN-54987/2.000000-1OPASESEN16B30f370212037.5251.5-29409"
    query_image = data.reader.env_reader.read_series_at_path(query_path)
    x, y, z = 4, 4, 4
    emb = adapter.query_embedding(query_image, x, y, z)
    adapter.load_index()
    results = adapter.query_index(emb)
    print(results[0])
    # input_tensor = torch.randn(8, 1, 128, 128, 128).to("cuda")
    # output_tensor = adapter.conv(input_tensor)
    # print(f"Conv output shape is {output_tensor.shape}")
