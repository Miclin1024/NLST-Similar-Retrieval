import attrs
import torch
import numpy as np
import torchio as tio
from data.reader import *
from typing import Optional, Tuple
from inference.index import _IndexBuilderMixin


@attrs.define()
class IndexQueryResult:
    series_id: SeriesID = attrs.field()

    x: int = attrs.field()
    y: int = attrs.field()
    z: int = attrs.field()

    score: float = attrs.field()


class _InferenceQueryMixin(_IndexBuilderMixin):

    enforce_same_location = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def query_embedding(self, image: np.ndarray, loc: (int, int, int)) -> np.ndarray:
        image = np.reshape(image, (1, *image.shape))
        with torch.no_grad():
            image = torch.tensor(image).to("cuda")
            output_conv = self.conv(image)

            output_conv = torch.nn.functional.normalize(output_conv)
            output_conv = output_conv.cpu().numpy()

        x, y, z = loc
        result = output_conv[0, :, x, y, z]
        assert result.shape == (512,)
        return result

    def query_index(self, query: np.ndarray, loc: Optional[Tuple[int, int, int]] = None) -> [IndexQueryResult]:
        results = [self.query_index_item(query, key, loc) for key in self.index.keys()]
        results.sort(reverse=True, key=lambda x: x.score)
        results = results[:100]
        return results

    def query_index_item(self, query: np.ndarray, key: SeriesID, loc: Optional[Tuple[int, int, int]] = None) \
            -> IndexQueryResult:
        item = self.index[key]
        assert query.shape == (512,)
        assert item.shape == (512, 16, 8, 8)

        if self.enforce_same_location and loc is not None:
            x, y, z = loc
            score = float(np.dot(query, item[:, x, y, z]))
            result = IndexQueryResult(key, x, y, z, score)
            return result
        else:
            channel_scores = np.einsum("cijk,c->ijk", item, query)
            x, y, z = np.unravel_index(channel_scores.argmax(), channel_scores.shape)
            result = IndexQueryResult(key, x, y, z, channel_scores[x, y, z])

        return result
