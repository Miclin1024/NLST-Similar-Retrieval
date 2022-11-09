import attrs
from _types import *
import torch.utils.data
from reader import NLSTDataReader
from typing import Optional, Callable


@attrs.define(auto_attribs=True, slots=True, init=False)
class DatasetBase:
    _reader: NLSTDataReader
    train_ds: Optional[torch.utils.data.Dataset] = None
    validation_ds: Optional[torch.utils.data.Dataset] = None
    test_ds: Optional[torch.utils.data.Dataset] = None
    transform_train: Optional[Callable] = None
    transform_test: Optional[Callable] = None

    def __init__(self, manifest: int, ds_split: list[int]):
        self._reader = NLSTDataReader(manifest)
        dataset = NLSTDataset(self._reader, self._reader.series_list)
        self.train_ds, self.validation_ds, self.test_ds = torch.utils.data.random_split(
            dataset, ds_split)

    @property
    def instance_shape(self):
        instance = next(iter(self.train_ds))[0]
        return instance.shape


@attrs.define(slots=True, auto_attribs=True)
class NLSTDataset(torch.utils.data.Dataset):
    reader: NLSTDataReader
    series_list: list[SeriesID]
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, idx):
        series_id = self.series_list[idx]
        return self.reader.read_series(series_id)
