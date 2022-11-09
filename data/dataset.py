import math
import attrs
import numpy as np
from _types import *
import torch.utils.data
from typing import Optional, Callable
from data.reader import NLSTDataReader


@attrs.define(auto_attribs=True, slots=True, init=False)
class DatasetBase:
    _reader: NLSTDataReader
    train_ds: Optional[torch.utils.data.Dataset] = None
    validation_ds: Optional[torch.utils.data.Dataset] = None
    test_ds: Optional[torch.utils.data.Dataset] = None
    transform_train: Optional[Callable] = None
    transform_test: Optional[Callable] = None

    def __init__(self, manifest: int, ds_split: list[float]):
        self._reader = NLSTDataReader(manifest)
        dataset = NLSTDataset(self._reader, self._reader.series_list)
        total_length = len(dataset)
        train_length = math.floor(total_length * ds_split[0])
        validation_length = math.floor(total_length * ds_split[1])
        test_length = total_length - train_length - validation_length
        self.train_ds, self.validation_ds, self.test_ds = torch.utils.data.random_split(
            dataset, [train_length, validation_length, test_length])
        print(self.train_ds)

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

    # TODO: modify this to add data augmentation so key and query are 2 augmented versions of the same scan
    def __getitem__(self, idx):
        q_series_id = self.series_list[idx]
        k_series_id = q_series_id
        while k_series_id == q_series_id:
            k_series_id = np.random.choice(self.series_list)
        # x = channel, depth, width, height
        x = self.reader.read_series(q_series_id)
        x = (
            np.array([
                x[0],
                self.reader.read_series(k_series_id)[0]
            ]),
            x[1],
        )

        return x


if __name__ == '__main__':
    database = DatasetBase(manifest=1663396252954, ds_split=[.3, .3, .4])
    print(len(database.test_ds))
