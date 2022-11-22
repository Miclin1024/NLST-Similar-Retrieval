import math
import attrs
import numpy as np
from definitions import *
import torchio as tio
import torch.utils.data
from typing import Optional, Callable
from data.reader import NLSTDataReader


@attrs.define(auto_attribs=True, slots=True, init=False)
class DatasetManager:
    _reader: NLSTDataReader
    train_ds: torch.utils.data.Dataset
    validation_ds: torch.utils.data.Dataset
    test_ds: Optional[torch.utils.data.Dataset]
    transform_train: Callable
    transform_validation: Callable
    transform_test: Callable

    @staticmethod
    def default_moco_transform() -> Callable:
        transforms = [
            tio.RandomElasticDeformation(
                num_control_points=8, locked_borders=2),
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=15,
                translation=20,
                isotropic=True,
                center="image",
            ),
            tio.RandomNoise(),
            tio.RandomBlur(),
            tio.ZNormalization(),
        ]
        return tio.Compose(transforms)

    def __init__(self, manifest: int, ds_split: list[float],
                 transform_train: Callable = default_moco_transform(),
                 transform_validation: Callable = default_moco_transform(),
                 transform_test: Callable = default_moco_transform()):
        self._reader = NLSTDataReader(manifest)
        total_length = len(self._reader.series_list)
        train_idx = math.floor(total_length * ds_split[0])
        validation_idx = math.floor(total_length * ds_split[1]) + train_idx
        series_list = np.random.permutation(self._reader.series_list)
        split_series = np.split(series_list, [train_idx, validation_idx])

        self.transform_train = transform_train
        self.transform_validation = transform_validation
        self.transform_test = transform_test

        self.train_ds = NLSTDataset(
            self._reader, effective_series_list=split_series[0], train=True,
            transform=transform_train
        )
        self.validation_ds = NLSTDataset(
            self._reader, effective_series_list=split_series[1], train=False,
            transform=transform_validation,
        )
        if len(split_series[2]) != 0:
            self.test_ds = NLSTDataset(
                self._reader, effective_series_list=split_series[2], train=False,
                transform=transform_test
            )
        else:
            self.test_ds = None

    @property
    def instance_shape(self):
        instance = next(iter(self.train_ds))[0]
        return instance.shape


@attrs.define(slots=True, auto_attribs=True)
class NLSTDataset(torch.utils.data.Dataset):
    reader: NLSTDataReader
    effective_series_list: list[SeriesID]
    train: bool
    transform: Optional[Callable] = torch.nn.Identity()

    def __len__(self):
        return len(self.effective_series_list)

    def __getitem__(self, idx):
        q_series_id = self.effective_series_list[idx]
        k_series_id = q_series_id
        while k_series_id == q_series_id:
            k_series_id = np.random.choice(self.effective_series_list)
        # x = channel, depth, width, height
        slice_image, target = self.reader.read_series(q_series_id)
        slice_tensor = slice_image.tensor
        if self.train:
            slice_tensor = torch.stack(
                (self.transform(slice_tensor), self.transform(slice_tensor))
            )

        return slice_tensor, target


if __name__ == '__main__':
    database = DatasetManager(manifest=1663396252954, ds_split=[.8, .2, .0])
    print(len(database.train_ds))
