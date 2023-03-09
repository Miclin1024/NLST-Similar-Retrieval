import math
import attrs
import itertools
import numpy as np
from definitions import *
import torchio as tio
import torch.utils.data
from typing import Optional, Callable
from data.reader import NLSTDataReader


def default_moco_transform() -> Callable:
    transforms = [
        tio.RandomFlip(),
        tio.RandomBiasField(),
        tio.RandomElasticDeformation(max_displacement=5),
        # tio.RandomAffine(
        #     scales=(1.0, 1.0),
        #     degrees=45,
        #     translation=0,
        #     isotropic=True,
        #     center="image",
        # ),
        tio.RandomNoise(std=(0, 0.1)),
        # tio.RandomBlur(),

        tio.ZNormalization(),
    ]
    return tio.Compose(transforms)


class DatasetManager:
    reader: NLSTDataReader
    train_ds: torch.utils.data.Dataset
    validation_ds: torch.utils.data.Dataset
    test_ds: Optional[torch.utils.data.Dataset]
    transform_train: Callable
    transform_validation: Callable
    transform_test: Callable
    train_series: list[SeriesID]
    val_series: list[SeriesID]
    test_series: list[SeriesID]

    def __init__(self, manifests: [int], ds_split=None,
                 transform_train: Callable = default_moco_transform(),
                 transform_validation: Callable = default_moco_transform(),
                 transform_test: Callable = default_moco_transform(),
                 default_access_mode: str = "cached"):

        if ds_split is None:
            ds_split = [.8, .2, .0]

        self.reader = NLSTDataReader(manifests, default_access_mode=default_access_mode)
        patient_index = self.reader.patient_series_index
        patients_list = list(patient_index.keys())
        np.random.shuffle(patients_list)
        patients_count = len(patients_list)

        train_idx = math.floor(patients_count * ds_split[0])
        validation_idx = math.floor(patients_count * ds_split[1]) + train_idx

        self.transform_train = transform_train
        self.transform_validation = transform_validation
        self.transform_test = transform_test

        self.train_series = list(itertools.chain(
            *[patient_index[pid] for pid in patients_list[:train_idx]]
        ))
        self.val_series = list(itertools.chain(
            *[patient_index[pid] for pid in patients_list[train_idx:validation_idx]]
        ))

        if validation_idx < patients_count:
            self.test_series = list(itertools.chain(
                *[patient_index[pid] for pid in patients_list[validation_idx:]]
            ))
        else:
            self.test_series = []
        
        self.transform_train = transform_train
        self.transform_validation = transform_validation
        self.transform_test = transform_test

        self.train_ds = NLSTDataset(
            self.reader, effective_series_list=self.train_series, train=True,
            transform=transform_train
        )
        self.validation_ds = NLSTDataset(
            self.reader, effective_series_list=self.val_series, train=False,
            transform=transform_validation
        )
        if len(self.test_series) != 0:
            self.test_ds = NLSTDataset(
                self.reader, effective_series_list=self.test_series, train=False,
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
        stacked_tensor = torch.stack(
            (self.transform(slice_tensor).to(torch.float16), self.transform(slice_tensor).to(torch.float16))
            # (slice_tensor, self.transform(slice_tensor))
        )
        return stacked_tensor, target


if __name__ == '__main__':
    manager = DatasetManager(manifests=[1632928843386])
