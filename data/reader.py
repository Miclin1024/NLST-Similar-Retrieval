import os
import cv2
import glob
import attrs
import torch
import numpy as np
import pandas as pd
from utils import *
import torchio as tio
from typing import Tuple
from definitions import *
from pydicom import dcmread
from dotenv import load_dotenv
import matplotlib.pyplot as plt


# Use a .env file to indicate where the manifests are stored
load_dotenv()
DATA_FOLDER = os.getenv("DATA_FOLDER") or "data/"


@attrs.define(slots=True, auto_attribs=True, init=False)
class NLSTDataReader:
    manifest_folder: str
    manifest: pd.DataFrame
    metadata: pd.DataFrame
    patient_series_index: dict[PatientID, list[SeriesID]]
    series_list: list[SeriesID]
    target_meta_key: str
    # if True, cut short the metadata list to 100 scans to speed up training and testing of code
    test_mode: bool = False 

    def __init__(self, manifest: int, target_meta_key: str = "weight", test_mode: bool=False):
        self.metadata = pd.read_csv(
            os.path.join(ROOT_DIR, "metadata/nlst_297_prsn_20170404.csv"),
            dtype={201: "str", 224: "str", 225: "str"}
        )
        self.target_meta_key = target_meta_key

        manifest_folder = glob.glob(f"{DATA_FOLDER}manifest*{manifest}", recursive=False)
        if len(manifest_folder) > 0:
            self.manifest_folder = manifest_folder[0]

        else:
            raise ValueError(f"Cannot locate the manifest {manifest}. "
                             "Have you configured your data folder and make "
                             "sure that the manifest folder is in there?")
        self.manifest = pd.read_csv(
            os.path.join(self.manifest_folder, "metadata.csv")
        )
        # Remove localizer series. Remove duplicates done on the same
        # patient on the same date (the scans use different post-processing kernels).
        self.manifest = self.manifest[self.manifest["Number of Images"] > 3]\
            .drop_duplicates(subset=["Subject ID", "Study Date"], keep="first")
        
        self.test_mode = test_mode
        
        # cut the metadata short for testing 
        if self.test_mode:
            self.manifest = self.manifest.head(100)
            
        index = {}
        # Build an index with patient id, scans are ordered by their dates
        # Using iterrows() over itertuple() due to space within column names
        for _, row in self.manifest.iterrows():
            patient_id = row["Subject ID"]
            if patient_id not in index.keys():
                index[patient_id] = []
            year = int(row["Study Date"].split("-")[-1])
            series_id = row["Series UID"]
            insert_loc = 0
            for _, current_year in index[patient_id]:
                if current_year < year:
                    insert_loc += 1
                else:
                    break
            index[patient_id].insert(insert_loc, (series_id, year))

        self.series_list = []
        self.patient_series_index = {}
        for patient_id in index.keys():
            patient_list = list(map(
                lambda x: x[0], index[patient_id]
            ))
            self.patient_series_index[patient_id] = patient_list
            self.series_list.extend(patient_list)

    def __len__(self):
        return len(self.series_list)

    def read_series(self, series_id: SeriesID) -> Tuple[tio.Image, dict]:
        manifest_row = self.manifest[self.manifest["Series UID"] == series_id].iloc[0].to_dict()
        path = manifest_row["File Location"]
        series_folder = os.path.join(self.manifest_folder, path)
        image = self.preprocess(tio.ScalarImage(series_folder))

        pid: PatientID = manifest_row["Subject ID"]
        metadata_row = self.metadata[self.metadata["pid"] == pid].iloc[0].to_dict()

        weight = metadata_row["weight"]
        if np.isnan(weight):
            weight = 183

        height = metadata_row["height"]
        if np.isnan(height):
            height = 68

        bmi = weight / (height ** 2) * 703
        bmi_range = np.array([18.5, 25, 30, 35, 40])

        return image, {
            "pid": pid,
            "bmi_category": np.count_nonzero(bmi > bmi_range),
            "series_id": series_id
        }

    def read_patient(self, patient_id: PatientID) -> Tuple[tio.Image, dict]:
        patient_series_list = self.patient_series_index[patient_id]
        return self.read_series(patient_series_list[0])

    @staticmethod
    def preprocess(image: tio.Image) -> tio.Image:
        transforms = [
            tio.ZNormalization(),
            tio.Resample((2.8, 2.8, 2.5)),
            tio.CropOrPad((128, 128, 128)),
        ]
        return tio.Compose(transforms)(image)


if __name__ == '__main__':
    dataset = NLSTDataReader(manifest=1663396252954)
    print(dataset.read_series(dataset.series_list[0]))

