import os
import cv2
import glob
import attrs
import torch
import itertools
import numpy as np
import pandas as pd
from utils import *
import torchio as tio
from tqdm import tqdm
from typing import Tuple, Optional
from definitions import *
from pydicom import dcmread
from dotenv import load_dotenv


# Use a .env file to indicate where the manifests are stored
load_dotenv(os.path.join(ROOT_DIR, ".env"))
DATA_FOLDER = os.getenv("DATA_FOLDER") or "data/"
PREPROCESS_FOLDER = os.path.join(ROOT_DIR, "data", "cache")
# The list of Patient ID to ignore, due to faulty or corrupted local data entry.
PATIENT_EXCLUDE_SET = {101224, 102386, 105030, 115933, 200001}
os.makedirs(PREPROCESS_FOLDER, exist_ok=True)


@attrs.define(slots=True, auto_attribs=True, init=False)
class NLSTDataReader:
    manifest_folder: str
    manifest: pd.DataFrame
    metadata: pd.DataFrame
    patient_series_index: dict[PatientID, list[SeriesID]]
    series_list: list[SeriesID]
    target_meta_key: str
    # if True, cut short the metadata list to 100 scans to speed up training and testing of code
    default_access_mode: str = "cached"
    shape = (128, 128, 128)

    def __init__(self, manifests: [int], target_meta_key: str = "weight",
                 default_access_mode: str = "cached", head: Optional[int] = None):
        print(f"Initializing data reader with {len(manifests)} manifests...")
        print(f"Default access mode is {default_access_mode}, shape is {NLSTDataReader.shape}")
        self.metadata = pd.read_csv(
            os.path.join(ROOT_DIR, "metadata/nlst_297_prsn_20170404.csv"),
            dtype={201: "str", 224: "str", 225: "str"}
        )
        self.metadata.set_index("pid", inplace=True)
        self.target_meta_key = target_meta_key
        self.default_access_mode = default_access_mode

        manifest_dfs = []
        for manifest in manifests:
            manifest_folder = glob.glob(f"{DATA_FOLDER}manifest*{manifest}", recursive=False)
            if len(manifest_folder) > 0:
                manifest_folder = manifest_folder[0]
            else:
                raise ValueError(f"Cannot locate the manifest {manifest}. "
                                 "Have you configured your data folder and make "
                                 "sure that the manifest folder is in there?")
            df = pd.read_csv(
                os.path.join(manifest_folder, "metadata.csv")
            )
            df["Manifest Folder"] = str(manifest_folder)
            manifest_dfs.append(df)

        self.manifest = pd.concat(manifest_dfs, ignore_index=True)

        # Remove localizer series and series. Remove duplicates done on the same
        # patient on the same date (the scans use different post-processing kernels).
        self.manifest = self.manifest[self.manifest["Number of Images"] > 3]\
            .drop_duplicates(subset=["Subject ID", "Study Date"], keep="first")

        exclude_set = PATIENT_EXCLUDE_SET
        exclude_set.update(set(self.manifest[self.manifest["Number of Images"] < 100]["Subject ID"]))

        grouped = self.manifest.groupby("Subject ID").size()
        single_scan_patients = grouped[grouped == 1]
        exclude_set.update(set(single_scan_patients.index.tolist()))

        # Remove rows belongs to the exclude set.
        self.manifest = self.manifest[~self.manifest["Subject ID"].isin(exclude_set)]

        self.manifest.set_index("Series UID", inplace=True)
            
        index = {}
        # Build an index with patient id, scans are ordered by their dates
        # Using iterrows() over itertuple() due to space within column names
        for series_id, row in self.manifest.iterrows():
            patient_id = row["Subject ID"]
            if patient_id not in index.keys():
                index[patient_id] = []
            year = int(row["Study Date"].split("-")[-1])
            insert_loc = 0
            for _, current_year in index[patient_id]:
                if current_year < year:
                    insert_loc += 1
                else:
                    break
            index[patient_id].insert(insert_loc, (series_id, year))

        if head is not None:
            index = dict(itertools.islice(index.items(), head))

        self.series_list = []
        self.patient_series_index = {}
        for patient_id in index.keys():
            patient_list = list(map(
                lambda x: x[0], index[patient_id]
            ))
            self.patient_series_index[patient_id] = patient_list
            self.series_list.extend(patient_list)

        print(f"Reader initialized with {len(self.series_list)} series ({len(exclude_set)} patients excluded)")

    def __len__(self):
        return len(self.series_list)

    def perform_preprocessing(self):
        for series_id in tqdm(
                self.series_list,
                desc="Preprocessing original data and save for later"
        ):
            cache_path = os.path.join(PREPROCESS_FOLDER, f"{series_id}.nii.gz")
            # Assume no need to update if file exists
            if os.path.isfile(cache_path):
                continue
            manifest_row = self.manifest.loc[series_id].to_dict()
            path = manifest_row["File Location"]
            series_folder = os.path.join(manifest_row["Manifest Folder"], path)
            image = self.preprocess(tio.ScalarImage(series_folder))
            image.save(cache_path)

    def read_series(self, series_id: SeriesID, method: Optional[str] = None) -> Tuple[tio.Image, dict]:
        if method is None:
            method = self.default_access_mode
        manifest_row = self.manifest.loc[series_id].to_dict()

        if method == "cached":
            path = os.path.join(PREPROCESS_FOLDER, f"{series_id}.nii.gz")
            if not os.path.exists(path):
                print("Reader cache miss, falling back to direct read...")
                direct_path = os.path.join(manifest_row["Manifest Folder"], manifest_row["File Location"])
                image = self.preprocess(tio.ScalarImage(direct_path))
                image.save(path)
            else:
                image = tio.ScalarImage(path)
        elif method == "direct":
            path = os.path.join(manifest_row["Manifest Folder"], manifest_row["File Location"])
            image = self.preprocess(tio.ScalarImage(path))
        else:
            raise ValueError(f"Unknown read method {method}. It can either be 'cached' or 'direct")

        pid: PatientID = manifest_row["Subject ID"]
        metadata_row = self.metadata.loc[pid].to_dict()

        return image, {
            "pid": pid,
            "series_id": series_id
        }

    def read_patient(self, patient_id: PatientID) -> Tuple[tio.Image, dict]:
        patient_series_list = self.patient_series_index[patient_id]
        return self.read_series(patient_series_list[0])

    def read_patient_id(self, series_id: SeriesID) -> PatientID:
        manifest_row = self.manifest.loc[series_id].to_dict()
        return manifest_row["Subject ID"]

    def same_patient_scans(self, series_id: SeriesID) -> [SeriesID]:
        pid = self.read_patient_id(series_id)
        scans = self.patient_series_index[pid]
        return [scan for scan in scans if scan != series_id]

    def scan_year(self, series_id: SeriesID) -> int:
        date: str = self.manifest.loc[series_id].to_dict()["Study Date"]
        return int(date.split("-")[-1])

    def original_folder(self, series_id: SeriesID) -> str:
        manifest_row = self.manifest.loc[series_id].to_dict()
        return os.path.join(manifest_row["Manifest Folder"], manifest_row["File Location"])

    @staticmethod
    def preprocess(image: tio.Image) -> tio.Image:
        transforms = [
            tio.ZNormalization(),
            # Original: shape: (1, 512, 512, 126); spacing: (0.70, 0.70, 2.50)
            tio.Resample((2.8, 2.8, 2.5)),
            tio.CropOrPad(NLSTDataReader.shape),
        ]
        return tio.Compose(transforms)(image)


env_reader = NLSTDataReader(
    manifests=list(map(lambda elem: int(elem), os.environ.get("MANIFEST_ID").split(","))))


if __name__ == '__main__':
    print(env_reader.read_series(env_reader.series_list[0], method="direct"))
