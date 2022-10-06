import os
import glob
import pydicom
import cv2
import numpy as np
import pandas as pd
from utils import *
import matplotlib.pyplot as plt

DATA_FOLDER = os.path.join(os.path.curdir, "data")
TRANSFORM_SHAPE = (128, 128, 128)


class NLSTDataReader:
    def __init__(self, manifest: int):
        self.manifest_folder = os.path.join(DATA_FOLDER, f"manifest-{manifest}")
        self.metadata = pd.read_csv(
            os.path.join(self.manifest_folder, "metadata.csv")
        )

    def read_patient(self, sid: int, output="pixel") \
            -> dict[str, np.ndarray | str | pydicom.FileDataset]:
        metadata = self.metadata
        patient_meta = metadata[
            (metadata["Subject ID"] == sid) &
            (metadata["Number of Images"] > 3)
        ].drop_duplicates(subset=["Study Date"], keep="first")

        assert len(patient_meta.index) != 0, f"Patient {sid} not found"

        patient_meta = patient_meta.reset_index(drop=True)

        if output == "uid":
            return {row["Study Date"]: row["Series UID"]
                    for _, row in patient_meta.iterrows()}
        else:
            return {row["Study Date"]: self.read_series(
                        row["File Location"], use_full_dcm=(output == "dcm")
                    ) for _, row in patient_meta.iterrows()}

    def read_uid(self, uid: str, output="pixel") -> np.ndarray:
        metadata = self.metadata
        path = metadata[metadata["Series UID"] == uid]["File Location"].iloc[0]
        return self.read_series(path, use_full_dcm=(output == "dcm"))

    def read_all_patients(self, output="pixel") \
            -> dict[str, dict[str, np.ndarray | str | pydicom.FileDataset]]:
        metadata = self.metadata
        patients_meta = metadata[
            metadata["Number of Images"] > 3
        ].drop_duplicates(subset=["Subject ID", "Study Date"], keep="first")
        patients_meta = patients_meta.reset_index(drop=True)

        patients_data = {}
        for _, row in patients_meta.iterrows():
            sid = row["Subject ID"]
            study_date = row["Study Date"]
            if sid not in patients_data:
                patients_data[sid] = {}
            if output == "uid":
                patients_data[sid][study_date] = row["Series UID"]
            else:
                patients_data[sid][study_date] = self.read_series(
                    row["File Location"], use_full_dcm=(output == "dcm"))

        return patients_data

    def read_series(self, path: str, use_full_dcm=False) -> np.ndarray:
        series_folder = os.path.join(self.manifest_folder, path)
        if use_full_dcm:
            return np.array([
                pydicom.dcmread(file)
                for file in glob.glob(f"{series_folder}/*")
            ])
        else:
            return self.transform(np.array([
                pydicom.dcmread(file).pixel_array
                for file in glob.glob(f"{series_folder}/*")
            ]))

    @staticmethod
    def transform(series_array: np.ndarray) -> np.ndarray:
        # Slice or pad the first axis
        slice_diff = TRANSFORM_SHAPE[0] - series_array.shape[0]
        if slice_diff > 0:
            slice_pad = (slice_diff // 2 + slice_diff % 2, slice_diff // 2)
            result = np.pad(series_array, (slice_pad, (0, 0), (0, 0)), mode="edge")
        elif slice_diff < 0:
            slice_diff *= -1
            begin_index = slice_diff // 2 + slice_diff % 2
            end_index = series_array.shape[0] - slice_diff // 2
            result = series_array[begin_index:end_index]
        else:
            result = series_array

        size = (TRANSFORM_SHAPE[1], TRANSFORM_SHAPE[2])
        return np.array([
            cv2.resize(pixel_array, dsize=size, interpolation=cv2.INTER_CUBIC)
            for pixel_array in result
        ])

    def visualize(self, sid: int, window_width: int = 400, window_center: int = 40, date: str = None):
        patient_data = self.read_patient(sid, output="dcm")
        for date in sorted(patient_data.keys()):
            image_series = patient_data[date]
            print(f"{date} (5/{len(image_series)} slices chosen uniformly within each scan):")
            step_size = len(image_series) // 4

            fig, axs = plt.subplots(1, 5)
            for index in range(5):
                dicom_data = image_series[index * step_size]
                _, _, intercept, slope = get_windowing(dicom_data)
                image = window_image(dicom_data.pixel_array,
                                     window_center, window_width, intercept, slope)
                axs[index].imshow(image, cmap=plt.cm.gray)
                axs[index].axis("off")

            plt.show()


if __name__ == '__main__':
    dataset = NLSTDataReader(manifest=1663396252954)
    uid = dataset.read_patient(100002, output="uid")["01-02-1999"]
    print(dataset.read_uid(uid))

