import os
import cv2
import glob
import attrs
import pandas as pd
from utils import *
from _types import *
from pydicom import dcmread
from dotenv import load_dotenv
import matplotlib.pyplot as plt


# Use a .env file to indicate where the manifests are stored
load_dotenv()
DATA_FOLDER = os.getenv("DATA_FOLDER") or "data/"
TRANSFORM_SHAPE = (128, 128, 128)


@attrs.define(slots=True, auto_attribs=True, init=False)
class NLSTDataReader:
    manifest_folder: str
    metadata: pd.DataFrame
    patient_series_index: dict[PatientID, list[SeriesID]]
    series_list: list[SeriesID]

    def __init__(self, manifest: int):
        manifest_folder = glob.glob(f"{DATA_FOLDER}manifest*{manifest}", recursive=False)
        if len(manifest_folder) > 0:
            self.manifest_folder = manifest_folder[0]

        else:
            raise ValueError("Cannot locate the manifest. "
                             "Have you configured your data folder and make "
                             "sure that the manifest folder is in there?")
        self.metadata = pd.read_csv(
            os.path.join(self.manifest_folder, "metadata.csv")
        )
        # Remove localizer series. Remove duplicates done on the same
        # patient on the same date (the scans use different post-processing kernels).
        self.metadata = self.metadata[self.metadata["Number of Images"] > 3]\
            .drop_duplicates(subset=["Subject ID", "Study Date"], keep="first")

        index = {}
        # Build an index with patient id, scans are ordered by their dates
        # Using iterrows() over itertuple() due to space within column names
        for _, row in self.metadata.iterrows():
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

    def read_series(self, series_id: SeriesID) -> (np.ndarray, dict):
        row = self.metadata[self.metadata["Series UID"] == series_id].iloc[0].to_dict()
        path = row["File Location"]
        series_folder = os.path.join(self.manifest_folder, path)
        pixel_data = self.transform(np.array([
            dcmread(file).pixel_array
            for file in glob.glob(f"{series_folder}/*")
        ])).astype("int32")
        return pixel_data, row

    def read_patient(self, patient_id: PatientID) -> (np.ndarray, dict):
        patient_series_list = self.patient_series_index[patient_id]
        return self.read_series(patient_series_list[0])

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
        result = np.array([
            cv2.resize(pixel_array, dsize=size,
                       interpolation=cv2.INTER_LINEAR_EXACT)
            for pixel_array in result
        ])
        # Transform from (128, 128, 128) to (1, 128, 128, 128) so that after
        # apply batches our shape would be (B, C_in, D, H, W)
        result = result.reshape((1, -1, *size))
        return result

    def visualize(self, patient_id: PatientID, window_width: int = 400, window_center: int = 40, date: str = None):
        patient_series_id = self.patient_series_index[patient_id]
        for series_id in patient_series_id:
            pixel_data, _ = self.read_series(series_id)
            print(f"{date} (5/{len(pixel_data)} slices chosen uniformly within each scan):")
            step_size = len(pixel_data) // 4

            fig, axs = plt.subplots(1, 5)
            for index in range(5):
                dicom_data = pixel_data[index * step_size]
                _, _, intercept, slope = get_windowing(dicom_data)
                image = window_image(dicom_data.pixel_array,
                                     window_center, window_width, intercept, slope)
                axs[index].imshow(image, cmap=plt.cm.gray)
                axs[index].axis("off")

            plt.show()


if __name__ == '__main__':
    dataset = NLSTDataReader(manifest=1663396252954)
    print(dataset.read_patient(100002)[0].shape)

