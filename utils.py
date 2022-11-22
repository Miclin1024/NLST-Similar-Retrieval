import torch
import random
import pydicom
import numpy as np
from PIL import ImageFilter


# Reference: https://www.kaggle.com/code/redwankarimsony/ct-scans-dicom-files-windowing-explained/notebook
# Function to take care of the translation and windowing.
def window_image(pixel_data: np.ndarray,
                 window_center: int, window_width: int,
                 intercept: int, slope: int, rescale=True):
    pixel_data = (pixel_data * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    pixel_data[pixel_data < img_min] = img_min
    pixel_data[pixel_data > img_max] = img_max
    if rescale:
        pixel_data = (pixel_data - img_min) / (img_max - img_min) * 255.0
    return pixel_data


def get_first_of_dicom_field_as_int(x):
    # Get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data: pydicom.FileDataset) -> (int, int, int, int):
    dicom_fields = [data[('0028', '1050')].value,  # Window Center
                    data[('0028', '1051')].value,  # Window Width
                    data[('0028', '1052')].value,  # Intercept
                    data[('0028', '1053')].value]  # Slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
