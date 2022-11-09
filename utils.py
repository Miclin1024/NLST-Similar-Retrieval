import torch
import random
import pydicom
import numpy as np
from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


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


def log_softmax_with_factors(logits: torch.Tensor, log_factor: float = 1, neg_factor: float = 1) -> torch.Tensor:
    exp_sum_neg_logits = torch.exp(logits).sum(dim=-1, keepdim=True) - torch.exp(logits)
    softmax_result = logits - log_factor * torch.log(torch.exp(logits) + neg_factor * exp_sum_neg_logits)
    return softmax_result
