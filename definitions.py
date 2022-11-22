from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
import os


Tensor = torch.Tensor
PatientID = int
SeriesID = str

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
