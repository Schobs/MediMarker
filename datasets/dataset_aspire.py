from abc import abstractclassmethod
import os
from pathlib import Path
from typing import List
import warnings

import imgaug as ia
import imgaug.augmenters as iaa
import nibabel as nib
import numpy as np
import torch
from imgaug.augmentables import Keypoint, KeypointsOnImage
from PIL import Image
from torch.utils import data
from torchvision import transforms
from transforms.transformations import (
    HeatmapsToTensor,
    NormalizeZScore,
    ToTensor,
    normalize_cmr,
)
from transforms.dataloader_transforms import get_aug_package_loader

from transforms.generate_labels import LabelGenerator, generate_heatmaps
from utils.data.load_data import get_datatype_load, load_aspire_datalist

from time import time

import multiprocessing as mp
import ctypes



from abc import ABC, abstractmethod, ABCMeta

from datasets.dataset_generic import DatasetBase



class DatasetAspire(DatasetBase):
    """
    A custom dataset superclass for loading landmark localization data

    Args:
        name (str): Dataset name.
        split (str): Data split type (train, valid or test).
        image_path (str): local directory of image path (default: "./data").
        annotation_path (str): local directory to file path with annotations.
        annotation_set (str): which set of annotations to use [junior, senior, challenge] (default: "junior")
        image_modality (str): Modality of image (default: "CMRI").


    References:
        #TO DO
    """

    # Additional sample attributes we want to log for each sample in the dataset, these are accessable without instantiating the class.
    additional_sample_attribute_keys = ["patient_id", "suid"]

    def __init__(self, **kwargs):

        # super(DatasetBase, self).__init__()
        super(DatasetAspire, self).__init__(
            **kwargs,
            additional_sample_attribute_keys=DatasetAspire.additional_sample_attribute_keys
        )
