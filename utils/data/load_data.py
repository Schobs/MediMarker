import os
import json
from typing import Optional
import nibabel as nib
from PIL import Image
import pydicom as dicom
import numpy as np
from transforms.transformations import (
    normalize_cmr,
)

import pandas as pd


def get_datatype_load(im_path):
    """Decides the image load function based on the suffix of the image path.

    Args:
        im_path (str): The path to an image

    Returns:
        lambda function: lambda function that loads the image depending on the suffix of the image path.
    """
    if "nii.gz" in im_path:
        return lambda pth: Image.fromarray(nib.load(pth).get_fdata())
    elif "dcm" in im_path:
        return lambda pth: Image.fromarray(dicom.dcmread(pth).pixel_array)
    elif "npz" in im_path:
        return lambda pth: Image.fromarray(np.load(pth)["arr_0"])
    else:
        return lambda pth: Image.open(pth)


def _compute_path(base_dir, element):

    if isinstance(element, str):
        return os.path.normpath(os.path.join(base_dir, element))
    elif isinstance(element, list):
        for e in element:
            if not isinstance(e, str):
                raise ValueError("file path must be a string.")
        return [os.path.normpath(os.path.join(base_dir, e)) for e in element]
    else:
        raise ValueError("file path must be a string or a list of string.")


def _append_paths(base_dir, items):
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("data item must be dict.")
        for k, v in item.items():
            if k == "image":
                item[k] = _compute_path(base_dir, v)
            elif k == "label":
                item[k] = _compute_path(base_dir, v)
            elif k == "coordinates":
                item[k] = v
    return items


def load_aspire_datalist(
    data_list_file_path: str,
    data_list_key: str = "training",
    base_dir: Optional[str] = None,
):
    """Load image/label paths of decathalon challenge from JSON file


    Args:
        data_list_file_path: the path to the json file of datalist.
        data_list_key: the key to get a list of dictionary to be used, default is "training".
        base_dir: the base directory of the dataset, if None, use the datalist directory.

    Raises:
        ValueError: data list file {data_list_file_path} does not exist.
        ValueError: data list {data_list_key} not specified in '{data_list_file_path}'.

    Returns a list of data items, each of which is a dict keyed by element names, for example:

    .. code-block::

        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': 0},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': 1}
        ]

    """
    if not os.path.isfile(data_list_file_path):
        raise ValueError(f"data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    if data_list_key not in json_data:
        raise ValueError(
            f"data list {data_list_key} not specified in '{data_list_file_path}'."
        )
    expected_data = json_data[data_list_key]

    if base_dir is None:
        base_dir = os.path.dirname(data_list_file_path)

    return _append_paths(base_dir, expected_data)


def load_and_resize_image(image_path, coords, load_im_size, data_type_load):
    """Load image and resize it to the specified size. Also resize the coordinates to match the new image size.

    Args:
        image_path (str): _description_
        coords ([ints]): _description_

    Returns:
        _type_: _description_
    """

    original_image = data_type_load(image_path)
    original_size = np.expand_dims(np.array(list(original_image.size)), 1)
    if list(original_image.size) != load_im_size:
        resizing_factor = [
            list(original_image.size)[0] / load_im_size[0],
            list(original_image.size)[1] / load_im_size[1],
        ]
        resized_factor = np.expand_dims(np.array(resizing_factor), axis=0)
    else:
        resizing_factor = [1, 1]
        resized_factor = np.expand_dims(np.array(resizing_factor), axis=0)

    # potentially resize the coords
    coords = np.round(coords * [1 / resizing_factor[0], 1 / resizing_factor[1]])
    image = np.expand_dims(
        normalize_cmr(original_image.resize(load_im_size)), axis=0
    )

    return resized_factor, original_size, image, coords


def maybe_get_coordinates_from_xlsx(datapath, uids, landmarks_to_return, sheet_name=None):
    """
        Read csv file of data, returns samples whesplitre the value of the "split" column
        is contained in the "fold" variable. The columns cols_to_return are returned.

    Args:
        datapath (str): Path to csv file of uncertainty results,
        split (str): column name for split e.g. Validation, testing,
        fold (int or [int]]): fold/s contained in the split column to return,
        cols_to_return ([str]): Which columns to return (default="All").


    Returns:
        [pandas dataframe, pandas dataframe]: dataframe selected
    """
    if datapath is None:
        return None
    
    if sheet_name is None:
        sheet_name = 0

    datafame = pd.read_excel(datapath, sheet_name=sheet_name)

    if isinstance(landmarks_to_return, int):
        cols_to_return = ["L"+str(landmarks_to_return)]
    elif isinstance(landmarks_to_return, (list, pd.core.series.Series, np.ndarray)):
        cols_to_return = ["L"+str(i) for i in landmarks_to_return]
    else:
        return ValueError("landmarks_to_return must be int or list of ints")

    filtered_df = datafame[datafame['uid'].isin(uids)]
    assert return_data.shape[0] == len(uids), "Not all uids found in csv file"

    return_data = filtered_df.loc[:, [uids] + cols_to_return]

    return_dict = return_data.apply(lambda x: {'uid': x.uid, 'landmarks': x.tolist()[1:]}, axis=1).tolist()

    # return_data.to_dict(orient='list')

    return return_dict
