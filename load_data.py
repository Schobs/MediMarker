import os
import json
from typing import Optional


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
        raise ValueError(f"data list {data_list_key} not specified in '{data_list_file_path}'.")
    expected_data = json_data[data_list_key]
 
    if base_dir is None:
        base_dir = os.path.dirname(data_list_file_path)

    return _append_paths(base_dir, expected_data)