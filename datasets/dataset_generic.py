

from datasets.dataset_base import DatasetBase
# class DatasetMeta(data.Dataset):
#    pass

class DatasetGeneric(DatasetBase):
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
    additional_sample_attribute_keys = []

    def __init__(
        self, **kwargs
        ):
        
 
        # super(DatasetBase, self).__init__()
        super(DatasetGeneric, self).__init__(**kwargs)
     

    # def add_additional_sample_attributes(self, data):   
    #     return
        #Extended dataset class can add more attributes to each sample here
        # return data
