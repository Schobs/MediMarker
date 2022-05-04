import os
from pathlib import Path
from typing import List

import imgaug as ia
import imgaug.augmenters as iaa
import nibabel as nib
import numpy as np
import torch
from imgaug.augmentables import Keypoint, KeypointsOnImage
from PIL import Image
from torch.utils import data
from torchvision import transforms
from transforms.transformations import (HeatmapsToTensor, NormalizeZScore, ToTensor,
                             normalize_cmr)
from transforms.dataloader_transforms import get_aug_package_loader

from transforms.generate_labels import LabelGenerator, generate_heatmaps, get_downsampled_heatmaps
from load_data import get_datatype_load, load_aspire_datalist
from visualisation import (visualize_heat_pred_coords, visualize_image_target,
                           visualize_image_trans_coords,
                           visualize_image_trans_target)
from time import time

import multiprocessing as mp
import ctypes
# import albumentations as A
# import albumentations.augmentations.functional as F
# from albumentations.pytorch import ToTensorV2



class DatasetBase(data.Dataset):
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

    def __init__(
        self,
        landmarks,
        sigmas,
        LabelGenerator,
        hm_lambda_scale: float,
        annotation_path: str,
        generate_hms_here: bool,
        sample_patches: bool,
        image_modality: str= "CMRI",
        split: str ="training",
        root_path: str = "./data",
        cv: int = -1,
        cache_data: bool = False,
        debug: bool = False,
        input_size=  [512,512],
        sample_patch_size = [512, 512],
        sample_patch_bias = 0.66,
        original_image_size = [512,512],
        num_res_supervisions: int = 5,
        data_augmentation_strategy: str = None,
        data_augmentation_package: str = None,
        ):
        
 
        super(DatasetBase, self).__init__()

        self.LabelGenerator = LabelGenerator

        self.data_augmentation_strategy = data_augmentation_strategy
        self.hm_lambda_scale = hm_lambda_scale
        self.data_augmentation_package = data_augmentation_package

        self.generate_hms_here = generate_hms_here
        self.sample_patches = sample_patches
        self.sample_patch_size = sample_patch_size
        self.sample_patch_bias = sample_patch_bias

        #if we're sampling patches w/o aug we still need to center crop so change the aug strategy
        if self.sample_patches and self.data_augmentation_strategy == None:
            self.data_augmentation_package = "imgaug"
            self.data_augmentation_strategy = "CenterCropOnly"
            print("Only Augmentation is CenterCropOnly after patch sample for %s split." % split)


        if self.data_augmentation_strategy == None:
            print("No data Augmentation for %s split." % split)
        else:
            #Get data augmentor for the correct package
            self.aug_package_loader = get_aug_package_loader(self.data_augmentation_package)
            #Get specific data augmentation strategy
            self.transform = self.aug_package_loader(self.data_augmentation_strategy, input_size)
            print("Using data augmentation package %s and strategy %s." % (data_augmentation_package, self.data_augmentation_strategy) )


        self.heatmaps_to_tensor = transforms.Compose([
                HeatmapsToTensor()
            ])
        

        self.root_path = Path(root_path)
        self.annotation_path = Path(annotation_path)
        self.image_modality = image_modality
        self.landmarks = landmarks
        self.split = split
        self.sigmas= sigmas

        self.cv=cv
        self.cache_data = cache_data
        self.debug = debug
        
        self.input_size = input_size

        #If we are sampling patches we want to load in the full resolution image.
        if self.sample_patches:
            self.load_im_size = original_image_size
            self.downscale_factor = 1
        else:
            self.load_im_size = input_size
            self.downscale_factor = [original_image_size[0]/input_size[0], original_image_size[1]/input_size[1]]


        self.original_image_size = original_image_size
        self.num_res_supervisions = num_res_supervisions


        #Lists to save the image paths (or images if caching), target coordinates (scaled to input size), and full resolution coords.
        self.images = []
        self.target_coordinates = []
        self.full_res_coordinates = [] #full_res will be same as target if input and original image same size
        self.image_paths = []
        self.uids = []



        if cv >= 0:
            label_std = os.path.join('fold' + str(self.cv) +'.json' )
            print("Loading %s data for CV %s " % (self.split, os.path.join(self.annotation_path, label_std)))
            datalist = load_aspire_datalist(os.path.join(self.annotation_path, label_std), data_list_key=self.split, base_dir=self.root_path)
        
        else:
            raise NotImplementedError("Only cross validation is currently implemented. Put all training data as fold1 and use cv=1 instead.")

        # based on first image extenstion, get the load function.
        self.datatype_load = get_datatype_load(datalist[0]["image"]) 
        if self.cache_data:
            #If cached, no load function needed.
            self.load_function = lambda img: img 

            for idx, data in enumerate(datalist):

                interested_landmarks = (np.array(data["coordinates"])[self.landmarks,:2] / self.downscale_factor)
                expanded_image = np.expand_dims(normalize_cmr(self.datatype_load(data["image"]).resize( self.load_im_size)), axis=0)

               

                self.images.append(expanded_image)
                self.target_coordinates.append(interested_landmarks)
                self.full_res_coordinates.append(np.array(data["coordinates"])[self.landmarks,:2] )
                self.image_paths.append(data["image"])
                self.uids.append(data["id"])

            print("Cached all %s data in memory. Length of %s " % (self.split, len(self.images)))
        else:
            self.load_function = lambda pth_: np.expand_dims(normalize_cmr(self.datatype_load(pth_).resize(self.load_im_size)), axis=0)

            for idx, data in enumerate(datalist):

                interested_landmarks = (np.array(data["coordinates"])[self.landmarks,:2] / self.downscale_factor)
                self.images.append(data["image"]) #just appends the path, the load_function will load it later.
                self.target_coordinates.append(interested_landmarks)
                self.full_res_coordinates.append(np.array(data["coordinates"])[self.landmarks,:2] )
                self.image_paths.append(data["image"])
                self.uids.append(data["id"])



            print("Not caching %s image data in memory, will load on the fly. Length of %s " % (self.split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, index):
        """ Implement method to get a data sample. Also give an option to debug here.
            


        Args:
            index (_type_): _description_

        Returns:
            It must return a dictionary with the keys: 
            sample = {
            "image": torch.from_numpy(image) - torch tensor of the image 
            "label": heatmaps,  - if self.generate_hms_here bool -> (torch tensor of heatmaps); else -> [] 
            "target_coords": list of target coords of for landmarks, same scale as input to network , 
            "full_res_coords": original list of coordinates, same scale as original image (may be same as target_coords)
            "image_path": im_path: string to orignal image path 
            "uid":this_uid : string of sample's unique id
             }
        """
        
    
        hm_sigmas = self.sigmas
        image = self.load_function(self.images[index])
        coords = self.target_coordinates[index]
        full_res_coods = self.full_res_coordinates[index]
        im_path = self.image_paths[index]
        run_time_debug= False
        this_uid = self.uids[index]
        # print("load image time", time()- soo)

        so = time()

        # if self.sample_input_patch()

        #Do data augmentation
        if self.data_augmentation_strategy != None:


            untransformed_im = image
            untransformed_coords = coords[:,:2]
            if self.sample_patches:
                untransformed_im, untransformed_coords, landmarks_in_indicator = self.sample_patch(untransformed_im, untransformed_coords)
          

            kps = KeypointsOnImage([Keypoint(x=coo[0], y=coo[1]) for coo in untransformed_coords], shape=untransformed_im[0].shape )

            transformed_sample = self.transform(image=untransformed_im[0], keypoints=kps) #list where [0] is image and [1] are coords.
            input_image = normalize_cmr(transformed_sample[0], to_tensor=True)
            input_coords = np.array([[coo.x_int, coo.y_int] for coo in transformed_sample[1]])
            
            #Recalculate indicators incase transform pushed out/in coords.
            landmarks_in_indicator = [1 if ((0 <= xy[0] <= self.input_size[0] ) and (0 <= xy[1] <= self.input_size[1] )) else 0 for xy in input_coords  ]
            
        else:
            # if self.sample_patches:
            #     raise NotImplementedError("need to implement sampling patches without data augmentation.")
            untransformed_im = image
            untransformed_coords = coords
            input_coords = coords
            input_image = torch.from_numpy(image)

            if self.sample_patches:
                untransformed_im, untransformed_coords, landmarks_in_indicator = self.sample_patch(untransformed_im, untransformed_coords)
                input_coords = np.array(untransformed_coords)
                input_image = normalize_cmr(untransformed_im, to_tensor=True)
        
        if self.generate_hms_here:
            
            label = self.LabelGenerator.generate_labels(input_coords, landmarks_in_indicator,  self.input_size, hm_sigmas,  self.num_res_supervisions, self.hm_lambda_scale)
            # heatmaps = self.heatmaps_to_tensor(generate_heatmaps(trans_kps, self.input_size, hm_sigmas,  self.num_res_supervisions, self.hm_lambda_scale))  
        else:
            label = []

        sample = {"image":input_image , "label":label, "target_coords": input_coords, 
            "full_res_coords": full_res_coods, "image_path": im_path, "uid":this_uid  }

        #If coordinates are cutoff by augmentation throw a run time error. 
        if len(np.array(input_coords)) <len(coords):
            print("some coords have been cut off! You need to change the data augmentation, it's too strong.")
            run_time_debug = True

        

        if (self.debug or run_time_debug):
            self.LabelGenerator.debug_sample(sample,untransformed_im, untransformed_coords)
          
    
        return sample

    def generate_labels(self, landmarks, sigmas):
        """ Generate heatmap labels using same method as in _get_item__ 

        Args:
            landmarks (_type_): _description_
            sigmas (_type_): _description_

        Returns:

        """


        return self.LabelGenerator.generate_labels(landmarks, self.input_size, sigmas,  self.num_res_supervisions, self.hm_lambda_scale)
        

    def sample_patch(self, image, landmarks):
        
        z_rand = np.random.uniform(0, 1)
        landmarks_in_indicator = []

        if z_rand >= (1-self.sample_patch_bias):         
            while 1 not in landmarks_in_indicator:
                landmarks_in_indicator = []

                y_rand = np.random.randint(0, self.original_image_size[1]-self.sample_patch_size[1])
                x_rand = np.random.randint(0, self.original_image_size[0] -self.sample_patch_size[0])

                for lm in landmarks:
                    landmark_in = 0
                    # if y_rand <= lm[1]-4 and y_rand+self.sample_patch_size[1]  <= lm[1]+4 :
                    #     if x_rand <= lm[0]-4 and  lm[0]-4 <= x_rand + self.sample_patch_size[0]:

                    if y_rand  <= lm[1]<= y_rand+self.sample_patch_size[1]:
                        if x_rand <= lm[0] <= x_rand +self.sample_patch_size[0]:
                            landmark_in = 1
                    
                    landmarks_in_indicator.append(landmark_in)

        else:
            y_rand = np.random.randint(0, self.original_image_size[1]-self.sample_patch_size[1])
            x_rand = np.random.randint(0, self.original_image_size[0] -self.sample_patch_size[0])

            for lm in landmarks:
                landmark_in = 0
                if y_rand <= lm[1]<= y_rand+self.sample_patch_size[1]:
                    if x_rand <= lm[0] <= x_rand +self.sample_patch_size[0]:
                        landmark_in = 1
                landmarks_in_indicator.append(landmark_in)


        # print("y and x rands@ ", y_rand, x_rand)
        normalized_landmarks = [[lm[0]-x_rand, lm[1]-y_rand] for lm in landmarks]

        #Adding some padding to allow data augmentations and then center crop the desired size afterwards.
        safe_crop_y = [max(0,y_rand-16), min(self.original_image_size[1], y_rand+self.sample_patch_size[1]+16)]
        safe_crop_x = [max(0,x_rand-16), min(self.original_image_size[0], x_rand+self.sample_patch_size[0]+16)]

        # print("safe crops: ", safe_crop_y, safe_crop_x)
        #Pytorch has loaded the images y-x axis way round
        cropped_image_padded = image[:, safe_crop_y[0]:safe_crop_y[1], safe_crop_x[0]:safe_crop_x[1]]

        return cropped_image_padded, normalized_landmarks, landmarks_in_indicator