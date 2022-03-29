import torch
from torchvision import transforms
import numpy as np
import nibabel as nib

from torch.utils import data
from pathlib import Path
from typing import List
import os

from transformations import normalize_cmr, ToTensor, NormalizeZScore, HeatmapsToTensor
from visualisation import visualize_image_target, visualize_image_trans_target, visualize_image_trans_coords
from load_data import load_aspire_datalist
from PIL import Image
from generate_labels import generate_heatmaps, get_downsampled_heatmaps

# import albumentations as A
# import albumentations.augmentations.functional as F
# from albumentations.pytorch import ToTensorV2

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

class ASPIRELandmarks(data.Dataset):
    """
    A custom dataset for loading and processing the ASPIRE Cardiac landmark dataset [1]

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
        hm_lambda_scale: float,
        annotation_path: str,
        image_modality: str= "CMRI",
        split: str ="training",
        root_path: str = "./data",
        sigma: float = 3.0,
        cv: int = -1,
        cache_data: bool = False,
        normalize: bool = True,
        debug: bool = False,
        input_size=  [512,512],
        original_image_size = [512,512],
        num_res_supervisions: int = 5,
        data_augmentation: str = None,
        ):
        
 
        super(ASPIRELandmarks, self).__init__()

        self.data_augmentation = data_augmentation
        self.hm_lambda_scale = hm_lambda_scale

        # self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.heatmaps_to_tensor = transforms.Compose([
                HeatmapsToTensor()
            ])
        if self.data_augmentation == None:
            self.transform = transforms.Compose([
                # ToTensor()
            ])
            

        # elif self.data_augmentation == "ALL":
        #     self.transform = iaa.Sequential([
        #             iaa.Sometimes(
        #                 0.75,
        #                 iaa.Affine(
        #                     rotate=(-45,45),
        #                     scale=(0.8, 1.2),
        #                 ),
        #                 iaa.Affine(translate_percent={"x": (-0.-7, 0.07), "y": (-0.07, 0.07)})

        #             ),
        #             iaa.flip.Flipud(p=0.5),
        #             iaa.Sometimes(
        #                 0.5,
        #                 iaa.ElasticTransformation(alpha=(0, 100.0), sigma=(5,10))
        #             ),

        #             iaa.Sometimes(
        #                 0.5,
        #                 iaa.OneOf([
        #                     iaa.GaussianBlur((0, 0.2)),
        #                     iaa.AverageBlur(k=(3, 5)),
        #                     iaa.MedianBlur(k=(3, 5)),
        #                     iaa.AveragePooling(2)
        #                 ])
        #             ),

        #              iaa.Sometimes(
        #                 0.75,
        #                 iaa.SomeOf(
        #                     (0,3), 
        #                     [
        #                         iaa.Sharpen(alpha=(0, 0.75), lightness=(0, 0.5)),
        #                         iaa.Emboss(alpha=(0, 0.5), strength=(0, 1)),
        #                         iaa.LinearContrast((0.4, 1.6))                            
        #                     ]
        #                 )
        #             )
        #         ])
        elif self.data_augmentation =="AffineSimple":
 
            self.transform = iaa.Sequential([
                iaa.Sometimes(
                    0.75,
                    iaa.Affine(
                        rotate=(-45,45),
                        scale=(0.8, 1.2),
                    ),
                ),
                iaa.flip.Flipud(p=0.5),
            ])

        elif self.data_augmentation =="AffineComplex":
            self.transform = iaa.Sequential([
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    )
                ),
                iaa.flip.Flipud(p=0.5),

            ])
        elif self.data_augmentation == "AffineComplexElastic":
            self.transform = iaa.Sequential([
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    )
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(
                    0.5,
                    iaa.ElasticTransformation(alpha=(0,200), sigma=(9,13))
                ),

            ])
        elif self.data_augmentation == "AffineComplexElasticLight":
            self.transform = iaa.Sequential([
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    )
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(
                    0.5,
                    iaa.ElasticTransformation(alpha=(0,50), sigma=(5,10))
                ),

            ])
        elif self.data_augmentation == "AffineComplexElasticBlur":
            self.transform = iaa.Sequential([
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    )
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(
                    0.5,
                    iaa.ElasticTransformation(alpha=(0, 50), sigma=(2,5))
                ),

                iaa.Sometimes(
                    0.5,
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 0.2)),
                        iaa.AverageBlur(k=(3, 5)),
                        iaa.MedianBlur(k=(3, 5)),
                        iaa.AveragePooling(2)
                    ])
                ),

            ])
        elif self.data_augmentation == "AffineComplexElasticBlurSharp":
            self.transform = iaa.Sequential([
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    )
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(
                    0.5,
                    iaa.ElasticTransformation(alpha=(0, 50), sigma=(2,5))
                ),

                iaa.Sometimes(
                    0.5,
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 0.2)),
                        iaa.AverageBlur(k=(3, 5)),
                        iaa.MedianBlur(k=(3, 5)),
                        iaa.AveragePooling(2)
                    ])
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.SomeOf(
                        (0,3), 
                        [
                            iaa.Sharpen(alpha=(0, 0.75), lightness=(0, 0.5)),
                            iaa.Emboss(alpha=(0, 0.5), strength=(0, 1)),
                            iaa.LinearContrast((0.4, 1.6))                            
                        ]
                    )
                )
            ])
        else:
            raise ValueError("transformations mode for dataaugmentation not recognised, try None, V1, V2, V3 or V4")

        self.root_path = Path(root_path)
        self.annotation_path = Path(annotation_path)
        self.image_modality = image_modality
        self.landmarks = landmarks
        self.split = split
        self.sigma=sigma
        self.cv=cv
        self.cache_data = cache_data
        self.normalize = normalize
        self.debug = debug
        self.input_size = input_size

        self.original_image_size = original_image_size
        self.num_res_supervisions = num_res_supervisions


        if self.cache_data:
            self.images = []
            self.labels = []
            self.target_coordinates = []
            self.full_res_coordinates = []




        if cv > 0:
            # label_std = os.path.join('fold' + str(self.cv) + '_' + str(self.sigma) + 'std.json' )
            label_std = os.path.join('fold' + str(self.cv) +'.json' )

            datalist = load_aspire_datalist(os.path.join(self.annotation_path, label_std), data_list_key=self.split, base_dir=self.root_path)
        
        else:
            raise NotImplementedError("Only cross validation is currently implemented")
        #   /mnt/tale_all/data/CMRI/ASPIRE/cardiac4ch_labels_VPnC_CV
        if self.cache_data:
            for idx, data in enumerate(datalist):
                # print("data@ ", data)
                # print("root dir, im dir and join dir", self.root_path, os.path.join(self.root_path, data['image']) )

                #/mnt/bess/shared/tale2/Shared/schobs/data/ISBI2015_landmarks/images
                downscale_factor = [original_image_size[0]/input_size[0], original_image_size[1]/input_size[1]]
                interested_landmarks = np.rint(np.array(data["coordinates"])[self.landmarks,:2] / downscale_factor)
                # generated_heatmaps = generate_heatmaps(interested_landmarks, self.input_size, self.sigma,  self.num_res_supervisions, self.hm_lambda_scale)
                # multi_level_heatmaps = get_downsampled_heatmaps(generated_heatmap, self.num_res_supervisions)
                
                if "nii.gz" in data["image"]:
                    expanded_image = np.expand_dims(normalize_cmr(Image.fromarray(nib.load(data["image"]).get_fdata()).resize( self.input_size)), axis=0)
                else:
                    expanded_image = np.expand_dims(normalize_cmr(Image.open(data["image"]).resize( self.input_size)), axis=0)

                # print("image shape and gen heatmap shape: ", expanded_image.shape, generated_heatmaps.shape)
                self.images.append(expanded_image)
                # self.labels.append(generated_heatmaps)
                self.target_coordinates.append(interested_landmarks)
                self.full_res_coordinates.append(np.array(data["coordinates"])[self.landmarks,:2] )

                # if self.debug and self.data_augmentation==None:
                #     from utils.heatmap_manipulation import get_coords
                #     print("coordinates: ", interested_landmarks)
                #     landmarks_from_label = get_coords(torch.from_numpy(np.expand_dims(generated_heatmaps[-1], axis=0)))
                #     print("landmarks from heatmap label: ", landmarks_from_label)
                #     # print("heatmap: ", generated_heatmaps.shape)
                #     visualize_image_target(np.squeeze(expanded_image), generated_heatmaps, interested_landmarks)
                #     # visualize_image_target(self.images[-1], self.labels[-1])
            print("Cached all %s data in memory. Length of %s " % (self.split, len(self.images)))
        else:

            # for idx, data in enumerate(datalist):
            #     downscale_factor = [original_image_size[0]/input_size[0], original_image_size[1]/input_size[1]]

            #     interested_landmarks = np.rint(np.array(data["coordinates"])[self.landmarks,:2] / downscale_factor)


            raise NotImplementedError("Not caching dataset not implemented yet. set cache_data=true for now.")

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        image = self.images[index]
        # label = self.labels[index]
        coords = self.target_coordinates[index]
        full_res_coods = self.full_res_coordinates[index]

        # print("iamge type %s min max %s %s and mean and std: %s %s " % ((image).dtype, image.min(), image.max(), np.mean(image), np.std(image)))
        # print("OG Coords", coords[:,:2])
        # if self.transform_complex:
        #     # coords_to_transform = [(x[0], x[1]) for x in coords[:,:2] ]
        #     # print("tuple coords", coords_to_transform)

        #     transformed_sample = self.transform(image=image[0], keypoints=coords[:,:2])
        #     heatmaps = generate_heatmaps(np.array(transformed_sample["keypoints"]), self.image_size, self.sigma,  self.num_res_supervisions)  
        #     sample = {"image":transformed_sample['image'], "label":heatmaps, "target_coords": transformed_sample['keypoints']  }

        #     print("trans image shape", sample["image"].shape, "trans targ coords: ", sample["target_coords"])

        #     visualize_image_trans_target(np.squeeze(image), sample["image"][0], heatmaps[-1])

        if self.data_augmentation != None:

            kps = KeypointsOnImage([Keypoint(x=coo[0], y=coo[1]) for coo in coords[:,:2]], shape=image[0].shape )
            transformed_sample = self.transform(image=image[0], keypoints=kps)
            # print("seps \n", transformed_sample[0], " \n ", transformed_sample[1])

            trans_kps = np.array([[coo.x_int, coo.y_int] for coo in transformed_sample[1]])
            # print("transformed sample: ", transformed_sample)
            # print("seps \n", transformed_sample[0], " \n ", transformed_sample[1])
            # print("trans kps: ", trans_kps)
            heatmaps = self.heatmaps_to_tensor(generate_heatmaps(trans_kps, self.input_size, self.sigma,  self.num_res_supervisions, self.hm_lambda_scale))  
            sample = {"image":normalize_cmr(transformed_sample[0], to_tensor=True) , "label":heatmaps, "target_coords": trans_kps, "full_res_coords": full_res_coods  }

            # print("genned heatmaps")
            # sample["image"] = normalize_cmr(sample["image"])

            run_time_debug= False
            if len(np.array(transformed_sample[1])) <len(coords):
                print("some coords have been cut off!")
                # print("transformation: ", transformed_sample['replay'])
                run_time_debug = True


            if self.debug or run_time_debug:
                from utils.heatmap_manipulation import get_coords

                print("og image sahpe: ", image.shape, "trans image shape", sample["image"].shape, "trans targ coords: ", sample["target_coords"])
                print("len of hetamps ", len(heatmaps), " and shape: ", heatmaps[-1].shape, " and hm exp shape ", np.expand_dims(heatmaps[-1], axis=0).shape)
                landmarks_from_label = get_coords(torch.from_numpy(np.expand_dims(heatmaps[-1], axis=0)))
                print("landmar!=s from heatmap label: ", landmarks_from_label)
                # visualize_image_target(sample["image"][0], heatmaps[-1], np.rint(transformed_sample['keypoints']))

                visualize_image_trans_coords(image[0], sample["image"][0], sample['target_coords'])

        else:
            label = self.heatmaps_to_tensor(generate_heatmaps(coords, self.input_size, self.sigma,  self.num_res_supervisions, self.hm_lambda_scale))
            sample = {"image": normalize_cmr(image, to_tensor=True), "label": label,  "target_coords": coords, "full_res_coords": full_res_coods}

            sample = self.transform(sample)
            # sample["image"] = normalize_cmr(sample["image"])

            
        # else:
        #     sample = {"image": image, "label": label,  "target_coords": coords}
        return sample


    
