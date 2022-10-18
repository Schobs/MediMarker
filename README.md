# LaNNU-Net

LannU-Net is an out-of-the-box automated pipeline for landmark localization. It is inspired by [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), but tailored for landmark localization rather than biomedical image segmentation. We also support uncertainty estimation for model predictions.

As a user, all you need to do is provide your data in the correct format following simple instructions and the pipeline will take care of the rest.

As a researcher/developer, you can extend this framework to add your own models, loss functions, training schemes etc. by extending a few classes. I provide easy instructions with examples on exactly what you need to do. You can use this framework to evaluate your own models over many datasets in a controlled environment.

The pipeline is simple, and based on the [U-Net architecture](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf). It automatically configures the size of the architecture based on the size of your images. By default, we use heatmap regression for landmark localization. We support ensemble learning, and uncertainty estimation.



For advanced users, we provide the following features:
- Support for new arhcitectures by extending a few classes.
- Data augmentation strength.
- Regress Sigma for Gaussian loss (Beta).
- Patch-based sampling (Beta).


#Table of Contents
- [LaNNU-Net](#lannu-net)
- [Installation](#installation)
- [Using Your Own Dataset](#using-your-own-dataset)
  - [1) Expected Directory Format](#1-expected-directory-format)
  - [2) Create a JSON file](#2-create-a-json-file)
  - [3) Create a .yaml config file](#3-create-a-yaml-config-file)
- [Train your model!](#train-your-model)
  - [Training](#training)
  - [Inference (Testing)](#inference-testing)
- [Advanced Configuration](#advanced-configuration)
  - [Advanced JSON File](#advanced-json-file)
  - [Advanced Dataset Configuration](#advanced-dataset-configuration)
  - [Advanced Config File](#advanced-config-file)
- [Advanced Features](#advanced-features)
  - [Ensembling](#ensembling)
  - [Uncertainty Estimation](#uncertainty-estimation)
  - [Changing the Model Architecture](#changing-the-model-architecture)
  - [Changing the Loss Function](#changing-the-loss-function)
  - [Changing Training Schedule](#changing-training-schedule)
  - [Data Augmentation](#data-augmentation)
  - [Patch-based Training](#patch-based-training)
  - [Sigma Regression](#sigma-regression)
  - [Logging](#logging)
  - [Inference: Fitting a Gaussian to the Predicted Heatmap](#inference-fitting-a-gaussian-to-the-predicted-heatmap)
  - [Debugging](#debugging)
- [For Developers: LannU-Net Structure](#for-developers-lannu-net-structure)

# Installation


# Using Your Own Dataset
As a user, all you need to do is provide your data in the correct format, fill in a config file, and LannU-Net will take care of the rest. 


## 1) Expected Directory Format
Currently, we support dicom, npz and png image formats. However, all images in one dataset should have the same format. The expected directory format is as follows:

    root_image_folder
        ↳ aribitary_subroute_1
            ↳ image_1.png
            ↳ image_2.png
            ↳ ...
        ↳ aribitary_subroute_21
            ↳ aribitary_subroute_22
                ↳ image_3.png
            ↳ aribitary_subroute_23
                ↳ image_4.png
        ↳ image_5.png
        ↳ ...


Essentially, the exact directory format does not matter, as long as all images have a common root folder (root_image_folder). However, for simplicity I recommend keeping the directories conisistent since it will make the JSON generation easier. In the JSON file, you need to specify the route from the root_image_folder to each sample's image.



## 2) Create a JSON file
The JSON file contains all the dataset information LannU-Net needs for the training, validation and (optionally) testing of the dataset.

The *minimum* JSON file should include the name of the dataset, a short description, a fold number, and a training, validation & (optionally) testing lists. The 3 lists include the individual samples of the dataset, where each sample is a dictionary. Samples in the *training* list will be used for training, *validation* for validation and *testing* for testing.

    {
        "name": "EXAMPLE DATASET NAME",
        "description": "Cardiac MRI 4ch View Images.",
        "fold": 0,
        "training": [],
        "validation": [],
        "testing": []
    }



Here is an example of a dataset JSON file. The *training* list shows the minumum information needed per sample: a unique id, a 2D list of coordinates (landmarks), and the path to the image. The *validation* and *testing* lists should follow the same format. The image path ("image") should connect the root_image_folder from above in subsection *1) Expected Directory Format* to the sample's image. If you want to add more attributes to each sample see [[Advanced JSON format]].


    
    {
        "name": "ASPIRE BASELINE + FOLLOWUP",
        "description": "Cardiac MRI 4ch View Images.",
        "fold": 0,
        "training": 
            [
                {
                    "id": "PHD_430",
                    "coordinates": 
                        [
                            [320.0, 316.0], 
                            [228.0, 339.0],
                            [259.0, 220.0], 
                            [340.0, 220.0],    
                        ],
                    "image": "PAH-Baseline/Proc/4ch/PHD_430/phase_1.npz",             
                },
                {
                    "id": "PHD_431",
                    "coordinates": 
                        [
                            [320.0, 316.0], 
                            [227.0, 329.0],
                            [239.0, 220.0], 
                            [360.0, 210.0],    
                        ],
                    "image": "PAH-Baseline/Proc/4ch/PHD_431/phase_1.npz",               
                }
            ],
        "validation": 
            [
                . . .
            ],
        "testing": 
            [
                . . .
            ]
    }

If your testing set has no annotations, set the coordinates of each sample to null and add an has_annotation bool and set it to false:

    "testing": 
        [
            {
                "id": "1.2.840.113619.2.312.2807.4259692.12440.1659507727.690",
                "image": "PAH-Large/4ch 3/155276/7/DICOM/MR000000.dcm",
                "has_annotation": false,
                "coordinates": null,
            },
            {
                "id": "1.3.1.45.43.34621389534287095312.23",
                "image": "PAH-Large/4ch 3/23512/7/DICOM/14.dcm",
                "has_annotation": false,
                "coordinates": null,
            }
        ]   

<!-- &#8627; -->

If you are performing cross-validation (CV), make a JSON file for each fold and **you MUST set the name of each file to fold0.json, fold1.json, etc.** These files should be in the same root folder. If you are not using CV, it does not matter what the name is. Folder Structure example:

    root_annotation_folder
        ↳ fold0.json
        ↳ fold1.json
        ↳ fold2.json
        ↳ fold3.json
        ↳ fold4.json

## 3) Create a .yaml config file
Now your dataset is in the correct directory structure and you have the JSON annotation files, you just need to edit it a yaml based config file and you're ready to go! We will override the default config parameters in the config.py file. Under the /configs/ directory, create a yaml file e.g. my_dataset_config.yaml and fill in the following options:


    OUTPUT_DIR: "/path_to_output_folder/results"

    SOLVER:
        DATA_LOADER_BATCH_SIZE: 32

    DATASET:
        ROOT: '/path_to_image_root/root_image_folder'
        SRC_TARGETS: '/path_to_annotation_root/root_annotation_folder/no_cv_annotations.json'
        LANDMARKS : [0,1,2,3,]
    
    TRAINER:
        INFERENCE_ONLY: False
        CACHE_DATA: False
        FOLD: -1 #no cv

    SAMPLER:
        INPUT_SIZE : [512,512]


- OUTPUT_DIR: The path to the folder where the model checkpoints and results will be saved. 
- DATASET_ROOT.ROOT: The path to the root_image_folder from above in subsection *1) Expected Directory Format*.
- DATASET.SRC_TARGETS: The path to the root_annotation_folder from above in subsection *2) Create a JSON file*. If you are not using CV, specify the exact JSON file like in the example above. If you are using CV, just specify the root_annotation_folder and the TRAINER.FOLD information will be used to select the correct JSON file.
- DATASET.LANDMARKS: The indicies of the landmarks in the JSON file. For example, if the JSON file has 4 landmarks, and you want to use the first 3, then the list should be [0,1,2].
- TRAINER.FOLD: The fold number to use for training. This number will match the number in fold0.json, fold1.json etc. **If you are using cross-validation, set this to -1**.
- TRAINER.CACHE_DATA: If you are using a small dataset, it is recommended to set this to True. This will cache the dataset in memory, which will speed up training. However, if you are using a large dataset, you may run out of memory. *Try as True first.*
- TRAINER.INFERENCE_ONLY: If you want to train first, set this to False. If you have already trained a model and want to run inference on a new dataset, set this to True.
- SOLVER.DATA_LOADER_BATCH_SIZE: The batch size to use for training. This should be set to the largest batch size your GPU can handle. First, try high and go smaller if you run out of memory. This will be automated in future.
- SAMPLER.INPUT_SIZE: The size you want to resize the images to for training. Regardless of resolution uniformity in the dataset, you need to resize them to a common size. If possible, use the median size of the dataset. If the median resolution is too big to fit into memory, [512, 512] is a good bet. Alternatively you can use full-resolution images via patch-based training, explained in [[Patch-based Training]]. This will be automated in future.

There are many other options you can set in the config file such as data augmentation scheme, learning schedule, architecture choice & parameters, loss function, sigma regression etc. See [[Config File]] for more information.


# Train your model!

## Training 
You are now ready to train your landmark localization model! From /LannU-Net/ run:
    
    python main.py --cfg /configs/my_dataset_config.yaml


## Inference (Testing) 
If you included a *testing* list in your JSON, inference will be completed after training and the results will be saved in the OUTPUT_DIR.

 If you did not include a *testing* list, you can run inference on a separate json file by:
1) Setting TRAINER.INFERENCE_ONLY= True. 
2) Setting DATASET.SRC_TARGETS to the path of the new JSON file with the *testing* list.
3) Setting TRAINER.FOLD to -1 (remember, if it is not -1 it will automatically try to find the fold0.json, fold1.json etc. files from the SRC_TARGETS folder). 

For example:


    TRAINER:
        INFERENCE_ONLY: True
        FOLD: -1 #no cv

    DATASET:
        SRC_TARGETS: '/path_to_annotation_root/root_annotation_folder/testing_only.json

        


If you want to perform inference on a specific model checkpoint, also add the following to the yaml file:


    MODEL:
        CHECKPOINT: '/path_to_checkpoint/your_model_checkpoint.model'

*If you omit MODEL.CHECKPOINT or set MODEL.CHECKPOINT=None, LannU-Net will perform inference over all model checkpoints in the OUTPUT_DIR directory.*

If you did include a *testing* list in your JSON, and simply want to re-run the inference, you can soley perform inference by leaving the yaml file the same and just changing TRAINER.INFERENCE_ONLY = True. You can also pick and choose which model checkpoint to perform inference on too. 

Note, here you are *editing* the yaml config file from training, not creating one with only these fields. Alternatively, you can copy the training yaml and have two seperate yaml files: one for training, one for testing. Just remember to add the correct one to the command line as the cfg parameter when running the programme.

Now, Run the same command as above.

    python main.py --cfg /configs/my_dataset_config.yaml


# Advanced Configuration
## Advanced JSON File
## Advanced Dataset Configuration
## Advanced Config File

# Advanced Features
## Ensembling
## Uncertainty Estimation
## Changing the Model Architecture
## Changing the Loss Function
## Changing Training Schedule
## Data Augmentation
## Patch-based Training
## Sigma Regression
## Logging
## Inference: Fitting a Gaussian to the Predicted Heatmap
## Debugging





# For Developers: LannU-Net Structure
LannU-net is modular and can be easily extended. Here I will explain the control flow of the framework and how you can extend it.

The main components are:

