# MediMarker: AI Powered Anatomical Landmark Localisation

MediMarker is an out-of-the-box automated pipeline for landmark localization. We also support uncertainty estimation for model predictions.

As a user, all you need to do is provide your data in the correct format and the pipeline will take care of the rest. You can also use our pre-trained models for inference.

The pipeline is simple, and the default model is based on the [U-Net architecture](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf). It automatically configures the size of the architecture based on the size of your images. By default, we use heatmap regression for landmark localization. We support ensemble learning, and uncertainty estimation.

As a researcher/developer, you can extend this framework to add your own models, loss functions, training schemes etc. by extending a few classes. The advantage of this is that there is a *lot* of code you don't have to write that is specific to landmark localization and you can concentrate on implementing the important stuff. 

I provide easy instructions with examples on exactly what you need to do. You can use this framework to evaluate your own models over many datasets in a controlled environment.  So far, beyond U-Net we have also added PHD-Net, which is a completely different paradigm of landmark localization, but can be integrated seamlessly with this framework. In our gaussian_process branch, we have also added a Convolutional Gaussian Process model for landmark localization.




For advanced users, we provide the following features:
- Pretrained models.
- Support for new architectures by extending a few classes.
- Convolutional Gaussian Proccesses for landmark localization.
- Extensive, easy configuration using .yaml files.
- Data augmentation strength.
- Regressing Sigma for Gaussian Heatmap (Beta).
- Fit Gaussian Covariance Matrix at inference.
- Patch-based sampling (Beta).
- Ensembling.
- Uncertainty Estimation: Maximum-Heatmap Activation, Ensembling, Test Time Augmentation, MC-Dropout, and more.
- Evaluation Metrics (Point Error, Success Detection Rate, Negative Loss Preditive Density, and more.)
- Comet.ML logging


For Gaussian Processes check the gaussian_process branch. For transformers and resnet check the tom branch. 
# Table of Contents
- [MediMarker](#MediMarker)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Running an Example](#running-an-example)
  - [Inference (Testing)](#inference-testing)
- [Using Your Own Dataset](documentation/readme/using_own_dataset.md)
  - [Expected Directory Format](documentation/readme/using_own_dataset.md#1-expected-directory-format)
  - [Create a JSON file](documentation/readme/using_own_dataset.md#2-create-a-json-file)
  - [Create a .yaml config file](documentation/readme/using_own_dataset.md#3-create-a-yaml-config-file)
- [Train your model!](documentation/readme/train_own_model.md)
  - [Training](documentation/readme/train_own_model.md#training)
  - [Inference (Testing)](documentation/readme/inference.md)
  - [Evaluation](documentation/readme/evaluation.md)

- [Implemented Models](documentation/readme/implemented_models.md)
  - [U-Net](documentation/readme/implemented_models.md#u-net)
  - [PHD-Net](documentation/readme/implemented_models.md#phd-net)
- [Advanced Configuration](documentation/readme/advanced_yaml_config.md)
  - [Advanced Dataset Configuration](documentation/readme/advanced_yaml_config.md#advanced-dataset-configuration)
  - [Changing the .yaml Config File](documentation/readme/advanced_yaml_config.md#changing-the-yaml-config-file)
    - [DATASET](documentation/readme/advanced_yaml_config.md#dataset)
    - [SAMPLER](documentation/readme/advanced_yaml_config.md#sampler)
    - [SOLVER](documentation/readme/advanced_yaml_config.md#solver)
    - [TRAINER](documentation/readme/advanced_yaml_config.md#trainer)
    - [MODEL](documentation/readme/advanced_yaml_config.md#model)
      - [MODEL.UNET](documentation/readme/advanced_yaml_config.md#modelunet)
      - [MODEL.PHD-NET](documentation/readme/advanced_yaml_config.md#modelphd-net)
    - [INFERENCE](documentation/readme/advanced_yaml_config.md#inference)
    - [OUTPUT](documentation/readme/advanced_yaml_config.md#output)
    - [SSH](documentation/readme/advanced_yaml_config.md#ssh)

- [Ensembling and Uncertainty](documentation/readme/ensembling_and_uncertainty.md#ensembling-and-uncertainty)
- [Adding Your Own Model Architecture and Task](documentation/readme/adding_new_models.md)
  - [The Model Class](documentation/readme/adding_new_models.md#the-model-class)
  - [The Model Trainer Class](documentation/readme/adding_new_models.md#the-model-trainer-class)
  - [The Label Generator Class](documentation/readme/adding_new_models.md#the-label-generator-class)
  - [Changing the Loss Function](documentation/readme/adding_new_models.md#changing-the-loss-function)
  - [Changing the Training Schedule](documentation/readme/adding_new_models.md#changing-the-training-schedule)
  - [Data Augmentation](documentation/readme/adding_new_models.md#data-augmentation)
  - [Full Image vs. Patch-based Training](documentation/readme/adding_new_models.md#full-image-vs-patch-based-training)
  -  [Sigma Regression](documentation/readme/adding_new_models.md#sigma-regression)
  - [Uploading your trained models](documentation/readme/adding_new_models.md#uploading-your-trained-model)

- [Logging](documentation/readme/logging.md)
- [Debugging](documentation/readme/debugging.md)
    


# Installation
1) Clone the repo
2) cd into the repo
3) Create and activate conda environment
```
conda create --name my_env
conda activate my_env
```
3) Install from the environments .yaml file
```
conda env update --name my_env --file requirements/environment.yml
```

That's it!

# Running an Example
If you have the [ISBI 2015 Cephalometric landmarking dataset](https://www.sciencedirect.com/science/article/pii/S1361841516000190) accessible, you can run to perform inference on a pretrained model with the default U-Net model, or easily train your own. In MediMarker, we use .yaml files to configure the pipeline. You can find the .yaml files in the configs/examples folder. 

To run inference, run:

    python main.py --cfg configs/examples/U-Net_Classic/Cephalometric/unet_cephalometric_fold0.yaml

To train a model on this dataset, you can run:

    python main.py --cfg configs/examples/U-Net_Classic/Cephalometric/unet_cephalometric_fold0_train.yaml


For students at the University of Sheffield using the Bessemer on the HPC, you have to load conda and CUDA. I have written a script to do so. Run the following:

```
    cd scripts/scripts_bess
    source run_train_config.sh --cfg configs/examples/U-Net_Classic/Cephalometric/unet_cephalometric_fold0.yaml
```

# Inference (Testing) 
If you included a *testing* list in your JSON (this is the case for the above example), inference will be completed after training and the results will be saved in the OUTPUT.OUTPUT_DIR (defined in the .yaml file). If you cancel training early or want to re-run inference, change your .yaml file as follows:

1) Set TRAINER.INFERENCE_ONLY=True. 
2) Either define the model checkpoint you want to use in MODEL.CHECKPOINT, or leave it as None and MediMarker will automatically find the latest checkpoints in the OUTPUT.OUTPUT_DIR.
3) Re-run the script.


 If you did not include a *testing* list (e.g. you want to use a pre-trained model on your own data), you can run inference on a separate json file if you change your .yaml file as follows:

1) Set TRAINER.INFERENCE_ONLY=True. 
2) Either define the model checkpoint you want to use in MODEL.CHECKPOINT, or leave it as None and MediMarker will automatically find the latest checkpoints in the OUTPUT.OUTPUT_DIR.
3) Setting DATASET.SRC_TARGETS to the path of the new JSON file with the *testing* list.
4) Setting TRAINER.FOLD to -1 (if it is not -1 it will automatically try to find the fold0.json, fold1.json etc. files from the SRC_TARGETS folder).

**Please see [Using Your Own Dataset](documentation/readme/using_own_dataset.md) for more details.**






