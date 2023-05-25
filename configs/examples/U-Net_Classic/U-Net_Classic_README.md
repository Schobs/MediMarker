# U-Net Classic Example

This is an example of how to use LannU-Net to train a U-Net model on the ISBI 2015 Challenge Cephalometric Landmark Detection dataset. 

Details on the model, as well as the results from these trained models can be found under [Implemented Models - U-Net](../../../documentation/readme/implemented_models.md#U-Net).

## Run Inference
To run inference for a model trained on folds 0,1 & 2 of the dataset and evaluated onf fold 3, run

    python main.py --cfg /configs/examples/U-Net_Classic/Cephalometric/unet_cephalometric_fold0.yaml

To run a different fold, just change the config link appropriately.

To get some visualizations of the inference, change INFERENCE.DEBUG=True. This will show you some matplotlib visualizations.


To run inference on your own dataset, simply change the DATASET.ROOT and DATASET.SRC (you do not need ground truth annotations). See [Using your own Dataset](../../../documentation/readme/using_own_dataset.md) for more details.


## Run Training

If you would like to try training yourself, you can run the following command:

    python main.py --cfg configs/examples/U-Net_Classic/Cephalometric/unet_cephalometric_fold0_train.yaml

If you would like to change the fold, change TRAINER.FOLD In the .yaml file.

If you would like to continue training from one of the pretrained models, add the config option for MODEL.CHECKPOINT in the .yaml file e.g.

MODEL:
  CHECKPOINT: "./LaNNU-Net/model_zoo/Cephalometric/U-Net/model_best_valid_coord_error_fold0.model"
  MODEL_GDRIVE_DL_PATH: "https://drive.google.com/file/d/1wrniAqjrNGhw_q5N2Z26xknLt5HiQFfz/view?usp=share_link"

Just make sure you're using the correct model for the fold you are on!

You can also change this example to train on your own dataset. Simply change the DATASET.ROOT and DATASET.SRC (you **do** need ground truth annotations for training). See [Using your own Dataset](../../../documentation/readme/using_own_dataset.md) for more details.


## Customization

If you would like to change general configuration options such as data augmentation, learning rate schedules etc, please see [Advanced Configuration](../../../documentation/readme/advanced_yaml_config.md).

If you would like to change model-specific configuration options such as the number of layers, number of filters etc, please see [MODEL.UNET Config Options](../../../documentation/readme/advanced_yaml_config.md#modelunet) for details.