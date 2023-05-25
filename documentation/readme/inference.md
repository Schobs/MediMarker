# Inference

If you included a *testing* list in your JSON file, inference will be completed after training and the results will be saved in the OUTPUT.OUTPUT_DIR (defined in the .yaml file). If you cancel training early or want to re-run inference, change your .yaml file as follows:

1) Set TRAINER.INFERENCE_ONLY=True in your_config.yaml. 
2) Either define the model checkpoint you want to use in MODEL.CHECKPOINT, or leave it as None and LannU-Net will automatically find the latest checkpoints in the OUTPUT.OUTPUT_DIR.
3) Re-run the main script e.g.

```
    python main.py --cfg configs/your_config.yaml
```

 If you did not include a *testing* list, or you want to use a pre-trained model on your own data, you can run inference on a separate json file if you change your .yaml file as follows:

1) Set TRAINER.INFERENCE_ONLY=True in your_config.yaml. 
2) Either define the model checkpoint you want to use in MODEL.CHECKPOINT, or leave it as None and LannU-Net will automatically find the latest checkpoints in the OUTPUT.OUTPUT_DIR.
3) Setting DATASET.SRC_TARGETS to the path of the new JSON file with the *testing* list. If the testing list has GT annotations, evaluation will be performed. If not, it will just make the predictions and uncertainty estimates. See [Using Your Own Dataset](using_own_dataset.md) for more information on the JSON file format.
4) Setting TRAINER.FOLD to -1 (if it is not -1 it will automatically try to find the fold0.json, fold1.json etc. files from the SRC_TARGETS folder).


## Ensembling and Uncertainty Estimation
You can also create  a deep ensemble by first training multiple networks in different output directories. LannU-Net can then run inference by automatically combining predictions using heatmap averaging and coordiante prediction averaging. It can then estimate uncertainty.
See [Ensembling and Uncertainty](ensembling_and_uncertainty.md#ensembling-and-uncertainty) for more information on how to use this feature.

## Inference: Fitting a Gaussian to the Predicted Heatmap

At inference we can fit a Gaussian using a robust least squares method onto the predicted heatmap. Then, the landmark is extracted. In the config change INFERENCE.FIT_GAUSS = True. *This is very slow*

