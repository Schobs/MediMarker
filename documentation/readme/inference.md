## Inference: Fitting a Gaussian to the Predicted Heatmap

At inference we can fit a Gaussian using a robust least squares method onto the predicted heatmap. Then, the landmark is extracted. In the config change INFERENCE.FIT_GAUSS = True. *This is very slow*

