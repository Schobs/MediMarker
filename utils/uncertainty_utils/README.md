# Test time augmentation transformations

The module `tta.py` contains all the functionality for applying test time augmentation to model inference within the framework. Note that only a small number of transforms have been added, so feel free to add your own - be aware that each added transform will need to be inverted during the logging section so that coordinates are in the original image space. Transforms must also be able to invert heatmaps and signle points to cater for the heatmap and coordinate-based evaluation methods.

To use the TTA ensemble-like inference method, change the option for it in you config file as seen in the `config.py` base.