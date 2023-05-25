# Evaluation

After inference, some .xlsx files will be output, one for the individual samples and a summary file. It will save predictions, uncertainties (S-MHA, if using ensemble: E-MHA & E-CPV). The summary file will contain Success Detection Rate (SDR), and mean error.

The output file can be directly used with my Quantile Binning framework:
https://github.com/Schobs/Qbin

This will analyse S-MHA, E-MHA, E-CPV for your model. 

TODO: flesh this out.