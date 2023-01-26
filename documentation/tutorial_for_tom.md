# Tutorial For Tom

## 1. Install

1) Follow normal instructions to clone the repo.
2) Then do ```git checkout gaussian_process```
3) Make a conda environment using ```requirements/gp_environment.yml``` **NOT** ```requirements/environment.yml```


## 2. Data
Put your data somewhere (the stuff I shared with you on Google Drive)

## 3. Configure
Check out ```configs/configs_gp/configs_for_tom.yaml``` for your config file. The config is documented so hopefully you know what each one is doing. You will need to change your paths to the data, and the output.

## 4. Comet.ML
Make a Comet.ml account so you can track the experiments. You can add your keys to your config file.


## 4. Run
```bash
python main.py --cfg configs/configs_gp/configs_for_tom.yaml
```

## 5. Relevant files
- main.py : Where the code is run from
- model_trainer_gp.py: The most important file. This is where the model is trained and inference is performed
- dataset_base.py: The base class for the dataset. This is where the data is loaded and preprocessed.
- models/gp_model.py: The GP model


Hopefully that is enough to get you started. Let me know if you have any questions.
