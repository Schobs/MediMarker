

# Train your model!

## Training 
You are now ready to train your landmark localization model! From /LannU-Net/ run:
    
    python main.py --cfg /configs/my_dataset_config.yaml

For students at the University of Sheffield using the Bessemer on the HPC, you have to load conda and CUDA. I have written a script to do this and then run the command above. Run the following:
```
    cd scripts/scripts_bess
    source run_train_config.sh --cfg ../../configs/my_dataset_config.yaml
```
## Inference (Testing) 
If you included a *testing* list in your JSON, inference will be completed after training and the results will be saved in the OUTPUT.OUTPUT_DIR.

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

*If you omit MODEL.CHECKPOINT or set MODEL.CHECKPOINT=None, LannU-Net will perform inference over all model checkpoints in the OUTPUT.OUTPUT_DIR directory.*

If you did include a *testing* list in your JSON, and simply want to re-run the inference, you can solely perform inference by leaving the yaml file the same and just changing TRAINER.INFERENCE_ONLY = True. You can also pick and choose which model checkpoint to perform inference on too. 

Note, here you are *editing* the yaml config file from training, not creating one with only these fields. Alternatively, you can copy the training yaml and have two separate yaml files: one for training, one for testing. Just remember to add the correct one to the command line as the cfg parameter when running the programme.

Now, Run the same command as above.

    python main.py --cfg /configs/my_dataset_config.yaml