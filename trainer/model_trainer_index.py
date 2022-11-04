"""This holds dictionary to various model trainers. when adding a new model trainers., add it to this dictionary!"""



from trainer.model_trainer_unet import UnetTrainer
from trainer.model_trainer_phdnet import PHDNetTrainer
from trainer.model_trainer_GP import GPTrainer

MODEL_TRAINER_INDEX = {
    'PHD-Net': PHDNetTrainer,
    'U-Net': UnetTrainer,
    'GP': GPTrainer


}