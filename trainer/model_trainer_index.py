"""This holds dictionary to various model trainers. when adding a new model trainers., add it to this dictionary!"""


from trainer.model_trainer_unet import UnetTrainer
from trainer.model_trainer_phdnet import PHDNetTrainer
from trainer.model_trainer_GP import GPTrainer
from trainer.model_trainer_trunet import TrUNetTrainer
from trainer.model_trainer_unetr import UnetrTrainer

MODEL_TRAINER_INDEX = {"PHD-Net": PHDNetTrainer, "U-Net": UnetTrainer,
                       "GP": GPTrainer, "TrU-Net": TrUNetTrainer, "UNETR": UnetrTrainer}
