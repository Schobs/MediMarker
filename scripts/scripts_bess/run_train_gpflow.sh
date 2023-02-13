#!/bin/bash


module load Anaconda3/5.3.0

# module load libs/cudnn/7.5.0.56/binary-cuda-10.0.130
module load cuDNN/7.6.4.38-gcccuda-2019b

source activate gpflow

python ../../main.py  "$@" 


