#!/bin/bash
#mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

module load Anaconda3/5.3.0

# module load libs/cudnn/7.5.0.56/binary-cuda-10.0.130

# # module load cuDNN/7.6.4.38-gcccuda-2019b
# source activate tensorflowEnv

<<<<<<< HEAD
# module load cuDNN/8.0.4.30-CUDA-11.1.1 
module load cuDNN/8.0.4.30-CUDA-11.0.2
source activate tf

# source activate tensorflowEnv
=======
module load cuDNN/8.0.4.30-CUDA-11.1.1 
source activate tf

>>>>>>> 651b2a2c43e31295a7b56bfa4026a347d3f1911d


python ../../main.py  "$@" 


