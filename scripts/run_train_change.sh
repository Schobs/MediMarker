#!/bin/bash

# Function to set the value of CUDA_VISIBLE_DEVICES to the id of the least-utilised device in the system.
# This will not set the device if CUDA_VISIBLE_DEVICES has already been set, to prevent issues on the ShARC DGX-1 (Node 126) or other private nodes which use the device allocation model
function setCUDAVisibileDevices {
# If there is no value of CUDA_VISIBLE_DEVICE
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Query device info, to select the device id with the least chance of utilsiation. Sorts by free memory, then gpu utilsation, memory utilsiation, temperature and then id
    deviceid=$(nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits | sort -t, -k3,3nr -k4,4n -k5,5n -k6,6n -k1,1nr | head -n 1 | cut -d, -f1)
    # Output and export the cuda visible devices variable
    echo "Setting CUDA_VISIBLE_DEVICES=$deviceid"
    export CUDA_VISIBLE_DEVICES=$deviceid
else
    # If CUDA_VISIBLE_DEVICES is already set, output the value to the user.
    echo "CUDA_VISIBLE_DEVICES is already set. CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
}

setCUDAVisibileDevices
#load modules
module load apps/python/conda

# module load libs/cudnn/7.5.0.56/binary-cuda-10.0.130
# module load libs/cudnn/8.2.1.32/binary-cuda-11.3.0



source activate /data/acq19las/landmark_unet_env


python ../main.py --cfg ../configs/4CH_awl.yaml "$@"
