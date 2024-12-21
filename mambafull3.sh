#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J mamba_gradual_100
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 19:00
# request GB of system-memory
#BSUB -R "rusage[mem=50GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s233113@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o mamba_gradual_100_%J.out
#BSUB -e mamba_gradual_100_%J.err
# -- end of LSF options --

# Load the modules
module load python3/3.10.12
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
source /zhome/7a/2/203308/DL_project/venv2/bin/activate

python updated_cli_2.py --output_path=./mamba_1212_batch --model_type=mamba --epochs=20 --batch_size=32 --dropout=0.1 --lr=0.0001
