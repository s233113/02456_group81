#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J transformer1
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
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
#BSUB -o transformer1_%J.out
#BSUB -e transformer1_%J.err
# -- end of LSF options --

nvidia-smi
# Load the modules
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
source /zhome/eb/a/203134/venvdef/bin/activate

python cli.py --epochs=100 --output_path="./ehr_transformer/" --batch_size=16 --model_type=transformer --dropout=0.2 --attn_dropout=0.1 --layers=3 --heads=1 --pooling=max --lr=0.0001
