# Deep State Space Model for Mortality Classification using Electronic Health Records

Repository branched from https://github.com/gsn245/Selective_SSM_for_EHR_Classification

Group 81, members: Paula Gomez Plana Rodriguez (s233165), Michela Sbetta (s230255), Miguel Gonzalez-Valdes Tejero (s233139), Maria Gabriela Frascella (s233113)

# Background
This repository allows you to train and test a variety of electronic health record (EHR) classification models on mortality prediction for the Physionet 2012 Challenge (`P12`) dataset. More information on the dataset can be found here (https://physionet.org/content/challenge-2012/1.0.0/). Note that the data in the repository has already been preprocessed (outliers removed, normalized) in accordance with https://github.com/ExpectationMax/medical_ts_datasets/tree/master and saved as 5 randomized splits of train/validation/test data. Adam is used for optimization.

As part of the project, we've added some more code from the original repository, for two purposes:
1. Integrating the EHR mamba implementation from https://github.com/VectorInstitute/odyssey into our methods-
2. Creating a custom mamba implementation for the Physionet 2012 Challenge dataset.

# Environment
Mamba implementations have different requirements from the baseline models. Here's how to create both environments from the specific requirements file with venv:

### Baseline models

```
# CD into the project folder
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

### Mamba implementations

```
# CD into the project folder
module load python3/3.10.12
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirementsmamba.txt
```

# Other prerequesites
You should unzip the data files before running these, and change the output paths in the commands. Additionally, GPU is required for running the Mamba models.

# Run models 
4 baseline models have been implemented in `Pytorch` and can be trained/tested on `P12`. Each has a unique set of hyperparameters that can be modified, but the supervisor has gotten the best performance by running the following commands:

`transformer` (https://arxiv.org/abs/1706.03762):

`python cli.py --output_path=your/path/here --epochs=100 --batch_size=16 --model_type=transformer --dropout=0.2 --attn_dropout=0.1 --layers=3 --heads=1 --pooling=max --lr=0.0001` 


`seft` (https://github.com/BorgwardtLab/Set_Functions_for_Time_Series):

`python cli.py --output_path=your/path/here --model_type=seft --epochs=100 --batch_size=128 --dropout=0.4 --attn_dropout=0.3 --heads=2 --lr=0.01 --seft_dot_prod_dim=512 --seft_n_phi_layers=1 --seft_n_psi_layers=5 --seft_n_rho_layers=2 --seft_phi_dropout=0.3 --seft_phi_width=512 --seft_psi_width=32 --seft_psi_latent_width=128 --seft_latent_width=64 --seft_rho_dropout=0.0 --seft_rho_width=256 --seft_max_timescales=1000 --seft_n_positional_dims=16`

`grud` (https://github.com/PeterChe1990/GRU-D/blob/master/README.md):

`python cli.py --output_path=your/path/here --model_type=grud --epochs=100 --batch_size=32 --lr=0.0001 --recurrent_dropout=0.2 --recurrent_n_units=128`

`ipnets` (https://github.com/mlds-lab/interp-net):

`python cli.py --output_path=your/path/here --model_type=ipnets --epochs=100 --batch_size=32 --lr=0.001 --ipnets_imputation_stepsize=1 --ipnets_reconst_fraction=0.75 --recurrent_dropout=0.3 --recurrent_n_units=32` 


For running the mamba models, the following commands can be run:

1. EHR mamba:
 - `python updated_cli_2.py --output_path=./mamba --model_type=mamba --epochs=20 --batch_size=32 --dropout=0.1 --lr=0.0001`

2. Custom Mamba;
   - For the regular training over one split of the data: `python custom_mamba.py` (the settings are defined inside the script)
   - In order to tune some hyperparameters over one split of the data: `python mamba_hyperparams.py` (the settings are defined inside the script)

# Jupyter notebook

We have also included a Jupyter Notebook, which allows you to run the same commands.

Our experimental setup doesn't allow us to train and run the experiments in a notebook, since the HPC is needed.  

**There are some changes in the jupyter notebook that the user needs to READ in order to manage to run the models.**
