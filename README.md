## Overview

This is a deep-learning-based project for geometry prediction in DED 3D-printing. 
The input is co-axial images, parameter, timestamp, etc. And the output is WHA, and bead profiles. 

## Requirement
The python version is `Python 3.11`
```shell
pip install pyyaml bidict numpy pandas torch torchvision tensorboard tqdm 
```


## File Structure

``` 
./
├── config.yaml # defines hyper parameters, file paths, etc.
├── util.py     # to setup folders, load configs, etc.
├── data.py     # to generate dataset and dataloader
├── model.py    # to define all NN models 
├── train.py    # to train, valid, and save logs + checkpoints
├── test.py     # to testify trained models
├── multi_run.bat   # to run multi training cases at the same time
├── delay_run.bat   # to run the multi_run.bat with a time delay 
└── README.md   # requirement and instructions

dataset/
├── raw/
│   └── (original datasets)
└── processed/
    └── (preprocessed datasets)

output/
├── logs
│   └── (training and evaluation logs)
└── checkpoints/
    └── (saved models)
```

## Data
Please contact authors for dataset.
Note that pre-processed data is used to increase time efficiency in training.

## Train
The following shell formatting are supported.
```shell
python train.py
python train.py --config config.yaml
python train.py --config config.yaml --batch_size 32 --lr 0.01
python train.py --batch_size 32 --lr 0.01
```

To run multiple cases:
```shell
./multi_run.bat
```

To run the multi_run.bat with a time delay:
```shell
python delay_run.py
```

To view training results on tensorboard (adjust path accordingly):
```shell
tensorboard --logdir=C:\mydata\output\p2_ded_bead_profile\v13.1
```

To apply filters on tensorboard using regex
```
^(?=.*ABC)(?=.*5.0E-04)(?=.*loss_val).*
^(?=.*ABC)(?=.*5.0E-04)(?=.*mse_w_val).*
^(?=.*ABC)(?=.*101)(?=.*loss_val).*
```


## Test
Check the result of trained model.
```shell
python test.py
```

