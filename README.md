## Overview

This project develops a deep learning framework for geometry prediction in Directed Energy Deposition (DED) 3D printing.

**Inputs**:
- Co-axial melt pool images
- Process parameters (e.g., laser power, travel speed)
- Positional variables (e.g., absolute positions, velocities)

**Outputs**:
- Cross-sectional bead profiles
- Bead geometry descriptors (width, height, area)

For further details, see our publication: [DED bead geometry and profile prediction with multimodal spatio-temporal neural networks](https://doi.org/10.1016/j.addma.2025.104952)

**Keywords**: Additive manufacturing, Directed energy deposition, Bead geometry prediction, Multimodal data fusion, Spatio-temporal neural network, Machine learning

## Requirement
The python version is `Python 3.11`

The pytorch version is `2.3.0`

The cuda version is `cu121`

```shell
pip install pyyaml bidict numpy pandas torch torchvision tensorboard tqdm 
```


## File Structure

``` 
./
├── config.yaml # defines hyper parameters, dataset paths, output path, etc.
├── machine.yaml # defines local machine info, root dir., GPU settings, etc.
├── util.py     # to setup folders, load configs, and basic math functions, etc.
├── data.py     # to generate dataset and dataloader
├── model.py    # to define all NN models 
├── train.py    # to train, valid, test, and save logs + checkpoints
├── deploy.py     # to deploy the trained model to testing data
├── multi_run.bat   # to run multi training cases at the same time
├── _delay_run.bat   # to run the multi_run.bat with a time delay 
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
Public dataset (size: ~4GB): https://zenodo.org/records/17087718
- Note that this dataset (with ext of .pt) is pre-processed data, which is used to increase time efficiency in training.
- It is recommended to use the pre-processed dataset as it is already spatio-temporally aligned.
- Please check function `create_dataset` in the file `data.py` for its generation method.
- Please check class `MyDataset` and class `MyCombinedDataset` in the file `data.py` for its usage.  

Please contact authors to request the full dataset (size ~55 GB), which include
- raw images (~55 GB)
- point cloud (~9 GB)
- raw data entries with timestamps (~3 GB).

To ensure responsible sharing and compliance with academic standards, please send your request to ywang100@connect.hkust-gz.edu.cn with the following information:
1. Full name, institutional affiliation, position, and a valid institutional email
2. Intended research use (project title, objectives, academic/non-commercial)
3. Background/credibility (Google Scholar, GitHub, or relevant publications)
4. Data security plan (storage, access control, confidentiality/deletion policy)
5. Confirmation that the dataset will not be redistributed and will be properly cited
6. Requested subset, if not the full dataset (~67 GB total: raw images ~55 GB, point cloud ~9 GB, raw entries ~3 GB)
   
Once we review your request, we will provide access details along with a Data Use Agreement (DUA/EULA).

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
tensorboard --logdir=C:\mydata\output\p2_ded_bead_profile\v19.9
```

To apply filters on tensorboard using regex
```
^(?=.*ABC)(?=.*5.0E-04)(?=.*loss_val).*
^(?=.*ABC)(?=.*5.0E-04)(?=.*mse_w_val).*
^(?=.*ABC)(?=.*101)(?=.*loss_val).*
```



