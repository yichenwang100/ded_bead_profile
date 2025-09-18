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
├── config.yaml     # defines hyper parameters, dataset paths, output path, etc.
├── machine.yaml     # defines local machine info, root dir., GPU settings, etc.
├── util.py     # to setup folders, load configs, and basic math functions, etc.
├── data.py     # to generate dataset and dataloader
├── model.py     # to define all NN models 
├── train.py     # to train, valid, test, and save logs + checkpoints
├── deploy.py     # to deploy the trained model to testing data
├── multi_run.bat     # to run multi training cases at the same time
├── _delay_run.bat     # to run the multi_run.bat with a time delay 
└── README.md     # requirement and instructions

dataset/
├── 20240919     # pytorch data for supervised learning
├── 20241225     # pytorch data for transfer learning
├── Post_Data_20240919     # matlab ST-aligned data for supervised learning
├── Post_Data_20241225     # matlab ST-aligned data for transfer learning
└── DS_Data     # melt pool image with timestamp
    └── DS_Data_High     # for supervised learning
    └── DS_Data_Low     # for supervised learning
    └── CAD_Data     # for transfer learning

output/
├── logs     # logs with tensorboard
    └── train_val_stats.csv    # the train-val-test stats for each epoch
    └── epoch_loss_test    # loss of the testing process for each epoch
    └── epoch_loss_train     # loss of the training process for each epoch
    └── ...
├── best_model_stats.csv     # the train-val-test stats for each saved weights
├── best_model_wts.pth     # model weights of the best model
├── best_model_wts_ep_0.pth     # model weights of the best model up to the 0th epoch
├── best_model_wts_ep_5.pth     # model weights of the best model up to the 5th epoch
├── best_model_wts_ep_x.pth     # model weights of the best model up to the xth epoch
├── best_model_wts_ep_100.pth     # model weights of the best model up to the 100th epoch
├── config.yaml      # copy of the config file used for training
```

## Data
Public dataset (size: ~4GB): https://zenodo.org/records/17087718
- Note that this dataset (with ext of .pt) is pre-processed data, which is used to increase time efficiency in training.
- It is recommended to use the pre-processed dataset as it is already spatio-temporally aligned.
- Please check function `create_dataset` in the file `data.py` for its generation method.
- Please check class `MyDataset` and class `MyCombinedDataset` in the file `data.py` for its usage.  

Please contact authors to request the full dataset (size 72.4 GB), which include
|Entry|Size|
|:-|:-|
|Melt pool image with timestamp (supervised learning)|52.8 GB|
|Melt pool image with timestamp (transfer learning)|1.19 GB|
|Point cloud data (supervised learning)|8.82 GB|
|Point cloud data (transfer learning)|0.03 GB|
|MATLAB ST-aligned data (supervised learning)|5.42 GB|
|MATLAB ST-aligned data (transfer learning)|0.08 GB|
|Pytorch data (supervised learning)|3.96 GB|
|Pytroch data (transfer learning)|0.06 GB|

To ensure responsible sharing and compliance with academic standards, please send your request to ywang100@connect.hkust-gz.edu.cn with the following information:
1. Full name, institutional affiliation, position, and a valid institutional email
2. Intended research use (project title, objectives, academic/non-commercial)
3. Background/credibility (Google Scholar, GitHub, or relevant publications)
4. Data security plan (storage, access control, confidentiality/deletion policy)
5. Confirmation that the dataset will not be redistributed and will be properly cited
   
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



