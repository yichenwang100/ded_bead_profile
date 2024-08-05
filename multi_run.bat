@echo off
@REM start cmd /c python train.py

start cmd /k python train.py --extra_name ffd_ta.embed6.sampling_8.lr_1e-5adap0.96.loss_mse --criterion_option mse --sys_sampling_interval 8
start cmd /k python train.py --extra_name ffd_ta.embed6.sampling_8.lr_1e-5adap0.96.loss_iou --criterion_option iou --sys_sampling_interval 8
start cmd /k python train.py --extra_name ffd_ta.embed6.sampling_8.lr_1e-5adap0.96.loss_mse_iou --criterion_option mse_iou --sys_sampling_interval 8

@REM start cmd /k python train.py

