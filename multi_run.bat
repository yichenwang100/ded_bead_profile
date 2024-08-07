@echo off
@REM start cmd /c python train.py
@REM start cmd /k python train.py


@REM start cmd /c python train.py --extra_name ffd_ta.embed6.sampling_2.lr_1e-4adap0.96.loss_iou_0.5.dropout_0.3 --criterion_mse_lambda 0.5 --criterion_iou_lambda 0.5
@REM start cmd /c python train.py --extra_name ffd_ta.embed6.sampling_2.lr_1e-4adap0.96.loss_iou_0.1.dropout_0.3 --criterion_mse_lambda 0.9 --criterion_iou_lambda 0.1
@REM start cmd /c python train.py --extra_name ffd_ta.embed6.sampling_2.lr_1e-4adap0.96.loss_iou_0.01.dropout_0.3 --criterion_mse_lambda 0.99 --criterion_iou_lambda 0.01
@REM start cmd /c python train.py --extra_name ffd_ta.embed6.sampling_2.lr_1e-4adap0.96.loss_iou_0.0.dropout_0.3 --criterion_mse_lambda 1.0 --criterion_iou_lambda 0.0

start cmd /c python train.py