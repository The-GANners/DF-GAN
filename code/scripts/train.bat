@echo off
REM filepath: c:\Users\NMAMIT\Desktop\FYP AIDS 13\DF-GAN\code\scripts\train.bat

REM configs of different datasets
set cfg=%1

REM model settings
set imsize=256
set num_workers=4
set batch_size_per_gpu=32
set stamp=normal
set train=True

REM resume training
set resume_epoch=1
set resume_model_path=./saved_models/bird/base_z_dim100_bird_256_2022_06_04_23_20_33/

REM Use only a single GPU
set CUDA_VISIBLE_DEVICES=0

python src/train.py ^
      --stamp %stamp% ^
      --cfg %cfg% ^
      --batch_size %batch_size_per_gpu% ^
      --num_workers %num_workers% ^
      --imsize %imsize% ^
      --resume_epoch %resume_epoch% ^
      --resume_model_path %resume_model_path% ^
      --train %train% ^
      --multi_gpus False