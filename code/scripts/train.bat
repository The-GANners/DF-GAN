@echo off
REM filepath: c:\Users\nanda\OneDrive\Desktop\DF-GAN\code\scripts\train.bat

REM configs of different datasets
set cfg=%1

REM model settings
set imsize=256
set num_workers=8
set batch_size_per_gpu=20
set stamp=normal
set train=True

REM resume training
set resume_epoch=322
REM Remove the trailing backslash to prevent path issues
set resume_model_path=C:\Users\nanda\OneDrive\Desktop\DF-GAN\code\saved_models\flower\dfgan_normal_flower_256_2025_07_08_10_31_56
python src/train.py ^
      --stamp %stamp% ^
      --cfg %cfg% ^
      --batch_size %batch_size_per_gpu% ^
      --num_workers %num_workers% ^
      --imsize %imsize% ^
      --resume_epoch %resume_epoch% ^
      --resume_model_path "%resume_model_path%" ^
      --train %train% ^
      --multi_gpus False ^
      --alignment_interval 1