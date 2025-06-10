@echo off
:: configs of different datasets
set cfg=%1

:: model settings
set imgs_per_sent=16
set cuda=True
set gpu_id=0
set encoder_epoch=100
set example_captions=code\example_captions\coco.txt

python code\src\sample.py --cfg %cfg% --imgs_per_sent %imgs_per_sent% --cuda %cuda% --gpu_id %gpu_id% --encoder_epoch %encoder_epoch% --example_captions %example_captions%
