:: filepath: c:\Users\nanda\OneDrive\Desktop\DF-GAN\scripts\sample.bat
@echo off
:: configs of different datasets
set cfg=%1

:: model settings
set imgs_per_sent=16
set cuda=True
set gpu_id=0

python code\src\sample.py --cfg %cfg% --imgs_per_sent %imgs_per_sent% --cuda %cuda% --gpu_id %gpu_id%