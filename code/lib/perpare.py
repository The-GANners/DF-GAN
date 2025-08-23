import os, sys
import os.path as osp
import time
import random
import datetime
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split

from lib.utils import mkdir_p, get_rank, load_model_weights
from models.DAMSM import RNN_ENCODER, CNN_ENCODER
from models.GAN import NetG, NetD, NetC

###########   preparation   ############
def prepare_models(args):
    device = args.device
    local_rank = args.local_rank
    multi_gpus = args.multi_gpus
    data_dir = getattr(args, 'data_dir', getattr(args, 'DATA_DIR', None))

    # --- Force flower encoder paths and vocab size if using flower dataset ---
    if data_dir and ('flower' in data_dir.lower()):
        img_encoder_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/flower/DAMSMencoder/image_encoder100.pth")
        )
        text_encoder_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/flower/DAMSMencoder/text_encoder100.pth")
        )
        vocab_size = 5428  # Hardcode for flower dataset
        print(f"Forced flower image encoder path: {img_encoder_path}")
        print(f"Forced flower text encoder path: {text_encoder_path}")
        print(f"Forced flower vocab size: {vocab_size}")
    else:
        # Use encoder_epoch parameter from args instead of hardcoded 200
        encoder_epoch = args.encoder_epoch  # This allows specifying the epoch (default 100)
        img_encoder_path = os.path.join(args.data_dir, 'DAMSMencoder', f'image_encoder{encoder_epoch}.pth')
        text_encoder_path = os.path.join(args.data_dir, 'DAMSMencoder', f'text_encoder{encoder_epoch}.pth')
        
        print(f"Loading image encoder from: {img_encoder_path}")
        print(f"Loading text encoder from: {text_encoder_path}")

        # Check if the encoder files exist
        if not os.path.exists(img_encoder_path):
            print(f"Warning: {img_encoder_path} not found. Trying to use image_encoder100.pth instead.")
            img_encoder_path = os.path.join(args.data_dir, 'DAMSMencoder', 'image_encoder100.pth')
            
        if not os.path.exists(text_encoder_path):
            print(f"Warning: {text_encoder_path} not found. Trying to use text_encoder100.pth instead.")
            text_encoder_path = os.path.join(args.data_dir, 'DAMSMencoder', 'text_encoder100.pth')
            
        vocab_size = args.vocab_size

    # Print debug info
    print(f"Image encoder path: {img_encoder_path}")
    print(f"Image encoder exists: {os.path.exists(img_encoder_path)}")
    image_encoder = CNN_ENCODER(args.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(img_encoder_path, map_location='cpu')
    image_encoder = load_model_weights(image_encoder, state_dict, multi_gpus=False)
    image_encoder.to(device)
    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.eval()

    print(f"Text encoder path: {text_encoder_path}")
    print(f"Text encoder exists: {os.path.exists(text_encoder_path)}")
    text_encoder = RNN_ENCODER(vocab_size, nhidden=args.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(text_encoder_path, map_location='cpu')
    text_encoder = load_model_weights(text_encoder, state_dict, multi_gpus=False)
    # Move text encoder to the configured device (supports CPU or CUDA)
    text_encoder.to(device)
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    netG = NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size).to(device)
    netC = NetC(args.nf, args.cond_dim).to(device)
    return image_encoder, text_encoder, netG, netD, netC


def prepare_dataset(args, split, transform):
    imsize = args.imsize
    # --- Fix: Use CONFIG_NAME if present, else fallback to DATASET_NAME ---
    config_name = getattr(args, 'CONFIG_NAME', getattr(args, 'DATASET_NAME', ''))
    if transform is not None:
        image_transform = transform
    elif config_name.find('CelebA') != -1:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    # train dataset
    from lib.datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, transform=image_transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    # train dataset
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    # test dataset
    val_dataset = prepare_dataset(args, split='val', transform=transform)
    return train_dataset, val_dataset


def prepare_dataloaders(args, transform=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform)
    # Always use single GPU/CPU DataLoader
    train_sampler = None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True,
        num_workers=num_workers, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, drop_last=True,
        num_workers=num_workers, shuffle=True)
    return train_dataloader, valid_dataloader, \
            train_dataset, valid_dataset, train_sampler


