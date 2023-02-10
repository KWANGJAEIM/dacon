import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
from Dataset.ActionDataset import ActionDataset

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore') 

import numpy as np
import random 
import os
import pandas as pd
from tqdm.auto import tqdm 
import wandb
from utils.seed import seed_everything

from train import train, validation
from model.model import Actionclassfier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'IMG_SIZE':128,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':16,
    'SEED':41,
    "ROOT_DIR": "c:\\Users\\windowadmin5\\Desktop\\kj\\ActionClassification\\Dataset"
}

wandb.login()

wandb.init( project="dacon", entity="msn6385")
wandb.config.update(CFG,allow_val_change=True)
wandb.run.name = f'3dresnet{CFG["EPOCHS"]}_{CFG["LEARNING_RATE"]}_{CFG["BATCH_SIZE"]}'

seed_everything(CFG['SEED']) # Seed 고정

df = pd.read_csv('./train.csv')

df["video_path"] = list(map(lambda x: x.split(".")[1]+"."+x.split(".")[-1],df["video_path"]))
df["video_path"]= list(map(lambda x:CFG["ROOT_DIR"]+x, df["video_path"]))

train_df, val_df, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])


backbone = torchvision.models.video.r3d_18(pretrained=True)


 


train_dataset = ActionDataset(train_df['video_path'].values, train_df['label'].values)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = ActionDataset(val_df['video_path'].values, val_df['label'].values)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

model =Actionclassfier()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
wandb.watch(model, log_freq =100)
infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)