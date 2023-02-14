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
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings(action='ignore') 

import numpy as np
import random 
import os
import pandas as pd
from tqdm.auto import tqdm 
import wandb
from utils.seed import seed_everything
from utils.get_df import get_df
from train import train, validation
from model.model import Actionclassfier

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from test import test 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'IMG_SIZE':128,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':41,
    "ROOT_DIR": "c:\\Users\\windowadmin5\\Desktop\\kj\\ActionClassification\\Dataset",
    "check_dir": "c:\\Users\\windowadmin5\\Desktop\\kj\\ActionClassification\\checkpoint\\",
    "model_name":"3d_resnet"
}

wandb.login()

wandb.init( project="dacon")
wandb.config.update(CFG)
wandb.run.name = f'3dresnet{CFG["EPOCHS"]}_{CFG["LEARNING_RATE"]}_{CFG["BATCH_SIZE"]}'

seed_everything(CFG['SEED']) # Seed 고정

df = get_df('./train.csv')

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

checkpoint_callback = ModelCheckpoint(
    monitor='val_score',
    dirpath=CFG['check_dir'],
    filename=f'{CFG["model_name"]}'+'-{epoch:02d}-{train_loss:.4f}-{val_score:.4f}',
    mode='max'
)
early_stop_callback = EarlyStopping(
    monitor="train_loss",
    patience=3,
    verbose=False,
    mode="min"
)

pl_video_model = model

wandb_logger = WandbLogger()

trainer = pl.Trainer(
    logger = wandb_logger,
    max_epochs=100,
    accelerator='auto', 
    precision=16,
    callbacks=[early_stop_callback, checkpoint_callback]
                    
)
train_dataset = ActionDataset(train_df['video_path'].values, train_df['label'].values)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = ActionDataset(val_df['video_path'].values, val_df['label'].values)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

trainer.fit(pl_video_model, train_loader, val_loader)
 
test_df = get_df('./test.csv')

test_dataset = ActionDataset(test_df['video_path'].values, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

preds = test(model, test_loader, device)

submit = pd.read_csv('sample_submission.csv')
submit['label'] = preds
submit.head()
submit.to_csv(f'./baseline_submit.csv {CFG["EPOCHS"]}_{CFG["LEARNING_RATE"]}_{CFG["BATCH_SIZE"]}', index=False)