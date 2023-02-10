import gc
gc.collect()

from tqdm.auto import tqdm 
import torch.nn as nn
import numpy as np
import torch 
import wandb 
from sklearn.metrics import f1_score

CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'IMG_SIZE':128,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':8,
    'SEED':41
}

wandb.login()

wandb.init( project="dacon", entity="msn6385")
wandb.config.update(CFG)
wandb.run.name = f'3dresnet{CFG["EPOCHS"]}_{CFG["LEARNING_RATE"]}_{CFG["BATCH_SIZE"]}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, train_loader, val_loader, scheduler, device=device):
    model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(videos)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)

        wandb.log({"train_loss": _train_loss, "val_loss":_val_loss, "val_acc":_val_score})
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        

        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    
    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score