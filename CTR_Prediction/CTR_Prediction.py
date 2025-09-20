from CTRModules import *

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from torchmetrics.classification import AUROC

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import gc

DATA_PATH = "F:/CTR_Prediction/"

X_train = pd.read_csv(DATA_PATH + "X_train.csv")
X_val = pd.read_csv(DATA_PATH + "X_val.csv")
y_train = pd.read_csv(DATA_PATH + "y_train.csv")
y_val = pd.read_csv(DATA_PATH + "y_val.csv")

X_train_seq =  X_train.pop("seq")
X_val_seq = X_val.pop("seq")




BATCH_SIZE = 128

trainDS = CTRDataset(X_train, X_train_seq, y_train)
valDS = CTRDataset(X_val, X_val_seq, y_val)

trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_train)
valDL = DataLoader(valDS, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_train)




EPOCH = 300
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-3

hidden_dim = [512, 256, 128]

num_embeddings = 600 + 1
transformer_dim = 32
nhead = 4
num_encoder_layers = 1

model = DCNv2(input_dim=X_train.shape[1],
              num_layers=2,
              hidden_dim=hidden_dim,
              dropout=0.2,
              num_embeddings=num_embeddings,
              transformer_dim=transformer_dim,
              nhead=nhead,
              num_encoder_layers=num_encoder_layers
              ).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR)

weight = torch.tensor([2])

loss_fn = nn.BCELoss(weight=weight).to(DEVICE)
score_fn = AUROC(task="binary")

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

data_size = len(X_train)
val_data_size = len(X_val)



del X_train, X_train_seq, X_val, X_val_seq, y_train, y_val

gc.collect()
torch.cuda.empty_cache()



loss, score = training(model, trainDL, valDL, optimizer, EPOCH,
                       data_size, val_data_size, loss_fn, score_fn, 
                       scheduler, DEVICE)
