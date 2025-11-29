import os
import random

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

from tqdm import tqdm
from typing import List

MODEL_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"

BASE_DIR = "./genomic_language_model"

TEST_PATH = os.path.join(BASE_DIR, "test.csv")
# LORA_PATH = os.path.join(BASE_DIR, "plant_nucleotide.csv")
TRIPLET_PATH = os.path.join(BASE_DIR, "triplet_data.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

OUTPUT_PATH = "submission.csv"

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = 5
LR = 5e-5

target_layers = ["encoder.layer.{}.attention.self.query".format(i) for i in range(28, 32)]
target_layers += ["encoder.layer.{}.attention.self.key".format(i) for i in range(28, 32)]
target_layers += ["encoder.layer.{}.attention.self.value".format(i) for i in range(28, 32)]
target_layers += ["encoder.layer.{}.attention.self.output".format(i) for i in range(28, 32)]

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=target_layers,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

def set_seed(seed: int=7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

def l2_normalize(x: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # L2 normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        # Cosine distance: 1 - cos_sim
        pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
        neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)

        # Triplet loss = max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()

class LoRADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tok = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "anchor": row["seq"],
            "pos": row["positive"],
            "neg": row["negative"]
        }

def get_embedding(backbone, seq_list, tokenizer, max_len, device):
    tokens = tokenizer(
        seq_list, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=max_len
    ).to(device)

    out = backbone(**tokens, output_hidden_states=True)
    h = out.hidden_states[-1]           # 마지막 layer hidden state (B,L,D)
    pooled = h.mean(dim=1)              # mean pooling
    emb = F.normalize(pooled, dim=-1)   # projection head 없이 방향만 정규화

    return emb


def validating(model, valDL, data_size, tokenizer, max_len, loss_fn, device):
    model.eval()

    loss_total = 0

    use_amp = (DEVICE == "cuda")
    
    with torch.no_grad():
        for batch in tqdm(valDL):
            anchor = batch["anchor"]
            pos = batch["pos"]
            neg = batch["neg"]

            batch_size = len(batch)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                anchor = get_embedding(model, anchor, tokenizer, max_len, device)
                pos = get_embedding(model, pos, tokenizer, max_len, device)
                neg = get_embedding(model, neg, tokenizer, max_len, device)

                loss = loss_fn(anchor, pos, neg)

            loss_total += loss.item() * batch_size
    
    avg_loss = loss_total / data_size

    return avg_loss

    
def training(model, trainDL, valDL, optimizer, epoch,
            data_size, val_data_size, tokenizer, max_len,
            loss_fn, scheduler, device):
    SAVE_PATH = "./saved_models"
    os.makedirs(SAVE_PATH, exist_ok=True)

    BREAK_CNT_LOSS = 0
    LIMIT_VALUE = 3

    LOSS_HISTORY = [[], []]

    use_amp = (DEVICE == "cuda")

    for count in range(1, epoch + 1):
        model.train()

        SAVE_LORA_WEIGHT = os.path.join(SAVE_PATH, f"lora_weights")

        loss_total = 0

        for batch in tqdm(trainDL):
            anchor = batch["anchor"].to(device)
            pos = batch["pos"].to(device)
            neg = batch["neg"].to(device)

            batch_size = len(batch)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                anchor = get_embedding(model, anchor, tokenizer, max_len, device)
                pos = get_embedding(model, pos, tokenizer, max_len, device)
                neg = get_embedding(model, neg, tokenizer, max_len, device)

                loss = loss_fn(anchor, pos, neg)

            loss_total += loss.item() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss = validating(model, valDL, val_data_size, tokenizer, max_len, loss_fn, device)

        LOSS_HISTORY[0].append(loss_total / data_size)
        LOSS_HISTORY[1].append(val_loss)

        print(f"[{count} / {epoch}]\n - TRAIN LOSS : {LOSS_HISTORY[0][-1]}")
        print(f"VAL LOSS : {LOSS_HISTORY[1][-1]}")

        scheduler.step(val_loss)

        if len(LOSS_HISTORY[0]) >= 2:
            if LOSS_HISTORY[0][-1] >= LOSS_HISTORY[0][-2]: BREAK_CNT_LOSS += 1
        
        if len(LOSS_HISTORY[0]) == 1:
            model.save_pretrained(SAVE_LORA_WEIGHT)
        
        else:
            if LOSS_HISTORY[0][-1] < min(LOSS_HISTORY[0][:-1]):
                model.save_pretrained(SAVE_LORA_WEIGHT)
        
        if BREAK_CNT_LOSS > LIMIT_VALUE:
            print(f"성능 및 손실 개선이 없어서 {count} EPOCH에 학습 중단")
            break
    
    return LOSS_HISTORY
    


def main():
    lora_df = pd.read_csv(TRIPLET_PATH)[:100000]

    train_df, val_df = train_test_split(lora_df, test_size=0.1, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = get_peft_model(model, lora_config)

    print(f"학습 파라미터 수 : {model.print_trainable_parameters()}")

    loss_fn = TripletLoss()

    max_seq_len = lora_df["seq"].str.len().max()
    MODEL_CAP = tokenizer.model_max_length

    MAX_LEN = min(MODEL_CAP, max_seq_len)

    train_dataset = LoRADataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = LoRADataset(val_df, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    data_size = len(train_dataset)
    val_data_size = len(val_dataset)

    loss = training(model=model, trainDL=train_loader, valDL=val_loader,
                    optimizer=optimizer, epoch=EPOCH, data_size=data_size,
                    val_data_size=val_data_size, tokenizer=tokenizer,
                    max_len=MAX_LEN, loss_fn=loss_fn, scheduler=scheduler,
                    device=DEVICE)
    
    print("LoRA 파인튜닝 완료")

if __name__ == "__main__":
    main()

