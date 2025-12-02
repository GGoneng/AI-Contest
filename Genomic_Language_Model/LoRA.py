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
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
BASE_DIR = "./genomic_language_model"
TRIPLET_PATH = os.path.join(BASE_DIR, "triplet_data.csv")
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = 10
LR = 1e-4


def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        pos_cos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
        neg_cos = 1 - F.cosine_similarity(anchor, negative, dim=-1)

        pos_dist = pos_cos * 100
        neg_dist = neg_cos * 100

        loss = F.relu(pos_dist - neg_dist + self.margin) * 3
        
        print("Anchor-Positive Cosine Dist:", pos_cos.mean().item())
        print("Anchor-Negative Cosine Dist:", neg_cos.mean().item())
        print("Loss:", loss.mean().item())

        return loss.mean()

class TripletDataset(Dataset):
    def __init__(self, df):
        self.df = df

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
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
        emb = backbone(tokens)
    return emb

class EmbeddingModel(nn.Module):
    def __init__(self, backbone, target_dim=2048):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(backbone.config.hidden_size, target_dim)

    def forward(self, tokens):
        out = self.backbone(**tokens, output_hidden_states=True)
        h = out.hidden_states[-1]
        pooled = h.mean(dim=1)
        emb = self.proj(pooled)
        emb = F.normalize(emb, dim=-1)
        return emb


def validating(model, valDL, tokenizer, max_len, loss_fn, device):
    model.eval()
    loss_total = 0
    use_amp = (DEVICE == "cuda")

    with torch.no_grad():
        for batch in tqdm(valDL):
            anchor = batch["anchor"]
            pos = batch["pos"]
            neg = batch["neg"]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                anchor = get_embedding(model, anchor, tokenizer, max_len, device)
                pos = get_embedding(model, pos, tokenizer, max_len, device)
                neg = get_embedding(model, neg, tokenizer, max_len, device)

                loss = loss_fn(anchor, pos, neg)
            loss_total += loss.item()
    return loss_total

def training(model, trainDL, valDL, optimizer, 
            epoch, tokenizer, max_len,
            loss_fn, scheduler, device):
    SAVE_PATH = "./saved_models"
    os.makedirs(SAVE_PATH, exist_ok=True)
    BREAK_CNT_LOSS = 0
    LIMIT_VALUE = 3
    LOSS_HISTORY = [[], []]
    use_amp = (DEVICE == "cuda")

    for count in range(1, epoch + 1):
        model.train()
        SAVE_WEIGHT = os.path.join(SAVE_PATH, "model_weights")
        loss_total = 0

        for batch in tqdm(trainDL):
            anchor = batch["anchor"]
            pos = batch["pos"]
            neg = batch["neg"]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                anchor = get_embedding(model, anchor, tokenizer, max_len, device)
                pos = get_embedding(model, pos, tokenizer, max_len, device)
                neg = get_embedding(model, neg, tokenizer, max_len, device)
                loss = loss_fn(anchor, pos, neg)
            loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = validating(model, valDL, tokenizer, max_len, loss_fn, device)

        LOSS_HISTORY[0].append(loss_total / len(trainDL))
        LOSS_HISTORY[1].append(val_loss / len(valDL))

        print(f"[{count} / {epoch}]\n - TRAIN LOSS : {LOSS_HISTORY[0][-1]}")
        print(f"VAL LOSS : {LOSS_HISTORY[1][-1]}")

        scheduler.step(val_loss)

        if len(LOSS_HISTORY[0]) >= 2 and LOSS_HISTORY[0][-1] >= LOSS_HISTORY[0][-2]:
            BREAK_CNT_LOSS += 1

        if len(LOSS_HISTORY[0]) == 1 or LOSS_HISTORY[0][-1] < min(LOSS_HISTORY[0][:-1]):
            torch.save(model.state_dict(), SAVE_WEIGHT)

        if BREAK_CNT_LOSS > LIMIT_VALUE:
            print(f"성능 및 손실 개선이 없어서 {count} EPOCH에 학습 중단")
            break

    return LOSS_HISTORY

def freeze_all(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

def unfreeze_layers(model, target_layers):
    for name, param in model.named_parameters():
        if any(t in name for t in target_layers):
            param.requires_grad = True

def main():
    ft_df = pd.read_csv(TRIPLET_PATH)[:1000]
    train_df, val_df = train_test_split(ft_df, test_size=0.1, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

    target_layers = []
    for i in range(20, 32):
        target_layers += [
            f"esm.encoder.layer.{i}.attention.self.query",
            f"esm.encoder.layer.{i}.attention.self.key",
            f"esm.encoder.layer.{i}.attention.self.value",
            f"esm.encoder.layer.{i}.attention.self.output.dense",
            f"esm.encoder.layer.{i}.intermediate.dense",
            f"esm.encoder.layer.{i}.output.dense",
            f"esm.encoder.layer.{i}.LayerNorm"
        ]

    freeze_all(backbone)
    unfreeze_layers(backbone, target_layers)

    model = EmbeddingModel(backbone).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"학습 파라미터 수: {trainable:,} / {total:,}")

    for name, p in model.named_parameters():
        if p.requires_grad:
            print("TRAIN:", name)

    loss_fn = TripletLoss()
    max_seq_len = ft_df["seq"].str.len().max()
    MODEL_CAP = tokenizer.model_max_length
    MAX_LEN = min(MODEL_CAP, max_seq_len)

    train_dataset = TripletDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TripletDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    loss = training(
        model=model,
        trainDL=train_loader,
        valDL=val_loader,
        optimizer=optimizer,
        epoch=EPOCH,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=DEVICE
    )

    print("파인튜닝 완료")

if __name__ == "__main__":
    main()
