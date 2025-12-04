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

from scipy.stats import pearsonr


MODEL_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
BASE_DIR = "./genomic_language_model"
TRIPLET_PATH = os.path.join(BASE_DIR, "triplet_data.csv")
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = 10


def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)

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
            "neg": row["negative"],
            "pos_mut": row["pos_mutations"],
            "neg_mut": row["neg_mutations"]
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

class CompositeMetricLoss(nn.Module):
    def __init__(self, margin=0.2, lambda_reg=0.3, lambda_metric=0.5):
        super().__init__()
        self.margin = margin
        self.lambda_reg = lambda_reg
        self.lambda_metric = lambda_metric

    def forward(self, anchor, positive, negative, pos_mutations, neg_mutations):
        # --- Cosine distance ---
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)

        pos_dist = 1 - pos_sim
        neg_dist = 1 - neg_sim

        # --- Triplet Loss ---
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin).mean()

        # --- Regression Loss (mutations vs distance) ---
        pos_mut = torch.tensor(pos_mutations, dtype=torch.float, device=anchor.device)
        neg_mut = torch.tensor(neg_mutations, dtype=torch.float, device=anchor.device)

        mut_counts = torch.cat([pos_mut, neg_mut])
        dists = torch.cat([pos_dist, neg_dist])

        mut_norm = mut_counts / mut_counts.max()
        regression_loss = F.mse_loss(dists, mut_norm)

        # --- Metric Alignment Loss ---
        cd = (pos_dist.mean() + neg_dist.mean()) / 2
        cdd = (neg_dist.mean() - pos_dist.mean()) / 2

        # PCC 계산 (batch 단위)
        if len(set(mut_counts.tolist())) > 1:
            pcc = pearsonr(mut_counts.detach().cpu().numpy(),
                           dists.detach().cpu().numpy())[0]
        else:
            pcc = 0.0

        # === 정규화 ===
        cd_norm = cd / 2
        cdd_norm = (cdd + 1) / 2
        pcc_norm = (pcc + 1) / 2 

        metric_score = (cd_norm + cdd_norm + pcc_norm) / 3
        metric_loss = 1 - metric_score  # 점수를 최대화 → loss는 최소화

        # --- Total Loss ---
        total_loss = triplet_loss + self.lambda_reg * regression_loss + self.lambda_metric * metric_loss
        return total_loss

def normalize_metrics(values):
    arr = np.array(values)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)


def validating(model, valDL, tokenizer, max_len, loss_fn, device):
    model.eval()
    loss_total = 0
    cd_list, cdd_list, pcc_list = [], [], []
    use_amp = (DEVICE == "cuda")

    with torch.no_grad():
        for batch in tqdm(valDL):
            anchor = batch["anchor"]
            pos = batch["pos"]
            neg = batch["neg"]
            pos_mut = batch["pos_mut"]
            neg_mut = batch["neg_mut"]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                ea = get_embedding(model, anchor, tokenizer, max_len, device)
                ep = get_embedding(model, pos, tokenizer, max_len, device)
                en = get_embedding(model, neg, tokenizer, max_len, device)

                loss = loss_fn(ea, ep, en, pos_mut, neg_mut)

            loss_total += loss.item()

            pos_sim = F.cosine_similarity(ea, ep)
            neg_sim = F.cosine_similarity(ea, en)
            pos_dist = 1 - pos_sim
            neg_dist = 1 - neg_sim

            cd = (pos_dist.mean().item() + neg_dist.mean().item()) / 2
            cdd = (neg_dist.mean().item() - pos_dist.mean().item()) / 2

            mut_counts = np.array(list(pos_mut) + list(neg_mut))
            dists = np.concatenate([pos_dist.cpu().numpy(), neg_dist.cpu().numpy()])
            if len(set(mut_counts.tolist())) > 1:
                pcc = pearsonr(mut_counts, dists)[0]
            else:
                pcc = 0.0

            cd_list.append(cd)
            cdd_list.append(cdd)
            pcc_list.append(pcc)

    print(f"VAL LOSS: {loss_total/len(valDL):.4f} | "
          f"CD: {np.mean(cd_list):.4f} | "
          f"CDD: {np.mean(cdd_list):.4f} | "
          f"PCC: {np.mean(pcc_list):.4f}")

    return loss_total

def training(model, trainDL, valDL, optimizer, 
            epoch, tokenizer, max_len,
            loss_fn, scheduler, device, scaler):
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
            pos_mut = batch["pos_mut"]
            neg_mut = batch ["neg_mut"]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                anchor = get_embedding(model, anchor, tokenizer, max_len, device)
                pos = get_embedding(model, pos, tokenizer, max_len, device)
                neg = get_embedding(model, neg, tokenizer, max_len, device)
                loss = loss_fn(anchor, pos, neg, pos_mut, neg_mut)
            loss_total += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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
    ft_df = pd.read_csv(TRIPLET_PATH)
    train_df, val_df = train_test_split(ft_df, test_size=0.1, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

    target_layers = []
    for i in range(26, 32):
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

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model = EmbeddingModel(backbone).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"학습 파라미터 수: {trainable:,} / {total:,}")

    for name, p in model.named_parameters():
        if p.requires_grad:
            print("TRAIN:", name)

    loss_fn = CompositeMetricLoss()
    max_seq_len = ft_df["seq"].str.len().max()
    MODEL_CAP = tokenizer.model_max_length
    MAX_LEN = min(MODEL_CAP, max_seq_len)

    train_dataset = TripletDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TripletDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    head_params = list(model.proj.parameters())
    backbone_params = [p for n,p in model.named_parameters() if p.requires_grad and ('proj' not in n)]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=1e-4)

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
        device=DEVICE,
        scaler=scaler
    )

    print("파인튜닝 완료")

if __name__ == "__main__":
    main()
