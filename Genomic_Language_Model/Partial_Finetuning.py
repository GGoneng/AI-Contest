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
HEAD_EPOCH = 3
FT_EPOCH = 1

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
            "anchor": row["anchor"],
            "pos": row["positive"],
            "neg": row["negative"],
            "pos_mut": row["pos_mutations"],
            "neg_mut": row["neg_mutations"]
        }

def get_embedding(model, seq_list, tokenizer, max_len, device):
    tokens = tokenizer(
        seq_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
        emb = model(tokens)
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
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)

        pos_dist = 1 - pos_sim
        neg_dist = 1 - neg_sim

        triplet_loss = F.relu(pos_dist - neg_dist + self.margin).mean()

        pos_mut = torch.tensor(pos_mutations, dtype=torch.float, device=anchor.device)
        neg_mut = torch.tensor(neg_mutations, dtype=torch.float, device=anchor.device)

        mut_counts = torch.cat([pos_mut, neg_mut])
        dists = torch.cat([pos_dist, neg_dist])

        mut_norm = mut_counts / mut_counts.max()
        regression_loss = F.mse_loss(dists, mut_norm)

        cd = (pos_dist.mean() + neg_dist.mean()) / 2
        cdd = (neg_dist.mean() - pos_dist.mean()) / 2

        if len(set(mut_counts.tolist())) > 1:
            pcc = pearsonr(mut_counts.detach().cpu().numpy(),
                           dists.detach().cpu().numpy())[0]
        else:
            pcc = 0.0

        cd_norm = cd / 2
        cdd_norm = (cdd + 1) / 2
        pcc_norm = (pcc + 1) / 2 

        metric_score = (cd_norm + cdd_norm + pcc_norm) / 3
        metric_loss = 1 - metric_score 

        total_loss = triplet_loss + self.lambda_reg * regression_loss + self.lambda_metric * metric_loss
        return total_loss


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
            loss_fn, device, scaler):
    SAVE_PATH = "./saved_models"

    os.makedirs(SAVE_PATH, exist_ok=True)
    LOSS_HISTORY = [[], []]
    use_amp = (DEVICE == "cuda")

    for count in range(1, epoch + 1):
        model.train()
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

    return LOSS_HISTORY

def freeze_all(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

def unfreeze_layers(model, target_layers):
    for name, param in model.named_parameters():
        if any(t in name for t in target_layers):
            param.requires_grad = True

def train_head_only(ft_df, tokenizer, backbone):
    train_df, val_df = train_test_split(ft_df, test_size=0.1, shuffle=True)

    freeze_all(backbone)

    model = EmbeddingModel(backbone).to(DEVICE)

    head_params = list(model.proj.parameters())
    optimizer = optim.AdamW([
        {'params': head_params, 'lr': 5e-5}, 
    ], weight_decay=1e-4)

    loss_fn = CompositeMetricLoss(margin=0.2, lambda_reg=0.2, lambda_metric=0.3)

    max_seq_len = ft_df["anchor"].str.len().max()
    MODEL_CAP = tokenizer.model_max_length
    MAX_LEN = min(MODEL_CAP, max_seq_len)

    train_dataset = TripletDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TripletDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    print("=== Phase 1: Head-only training (backbone freeze) ===")
    training(
        model=model,
        trainDL=train_loader,
        valDL=val_loader,
        optimizer=optimizer,
        epoch=HEAD_EPOCH,                       
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        loss_fn=loss_fn,
        device=DEVICE,
        scaler=scaler
    )

    # 이 시점의 가중치 저장
    torch.save(model.state_dict(), "./saved_models/model_head_only.pt")
    return model, MAX_LEN

def train_partial_ft(ft_df, tokenizer, backbone, max_len):
    train_df, val_df = train_test_split(ft_df, test_size=0.1, shuffle=True)

    model = EmbeddingModel(backbone).to(DEVICE)
    state_dict = torch.load("./saved_models/model_head_only.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)

    target_layers = []
    for i in range(31, 32):
        target_layers += [
            f"esm.encoder.layer.{i}.attention.self.query",
            f"esm.encoder.layer.{i}.attention.self.key",
            f"esm.encoder.layer.{i}.attention.self.value",
            f"esm.encoder.layer.{i}.attention.self.output.dense",
            f"esm.encoder.layer.{i}.intermediate.dense",
            f"esm.encoder.layer.{i}.output.dense",
            f"esm.encoder.layer.{i}.LayerNorm"
        ]

    freeze_all(model.backbone)
    unfreeze_layers(model.backbone, target_layers)

    for p in model.proj.parameters():
        p.requires_grad = True

    head_params = [p for n, p in model.named_parameters() if p.requires_grad and ('proj' in n)]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and ('proj' not in n)]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 5e-7},  
        {'params': head_params, 'lr': 5e-5},
    ], weight_decay=0.0) 

    loss_fn = CompositeMetricLoss(margin=0.2, lambda_reg=0.1, lambda_metric=0.2)

    train_dataset = TripletDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TripletDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    print("=== Phase 2: Partial fine-tuning (last layers only) ===")
    training(
        model=model,
        trainDL=train_loader,
        valDL=val_loader,
        optimizer=optimizer,
        epoch=FT_EPOCH,        
        tokenizer=tokenizer,
        max_len=max_len,
        loss_fn=loss_fn,
        device=DEVICE,
        scaler=scaler
    )

    torch.save(model.state_dict(), "./saved_models/model_partial_ft.pt")
    return model

def main():
    ft_df = pd.read_csv(TRIPLET_PATH) 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

    model_head, max_len = train_head_only(ft_df, tokenizer, backbone)

    model_final = train_partial_ft(ft_df, tokenizer, backbone, max_len)

    print("파인튜닝 완료")


if __name__ == "__main__":
    main()
