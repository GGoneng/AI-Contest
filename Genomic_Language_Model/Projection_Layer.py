import os, random
import numpy as np
import pandas as pd
from typing import List 

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel

from scipy.stats import pearsonr

from tqdm import tqdm

SEED = 7
MODEL_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"

DATA_DIR = "./genomic_language_model/"
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TRIPLET_PATH = os.path.join(DATA_DIR, "triplet_data.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

OUT_PATH = "submission_v16.csv"

OUTPUT_DIM = 2048
LAST_N_LAYERS = 12
MAX_LENGTH = 512
USE_FP16 = True
BATCH_SIZE_TR = 32
BATCH_SIZE_INFER = 32
NUM_WORKERS = 2

TRAIN_EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-4

def set_seed(seed: int=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.att = nn.Linear(hidden_size, 1)

    def forward(self, feats, mask):
        scores = self.att(feats).squeeze(-1)
        mask_bool = mask.to(torch.bool)

        fill = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask_bool, fill)

        weights = F.softmax(scores, dim=-1)

        pooled = torch.sum(feats * weights.unsqueeze(-1), dim=1)
        return pooled

class RobustModel(nn.Module):
    def __init__(self, hidden_size: int, last_n: int, out_dim: int):
        super().__init__()
        self.last_n = last_n
        self.layer_weights = nn.Parameter(torch.zeros(last_n)) 
        self.pool = AttentionPooling(hidden_size)

        self.linear1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.final = nn.Linear(hidden_size, out_dim)


    def forward(self, hidden_states: List[torch.Tensor], mask: torch.Tensor):
        stack = torch.stack(hidden_states[-self.last_n:], dim=0)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        feat = (stack * w).sum(dim=0) 
        
        mean_emb = self.pool(feat, mask)
        
        x = self.linear1(mean_emb)
        x = self.linear2(x) + x
        x = self.linear2(x) + x
        x = self.linear3(x)
        x = self.final(x)

        return l2_normalize(x)

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
        cdd = neg_dist.mean() - pos_dist.mean()

        # PCC 계산 (batch 단위)
        if len(set(mut_counts.tolist())) > 1:
            pcc = pearsonr(mut_counts.detach().cpu().numpy(),
                           dists.detach().cpu().numpy())[0]
        else:
            pcc = 0.0

        # === 개선된 정규화 ===
        # batch 단위 분포에 대해 min-max scaling
        cd_norm = (cd - pos_dist.min()) / (pos_dist.max() - pos_dist.min() + 1e-12)
        cdd_norm = (cdd - neg_dist.min()) / (neg_dist.max() - neg_dist.min() + 1e-12)
        pcc_norm = (pcc + 1) / 2  # [-1,1] → [0,1]

        metric_score = (cd_norm + cdd_norm + pcc_norm) / 3
        metric_loss = 1 - metric_score  # 점수를 최대화 → loss는 최소화

        # --- Total Loss ---
        total_loss = triplet_loss + self.lambda_reg * regression_loss + self.lambda_metric * metric_loss
        return total_loss

def normalize_metrics(values):
    arr = np.array(values)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_df = pd.read_csv(TEST_PATH)
    sequences = test_df['seq'].tolist()

    triplet_df = pd.read_csv(TRIPLET_PATH)
    # anchor, positive, negative, pos_mutations, neg_mutations 컬럼이 있다고 가정
    triplet_data = triplet_df.values.tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    model = RobustModel(backbone.config.hidden_size, LAST_N_LAYERS, OUTPUT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    loss_fn = CompositeMetricLoss()

    print("학습 시작")
    model.train()

    bs = BATCH_SIZE_TR
    for epoch in range(TRAIN_EPOCHS):
        random.shuffle(triplet_data)
        epoch_loss = []
        all_pos_dists, all_neg_dists = [], []
        all_pos_mut, all_neg_mut = [], []
    
        for i in tqdm(range(0, len(triplet_data), bs)):
            batch = triplet_data[i:i+bs]
            if len(batch) < 2: continue
    
            anchors, poss, negs, pos_muts, neg_muts = zip(*batch)
            all_seqs = list(anchors) + list(poss) + list(negs)
    
            enc = tokenizer(all_seqs, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt").to(device)
    
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                with torch.no_grad():
                    out = backbone(**enc, output_hidden_states=True)
    
                embs = model(out.hidden_states, enc['attention_mask'])
                B = len(batch)
                ea, ep, en = embs[:B], embs[B:2*B], embs[2*B:]
    
                loss = loss_fn(ea, ep, en, pos_muts, neg_muts)
    
                pos_sim = F.cosine_similarity(ea, ep)
                neg_sim = F.cosine_similarity(ea, en)
    
                pos_dist = (1 - pos_sim).detach().cpu().numpy()
                neg_dist = (1 - neg_sim).detach().cpu().numpy()
    
                all_pos_dists.extend(pos_dist.tolist())
                all_neg_dists.extend(neg_dist.tolist())
                all_pos_mut.extend(pos_muts)
                all_neg_mut.extend(neg_muts)
    
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            epoch_loss.append(loss.item())
    
        # ====== 에포크 끝난 후 평가 지표 계산 ======
        cd = (np.mean(all_pos_dists) + np.mean(all_neg_dists)) / 2
        cdd = np.mean(all_neg_dists) - np.mean(all_pos_dists)
    
        mut_counts = np.array(all_pos_mut + all_neg_mut)
        dists = np.array(all_pos_dists + all_neg_dists)
        pcc = pearsonr(mut_counts, dists)[0] if len(set(mut_counts)) > 1 else 0.0
    
        cd_norm = normalize_metrics(all_pos_dists + all_neg_dists).mean()
        cdd_norm = normalize_metrics(all_neg_dists).mean() - normalize_metrics(all_pos_dists).mean()
        pcc_norm = (pcc + 1) / 2
    
        final_score = (cd_norm + cdd_norm + pcc_norm) / 3
    
        print(f"[Epoch {epoch+1}] Loss={np.mean(epoch_loss):.4f} | "
              f"CD={cd:.4f} | CDD={cdd:.4f} | PCC={pcc:.4f} | "
              f"FinalScore={final_score:.4f}")

    print("추론 시작")
    model.eval()
    sub_df = pd.read_csv(SAMPLE_SUB_PATH)
    embeddings = []

    for i in tqdm(range(0, len(sequences), BATCH_SIZE_INFER)):
        batch = sequences[i:i+BATCH_SIZE_INFER]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            out = backbone(**enc, output_hidden_states=True)
            emb = model(out.hidden_states, enc['attention_mask'])
            embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    col_names = [f"emb_{i:04d}" for i in range(OUTPUT_DIM)]
    emb_df = pd.DataFrame(embeddings, columns=col_names)
    final_df = pd.concat([sub_df[['ID']], emb_df], axis=1)
    final_df.to_csv(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}")

if __name__ == "__main__":
    main()