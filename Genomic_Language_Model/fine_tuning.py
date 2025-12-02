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
SAVE_LORA_WEIGHT = os.path.join(DATA_DIR, "lora_weights")
OUT_PATH = "submission_v12.csv"

OUTPUT_DIM = 2048
LAST_N_LAYERS = 12
MAX_LENGTH = 512
USE_FP16 = True
BATCH_SIZE_TR = 32
BATCH_SIZE_INFER = 32
NUM_WORKERS = 2

TRAIN_EPOCHS = 5
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
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
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
        
        x = self.linear1(mean_emb) + mean_emb
        x = self.linear2(x) + x
        x = self.final(x)

        return l2_normalize(x)


# class InfoNCELoss(nn.Module):
#     def __init__(self, temperature=0.03):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, anchor, positive, negative):
#         pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature 
#         neg_sim = negative @ anchor.T / self.temperature              

#         mask = torch.eye(len(anchor), device=anchor.device).bool()
#         neg_sim = neg_sim.masked_fill(mask, float('-inf'))

#         logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) 

#         labels = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)

#         loss = F.cross_entropy(logits, labels)

#         return loss

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        # negatives shape: (B, K, dim)
        B, K, D = negatives.shape

        # pos similarity: (B,)
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        # anchor: (B, dim) → (B, 1, dim)
        anc = anchor.unsqueeze(1)

        # neg similarity: (B, K)
        neg_sim = torch.sum(anc * negatives, dim=-1) / self.temperature

        # logits = [pos | negs] → (B, K+1)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_df = pd.read_csv(TEST_PATH)
    sequences = test_df['seq'].tolist()

    triplet_df = pd.read_csv(TRIPLET_PATH)
    # anchor, positive, negative, pos_mutations, neg_mutations 컬럼이 있다고 가정
    triplet_data = triplet_df.values.tolist()[:50000]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    model = RobustModel(backbone.config.hidden_size, LAST_N_LAYERS, OUTPUT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    loss_fn = InfoNCELoss()

    print("학습 시작")
    model.train()

    bs = BATCH_SIZE_TR
    for epoch in range(TRAIN_EPOCHS):
        random.shuffle(triplet_data)
        epoch_loss = []

        # 평가용 기록
        all_pos_dists, all_neg_dists = [], []
        all_pos_mut, all_neg_mut = [], []

        for i in tqdm(range(0, len(triplet_data), bs)):
            batch = triplet_data[i:i+bs]
            if len(batch) < 2: continue

            anchors, poss, negs, pos_muts, neg_muts = zip(*batch)
            all_seqs = list(anchors) + list(poss) + list(negs)

            enc = tokenizer(all_seqs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)

            with torch.cuda.amp.autocast(enabled=USE_FP16):
                with torch.no_grad():
                    out = backbone(**enc, output_hidden_states=True)

                embs = model(out.hidden_states, enc['attention_mask'])
                B = len(batch)
                ea, ep, en = embs[:B], embs[B:2*B], embs[2*B:]

                # POS/NEG similarity 기록
                pos_sim = (ea * ep).sum(dim=-1).detach().cpu().numpy()
                neg_sim = (ea * en).sum(dim=-1).detach().cpu().numpy()

                all_pos_dists.extend(pos_sim.tolist())
                all_neg_dists.extend(neg_sim.tolist())
                all_pos_mut.extend(pos_muts)
                all_neg_mut.extend(neg_muts)

                # Hard negative mining
                K = 5
                neg_sim_matrix = ea @ en.T
                diag_mask = torch.eye(B, device=ea.device).bool()
                neg_sim_matrix = neg_sim_matrix.masked_fill(diag_mask, float("-inf"))
                _, topk_idx = torch.topk(neg_sim_matrix, K, dim=1)
                en_topk = en[topk_idx]  # (B, K, dim)

                loss = loss_fn(ea, ep, en_topk)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            epoch_loss.append(loss.item())

        # ====== 에포크 끝난 후 평가 지표 계산 ======
        cd = (np.mean(all_pos_dists) + np.mean(all_neg_dists)) / 2
        cdd = np.mean(all_neg_dists) - np.mean(all_pos_dists)

        # PCC: 변이 개수 vs 거리
        mut_counts = np.array(all_pos_mut + all_neg_mut)
        dists = np.array(all_pos_dists + all_neg_dists)
        if len(set(mut_counts)) > 1:
            pcc, _ = pearsonr(mut_counts, dists)
        else:
            pcc = 0.0

        print(f"[Epoch {epoch+1}] Loss={np.mean(epoch_loss):.4f} | CD={cd:.4f} | CDD={cdd:.4f} | PCC={pcc:.4f}")

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