#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random
import numpy as np
import pandas as pd
from typing import List 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

SEED = 42
MODEL_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"

DATA_DIR = "./data"
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH = "submission_v1.csv"

OUTPUT_DIM = 2048
LAST_N_LAYERS = 24
MAX_LENGTH = 512
USE_FP16 = True
BATCH_SIZE_TR = 32
BATCH_SIZE_INFER = 32
NUM_WORKERS = 2

TRAIN_EPOCHS = 3
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
TRIPLET_MARGIN = 1.0

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)

_rc_map = str.maketrans("ACGT", "TGCA")
def reverse_complement(seq: str) -> str:
    return seq.translate(_rc_map)[::-1]

def apply_biological_snv(seq: str, num_mutations: int) -> str:
    bases = ['A', 'C', 'G', 'T']
    s = list(seq)
    idx_list = random.sample(range(len(s)), min(num_mutations, len(s)))
    for idx in idx_list:
        orig = s[idx]
        candidates = [b for b in bases if b != orig]
        s[idx] = random.choice(candidates)
    return "".join(s)

def generate_triplets(sequences: List[str], num_pairs: int):
    rows = []
    print(f"Generating {num_pairs} triplets")
    for _ in range(num_pairs):
        anchor = random.choice(sequences)
        if len(anchor) > MAX_LENGTH:
            start = random.randint(0, len(anchor) - MAX_LENGTH)
            anchor = anchor[start:start+MAX_LENGTH]
        
        if random.random() < 0.5:
            pos = anchor
        else:
            pos = reverse_complement(anchor)
            
        n_mut = random.randint(3, 15)
        neg = apply_biological_snv(anchor, n_mut)
        
        rows.append([anchor, pos, neg])
    return rows


class RobustModel(nn.Module):
    def __init__(self, hidden_size: int, last_n: int, out_dim: int):
        super().__init__()
        self.last_n = last_n
        self.layer_weights = nn.Parameter(torch.zeros(last_n)) 
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, out_dim)
        )

    def forward(self, hidden_states: List[torch.Tensor], mask: torch.Tensor):
        stack = torch.stack(hidden_states[-self.last_n:], dim=0)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        feat = (stack * w).sum(dim=0) 
        
        mask_expanded = mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(feat * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask
        
        return l2_normalize(self.proj(mean_emb))

def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    test_df = pd.read_csv(TEST_PATH)
    sequences = test_df['seq'].tolist()
    
    triplet_data = generate_triplets(sequences, 30000)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    backbone.to(device)
    backbone.eval()
    
    for p in backbone.parameters():
        p.requires_grad = False
        
    model = RobustModel(backbone.config.hidden_size, LAST_N_LAYERS, OUTPUT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    print("학습시작해용학습시작해용학습시작해용학습시작해용학습시작해용학습시작해용학습시작해용학습시작해용학습시작해용학습시작해용학습시작해용학습시작해용")
    model.train()
    
    bs = BATCH_SIZE_TR
    for epoch in range(TRAIN_EPOCHS):
        random.shuffle(triplet_data)
        epoch_loss = []
        
        for i in tqdm(range(0, len(triplet_data), bs)):
            batch = triplet_data[i:i+bs]
            if len(batch) < 2: continue
            
            anchors, poss, negs = zip(*batch)
            all_seqs = list(anchors) + list(poss) + list(negs)
            
            enc = tokenizer(all_seqs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                with torch.no_grad():
                    out = backbone(**enc, output_hidden_states=True)
                
                embs = model(out.hidden_states, enc['attention_mask'])
                B = len(batch)
                ea, ep, en = embs[:B], embs[B:2*B], embs[2*B:]
                
                dist_pos = 1 - F.cosine_similarity(ea, ep)
                dist_neg = 1 - F.cosine_similarity(ea, en)
                loss = F.relu(dist_pos - dist_neg + TRIPLET_MARGIN).mean()
                
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            epoch_loss.append(loss.item())
            
        print(f"Epoch {epoch+1} Loss: {np.mean(epoch_loss):.4f}")
        
    print("추론시작해용추론시작해용추론시작해용추론시작해용추론시작해용추론시작해용추론시작해용추론시작해용추론시작해용추론시작해용추론시작해용")
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
    print(f"저장해용: {OUT_PATH}")

if __name__ == "__main__":
    main()