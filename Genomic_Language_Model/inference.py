import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
BASE_DIR = "./genomic_language_model"

TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
SAVE_WEIGHT = "./saved_models/model_weights"

test_df = pd.read_csv(TEST_PATH)
sub_df = pd.read_csv(SUB_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
backbone = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

model = EmbeddingModel(backbone)
model.load_state_dict(torch.load(SAVE_WEIGHT, map_location=torch.device("cuda"), weights_only=True))

model = model.to(DEVICE)

model.eval()

max_seq_len = test_df["seq"].str.len().max()
MODEL_CAP = tokenizer.model_max_length
MAX_LEN = min(MODEL_CAP, max_seq_len)

def embed_sequence(model, tokenizer, seq, max_len, device):
    tokens = tokenizer(
        [seq],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            emb = model(tokens)  # [1, 2048]
    return emb.squeeze(0).cpu().numpy()  # [2048]

# 5) test seq → embedding
emb_list = []
for seq in test_df["seq"]:
    emb = embed_sequence(model, tokenizer, seq, MAX_LEN, DEVICE)
    emb_list.append(emb)

for i in range(len(emb_list[0])):
    sub_df[f"emb_{i:04d}"] = [emb[i] for emb in emb_list]

sub_df.to_csv("submission.csv", index=False)

print("저장 완료")

