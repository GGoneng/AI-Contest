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
LORA_PATH = os.path.join(BASE_DIR, "plant_nucleotide.csv")
TRIPLET_PATH = os.path.join(BASE_DIR, "fine_tuning_triplet.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

OUTPUT_PATH = "submission.csv"

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = 5
LR = 5e-5


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["query", "key", "value"],
    bias="none",
    task_type="CAUSAL_LM"
)

lora_args = TrainingArguments(
    output_dir="./lora_mlm_out",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    num_train_epochs=3,
    warmup_ratio=0.05,
    weight_decay=0.01,
    fp16=False,
    logging_steps=100,
    save_steps=2000,
    save_total_limit=2
)

def set_seed(seed: int=7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

def l2_normalize(x: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)

# class LoraDataset(Dataset):
#     def __init__(self, seqs, tokenizer, max_len: int=512):
#         self.seqs = seqs
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.seqs)

#     def __getitem__(self, idx):
#         seq = self.seqs[idx]

#         enc = self.tokenizer(
#             seq,
#             truncation=True,
#             max_length=self.max_len,
#             padding="longest",      # 절대 padding 하지마세요
#             return_tensors="pt" # 리스트 그대로
#         )
#         return {
#             "input_ids": enc["input_ids"],
#             "attention_mask": enc["attention_mask"]
#         }

class LoraDataset(Dataset):
    def __init__(self, seqs: List, tokenizer: AutoTokenizer, max_len: int=512, mask_prob: float=0.15):
        self.seqs = seqs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]

        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )

        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()

        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < self.mask_prob) & (input_ids != self.tokenizer.pad_token_id)

        input_ids[mask_arr] = self.tokenizer.mask_token_id
        labels[~mask_arr] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"][0],
            "labels": labels
        }
    
    
def validating(model, valDL, data_size, device):
    model.eval()

    loss_total = 0

    use_amp = (DEVICE == "cuda")

    with torch.no_grad():
        for batch in tqdm(valDL):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)


            batch_size = len(input_ids)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

            loss_total += loss.item() * batch_size
    
    avg_loss = loss_total / data_size

    return avg_loss

    
def training(model, trainDL, valDL, optimizer, epoch,
            data_size, val_data_size, scheduler, device):
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)


            batch_size = len(input_ids)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

            loss_total += loss.item() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss = validating(model, valDL, val_data_size, device)

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
    lora_df = pd.read_csv(LORA_PATH)

    train_df, val_df = train_test_split(lora_df, test_size=0.1, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = get_peft_model(model, lora_config)

    print(f"학습 파라미터 수 : {model.print_trainable_parameters()}")

    max_seq_len = lora_df["seq"].str.len().max()
    MODEL_CAP = tokenizer.model_max_length

    MAX_LEN = min(MODEL_CAP, max_seq_len)

    train_dataset = LoraDataset(train_df["seq"].tolist(), tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = LoraDataset(val_df["seq"].tolist(), tokenizer, MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    data_size = len(train_dataset)
    val_data_size = len(val_dataset)

    loss = training(model=model, trainDL=train_loader, valDL=val_loader,
                    optimizer=optimizer, epoch=EPOCH, data_size=data_size,
                    val_data_size=val_data_size, scheduler=scheduler, device=DEVICE)
    
    print("LoRA 파인튜닝 완료")

if __name__ == "__main__":
    main()

