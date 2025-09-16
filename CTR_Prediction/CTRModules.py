import torch
import torch.nn as nn

from torch.utils.data import Dataset

import os

import numpy as np

MAX_SEQ_LEN = 4000

class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        xw = torch.matmul(xl, self.weight)
        cross = x0 * (xw + self.bias) + xl
        return cross

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.cross_layer = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_layers)])
    
    def forward(self, x0):
        xl = x0
        for layer in self.cross_layer:
            xl = layer(x0, xl)
        return xl

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        
        layers = []
        
        input = input_dim

        for dim in hidden_dim:
            layers.append(nn.Linear(input, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input = dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class DCNv2(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, dropout, num_embeddings, transformer_dim, nhead, num_encoder_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        total_input_dim = input_dim + transformer_dim

        self.cross = CrossNetwork(total_input_dim, num_layers)
        self.deep = DeepNetwork(total_input_dim, hidden_dim, dropout)

        self.output = nn.Linear(total_input_dim + hidden_dim[-1], 1)

    def forward(self, x, seq):
        seq_emb = self.embedding(seq)  # (batch, seq_len, transformer_dim)
        seq_out = self.transformer(seq_emb)  # (batch, seq_len, transformer_dim)
        seq_vec = seq_out.mean(dim=1)  # sequence pooling
        x = torch.concat([x, seq_vec], dim=1)
        
        cross_out = self.cross(x)
        deep_out = self.deep(x)

        concat = torch.concat([cross_out, deep_out], dim=1)
        y = torch.sigmoid(self.output(concat))
        return y
    

class CTRDataset(Dataset):
    def __init__(self, feature, seq, target=None):
        self.feature = feature
        self.seq = seq
        self.target = target

    def __len__(self):
        return self.feature.shape[0]
    
    def __getitem__(self, index):
        featureTS = torch.tensor(self.feature.iloc[index].values, dtype=torch.float32)
        
        s = str(self.seq[index])

        if s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.array([0.0], dtype=np.float32)  # fallback


        seq = torch.from_numpy(arr)

        if self.target is not None:
            targetTS = torch.tensor(self.target.iloc[index].values, dtype=torch.float32)
            return featureTS, seq, targetTS
        
        else:
            return featureTS, seq
        

def map_to_unk(seq, max_id=600, unk_id=600):
    return [x if x <= max_id else unk_id for x in seq]


def collate_fn_train(batch):
    xs, seqs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)

    batch_seqs = []

    for s in seqs:
        seq_mapped = map_to_unk(s.tolist())
        if len(seq_mapped) > MAX_SEQ_LEN:
            seq_mapped = seq_mapped[-MAX_SEQ_LEN:]

        batch_seqs.append(torch.tensor(seq_mapped, dtype=torch.long))

    seqs_padded = nn.utils.rnn.pad_sequence(batch_seqs, batch_first=True, padding_value=0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)  # 빈 시퀀스 방지
    
    return xs, seqs_padded, seq_lengths, ys


def collate_fn_test(batch):
    xs, seqs = zip(*batch)
    xs = torch.stack(xs)
    batch_seqs = []

    for s in seqs:
        seq_mapped = map_to_unk(s.tolist())
        if len(seq_mapped) > MAX_SEQ_LEN:
            seq_mapped = seq_mapped[-MAX_SEQ_LEN:]

        batch_seqs.append(torch.tensor(seq_mapped, dtype=torch.long))

    seqs_padded = nn.utils.rnn.pad_sequence(batch_seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    
    return xs, seqs_padded, seq_lengths


    
# 모델 Test 함수
def testing(model, valDL, data_size, loss_fn, score_fn, device):
    # Dropout, BatchNorm 등 가중치 규제 비활성화
    model.eval()

    loss_total, score_total = 0, 0

    with torch.no_grad():
        # Val DataLoader에 저장된 Feature, Target 텐서로 학습 진행
        for featureTS, seq, _, targetTS in valDL:
            featureTS, seq, targetTS = featureTS.to(device), seq.to(device), targetTS.to(device)

            batch_size = len(targetTS)
    
            pre_val = model(featureTS, seq)
            
            loss = loss_fn(pre_val, targetTS).to(device)

            # DICE Score 확인 (클래스 객체와 예측값의 면적 비교)
            score = score_fn(pre_val, targetTS)

            loss_total += loss.item() * batch_size
            score_total += score * batch_size

    avg_loss = loss_total / data_size
    avg_score = score_total / data_size

    return avg_loss, avg_score


def training(model, trainDL, valDL, optimizer, epoch, 
             data_size, val_data_size, loss_fn, score_fn,
             scheduler, device):
    # 가중치 파일 저장 위치 정의
    SAVE_PATH = './saved_models'
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Early Stopping을 위한 변수
    BREAK_CNT_LOSS = 0
    LIMIT_VALUE = 50

    # Loss가 더 낮은 가중치 파일을 저장하기 위하여 Loss 로그를 담을 리스트
    LOSS_HISTORY, SCORE_HISTORY = [[], []], [[], []]

    for count in range(1, epoch  + 1):
        # GPU 환경에서 training과 testing을 반복하므로 eval 모드 -> train 모드로 전환
        # testing에서는 train 모드 -> eval 모드
        model.train()

        SAVE_WEIGHT = os.path.join(SAVE_PATH, f"best_model_weights.pth")

        loss_total, score_total = 0, 0

        # Train DataLoader에 저장된 Feature, Target 텐서로 학습 진행
        for featureTS, seq, _, targetTS in trainDL:
            featureTS, seq, targetTS = featureTS.to(device), seq.to(device), targetTS.to(device)

            batch_size = len(targetTS)

            # 결과 추론
            pre_val = model(featureTS, seq)

            # 추론값으로 Loss값 계산
            loss = loss_fn(pre_val, targetTS)

            # DICE Score 확인 (클래스 객체와 예측값의 면적 비교)
            score = score_fn(pre_val, targetTS)

            loss_total += loss.item() * batch_size
            score_total += score * batch_size

            # 이전 gradient 초기화
            optimizer.zero_grad()

            # 역전파로 gradient 계산
            loss.backward()

            # 계산된 gradient로 가중치 업데이트
            optimizer.step()
        
        # Val Loss, Score 계산
        val_loss, val_score = testing(model, valDL, val_data_size, loss_fn, score_fn, device)

        LOSS_HISTORY[0].append(loss_total / data_size)
        SCORE_HISTORY[0].append(score_total / data_size)

        LOSS_HISTORY[1].append(val_loss)
        SCORE_HISTORY[1].append(val_score)

        print(f"[{count} / {epoch}]\n - TRAIN LOSS : {LOSS_HISTORY[0][-1]}")
        print(f"- TRAIN DICE SCORE : {SCORE_HISTORY[0][-1]}")

        print(f"\n - TEST LOSS : {LOSS_HISTORY[1][-1]}")
        print(f"- TEST DICE SCORE : {SCORE_HISTORY[1][-1]}")

        # Val Score 결과로 스케줄러 업데이트
        scheduler.step(val_loss)

        # Early Stopping 구현
        if len(LOSS_HISTORY[1]) >= 2:
            if LOSS_HISTORY[1][-1] >= LOSS_HISTORY[1][-2]: BREAK_CNT_LOSS += 1
        
        if len(LOSS_HISTORY[1]) == 1:
            torch.save(model.state_dict(), SAVE_WEIGHT)

        else:
            if LOSS_HISTORY[1][-1] < min(LOSS_HISTORY[1][:-1]):
                torch.save(model.state_dict(), SAVE_WEIGHT)

        if BREAK_CNT_LOSS > LIMIT_VALUE:
            print(f"성능 및 손실 개선이 없어서 {count} EPOCH에 학습 중단")
            break
    
    return LOSS_HISTORY, SCORE_HISTORY