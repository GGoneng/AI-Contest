import torch
import torch.nn as nn

from torch.utils.data import Dataset

import os

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
    def __init__(self, input_dim, num_layers, hidden_dim, dropout):
        super().__init__()
        self.cross = CrossNetwork(input_dim, num_layers)
        self.deep = DeepNetwork(input_dim, hidden_dim, dropout)

        self.output = nn.Linear(input_dim + hidden_dim[-1], 1)

    def forward(self, x):
        cross_out = self.cross(x)
        deep_out = self.deep(x)

        concat = torch.concat([cross_out, deep_out], dim=1)
        y = torch.sigmoid(self.output(concat))
        return y
    

class CTRDataset(Dataset):
    def __init__(self, feature, target=None):
        self.feature = feature
        self.target = target

    def __len__(self):
        return self.feature.shape[0]
    
    def __getitem__(self, index):
        featureTS = torch.tensor(self.feature.iloc[index].values, dtype=torch.float32)
        
        if self.target is not None:
            targetTS = torch.tensor(self.target.iloc[index].values, dtype=torch.float32)
            return featureTS, targetTS
        
        else:
            return featureTS


    
# 모델 Test 함수
def testing(model, valDL, data_size, loss_fn, score_fn, device):
    # Dropout, BatchNorm 등 가중치 규제 비활성화
    model.eval()

    loss_total, score_total = 0, 0

    with torch.no_grad():
        # Val DataLoader에 저장된 Feature, Target 텐서로 학습 진행
        for featureTS, targetTS in valDL:
            featureTS, targetTS = featureTS.to(device), targetTS.to(device)

            batch_size = len(targetTS)
    
            pre_val = model(featureTS)
            
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
        for featureTS, targetTS in trainDL:
            featureTS, targetTS = featureTS.to(device), targetTS.to(device)

            batch_size = len(targetTS)

            # 결과 추론
            pre_val = model(featureTS)

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