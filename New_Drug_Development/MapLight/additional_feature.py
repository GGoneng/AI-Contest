# 데이터프레임 기반 데이터 처리
import pandas as pd

# 수치 계산 및 배열 연산
import numpy as np

# 분자 데이터 처리
import deepchem as dc

TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

smile_list = train_df["Canonical_Smiles"]
test_smile_list = test_df["Canonical_Smiles"]

mol2vec = dc.feat.Mol2VecFingerprint()
circular = dc.feat.CircularFingerprint()

mol2vec_features = mol2vec.featurize(smile_list)
test_mol2vec_features = mol2vec.featurize(test_smile_list)
circular_features = circular.featurize(smile_list)
test_circular_features = circular.featurize(test_smile_list)

mol2vec_features = np.nan_to_num(mol2vec_features, nan = 0, posinf = 0)
test_mol2vec_features = np.nan_to_num(test_mol2vec_features, nan = 0, posinf = 0)
circular_features = np.nan_to_num(circular_features, nan = 0, posinf = 0)
test_circular_features = np.nan_to_num(test_circular_features, nan = 0, posinf = 0)

combined = pd.read_csv("combined_features.csv", header = None)
test_combined = pd.read_csv("test_combined_features.csv", header = None)

combined = np.array(combined)
test_combined = np.array(test_combined)

combined = np.concatenate((combined, mol2vec_features, circular_features), axis = 1)
test_combined = np.concatenate((test_combined, test_mol2vec_features, test_circular_features), axis = 1)

np.savetxt("train_features.csv", combined, delimiter=",")
np.savetxt("test_features.csv", test_combined, delimiter=",")