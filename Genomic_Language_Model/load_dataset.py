from datasets import load_dataset
import pandas as pd

repo_id = 'kuleshov-group/cross-species-single-nucleotide-annotation'
tis = load_dataset(repo_id, data_files={'train': 'TIS/train.tsv', 'valid': 'TIS/valid.tsv', 'test_rice':'TIS/test_rice.tsv', 'test_sorghum':'TIS/test_sorghum.tsv', 'test_maize':'TIS/test_maize.tsv'})
tis_train = tis['train']

tis_train_df = tis_train.to_pandas()
