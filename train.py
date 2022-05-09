import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import shutils

from torch.utils.data import DataLoader, Dataset
import datasets, transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["WANDB_DISABLED"] = "true"

train_df = pd.read_csv(f"{CFG.input_path}train.csv")
titles = pd.read_csv('../input/cpc-codes/titles.csv')
train_df = train_df.merge(titles, left_on='context', right_on='code')

def create_folds(data, num_splits):
    data["fold"] = -1
    data.loc[:, "bins"] = pd.cut(
        data["score"], bins=5, labels=False
    )
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'fold'] = f
        data = data.drop("bins", axis=1)
    return data

tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)


class TrainDataset(Dataset):
    def __init__(self, df):
        self.inputs = df['input'].values.astype(str)
        self.targets = df['target'].values.astype(str)
        self.label = df['score'].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = self.inputs[item]
        targets = self.targets[item]
        label = self.label[item]

        return {
            **tokenizer(inputs, targets),
            'label': label.astype(np.float32)
        }