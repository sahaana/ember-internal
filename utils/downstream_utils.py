from typing import Callable, List, Dict, Tuple, Sequence, NewType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

from embedding_models import BertPooler
from transformers import DistilBertModel, BertModel, DistilBertTokenizer


def tokenize_ds(inputs: List,
                tokenizer: DistilBertTokenizer,
                max_length: int = 512) -> torch.Tensor:
    outputs = []
    masks = []
    out = tokenizer(list(inputs[0]), return_tensors='pt', padding=True,
                        max_length=max_length, truncation=True)
    labels = inputs[1]
    return out['input_ids'], out['attention_mask'], labels

class DownstreamRatingDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 data_col: str = 'merged_all',
                 label_col: str = 'averageRating', 
                 indexed = False):
        self.df = df.copy()[data_col]
        self.labels = df.copy()[label_col]
        self.n_samples = len(df)
        self.indexed = indexed
        if indexed:
            self.index = list(df.index)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.indexed:
            idx = self.index[idx]
        return self.df[idx], self.labels[idx]
    
    
class SingleBERTModel(nn.Module):
    def __init__(self, final_size, pooling, bert_path, model_type='distilbert', pool_activation=None): 
        super().__init__()
        if model_type == 'distilbert':
            self.bert = DistilBertModel.from_pretrained(bert_path, return_dict=True)
        else:
            self.bert = BertModel.from_pretrained(bert_path, return_dict=True)
        self.pooler = BertPooler(self.bert.config, final_size, pooling, pool_activation)

    def forward(self, a, a_mask):
        output_a = self.bert(a, attention_mask=a_mask).last_hidden_state
        output_a = self.pooler(output_a)
        return output_a 
    
