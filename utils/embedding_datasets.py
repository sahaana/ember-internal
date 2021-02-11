from typing import Callable, List, Dict, Tuple, Sequence, NewType

import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, BertTokenizer, DistilBertTokenizer


class EmberEvalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 data_col: str = 'merged_all'):
        self.df = df.copy()[data_col]
        self.n_samples = len(df)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.df[idx]
                 
class EmberTripletDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df_l: pd.DataFrame, 
                 df_r: pd.DataFrame,
                 n_samples: int,
                 data_col: str = 'merged_all', 
                 supervision: pd.DataFrame = None):
        self.df_l = df_l.copy()[data_col]
        self.df_r = df_r.copy()[data_col]
        self.n_samples = n_samples
        self.triplets = self.gen_triplets(supervision)
    
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        pass

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        a, p, n = self.triplets[idx]
        return self.df_l[a], self.df_r[p], self.df_r[n]
    
class DeepMatcherDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        positives = supervision[supervision['label'] == 1]
        negatives = supervision[supervision['label'] == 0]
        merged = pd.merge(positives, negatives, on='ltable_id')

        triples = set()
        positive_with_negative = set()

        # Get all records in l that have positive and and negative example from r
        for _, record in merged.iterrows():
            a = record.ltable_id
            p = record.rtable_id_x
            n = record.rtable_id_y
            triples.add((a,p,n))
            positive_with_negative.add((a,p))
            if len(triples) >= self.n_samples:
                break

        num_positive_missed = len(positives) - len(positive_with_negative)
        # Get all the records that have positive labels but were ignored because they didn't have a negative
        # Add a random negative to each positive example that didn't get one above. 
        while len(triples) < self.n_samples and num_positive_missed > 0:
            for _, record in positives.iterrows():
                a = record.ltable_id
                p = record.rtable_id
                if (a, p) not in positive_with_negative:
                    n = np.random.randint(0,len(self.df_r))
                    triples.add((a,p,n))
                    num_positive_missed -= 1
                if len(triples) >= self.n_samples or num_positive_missed <= 0:
                    break

        # just fill out the rest of the triples with one positive and one random
        while len(triples) < self.n_samples:
            for _, record in positives.iterrows():
                a = record.ltable_id
                p = record.rtable_id
                if (a, p) not in positive_with_negative:
                    n = np.random.randint(0,len(self.df_r))
                    triples.add((a,p,n))
                if len(triples) >= self.n_samples:
                    break  
        return list(triples)       
    

    
        