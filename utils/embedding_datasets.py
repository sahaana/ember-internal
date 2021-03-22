from typing import Callable, List, Dict, Tuple, Sequence, NewType

import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, BertTokenizer, DistilBertTokenizer


class EmberEvalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 data_col: str = 'merged_all',
                 indexed = False):
        self.df = df.copy()[data_col]
        self.n_samples = len(df)
        self.indexed = indexed
        if indexed:
            self.index = list(df.index)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.indexed:
            idx = self.index[idx]
        return self.df[idx]
                 
class EmberTripletDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df_l: pd.DataFrame, 
                 df_r: pd.DataFrame,
                 n_samples: int,
                 data_col: str = 'merged_all', 
                 supervision: pd.DataFrame = None,
                 params = None): #meant to pass in misc domain specific params
        self.df_l = df_l.copy()[data_col]
        self.df_r = df_r.copy()[data_col]
        self.n_samples = n_samples
        self.params = params
        self.triplets = self.gen_triplets(supervision)
    
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        pass

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        a, p, n = self.triplets[idx]
        #print(idx, a, p, n)
        return self.df_l[a], self.df_r[p], self.df_r[n]

class MARCODataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        """#triplets = supervision.to_numpy(dtype='object')
        #np.random.shuffle(triplets) 
        #return triplets[:self.n_samples]        
        triples = []
        a_col = "QID_a"
        p_col = "SID_p"
        n_col = "SID_n"
        if p_col not in supervision.columns:
            p_col = "PID_p"
            n_col = "PID_n"
            
        for _, record in supervision.iterrows():
            triples.append((record[a_col], record[p_col], record[n_col]))"""
        #print("am I here now??")
        triplets = supervision.to_numpy(dtype='object')
        #samples = np.random.choice(np.arange(len(triplets)), size=self.n_samples, replace=False)
        return triplets[:self.n_samples] 

    
class SQuADDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        """triples = []
        a_col = "QID_a"
        p_col = "SID_p"
        n_col = "SID_n"
        if p_col not in supervision.columns:
            p_col = "PID_p"
            n_col = "PID_n"
            
        for _, record in supervision.iterrows():
            triples.append((record[a_col], record[p_col], record[n_col]))
            if len(triples) >= self.n_samples:
                break"""
        triplets = supervision.to_numpy(dtype='object')
        #samples = np.random.choice(np.arange(len(triplets)), size=self.n_samples, replace=False)
        return triplets[:self.n_samples] 
    
class SQuADRandomDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        triples = set()
        a_col = "QID_a"
        q_col = "SID_p"
        
        while len(triples) < self.n_samples:
            for _, record in supervision.iterrows():
                random_negative = np.random.choice(self.df_r.index)
                if random_negative not in record[q_col]:
                    triples.add((record[a_col], np.random.choice(record[q_col]), random_negative))
                if len(triples) >= self.n_samples:
                    return list(triples)
        return list(triples)

class IMDBWikiDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        triples = set()
        a_col = "IMDB_ID"
        q_col = "QID"
        
        while len(triples) < self.n_samples:
            for _, record in supervision.iterrows():
                random_negative = np.random.choice(self.df_r.index)
                if record[q_col] != random_negative:
                    triples.add((record[a_col], record[q_col], random_negative))
                triples.add((record[a_col], record[q_col], random_negative))
                if len(triples) >= self.n_samples:
                    return list(triples)
        return list(triples)
    
class IMDBWikiHardNegativeDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        triples = set()
        a_col = "IMDB_ID"
        q_col = "QID"
        negatives = pd.read_pickle(self.params['negatives'])
        
        while len(triples) < self.n_samples:
            for idx, record in supervision.iterrows():
                negative_list = negatives.loc[idx][q_col]
                sampled_negative = np.random.choice(negative_list)
                if sampled_negative != record[q_col]:
                    triples.add((record[a_col], record[q_col], sampled_negative))
                if len(triples) >= self.n_samples:
                    return list(triples)
        return list(triples)
    
class IMDBFuzzyDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        triples = set()
        a_col = "FUZZY_ID"
        q_col = "tconst"
        
        while len(triples) < self.n_samples:
            for _, record in supervision.iterrows():
                random_negative = np.random.choice(self.df_r.index)
                if record[q_col] != random_negative:
                    triples.add((record[a_col], record[q_col], random_negative))
                if len(triples) >= self.n_samples:
                    return list(triples)
        return list(triples)
    
class IMDBFuzzyHardNegativeDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        triples = set()
        a_col = "FUZZY_ID"
        q_col = "tconst"
        negatives = pd.read_pickle(self.params['negatives'])
        
        while len(triples) < self.n_samples:
            for idx, record in supervision.iterrows():
                negative_list = negatives.loc[idx][q_col]
                sampled_negative = np.random.choice(negative_list)
                if sampled_negative != record[q_col]:
                    triples.add((record[a_col], record[q_col], sampled_negative))
                if len(triples) >= self.n_samples:
                    return list(triples)
        return list(triples)
    
class DMBlockedDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        triples = set()
        a_col = "ltable_id"
        q_col = "rtable_id"
        
        while len(triples) < self.n_samples:
            for _, record in supervision.iterrows():
                random_negative = np.random.choice(self.df_r.index)
                if random_negative not in record[q_col]:
                    triples.add((record[a_col], np.random.choice(record[q_col]), random_negative))
                if len(triples) >= self.n_samples:
                    return list(triples)
        return list(triples)

class DMHardNegativeBlockedDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        triples = set()
        a_col = "ltable_id"
        q_col = "rtable_id"
        negatives = pd.read_pickle(self.params['negatives'])
        
        while len(triples) < self.n_samples:
            for idx, record in supervision.iterrows():
                negative_list = negatives.loc[idx][q_col]
                #print(len(triples))
                sampled_negative = np.random.choice(negative_list)
                negatives.loc[idx][q_col].remove(sampled_negative)
                #print(len(negative_list), sampled_negative, record[q_col])
                if sampled_negative not in record[q_col]:
                    triples.add((record[a_col], np.random.choice(record[q_col]), sampled_negative))
                    #print(len(triples))
                if len(triples) >= self.n_samples:
                    return list(triples)
        return list(triples)
    
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
                    ##n = np.random.randint(0,len(self.df_r)) ## we np random choice this instead
                    n = np.random.choice(self.df_r.index) 
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
                    ##n = np.random.randint(0,len(self.df_r))  ## np randpm choice this
                    n = np.random.choice(self.df_r.index) 
                    triples.add((a,p,n))
                if len(triples) >= self.n_samples:
                    break  
        return list(triples)       
    

    
        