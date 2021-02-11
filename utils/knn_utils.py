from typing import Callable, List, Dict, Tuple, Sequence, NewType

import faiss
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.k = k

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))

    def kneighbors(self, X):
        return self.index.search(X.astype(np.float32), k=self.k)
    
def create_neib_mask(num_data, num_neib):
    return np.reshape([range(num_data)]*num_neib, (num_neib, num_data)).T


def knn_top_1_PRFS(dists: np.array, 
                   neibs: np.array, 
                   supervision: pd.DataFrame):
    l, r = np.where(dists == np.min(dists, axis=1)[:,None])
    top_index = defaultdict(set)
    for i,j in zip(l,r):
        top_index[i].add(neibs[i,j])
    
    predicted = []
    supervision = supervision.to_numpy()
    true = supervision[:,2]
    for left, right, label in supervision:
        if right in top_index[left]:
            predicted += [1]
        else:
            predicted += [0]
    return precision_recall_fscore_support(true, predicted, average = 'binary'), predicted
        
        

def knn_matching_accuracy(neib, k, train_idx, test_idx):
    accuracy_mask = create_neib_mask(neib.shape[0], k)
    
    #train acc
    train_neib = neib[train_idx]
    train_mask = accuracy_mask[train_idx]
    train_acc = np.sum(train_neib[:,:k] == train_mask[:,:k])
    
    #test acc
    test_neib = neib[test_idx]
    test_mask = accuracy_mask[test_idx]
    test_acc = np.sum(test_neib[:,:k] == test_mask[:,:k])   
    
    return train_acc, test_acc, train_acc/len(train_idx), test_acc/len(test_idx)
  
def find_perfect_recall(neib, k, data_idx):
    for i in range(k):
        if (knn_matching_accuracy(neib, i, data_idx, data_idx)[-1] == 1):
            break
    return i