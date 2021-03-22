from typing import Callable, List, Dict, Tuple, Sequence, NewType
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support


## Processing ############################################################################################


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

def compute_top_k_pd(routine, params, k_max, thresh = None):
    knn_results = defaultdict(list)
    for k in range(1, k_max+1):
        ret_avg, ret_count, all_avg, all_count, MRR, results, all_results, MRR_results = routine(*params, k=k, thresh=thresh)
        print(f"k: {k} \t ret avg: {ret_avg} \t ret_count: {ret_count} \t ret avg: {all_avg} \t ret_count: {all_count} \t MRR: {MRR}")
        knn_results['k'].append(k)
        knn_results['ret_avg'].append(ret_avg)
        knn_results['ret_count'].append(ret_count)
        knn_results['all_avg'].append(all_avg)
        knn_results['all_count'].append(all_count)
        knn_results['MRR'].append(MRR)

    knn_results = pd.DataFrame(knn_results)
    return knn_results


def process_results(results: List[pd.DataFrame]) -> Tuple[int, pd.DataFrame]:
    combined = pd.concat(results, axis=1)
    if len(results) > 1:
        averaged = pd.DataFrame({'k' : combined['k'].mean(axis=1), 
                                 'ret_avg': combined['ret_avg'].mean(axis=1), 
                                 'all_avg': combined['all_avg'].mean(axis=1),
                                 'MRR_avg': combined['MRR'].mean(axis=1),
                                 'ret_count': combined['ret_count'].mean(axis=1),
                                 'all_count': combined['all_count'].mean(axis=1),
                                }).set_index('k')
    else:
        averaged = pd.DataFrame({'k' : combined['k'],
                                 'ret_avg': combined['ret_avg'],
                                 'all_avg': combined['all_avg'],
                                 'MRR_avg': combined['MRR'],
                                 'ret_count': combined['ret_count'],
                                 'all_count': combined['all_count'],
                                }).set_index('k')
    return len(results), averaged


def return_metrics(res: pd.DataFrame, k_list: List[int], thresh_list: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame] :
    k_values = res.loc[k_list].copy()
    thresh_values = pd.DataFrame({'thresh': thresh_list, 
                                  "k_one": [res[res['ret_avg'] >= i].index[0] if len(res[res['ret_avg'] >= i]) > 0 else 300 for i in thresh_list],
                                  "k_all": [res[res['all_avg'] >= i].index[0] if len(res[res['all_avg'] >= i]) > 0 else 300 for i in thresh_list]}).set_index('thresh')
    return k_values, thresh_values


## KNN Modules ############################################################################################


def knn_deepmatcher_recall(dists: np.array, 
                           neibs: np.array, 
                           supervision: pd.DataFrame,
                           left_indexing: np.array,
                           right_indexing: np.array,
                           k: int = None,
                           thresh: float = None):
    neibs = right_indexing[neibs]
    if k is not None:
        l, r = np.where(dists <= np.max(dists[:,:k], axis=1)[:,None]) ## to get all equidistant mins
    else:
        l, r = np.where(dists <= thresh) 
    
    top_index = defaultdict(set)
    for i,j in zip(l,r):
        top_index[left_indexing[i]].add(neibs[i,j])
    
    predicted = []
    supervision = supervision.to_numpy()
    true = supervision[:,2].astype(int)
    for left, right, label in supervision:
        if right in top_index[left]:
            predicted += [1]
        else:
            predicted += [0]
    return *precision_recall_fscore_support(true, predicted, average = 'binary'), predicted

        
def knn_IMDB_wiki_recall(dists: np.array,
                         neibs: np.array,
                         supervision: pd.DataFrame,
                         left_indexing: np.array,
                         right_indexing: np.array,
                         k: int = None,
                         thresh: float = None):
    supervision = supervision.set_index('IMDB_ID')
    mode = "QID"
    if k is not None:
        neibs = right_indexing[neibs[:,:k]]
    else:
        pass # TODO
    
    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = left_indexing[idx]
        
        if qid in supervision.index:       
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0  
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results  


def knn_IMDB_fuzzy_recall(dists: np.array,
                          neibs: np.array,
                          supervision: pd.DataFrame,
                          left_indexing: np.array,
                          right_indexing: np.array,
                          k: int = None,
                          thresh: float = None):
    supervision = supervision.set_index('FUZZY_ID')
    mode = "tconst"
    if k is not None:
        neibs = right_indexing[neibs[:,:k]]
    else:
        pass # TODO
    
    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = left_indexing[idx]
        
        if qid in supervision.index:       
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0  
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results  


def knn_SQuAD_sent_recall(dists: np.array,
                          neibs: np.array,
                          supervision: pd.DataFrame,
                          left_indexing: np.array,
                          right_indexing: np.array,
                          k: int = None,
                          thresh: float = None):
    supervision = supervision.set_index('QID')
    mode = "SID"
    if k is not None:
        neibs = right_indexing[neibs[:,:k]]
    else:
        pass # TODO
    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = left_indexing[idx]
        
        if qid in supervision.index:       
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0  
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results  



def knn_DM_blocked_recall(dists: np.array,
                          neibs: np.array,
                          supervision: pd.DataFrame,
                          left_indexing: np.array,
                          right_indexing: np.array,
                          k: int = None,
                          thresh: float = None):
    supervision = supervision.set_index('ltable_id')
    mode = "rtable_id"
    if k is not None:
        neibs = right_indexing[neibs[:,:k]]
    else:
        pass # TODO
    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = left_indexing[idx]
        
        if qid in supervision.index:
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results  


def knn_MARCO_recall(dists: np.array,
                         neibs: np.array,
                         supervision: pd.DataFrame,
                         left_indexing: np.array,
                         right_indexing: np.array,
                         k: int = None,
                         thresh: float = None):
    supervision = supervision.set_index('QID')
    mode = "PID"
    if k is not None:
        neibs = right_indexing[neibs[:,:k]]
    else:
        pass # TODO
    
    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = left_indexing[idx]
        
        if qid in supervision.index:       
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0  
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results    




## BM25 Modules ############################################################################################


def bm25_deepmatcher_recall(bm25: pd.DataFrame, 
                            supervision: pd.DataFrame,
                            k: int = None,
                            thresh: float = None):
    supervision = supervision.set_index('ltable_id')
    bm25_index = np.array(bm25.index)
    
    neibs = bm25.to_numpy()[:,::-1]
    neibs = neibs[:,:k]
    if k is not None:
        l, r = np.where(neibs != None) #dumb hack
    else:
        pass
    
    top_index = defaultdict(set)
    for i,j in zip(l,r):
        top_index[bm25_index[i]].add(neibs[i,j])
    
    predicted = []
    true = []
    for left, (right, label) in supervision.iterrows():
        true += [label]
        if right in top_index[left]:
            predicted += [1]
        else:
            predicted += [0]
    return *precision_recall_fscore_support(true, predicted, average = 'binary'), predicted


def bm25_SQuAD_sent_recall(bm25: pd.DataFrame,
                           supervision: pd.DataFrame,
                           k: int,
                           thresh = None):
    supervision = supervision.set_index('QID')
    mode = "SID"
    neibs = bm25.to_numpy()[:,::-1]
    neibs = neibs[:, :k]
    #neibs = right_indexing[neibs[:,:k]]
    bm25_index = np.array(bm25.index)

    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = bm25_index[idx]
        
        if qid in supervision.index:       
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0  
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results    
   
def bm25_imdb_wiki_recall(bm25: pd.DataFrame,
                          supervision: pd.DataFrame,
                          k: int,
                          thresh = None):
    supervision = supervision.set_index('IMDB_ID')
    mode = 'QID'
    neibs = bm25.to_numpy()[:,::-1]
    neibs = neibs[:, :k]
    #neibs = right_indexing[neibs[:,:k]]
    bm25_index = np.array(bm25.index)

    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = bm25_index[idx]
        
        if qid in supervision.index:       
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0  
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results  
    
def bm25_imdb_fuzzy_recall(bm25: pd.DataFrame,
                           supervision: pd.DataFrame,
                           k: int,
                           thresh = None):
    supervision = supervision.set_index('FUZZY_ID')
    mode = 'tconst'
    neibs = bm25.to_numpy()[:,::-1]
    neibs = neibs[:, :k]
    #neibs = right_indexing[neibs[:,:k]]
    bm25_index = np.array(bm25.index)

    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = bm25_index[idx]
        
        if qid in supervision.index:       
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0  
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results  

  

def bm25_DM_joined_recall(bm25: pd.DataFrame,
                          supervision: pd.DataFrame,
                          k: int,
                          thresh = None):
    supervision = supervision.set_index('ltable_id')
    mode = "rtable_id"
    neibs = bm25.to_numpy()[:,::-1]
    neibs = neibs[:, :k]
    bm25_index = np.array(bm25.index)

    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = bm25_index[idx]
        
        if qid in supervision.index:       
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0  
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results  

## Unused I think... ############################################################################################


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