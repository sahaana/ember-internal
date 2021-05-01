from typing import Callable, List, Dict, Tuple, Sequence, NewType
from dataclasses import dataclass

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

import py_stringsimjoin as ssj
import py_stringmatching as sm

WS = sm.WhitespaceTokenizer(return_set=True)
TWO_GRAM = sm.QgramTokenizer(qval=2, return_set=True)


def simjoin_top_k_pd(routine, params, k_max, thresh = None, suppress=True, early_break=True):
    knn_results = defaultdict(list)
    for k in range(1, k_max+1):
        ret_avg, ret_count, all_avg, all_count, MRR, retrieved = routine(*params, k=k, thresh=thresh)
        if not suppress:
            print(f"k: {k} \t ret avg: {ret_avg} \t ret_count: {ret_count} \t ret avg: {all_avg} \t ret_count: {all_count} \t MRR: {MRR} \t retrieved: {retrieved}")
        knn_results['k'].append(k)
        knn_results['ret_avg'].append(ret_avg)
        knn_results['ret_count'].append(ret_count)
        knn_results['all_avg'].append(all_avg)
        knn_results['all_count'].append(all_count)
        knn_results['MRR'].append(MRR)
        knn_results['retrieved'].append(retrieved)
        
        if early_break and ret_avg == 1.0 and all_avg == 1.0:
            break

    knn_results = pd.DataFrame(knn_results).set_index('k')
    return knn_results


def return_simjoin_metrics(res: pd.DataFrame, k_list: List[int], thresh_list: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame] :
    if len(res) > 10:
        k_values = res.loc[k_list].copy()
    elif len(res) > 5:
        k_values = res.loc[k_list[:2]].copy()
    else:
        k_values = res.loc[[1,1]].copy()
    thresh_values = pd.DataFrame({'thresh': thresh_list, 
                                  "k_one": [res[res['ret_avg'] >= i].index[0] if len(res[res['ret_avg'] >= i]) > 0 else 300 for i in thresh_list],
                                  "k_all": [res[res['all_avg'] >= i].index[0] if len(res[res['all_avg'] >= i]) > 0 else 300 for i in thresh_list],
                                  #"k_one_eff": [res[res['ret_avg'] >= i]['retrieved'][0] if len(res[res['ret_avg'] >= i]) > 0 else 300 for i in thresh_list],
                                  #"k_all_eff": [res[res['all_avg'] >= i]['retrieved'][0] if len(res[res['all_avg'] >= i]) > 0 else 300 for i in thresh_list],     
                                 }).set_index('thresh')
    return k_values, thresh_values



def stack_padding(it, pad_val):

    def resize(row, size):
        new = np.array(row)
        #new.resize(size)
        new = np.pad(new, (0, size - len(new)), constant_values=pad_val)
        return new

    # find longest row length
    row_length = max(it, key=len).__len__()
    mat = np.array( [resize(row, row_length) for row in it] )

    return mat


def old_knn_sim_join_recall(dists: np.array,
                        neibs: np.array,
                        supervision: pd.DataFrame,
                        left_indexing: np.array,
                        comp_mode: str,
                        mode: str = "QID",
                        k: int = None,
                        thresh: float = None):
    
        
    if k is not None:
        if comp_mode == "LT":
            l, r = np.where(dists <= np.max(dists[:,:k], axis=1)[:,None]) ## to get all equidistant mins
        elif comp_mode == "GT":
            l, r = np.where(dists >= np.min(dists[:,:k], axis=1)[:,None]) ## to get all equidistant maxes
    else:
        if comp_mode == "LT":
            l, r = np.where(dists <= thresh) 
        if comp_mode == "GT":
            l, r = np.where(dists >= thresh) 
    results = []
    MRR_results = []
    all_captured = []
    num_retrieved_results = []
    
    candidate_matches = defaultdict(list)
    for i, j in zip(l,r):
        if neibs[i,j] != '-1':
            candidate_matches[left_indexing[i]].append(neibs[i, j])

    for idx, row in supervision.iterrows():
        true_matches = set(row[mode])
        candidates = candidate_matches[idx]
        num_retrieved = len(candidates)
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        all_matches = 1 if set(candidates) == true_matches else 0
        
        for entry in candidates:
            mrr_count += 1
            if entry in true_matches:
                first_relevant = min(mrr_count, first_relevant)
                matches += 1
                
        all_matches = 1 if len(true_matches) == matches else 0
        one_match = 1 if matches > 0 else 0  
            
        results.append(one_match)
        all_captured.append(all_matches)
        MRR_results.append(1./first_relevant)
        num_retrieved_results.append(num_retrieved)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), np.mean(num_retrieved_results) 


def knn_sim_join_recall_correct(dists: np.array,
                                neibs: np.array,
                                supervision: pd.DataFrame,
                                left_indexing: np.array,
                                comp_mode: str,
                                mode: str = "QID",
                                k: int = None,
                                thresh: float = None):
    candidate_matches = {}
    if k is not None:
        neibs = neibs[:,:k]
        for i in range(neibs.shape[0]):
            candidate_matches[left_indexing[i]] = neibs[i,:]
    else:
        pass # TODO
    
    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in supervision.iterrows():
        true_matches = set(row[mode])
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        if idx in candidate_matches:
            candidates = candidate_matches[idx]
            for entry in candidates:
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
           np.mean(MRR_results), k    


def knn_sim_join_recall(dists: np.array,
                        neibs: np.array,
                        supervision: pd.DataFrame,
                        left_indexing: np.array,
                        comp_mode: str,
                        mode: str = "QID",
                        k: int = None,
                        thresh: float = None):
    if k is not None:
        neibs = neibs[:,:k]
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
           np.mean(MRR_results), k  


def compute_joins(A, B, idA, idB, keyA, keyB, merged, compute = 5):
    if compute == 5:
        results = {'key_edit': ssj.edit_distance_join(A, B, idA, idB, keyA, keyB, 30, n_jobs=-5),
                   'key_jac_ws': ssj.jaccard_join(A, B, idA, idB, keyA, keyB, WS, 0.1, n_jobs=-5),
                   'key_jac_2g': ssj.jaccard_join(A, B, idA, idB, keyA, keyB, TWO_GRAM, 0.1, n_jobs=-5),
                   'mer_jac_ws': ssj.jaccard_join(A, B, idA, idB, merged, merged, WS, 0.1, n_jobs=-5),
                   'mer_jac_2g': ssj.jaccard_join(A, B, idA, idB, merged, merged, TWO_GRAM, 0.1, n_jobs=-5),
        }
    if compute == 3:
         results = {'key_edit': ssj.edit_distance_join(A, B, idA, idB, keyA, keyB, 30, n_jobs=-5),
                   'key_jac_ws': ssj.jaccard_join(A, B, idA, idB, keyA, keyB, WS, 0.1, n_jobs=-5),
                   'key_jac_2g': ssj.jaccard_join(A, B, idA, idB, keyA, keyB, TWO_GRAM, 0.1, n_jobs=-5),
                   'mer_jac_ws': ssj.jaccard_join(A, B, idA, idB, merged, merged, WS, 0.1, n_jobs=-5),
                   'mer_jac_2g': ssj.jaccard_join(A, B, idA, idB, merged, merged, TWO_GRAM, 0.1, n_jobs=-5),
        }  
            
    if compute == 2:
         results = {
                   'key_jac_ws': ssj.jaccard_join(A, B, idA, idB, keyA, keyB, WS, 0.1, n_jobs=-5),
                   'key_jac_2g': ssj.jaccard_join(A, B, idA, idB, keyA, keyB, TWO_GRAM, 0.1, n_jobs=-5),
        }    
            
    if compute == 1:
         results = {'key_edit': ssj.edit_distance_join(A, B, idA, idB, keyA, keyB, 30, n_jobs=-5),
        }    
            
            
    if compute == 0:
         results = {
                   'key_jac_2g': ssj.jaccard_join(A, B, idA, idB, keyA, keyB, TWO_GRAM, 0.1, n_jobs=-5),
        } 
    return results

def group_neighbors(joined, idA, idB, comp_mode, max_val=300):
    grouped = joined[[f'l_{idA}', f'r_{idB}', '_sim_score']].groupby(f'l_{idA}').aggregate(list)
    grouped['asc_sorted'] = grouped['_sim_score'].apply(lambda x: np.argsort(x)[:max_val])
    grouped['desc_sorted'] = grouped['_sim_score'].apply(lambda x: np.argsort(-1*np.array(x))[:max_val])

    grouped[f'r_{idB}'] = grouped[f'r_{idB}'].apply(lambda x: np.array(x))
    grouped['_sim_score'] = grouped['_sim_score'].apply(lambda x: np.array(x))

    if comp_mode == 'GT':
        grouped['highest_score_first'] = grouped.apply(lambda x: x['_sim_score'][x['desc_sorted']], axis=1)
        grouped['highest_tconst_first'] = grouped.apply(lambda x: x[f'r_{idB}'][x['desc_sorted']], axis=1)
        neibs = stack_padding(grouped['highest_tconst_first'].to_numpy(), pad_val=-1)
        dists = stack_padding(grouped['highest_score_first'].to_numpy(), pad_val=np.inf)
    elif comp_mode == 'LT':
        grouped['lowest_score_first'] = grouped.apply(lambda x: x['_sim_score'][x['asc_sorted']], axis=1)
        grouped['lowest_tconst_first'] = grouped.apply(lambda x: x[f'r_{idB}'][x['asc_sorted']], axis=1)
        neibs = stack_padding(grouped['lowest_tconst_first'].to_numpy(), pad_val=-1)
        dists = stack_padding(grouped['lowest_score_first'].to_numpy(), pad_val=-1)
    left_indexing = np.array(grouped.index)
    
    return neibs, dists, left_indexing
