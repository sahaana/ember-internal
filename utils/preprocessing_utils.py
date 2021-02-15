from typing import List
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from rank_bm25 import BM25Okapi
from tqdm.contrib.concurrent import process_map

def compute_BM25(corpus_df: pd.DataFrame, 
                 query_df: pd.DataFrame, 
                 data_col: str, 
                 f_name: str, 
                 reindex: False) -> np.array:
    pandarallel.initialize()
    base_path = "/lfs/1/sahaana/enrichment/data/Okapi25Queries"
    corpus = list(corpus_df[data_col].parallel_apply(lambda x: x.split()))
    indexed = BM25Okapi(corpus)
    bm25 = query_df[data_col].parallel_apply(lambda x: indexed.get_scores(x.split()))
    bm25 = np.vstack(bm25)    
    np.save(f"{base_path}/{f_name}.npy", bm25)
    final = np.argsort(bm25,axis=1)
 
    if not reindex:
        np.save(f"{base_path}/{f_name}_argsort.npy", final)
        print(f"Saved {f_name}")
        return final
    else: 
        corpus_indexes = np.array(corpus_df.index)
        query_index = np.array(query_df.index)
        
        final = corpus_indexes[final]
        np.save(f"{base_path}/{f_name}_argsort.npy", final)
        np.save(f"{base_path}/{f_name}_QIDs.npy", query_index)
        print(f"Saved {f_name}")
        return query_index, bm25, final 

    

def single_query_BM25(args) -> np.array:
    """args contains: corpus_df: pd.DataFrame, 
                      query_df: pd.DataFrame,
                      QID: int, 
                      PIDs.PID is List[int]"""
    corpus_df, query_df, QID, PIDs = args
 
    query = query_df.loc[QID].Query.split()
    corpus = [corpus_df.loc[i].Passage.split() for i in PIDs.PID]
    
    indexed = BM25Okapi(corpus)
    bm25 = indexed.get_scores(query)
    final = np.argsort(bm25)
    final = [PIDs.PID[i] for i in final]
    final = np.pad(final, (1001-len(final),0), mode='constant', constant_values=-1)
    bm25 = np.pad(bm25, (1001-len(bm25),0), mode='constant', constant_values=-1)
    
    return QID, bm25, final


def MARCO_1k_BM25(corpus_df: pd.DataFrame, 
                  query_df: pd.DataFrame,
                  bm25_1k: str,
                  grouped = False) -> np.array:
    base_path = "/lfs/1/sahaana/enrichment/data/Okapi25Queries/"
    grouped_queries = pd.read_pickle(bm25_1k)#[:200]
    
    entries = process_map(single_query_BM25,
                          [(corpus_df, query_df, QID, record) for (QID, record) in grouped_queries.iterrows()],
                          chunksize=int(len(grouped_queries)/40))
    #entries = [single_query_BM25((corpus_df, query_df, QID, record)) for (QID, record) in grouped_queries.iterrows()]
    qids = np.vstack([i[0] for i in entries])
    bm25 = np.vstack([i[1] for i in entries])
    bm25_argsort = np.vstack([i[2] for i in entries])
    np.save(f"{base_path}/MARCO_1k_QIDs.npy", bm25)
    np.save(f"{base_path}/MARCO_1k.npy", bm25)
    np.save(f"{base_path}/MARCO_1k_argsort.npy", bm25_argsort)
    return qids, bm25_argsort

def merge_columns(df: pd.DataFrame,
                  include_col: List[str],
                  new_col: str,
                  f_path: str,
                  separator = "[SEP]") -> pd.DataFrame:
    df[new_col] = ''
    for col in include_col:
        df[new_col] =  df[new_col] + f" {separator} " + f" {col} " + df[col].astype(str)
    df.to_pickle(f_path)
    return df

def reindex_deepmatcher(l_df: pd.DataFrame,
                        r_df: pd.DataFrame,
                        idx_df: pd.DataFrame) -> pd.DataFrame:
    df_updated = {}
    r_values = []
    l_values = []
    labels = []
    for idx, row in idx_df.iterrows():
        l_value = l_df[l_df['id'] == row['ltable_id']].index[0]
        r_value = r_df[r_df['id'] == row['rtable_id']].index[0]
        
        l_values += [l_value]
        r_values += [r_value]
        labels += [row['label']]
        
    df_updated['ltable_id'] = l_values
    df_updated['rtable_id'] = r_values
    df_updated['label'] = labels
    return pd.DataFrame(df_updated)