from typing import List
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from rank_bm25 import BM25Okapi

def compute_BM25(corpus_df: pd.DataFrame, 
                 query_df: pd.DataFrame, 
                 data_col: str, 
                 f_name: str) -> np.array:
    base_path = "/lfs/1/sahaana/enrichment/data/Okapi25Queries"
    corpus = list(corpus_df[data_col].apply(lambda x: x.split()))
    indexed = BM25Okapi(corpus)
    pandarallel.initialize()
    bm25 = query_df[data_col].parallel_apply(lambda x: indexed.get_scores(x.split()))
    bm25 = np.vstack(bm25)
    
    np.save(f"{base_path}/{f_name}.npy", bm25)
    final = np.argsort(bm25,axis=1)
    
    np.save(f"{base_path}/{f_name}_argsort.npy", final)
    print(f"Saved {f_name}")
    return final

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