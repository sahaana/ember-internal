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