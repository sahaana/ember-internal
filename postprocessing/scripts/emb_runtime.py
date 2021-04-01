import pandas as pd
import numpy as np
import argparse

import matplotlib.pyplot as plt
from collections import defaultdict

import os
import sys
import time

from typing import List

sys.path.append('/lfs/1/sahaana/enrichment/ember/utils')
from embedding_models import TripletSingleBERTModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from embedding_runner import eval_model


def compute_emb(input_df: pd.DataFrame, 
                data_col: str) -> np.array:
    model = TripletSingleBERTModel(200, 'CLS', 'distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    eval_data = DataLoader(EmberEvalDataset(input_df, data_col, indexed=True), 
                           batch_size=8,
                           shuffle = False
                          )
    
    start = time.time()
    left_index, left_embeddings = eval_model(model, tokenizer, eval_data, tokenizer_max_length=512)
    return time.time() - start



sys.path.append('/lfs/1/sahaana/enrichment/ember/utils')
base_path = "/lfs/1/sahaana/enrichment/ember/embedding"
config_base = "/lfs/1/sahaana/enrichment/ember/embedding/configs/"


def time_all(iters=25):
    timings = defaultdict(list)
    data_base = '/lfs/1/sahaana/enrichment/data'
    data_col = 'merged_all'
    model = TripletSingleBERTModel(200, 'CLS', 'distilbert-base-uncased')
    
    
    # Fuzzy Join
    A = data_base + "/main_fuzzy/test_tableA_processed.pkl"
    A = pd.read_pickle(A).reset_index()

    results = [compute_emb(A, data_col) for i in range(iters)]
    timings['data'].append('FJ')
    timings['results'].append(results)
    timings['avg'].append(np.mean(results))
    print('FJ Done')
    print(results)
    print()

          
    # IMDB_wiki
    A = data_base + "/imdb_wiki/dev_tableA_processed.pkl"
    A = pd.read_pickle(A).reset_index()

    
    results = [compute_emb(A, data_col) for i in range(iters)]
    timings['data'].append('IMDB-wiki')
    timings['results'].append(results)
    timings['avg'].append(np.mean(results))
    print('imdb_wiki Done')
    print(results)
    print()

    # SQUAD
    A = data_base + "/SQuAD/dev_tableA_processed.pkl"
    A = pd.read_pickle(A).reset_index()
    
    results = [compute_emb(A, data_col) for i in range(iters)]
    timings['data'].append('SQUAD')
    timings['results'].append(results)
    timings['avg'].append(np.mean(results))
    print('SQUAD Done')
    print(results)
    print()


    # Deepmatchers
    dm_data  = {0:"joined_abt_buy_exp_data", 
                1:"joined_amazon_google_exp_data", 
                2:"joined_beer_exp_data", 
                4:"joined_dblp_acm_exp_data", 
                5:"joined_dblp_scholar_exp_data", 
                6:"joined_dirty_dblp_acm_exp_data", 
                7:"joined_dirty_dblp_scholar_exp_data", 
                8:"joined_dirty_itunes_amazon_exp_data", 
                9:"joined_dirty_walmart_amazon_exp_data", 
                10:"joined_fodors_zagat_exp_data", 
                11:"joined_itunes_amazon_exp_data", 
                12:"joined_walmart_amazon_exp_data",
                30:"joined_company_exp_data"}

    dm_keys  = {0:"name", 
                1:"title", 
                2:"Beer_Name", 
                4:"title", 
                5:"title", 
                6:"title", 
                7:"title", 
                8:"Song_Name", 
                9:"title", 
                10:"name", 
                11:"Song_Name", 
                12:"title",
                30:"content" }
    for i in dm_data:
        print()
        print(dm_data[i])
        A = data_base + f"/dm_blocked/{dm_data[i]}/tableA_processed.pkl"

        results = [compute_emb(A, data_col) for i in range(iters)]
        timings['data'].append(dm_data[i])
        timings['results'].append(results)
        timings['avg'].append(np.mean(results))
        print(dm_data[i], " Done")
        print(results)
        print()
        
    print()

    MS Marco
    data_base = '/lfs/1/sahaana/enrichment/data/MSMARCO/'
    A = pd.read_pickle(data_base + 'dev_tableA_processed.pkl')      

    results = [compute_emb(A, data_col) for i in range(iters)]
    timings['data'].append(dm_data[i])
    timings['results'].append(results)
    timings['avg'].append(np.mean(results))
    print('marco Done')
    print(results)
    print()

    timings = pd.DataFrame(timings)
    timings.to_pickle(f'/lfs/1/sahaana/enrichment/ember/postprocessing/emb-iters_{iters}.pkl')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain MLM based on config.')
    parser.add_argument('-i', "--iters", required=True, type=int,
                        help="Number of iterations")
    args = parser.parse_args()
    time_all(args.iters)  
