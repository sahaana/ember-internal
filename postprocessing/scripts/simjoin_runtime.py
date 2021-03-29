import pandas as pd
import numpy as np
import argparse

import matplotlib.pyplot as plt
from collections import defaultdict

import os
import sys
import time

import py_stringsimjoin as ssj
import py_stringmatching as sm

WS = sm.WhitespaceTokenizer(return_set=True)
TWO_GRAM = sm.QgramTokenizer(qval=2, return_set=True)

sys.path.append('/lfs/1/sahaana/enrichment/ember/utils')
from simjoin_utils import knn_sim_join_recall, compute_joins, group_neighbors, simjoin_top_k_pd, return_simjoin_metrics

base_path = "/lfs/1/sahaana/enrichment/ember/embedding"
config_base = "/lfs/1/sahaana/enrichment/ember/embedding/configs/"


def compute_all(A, B, idA, idB, keyA, keyB, merged, iters):
    ke = []
    jkws = []
    jk2g = []
    jws = []
    j2g = []
    for _ in range(iters):
        start = time.time()
        ssj.edit_distance_join(A, B, idA, idB, keyA, keyB, 30, n_jobs=-1)
        ke.append(time.time() - start)
        
        start = time.time()
        ssj.jaccard_join(A, B, idA, idB, keyA, keyB, WS, 0.1, n_jobs=-1)
        jkws.append(time.time() - start)
        
        start = time.time()
        ssj.jaccard_join(A, B, idA, idB, keyA, keyB, TWO_GRAM, 0.1, n_jobs=-1)
        jk2g.append(time.time() - start)
        
        start = time.time()
        ssj.jaccard_join(A, B, idA, idB, merged, merged, WS, 0.1, n_jobs=-1)
        jws.append(time.time() - start)
        
        start = time.time()
        ssj.jaccard_join(A, B, idA, idB, merged, merged, TWO_GRAM, 0.1, n_jobs=-1)
        j2g.append(time.time() - start)
    return ke, jkws, jk2g, jws, j2g


def time_all(iters=25):
    timings = defaultdict(list)
    join_types = ['key_edit', 'key_jac_ws', 'key_jac_2g', 'mer_jac_ws', 'mer_jac_2g']
    data_base = '/lfs/1/sahaana/enrichment/data'
    
    # Fuzzy Join
    A = data_base + "/main_fuzzy/test_tableA_processed.pkl"
    B = data_base + "/main_fuzzy/test_tableB_processed.pkl"
    supervision = data_base + "/main_fuzzy/supervision_test.pkl"
    idA = 'FUZZY_ID'
    idB = 'tconst'
    keyA = 'primaryTitle'
    keyB = 'primaryTitle'
    merged = 'merged_all'
    
    A = pd.read_pickle(A).reset_index()
    B = pd.read_pickle(B).reset_index()
    supervision = pd.read_pickle(supervision)
    supervision = supervision.set_index(idA)

    results = compute_all(A, B, idA, idB, keyA, keyB, merged, iters)
    print('FJ')
    for j in range(len(join_types)):
        timings['data'].append('main_fuzzy')
        timings['type'].append(join_types[j])
        timings['results'].append(results[j])
        timings['avg'].append(np.mean(results[j]))
        print(timings['data'][-1], timings['type'][-1], timings['results'][-1], timings['avg'][-1])
    print()
    
          
    # IMDB_wiki
    A = data_base + "/imdb_wiki/dev_tableA_processed.pkl"
    B = data_base + "/imdb_wiki/dev_tableB_processed.pkl"
    supervision = data_base + "/imdb_wiki/supervision_test.pkl"
    idA = 'IMDB_ID'
    idB = 'QID'
    keyA = 'primaryTitle'
    keyB = 'title'
    merged = 'merged_all'
    
    A = pd.read_pickle(A).reset_index()
    B = pd.read_pickle(B).reset_index()
    supervision = pd.read_pickle(supervision)
    supervision = supervision.set_index(idA)
    
    results = compute_all(A, B, idA, idB, keyA, keyB, merged, iters)
    print('imdb_wiki')
    for j in range(len(join_types)):
        timings['data'].append('imdb_wiki')
        timings['type'].append(join_types[j])
        timings['results'].append(results[j])
        timings['avg'].append(np.mean(results[j]))  
        print(timings['data'][-1], timings['type'][-1], timings['results'][-1], timings['avg'][-1])
    print()

    # SQUAD
    A = data_base + "/SQuAD/dev_tableA_processed.pkl"
    B = data_base + "/SQuAD/dev_tableB_sent_processed.pkl"
    supervision = data_base + "/SQuAD/dev_sent_labels.pkl"
    idA = 'QID'
    idB = 'SID'
    keyA = 'question'
    keyB = 'sentence'
    merged = 'merged_all'
    
    A = pd.read_pickle(A).reset_index()
    B = pd.read_pickle(B).reset_index()
    supervision = pd.read_pickle(supervision)
    supervision = supervision.set_index(idA)
    
    results = compute_all(A, B, idA, idB, keyA, keyB, merged, iters)
    print('SQUAD')
    for j in range(len(join_types)):
        timings['data'].append('squad')
        timings['type'].append(join_types[j])
        timings['results'].append(results[j])
        timings['avg'].append(np.mean(results[j]))
        print(timings['data'][-1], timings['type'][-1], timings['results'][-1], timings['avg'][-1])
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
        B = data_base + f"/dm_blocked/{dm_data[i]}/tableB_processed.pkl"
        supervision = data_base + f"/dm_blocked/{dm_data[i]}/supervision_test.pkl"
        idA = 'id'
        idB = 'id'
        keyA = dm_keys[i]
        keyB = dm_keys[i]
        merged = 'merged_all'

        A = pd.read_pickle(A).reset_index()
        B = pd.read_pickle(B).reset_index()
        supervision = pd.read_pickle(supervision)
        supervision = supervision.set_index('ltable_id')
        results = compute_all(A, B, idA, idB, keyA, keyB, merged, iters)
        print(dm_data[i])
        for j in range(len(join_types)):
            timings['data'].append(dm_data[i])
            timings['type'].append(join_types[j])
            timings['results'].append(results[j])
            timings['avg'].append(np.mean(results[j]))
            print(timings['data'][-1], timings['type'][-1], timings['results'][-1], timings['avg'][-1])
        print()

    # MS Marco
    data_base = '/lfs/1/sahaana/enrichment/data/MSMARCO/'
    A = pd.read_pickle(data_base + 'dev_tableA_processed.pkl')
    B = pd.read_pickle(data_base + 'tableB_processed.pkl')
    supervision = pd.read_pickle(data_base + 'supervision_test.pkl').set_index('QID')    
    tk = pd.read_csv(data_base + 'top1000.dev', sep='\t', names = ['QID', 'PID', 'Query', 'Passage'])
    grouped_tk = tk.groupby('QID').agg(list)
    grouped_tk['Query'] = grouped_tk['Query'].apply(lambda x: x[0])
    grouped_tk = grouped_tk.reset_index()
    ke = []
    jkws = []
    jk2g = []
    jws = []
    j2g = []
 
    for _ in range(iters):
        tt = 0
        for idx, row in grouped_tk.iterrows():
                df_l = pd.DataFrame({'QID': [row.QID], 'Query': [row.Query]})
                df_r = pd.DataFrame({'PID': row.PID, 'Passage': row.Passage})
                start = time.time()
                scores = ssj.edit_distance_join(df_l, df_r, 'QID', 'PID', 'Query', 'Passage', 100, 
                                                l_out_attrs=['Query'], r_out_attrs=['Passage'], n_jobs=-3)
                tt += time.time() - start
        ke.append(tt)

        tt = 0
        for idx, row in grouped_tk.iterrows():
                df_l = pd.DataFrame({'QID': [row.QID], 'Query': [row.Query]})
                df_r = pd.DataFrame({'PID': row.PID, 'Passage': row.Passage})
                start = time.time()
                scores = ssj.jaccard_join(df_l, df_r, 'QID', 'PID', 'Query', 'Passage', WS, 0.05, 
                              l_out_attrs=['Query'], r_out_attrs=['Passage'], n_jobs=-3)
                tt += time.time() - start
        jkws.append(tt)
        jws.append(jkws[-1])
        
        tt = 0
        for idx, row in grouped_tk.iterrows():
                df_l = pd.DataFrame({'QID': [row.QID], 'Query': [row.Query]})
                df_r = pd.DataFrame({'PID': row.PID, 'Passage': row.Passage})
                start = time.time()
                scores = ssj.jaccard_join(df_l, df_r, 'QID', 'PID', 'Query', 'Passage', TWO_GRAM, 0.05, 
                          l_out_attrs=['Query'], r_out_attrs=['Passage'], n_jobs=-3)
                tt += time.time() - start
        jk2g.append(tt)
        j2g.append(jk2g[-1])
        
    results = (ke, jkws, jk2g, jws, j2g)
    print('marco')
    for j in range(len(join_types)):
        timings['data'].append('marco')
        timings['type'].append(join_types[j])
        timings['results'].append(results[j])
        timings['avg'].append(np.mean(results[j]))
        print(timings['data'][-1], timings['type'][-1], timings['results'][-1], timings['avg'][-1])
    print()

    timings = pd.DataFrame(timings)
    timings.to_pickle(f'/lfs/1/sahaana/enrichment/ember/postprocessing/simjoin-iters_{iters}.pkl')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain MLM based on config.')
    parser.add_argument('-i', "--iters", required=True, type=int,
                        help="Number of iterations")
    args = parser.parse_args()
    time_all(args.iters)  
