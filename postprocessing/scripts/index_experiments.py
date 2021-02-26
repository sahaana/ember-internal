import pandas as pd
import numpy as np
import argparse

import matplotlib.pyplot as plt
from collections import defaultdict

import os
import sys
import time

sys.path.append('/lfs/1/sahaana/enrichment/ember/utils')
from file_utils import load_config, get_sorted_files, get_config_knn_dir, get_alpha_sorted_files
from embedding_utils import param_header
from knn_utils import FaissKNeighbors


base_path = "/lfs/1/sahaana/enrichment/ember/embedding"
config_base = "/lfs/1/sahaana/enrichment/ember/embedding/configs/"



def time_all(iters=25):
    timings = defaultdict(list)
    for i in get_alpha_sorted_files(config_base):
        #if 'deepmatcher' in i:# or 'imdb' in i and 'OLD' not in i:
        if 'json' not in i or "MARCO" in i or "company" in i or 'deepmatcher' in i or 'OLD' in i: 
            continue
        conf = load_config(i)
        config_knn = get_config_knn_dir(i)

        print(conf['data'])
        print(conf['arch'], "\t", conf['bert_path'])
        print(i)
        files = get_sorted_files(config_knn)
        for j in files:
            if 'embeddings.pkl' in j:
                results = pd.read_pickle(j)
                right = results['right_embeddings']
                left = results['left_embeddings']

                time_left_index = []
                time_right_index = []
                
                #index left
                for _ in range(iters):
                    start = time.time()
                    knn = FaissKNeighbors(k=50)
                    knn.fit(left)
                    neib = knn.kneighbors(right)
                    time_left_index.append(time.time() - start)
                print("Indexing Left: ", np.mean(time_left_index))

                #index right
                for _ in range(iters):
                    start = time.time()
                    knn = FaissKNeighbors(k=50)
                    knn.fit(right)
                    neib = knn.kneighbors(left)
                    time_right_index.append(time.time() - start)
                print("Indexing Right: ", np.mean(time_right_index))
                print()
                print()
                break
        timings['indexing_left'].append(time_left_index.copy())
        timings['indexing_right'].append(time_right_index.copy())
        timings['arch'].append(conf['arch'])
        timings['data'].append(conf['data'])
        timings['model'].append(conf['bert_path'])
        timings['indexing_left_avg'].append(np.mean(time_left_index))
        timings['indexing_right_avg'].append(np.mean(time_right_index))

    timings = pd.DataFrame(timings)
    timings.to_pickle('/lfs/1/sahaana/enrichment/ember/postprocessing/indexing.pkl')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain MLM based on config.')
    parser.add_argument('-i', "--iters", required=True, type=int,
                        help="Number of iterations")
    args = parser.parse_args()
    time_all(args.iters)  
