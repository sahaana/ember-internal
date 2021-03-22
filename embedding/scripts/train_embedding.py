import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from types import SimpleNamespace   
from collections import defaultdict



import numpy as np
import pandas as pd

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertModel

sys.path.append('/lfs/1/sahaana/enrichment/ember/utils')
from embedding_datasets import IMDBWikiDataset, SQuADDataset, MARCODataset, DeepMatcherDataset, IMDBFuzzyDataset, EmberEvalDataset, DMBlockedDataset, DMHardNegativeBlockedDataset, IMDBWikiHardNegativeDataset, IMDBFuzzyHardNegativeDataset, SQuADRandomDataset
from embedding_models import TripletSingleBERTModel, TripletDoubleBERTModel, PreTrainedBERTModel
from embedding_utils import param_header, tokenize_batch  
from embedding_runner import train_emb_model, eval_model
from knn_utils import FaissKNeighbors, compute_top_k_pd, knn_IMDB_wiki_recall, knn_SQuAD_sent_recall, knn_MARCO_recall, knn_deepmatcher_recall, knn_IMDB_fuzzy_recall, knn_DM_blocked_recall  
from file_utils import load_config

dataset = { 
            'imdb_wiki': IMDBWikiDataset,
            'SQuAD_sent': SQuADDataset,
            'MSMARCO': MARCODataset,
            'deepmatcher': DeepMatcherDataset,
            'small_imdb_fuzzy': IMDBFuzzyDataset,
            'hard_imdb_fuzzy': IMDBFuzzyDataset,
            'main_fuzzy': IMDBFuzzyDataset,
            'hard_fuzzy': IMDBFuzzyDataset,
            'easy_fuzzy': IMDBFuzzyDataset,
            'dm_blocked': DMBlockedDataset
          }

HNdataset = { 
                'imdb_wiki': IMDBWikiHardNegativeDataset,
                #'MSMARCO': MARCODataset,
                #'deepmatcher': DeepMatcherDataset,
                #'small_imdb_fuzzy': IMDBFuzzyDataset,
                #'hard_imdb_fuzzy': IMDBFuzzyDataset,
                'main_fuzzy': IMDBFuzzyHardNegativeDataset,
                'hard_fuzzy': IMDBFuzzyHardNegativeDataset,
                'easy_fuzzy': IMDBFuzzyHardNegativeDataset,
                'dm_blocked': DMHardNegativeBlockedDataset
             }

randDataset = {'SQuAD_sent': SQuADRandomDataset}


knn_routine = {
                'imdb_wiki': knn_IMDB_wiki_recall,  #
                'SQuAD_sent': knn_SQuAD_sent_recall, #
                'MSMARCO': knn_MARCO_recall, #
                'deepmatcher': knn_deepmatcher_recall,
                'small_imdb_fuzzy': knn_IMDB_fuzzy_recall, #
                'hard_imdb_fuzzy': knn_IMDB_fuzzy_recall, #
                'main_fuzzy': knn_IMDB_fuzzy_recall, #
                'hard_fuzzy': knn_IMDB_fuzzy_recall, #
                'easy_fuzzy': knn_IMDB_fuzzy_recall, #
                'dm_blocked': knn_DM_blocked_recall, #
              }

model_arch = {
                'single-triplet': TripletSingleBERTModel, 
                'double-triplet': TripletDoubleBERTModel,
                'pretrained': PreTrainedBERTModel
             }

model_train = {
                'single-triplet': True, 
                'double-triplet': True,
                'pretrained': False
             }

def train_embedding(config):
    conf = SimpleNamespace(**config)
    if conf.data in ['imdb_wiki', 'SQuAD_sent', 'MSMARCO', 'deepmatcher', 'small_imdb_fuzzy', 'hard_imdb_fuzzy',
                     'main_fuzzy', 'hard_fuzzy', 'easy_fuzzy', 'dm_blocked']:
        left = pd.read_pickle(conf.datapath_l)
        right = pd.read_pickle(conf.datapath_r)
        train_supervision = pd.read_pickle(conf.train_supervision)
    if conf.data == 'MSMARCO':
        left = left.set_index('QID')
        right = right.set_index('PID')
        
    tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer)
    #bert_model = DistilBertModel.from_pretrained(conf.bert_path, return_dict=True)
         
    if 'negatives' in conf.model_name:
        train_data =  DataLoader(HNdataset[conf.data](left, right, conf.train_size, conf.column, train_supervision, params = {'negatives':conf.negatives}), 
                            batch_size=conf.batch_size,
                            shuffle = True
                            )  

    elif 'random' in conf.model_name:
        train_data =  DataLoader(randDataset[conf.data](left, right, conf.train_size, conf.column, train_supervision), 
                            batch_size=conf.batch_size,
                            shuffle = True
                            )    
    else: 
        train_data = DataLoader(dataset[conf.data](left, right, conf.train_size, conf.column, train_supervision), 
                                batch_size=conf.batch_size,
                                shuffle = True
                                )
    
    if conf.loss == 'triplet':
        loss = nn.TripletMarginLoss(margin=conf.tl_margin, p=conf.tl_p)
        model = model_arch[conf.arch](final_size = conf.final_size, pooling = conf.pool_type, bert_path = conf.bert_path)
        optimizer = optim.AdamW(model.parameters(), lr=conf.lr)#optim.SGD(model.parameters(), lr=lr)
    
    save_dir = param_header(conf.batch_size, conf.final_size, conf.lr, conf.pool_type, conf.epochs, conf.train_size)
    save_dir = f'models/{conf.model_name}/{save_dir}/'
    wandb.init(project=conf.model_name)
    
    print("Training Begins")
    last_saved = train_emb_model(model, 
                                 tokenizer, 
                                 tokenize_batch, 
                                 train_data, 
                                 loss, 
                                 optimizer, 
                                 conf.epochs, 
                                 save_dir, 
                                 conf.tokenizer_max_length,
                                 model_train[conf.arch])
    
    return last_saved

def perform_knn(config, latest_model_path):
    conf = SimpleNamespace(**config)
    
    model = model_arch[conf.arch](conf.final_size, conf.pool_type, conf.bert_path)
    model.load_state_dict(torch.load(latest_model_path))
    
    if conf.data in ['imdb_wiki', 'SQuAD_sent', 'MSMARCO', 'deepmatcher', 'small_imdb_fuzzy', 'hard_imdb_fuzzy',
                     'main_fuzzy', 'hard_fuzzy', 'easy_fuzzy', 'dm_blocked']:
        left = pd.read_pickle(conf.eval_datapath_l)
        right = pd.read_pickle(conf.eval_datapath_r)
        test_supervision = pd.read_pickle(conf.test_supervision)
    if conf.data == 'MSMARCO':
        left = left.set_index('QID')
        right = right.set_index('PID')
    
    tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer)   
    left_eval_data = DataLoader(EmberEvalDataset(left, conf.column, indexed=True), 
                                batch_size=conf.batch_size,
                                shuffle = False
                               )
    right_eval_data = DataLoader(EmberEvalDataset(right, conf.column, indexed=True), 
                                 batch_size=conf.batch_size,
                                 shuffle = False
                                )
    if conf.arch == 'double-triplet':
        left_index, left_embeddings = eval_model(model, tokenizer, left_eval_data, conf.tokenizer_max_length, mode='LEFT')
        right_index, right_embeddings = eval_model(model, tokenizer, right_eval_data, conf.tokenizer_max_length, mode='RIGHT')
    else:
        left_index, left_embeddings = eval_model(model, tokenizer, left_eval_data, conf.tokenizer_max_length)
        right_index, right_embeddings = eval_model(model, tokenizer, right_eval_data, conf.tokenizer_max_length)
    
    knn = FaissKNeighbors(k=conf.knn_k)
    knn.fit(right_embeddings)
    neib = knn.kneighbors(left_embeddings)
    
    knn_routine_params = (neib[0], neib[1], test_supervision, left_index, right_index)    
    knn_results = compute_top_k_pd(knn_routine[conf.data], knn_routine_params, k_max=conf.knn_k, thresh=None)   
    
    """knn_results = defaultdict(list)
    for k in range(1,conf.knn_k + 1):
        avg, count, MRR, results, MRR_results = knn_routine[conf.data](neib[0], neib[1], test_supervision, 
                                                                       left_index, right_index, k=k, thresh=None)
        print(f"k: {k} \t avg: {avg} \t count: {count} \t MRR: {MRR}")
        knn_results['k'].append(k)
        knn_results['avg'].append(avg)
        knn_results['count'].append(count)
        knn_results['MRR'].append(MRR)
        knn_results['results'].append(results)
        knn_results['MRR_results'].append(MRR_results)
    knn_results = pd.DataFrame(knn_results)"""
    
    save_dir = param_header(conf.batch_size, conf.final_size, conf.lr, conf.pool_type, conf.epochs, conf.train_size)
    save_dir = f'models/{conf.model_name}/{save_dir}/knn/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime('%H-%M-%d-%m-%y')
    save_path = f"{save_dir}/{timestamp}_knn_results.pkl"
    knn_results.to_pickle(save_path)
    print(f"Saved Results: {save_path}")
    
    embedding_out = {"left_index": left_index,
                     "left_embeddings": left_embeddings,
                     "right_index": right_index,
                     "right_embeddings": right_embeddings}
    
    with open(f'{save_dir}/{timestamp}_embeddings.pkl', 'wb') as handle:
        pickle.dump(embedding_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain MLM based on config.')
    parser.add_argument('-c', "--config", required=True,
                        help="Config file for script training")
    args = parser.parse_args()
    config = load_config(args.config)
    model = train_embedding(config)
    perform_knn(config, model)
