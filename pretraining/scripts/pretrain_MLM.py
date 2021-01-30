import sys
import json
import argparse
from types import SimpleNamespace    

import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from transformers import DistilBertConfig, DistilBertForMaskedLM, BertConfig, BertForMaskedLM
from transformers import pipeline, Trainer, TrainingArguments

sys.path.append('/lfs/1/sahaana/enrichment/context-enrichment/utils')
from MLM_utils import DataCollatorForEnrich, Okapi25MLMDataset
from data_utils import sequential_tt_split
from file_utils import load_config


def train_MLM(config):
    conf = SimpleNamespace(**config)
    
    data_l = pd.read_pickle(conf.datapath_l)
    data_r = pd.read_pickle(conf.datapath_r)
    bm25_argsort = np.load(conf.bm25_argsort_path)
    model_out = f"/lfs/1/sahaana/enrichment/context-enrichment/pretraining/models/{conf.model_name}"
    
    # Tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(f'{conf.model_type}-base-uncased')
    data_collator = DataCollatorForEnrich(tokenizer=bert_tokenizer, 
                                          mlm=True, 
                                          mlm_probability=conf.mlm_probability,
                                          masking=conf.mlm_masking,
                                          num_seps=conf.mlm_num_seps)
    
    # Model 
    if conf.model_type == 'distilbert':
        model_config = DistilBertConfig() 
        if conf.from_scratch:
            model = DistilBertForMaskedLM(config=model_config)
        else:
            model = DistilBertForMaskedLM(config=model_config).from_pretrained(f"distilbert-base-{conf.tokenizer_casing}")
    
    elif conf.model_type == 'bert':
        model_config = BertConfig()
        if conf.from_scratch:
            model = BertForMaskedLM(config=model_config)
        else:
            model = BertForMaskedLM(config=model_config).from_pretrained(f"bert-base-{conf.tokenizer_casing}")
    
    # Training Data
    train_idx, test_idx = sequential_tt_split(len(data_l), conf.num_train, conf.num_test)
    train_data_l = data_l.iloc[train_idx]
    test_data_l = data_l.iloc[test_idx]
    train_data_r = data_r.iloc[train_idx]
    test_data_r = data_r.iloc[test_idx]
    train_bm25 = bm25_argsort[train_idx]
    test_bm25 = bm25_argsort[test_idx]

    # Training Configs
    train_dataset = Okapi25MLMDataset(train_data_l, train_data_r, 
                                      bert_tokenizer, data_col=conf.data_column, 
                                      index_bm25=False, bm25_argsort=train_bm25)
    
    training_args = TrainingArguments(output_dir=model_out,
                                      overwrite_output_dir=True,
                                      num_train_epochs=conf.train_epochs,
                                      per_device_train_batch_size=conf.batch_size,
                                      save_steps=10_000)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset)
    
    # Train and save
    trainer.train()
    trainer.save_model(model_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain MLM based on config.')
    parser.add_argument('-c', "--config", required=True,
                        help="Config file for script training")
    args = parser.parse_args()
    config = load_config(args.config)
    train_MLM(config)
