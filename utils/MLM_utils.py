from dataclasses import dataclass
from typing import List, Union, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from pandarallel import pandarallel
from rank_bm25 import BM25Okapi

import torch 
from transformers import PreTrainedTokenizerBase, BatchEncoding

from data_utils import sample_excluding


@dataclass
class DataCollatorForEnrich:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    num_seps: int = 1 
    masking: str = "ALL"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        #print(batch["input_ids"])
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # alter special_tokens_mask so that we only mask from the specified region (before/after num_seps)
        # number of sep tokens. This is so we can mask either before or after a table break as needed
        # This will find all the sep tokens in each record, find the num_seps'th one, and make sure we
        # only mask from self.masking (before/after) that point. 
        if self.masking != "ALL":
            for ii, record in enumerate(inputs):
                idx = torch.nonzero(record == self.tokenizer.sep_token_id, as_tuple=False)[self.num_seps].item()
                if self.masking == "BEFORE":
                    special_tokens_mask[ii, idx:] = True
                elif self.masking == "AFTER":
                    special_tokens_mask[ii, :idx] = True
                
        #print(inputs)
        #print()
        #print(special_tokens_mask)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class DM_Okapi25MLMDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df_l: pd.DataFrame, 
                 df_r: pd.DataFrame,
                 tokenizer: PreTrainedTokenizerBase,
                 n_pairs_multiplier: int = 3,
                 data_col = 'merged',
                 max_len:int = 512, 
                 index_bm25 = True,
                 bm25_argsort = None):
        self.df_l = df_l.copy()[data_col]
        self.df_r = df_r.copy()[data_col]
        self.bm25_argsort = bm25_argsort 
        self.tokenizer = tokenizer
        self.n_pairs = len(df_l) * n_pairs_multiplier
        self.pairs = self.gen_pairs(n_pairs_multiplier) #needs to be after init_okapi25()
        self.max_len = max_len
        
    def gen_pairs(self, n_pairs_multiplier):
        pairs = []
        for i in range(len(self.df_l)):
            idx_l = i
            for j in range(1, 1 + n_pairs_multiplier):
                idx_r = self.bm25_argsort[i, -j]
                pairs.append([idx_l, idx_r])
        return np.array(pairs)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        l, r = self.pairs[idx]
        return {'input_ids': self.tokenizer(self.df_l[l] + ' [SEP] ' + self.df_r[r], 
                                            max_length=self.max_len, truncation=True)['input_ids']}    
    
class Okapi25MLMDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df_l: pd.DataFrame, 
                 df_r: pd.DataFrame,
                 tokenizer: PreTrainedTokenizerBase,
                 n_pairs_multiplier: int = 3,
                 data_col = 'merged',
                 max_len:int = 512, 
                 index_bm25 = True,
                 bm25_argsort = None):
        self.df_l = df_l.copy()[data_col]
        self.df_r = df_r.copy()[data_col]
        self.bm25_argsort = self.init_okapi25() if index_bm25 else bm25_argsort 
        self.tokenizer = tokenizer
        self.n_pairs = len(df_l) * n_pairs_multiplier
        self.pairs = self.gen_pairs(n_pairs_multiplier) #needs to be after init_okapi25()
        self.max_len = max_len
        
        
    def init_okapi25():
        corpus = list(self.df_r[data_col].apply(lambda x: x.split()))
        indexed = BM25Okapi(corpus)
        pandarallel.initialize()
        bm25 = self.df_r[data_col].parallel_apply(lambda x: indexed.get_scores(x.split()))
        return np.argsort(bm25,axis=1)
        
    def gen_pairs(self, n_pairs_multiplier):
        pairs = []
        for i in range(len(self.df_l)):
            idx_l = i
            offset = 0
            for j in range(1, 1 + n_pairs_multiplier):
                idx_r = self.bm25_argsort[i, -(j + offset)]
                while idx_r > len(self.df_l):
                    offset += 1
                    idx_r = self.bm25_argsort[i, -(j + offset)]
                pairs.append([idx_l, idx_r])
        return np.array(pairs)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        l, r = self.pairs[idx]
        return {'input_ids': self.tokenizer(self.df_l[l] + ' [SEP] ' + self.df_r[r], 
                                            max_length=self.max_len, truncation=True)['input_ids']}


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df_l: pd.DataFrame, 
                 df_r: pd.DataFrame,
                 tokenizer: PreTrainedTokenizerBase,
                 n_pairs_multiplier: int = 3,
                 data_col = 'merged',
                 max_len:int = 512):
        self.df_l = df_l.copy()[data_col]
        self.df_r = df_r.copy()[data_col]
        self.tokenizer = tokenizer
        self.n_pairs = len(df_l) * n_pairs_multiplier
        self.pairs = self.gen_pairs(n_pairs_multiplier)
        self.max_len = max_len
    
    def gen_pairs(self, n_pairs_multiplier):
        pairs = []
        num_points = int(self.n_pairs/n_pairs_multiplier)
        for i in range(num_points):
            idx_a = i
            idx_p = i
            pairs.append([idx_a, idx_p])
            seen_negatives = [i]
            for _ in range(n_pairs_multiplier):
                idx_n = sample_excluding(num_points, seen_negatives)
                seen_negatives.append(idx_n)
                pairs.append([idx_a, idx_n])
        return np.array(pairs)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        l, r = self.pairs[idx]
        return {'input_ids': self.tokenizer(self.df_l[l] + self.df_r[r], 
                                            max_length=self.max_len, truncation=True)['input_ids']}
