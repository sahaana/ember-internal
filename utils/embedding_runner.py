import numpy as np

import wandb
import torch
from embedding_utils import tokenize_single_data_batch, save_torch_model, to_cuda

def get_cuda_levels(batch_tokenizer):
    levels = {'tokenize_columns': 2,
              'tokenize_batch': 1, 
              'tokenize_string_categorical': 0,
              'tokenize_single_data_batch': 0}
    return levels[batch_tokenizer.__name__]

def train_emb_model(model, 
                    tokenizer,
                    batch_tokenizer, 
                    train_data, 
                    loss_func, 
                    optimizer, 
                    epochs, 
                    save_dir,
                    tokenizer_max_length = 512,
                    train_model = True):
    
    levels = get_cuda_levels(batch_tokenizer)
    model.cuda()
    for epoch in range(epochs):
        if train_model:
            wandb.log({"Epoch": epoch}) 
            model.train()

            for i, d in enumerate(train_data):
                batch = batch_tokenizer(d, tokenizer, tokenizer_max_length)
                inputs, masks = to_cuda(batch, levels=levels)
                a, p, n = inputs
                a_mask, p_mask, n_mask = masks
                optimizer.zero_grad()
                oa, op, on = model(a, p, n, a_mask, p_mask, n_mask)
                loss = loss_func(oa, op, on)
                loss.backward()
                optimizer.step()

                if (i % 100) == 0 : 
                    wandb.log({"train batch loss": loss.item()})
                if (i % 20000 == 0):
                    save_torch_model(save_dir, model)
        last_saved = save_torch_model(save_dir, model)
    return last_saved

def train_model(model, 
                tokenizer,
                batch_tokenizer, 
                train_data, 
                val_data, 
                loss_func, 
                optimizer, 
                epochs, 
                losses, 
                val_losses, 
                save_dir,
                compute_val = True, 
                tokenizer_max_length = 512):
    
    levels = get_cuda_levels(batch_tokenizer)
    model.cuda()
    for epoch in range(epochs):
        wandb.log({"Epoch": epoch}) 
        model.train()
        
        for i, d in enumerate(train_data):
            batch = batch_tokenizer(d, tokenizer, tokenizer_max_length)
            inputs, masks = to_cuda(batch, levels=levels)
            a, p, n = inputs
            a_mask, p_mask, n_mask = masks
            optimizer.zero_grad()
            oa, op, on = model(a, p, n, a_mask, p_mask, n_mask)
            loss = loss_func(oa, op, on)
            loss.backward()
            optimizer.step()
        
            wandb.log({"train batch loss": loss.item()})
            if i % 100 == 0:
                losses.append(loss.item())
        
        save_torch_model(save_dir, model)
        
        if compute_val:        
            model.eval()
            with torch.no_grad():
                v_loss = 0

                for i, d in enumerate(val_data):
                    batch = batch_tokenizer(d, tokenizer)
                    inputs, masks = to_cuda(batch, levels=levels)
                    a, p, n = inputs
                    a_mask, p_mask, n_mask = masks 
                    oa, op, on = model(a, p, n, a_mask, p_mask, n_mask) 
                    loss = loss_func(oa, op, on)
                    v_loss += loss.item()
                wandb.log({"val loss": v_loss})
                val_losses.append(v_loss)
                
                
def eval_model(model, 
               tokenizer,
               data,
               tokenizer_max_length = 512, 
               mode = None): #mode can be right or left too, but make sure to only use it with the double tower models
    embeddings = []
    model.eval()
    model.cuda()
    
    if mode == 'LEFT':
        emb_return = model.return_emb_l
    elif mode == 'RIGHT':
        emb_return = model.return_emb_r
    else:
        emb_return = model.return_emb
    for i, d in enumerate(data):
        batch = tokenize_single_data_batch(d, tokenizer, tokenizer_max_length)
        inputs, masks = to_cuda(batch, levels=0)
        out = emb_return(inputs, masks)
        embeddings.append(out.cpu().detach().numpy())
    embeddings = np.vstack(embeddings)
    if data.dataset.indexed:
        return np.array(data.dataset.index), embeddings
    return embeddings