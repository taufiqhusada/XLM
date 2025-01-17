import os
import torch

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))


from xlm.utils import AttrDict
from xlm.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from xlm.model.transformer import TransformerModel

from collections import OrderedDict

from xlm_indo_nlu_utils.data_loader_utils import AspectBasedSentimentAnalysisProsaDataset, AspectBasedSentimentAnalysisDataLoader

import random

import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import absa_metrics_fn

from xlm_indo_nlu_utils.model_utils import forward_sequence_multi_classification
from torch import nn

import argparse


NUM_LABELS = [3, 3, 3, 3, 3, 3]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Proj(nn.Module):
    def __init__(self):
        super(Proj, self).__init__()
        self.dropout = nn.Dropout(params.dropout)

        self.pooler = nn.Sequential(nn.Linear(1024, 1024), nn.Tanh())
        self.classifiers = nn.ModuleList([nn.Linear(1024, num_label) for num_label in NUM_LABELS])

    def forward(self, x):
        sequence_output = self.dropout(self.pooler(x))
        logits = []
        for classifier in self.classifiers:
            logits.append(classifier(sequence_output))
        return logits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/projectnb/statnlp/gkuwanto/XLM/dumped/baseline_para_0/q3v4i6kl9t/best-valid_mlm_ppl.pth", help="Model path")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of sentences per batch")
    parser.add_argument("--lr_optimizer_e", type=float, default=0.00001, help="LR Embedder (pretrained model) optimizer")
    parser.add_argument("--lr_optimizer_p", type=float, default=0.00001, help="LR Projection (classifier) optimizer")
    parser.add_argument("--n_epochs", type=int, default=25, help="Maximum number of epochs")
    
    set_seed(42)

    custom_params = parser.parse_args()

    model_path = custom_params.model_path
    reloaded = torch.load(model_path)
    params = AttrDict(reloaded['params'])
    print("Supported languages: %s" % ", ".join(params.lang2id.keys()))
    
    # build dictionary / update parameters
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(dico)
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)

    # build model / reload weights
    model = TransformerModel(params, dico, True, True)
    model.eval()

    reloaded_model = OrderedDict()
    for k, v in reloaded['model'].items():
          reloaded_model[k.replace('module.', '')] = v
    model.load_state_dict(reloaded_model)
    
    
    train_dataset_path = '/projectnb/statnlp/gik/XLM/IndoNLU/data/casa_absa-prosa/train_preprocess_bpe.csv'
    valid_dataset_path = '/projectnb/statnlp/gik/XLM/IndoNLU/data/casa_absa-prosa/valid_preprocess_bpe.csv'
    test_dataset_path = '/projectnb/statnlp/gik/XLM/IndoNLU/data/casa_absa-prosa/test_preprocess_masked_label_bpe.csv'
    
    train_dataset = AspectBasedSentimentAnalysisProsaDataset(train_dataset_path, dico, params, lowercase=True)
    valid_dataset = AspectBasedSentimentAnalysisProsaDataset(valid_dataset_path, dico, params, lowercase=True)
    test_dataset = AspectBasedSentimentAnalysisProsaDataset(test_dataset_path,dico, params, lowercase=True)

    train_loader = AspectBasedSentimentAnalysisDataLoader(dataset=train_dataset, params = params, max_seq_len=512, batch_size=custom_params.batch_size, num_workers=4, shuffle=True)  
    valid_loader = AspectBasedSentimentAnalysisDataLoader(dataset=valid_dataset,  params = params, max_seq_len=512, batch_size=custom_params.batch_size, num_workers=4, shuffle=False)  
    test_loader = AspectBasedSentimentAnalysisDataLoader(dataset=test_dataset,  params = params, max_seq_len=512, batch_size=custom_params.batch_size, num_workers=4, shuffle=False)
    
    w2i, i2w = AspectBasedSentimentAnalysisProsaDataset.LABEL2INDEX, AspectBasedSentimentAnalysisProsaDataset.INDEX2LABEL
    print(w2i)
    print(i2w)
    
    proj = Proj()
    
    optimizer_m = optim.Adam(model.parameters(), lr=custom_params.lr_optimizer_e)
    model = model.cuda()
    optimizer_p = optim.Adam(proj.parameters(),lr=custom_params.lr_optimizer_p)
    proj = proj.cuda()
    
    n_epochs = custom_params.n_epochs

    for epoch in range(n_epochs):
        model.train()
        proj.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0

        list_hyp, list_label = [], []

        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward model
            loss, logits, batch_hyp, batch_label = forward_sequence_multi_classification(proj, model, batch_data[:-1], i2w=i2w, num_labels = NUM_LABELS, device='cuda')
    #         print(loss)

            optimizer_m.zero_grad()
            optimizer_p.zero_grad()
            loss.backward()
            optimizer_m.step()
            optimizer_p.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            list_hyp += batch_hyp
            list_label += batch_label

            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f}".format((epoch+1),
                total_train_loss/(i+1)))


        # Calculate train metric
        metrics = absa_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {}".format((epoch+1),
            total_train_loss/(i+1),metrics))


        # Evaluate on validation
        model.eval()
        proj.eval()
        torch.set_grad_enabled(False)

        total_loss, total_correct, total_labels = 0, 0, 0

        list_hyp, list_label = [], []

        pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]        
            loss, logits, batch_hyp, batch_label = forward_sequence_multi_classification(proj, model, batch_data[:-1], i2w=i2w, num_labels = NUM_LABELS, device='cuda')

            # Calculate total loss
            valid_loss = loss.item()
            total_loss = total_loss + valid_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label

            pbar.set_description("VALID LOSS:{:.4f}".format(total_loss/(i+1)))

        metrics = absa_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
            total_loss/(i+1), metrics))
        
        
    # Evaluate on test
    model.eval()
    proj.eval()
    torch.set_grad_enabled(False)

    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        loss, logits, batch_hyp, batch_label = forward_sequence_multi_classification(proj, model, batch_data[:-1], i2w=i2w, num_labels = NUM_LABELS, device='cuda')
        list_hyp += batch_hyp

    # Save prediction
    df = pd.DataFrame({'label':list_hyp}).reset_index()
    # df.to_csv('pred.txt', index=False)

    print(df['label'])

    df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/output/pred-casa.csv', index=False)

    torch.save(model.state_dict(), '/projectnb/statnlp/gik/XLM/IndoNLU/output/casa_xlm_finetuned_model.pth')
    torch.save(proj.state_dict(), '/projectnb/statnlp/gik/XLM/IndoNLU/output/casa_proj.pth')
    
    
    
    
    
    
    
