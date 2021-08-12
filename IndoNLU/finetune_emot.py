import os
import torch

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))


from xlm.utils import AttrDict
from xlm.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from xlm.model.transformer import TransformerModel

from collections import OrderedDict

from xlm_indo_nlu_utils.data_loader_utils import EmotionDetectionDataset, EmotionDetectionDataLoader

import random

import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import document_sentiment_metrics_fn

from xlm_indo_nlu_utils.model_utils import forward_sequence_classification
from torch import nn

import argparse


NUM_LABELS = 5

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_bpe(sentences):
    # write sentences to tmp file
    with open('/tmp/sentences.bpe', 'w') as fwrite:
        for sent in sentences:
            fwrite.write(sent + '\n')
    
    # apply bpe to tmp file
    os.system('%s applybpe /tmp/sentences.bpe /tmp/sentences %s' % (fastbpe, codes))
    
    # load bpe-ized sentences
    sentences_bpe = []
    with open('/tmp/sentences.bpe') as f:
        for line in f:
            sentences_bpe.append(line.rstrip())
    
    return sentences_bpe


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
    

    reloaded_model = OrderedDict()
    for k, v in reloaded['model'].items():
          reloaded_model[k.replace('module.', '')] = v
    model.load_state_dict(reloaded_model)
    
    model.eval()
    
    # Below is one way to bpe-ize sentences
    codes = "" # path to the codes of the model
    fastbpe = os.path.join(os.getcwd(), 'tools/fastBPE/fast')
    
    # Below are already BPE-ized sentences

    sentences = [
         'warung ini dimiliki oleh pengusaha pabrik tahu yang sudah puluhan tahun terkenal membuat tahu putih di bandung . tahu berkualitas , dipadu keahlian memasak , dipadu kretivitas , jadilah warung yang menyajikan menu utama berbahan tahu , ditambah menu umum lain seperti ayam . semuanya selera indonesia . harga cukup terjangkau . jangan lewatkan tahu bletoka nya , tidak kalah dengan yang asli dari tegal !',
    ]

    # bpe-ize sentences
    sentences = to_bpe(sentences)

    # check how many tokens are OOV
    n_w = len([w for w in ' '.join(sentences).split()])
    n_oov = len([w for w in ' '.join(sentences).split() if w not in dico.word2id])

    # add </s> sentence delimiters
    sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]
    
    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent in sentences])

    langs = None
    tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
    print(tensor.size()[-1])
    
    model_output_size = tensor.size()[-1]
    proj = nn.Sequential(*[
        nn.Dropout(params.dropout),
        nn.Linear(model_output_size, NUM_LABELS)
    ]).cuda()
    
    
    train_dataset_path = './dataset/emot_emotion-twitter/train_preprocess.csv'
    valid_dataset_path = './dataset/emot_emotion-twitter/valid_preprocess.csv'
    test_dataset_path = './dataset/emot_emotion-twitter/test_preprocess_masked_label.csv'
    
    train_dataset = EmotionDetectionDataset(train_dataset_path, dico, params, lowercase=True)
    valid_dataset = EmotionDetectionDataset(valid_dataset_path, dico, params, lowercase=True)
    test_dataset = EmotionDetectionDataset(test_dataset_path,dico, params, lowercase=True)

    train_loader = EmotionDetectionDataLoader(dataset=train_dataset, params=params, max_seq_len=512, batch_size=custom_params.batch_size, num_workers=1, shuffle=True)  
    valid_loader = EmotionDetectionDataLoader(dataset=valid_dataset, params=params, max_seq_len=512, batch_size=custom_params.batch_size, num_workers=1, shuffle=False)  
    test_loader = EmotionDetectionDataLoader(dataset=test_dataset, params=params, max_seq_len=512, batch_size=custom_params.batch_size, num_workers=1, shuffle=False)
    
    w2i, i2w = EmotionDetectionDataset.LABEL2INDEX, EmotionDetectionDataset.INDEX2LABEL
    print(w2i)
    print(i2w)
    
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
            loss, logits, batch_hyp, batch_label = forward_sequence_classification(proj, model, batch_data[:-1], i2w=i2w, device='cuda')
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
        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
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
            loss, logits, batch_hyp, batch_label = forward_sequence_classification(proj, model, batch_data[:-1], i2w=i2w, device='cuda')

            # Calculate total loss
            valid_loss = loss.item()
            total_loss = total_loss + valid_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label

            pbar.set_description("VALID LOSS:{:.4f}".format(total_loss/(i+1)))

        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
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
        loss, logits, batch_hyp, batch_label = forward_sequence_classification(proj, model, batch_data[:-1], i2w=i2w, device='cuda')
        list_hyp += batch_hyp

    # Save prediction
    df = pd.DataFrame({'label':list_hyp}).reset_index()
    # df.to_csv('pred.txt', index=False)

    print(df['label'].value_counts())

    df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/output/pred-emot.csv', index=False)

    torch.save(model.state_dict(), '/projectnb/statnlp/gik/XLM/IndoNLU/output/emot_xlm_finetuned_model.pth')
    torch.save(proj.state_dict(), '/projectnb/statnlp/gik/XLM/IndoNLU/output/emot_proj.pth')
    
    
    
    
    
    
    
