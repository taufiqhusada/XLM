import os, sys
sys.path.append('/projectnb/statnlp/gkuwanto/indonlu/')
os.chdir('/projectnb/statnlp/gkuwanto/indonlu/')

import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


class DocumentSentimentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
    INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'}
    NUM_LABELS = 3
    
    def load_dataset(self, path): 
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = ['text','sentiment']
        df['sentiment'] = df['sentiment'].apply(lambda lab: self.LABEL2INDEX[lab])
        return df
    
    def __init__(self, dataset_path, dico, params, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.dico = dico
        self.params = params
        
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, sentiment = data['text'], data['sentiment']
        subwords = torch.LongTensor([self.dico.index(w) for w in text])
      
        return subwords, sentiment, text
    
    def __len__(self):
        return len(self.data)    
        
class DocumentSentimentDataLoader(DataLoader):
    def __init__(self,params,  max_seq_len=512, *args, **kwargs):
        super(DocumentSentimentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        self.params = params
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        sentiment_batch = np.zeros((batch_size, 1), dtype=np.int64)
        
        seq_list = []
        lengths = []

        word_ids = torch.LongTensor(max_seq_len, batch_size).fill_(self.params.pad_index)
        for i, (subwords, sentiment, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            word_ids[:len(subwords), i] = subwords
            sentiment_batch[i,0] = sentiment
            seq_list.append(raw_seq)
            
            lengths.append(len(subwords))
            
            
        lengths = torch.LongTensor(lengths)
        
        langs = None
         
        return word_ids, sentiment_batch, lengths, langs, seq_list