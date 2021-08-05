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
        text = '</s> ' + text + ' </s>'
        subwords = torch.LongTensor([self.dico.index(w) for w in text])
      
        return subwords, np.array(sentiment), text
    
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
    
    
#####
# Emotion Twitter
#####
class EmotionDetectionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'sadness': 0, 'anger': 1, 'love': 2, 'fear': 3, 'happy': 4}
    INDEX2LABEL = {0: 'sadness', 1: 'anger', 2: 'love', 3: 'fear', 4: 'happy'}
    NUM_LABELS = 5
    
    def load_dataset(self, path):
        # Load dataset
        dataset = pd.read_csv(path)
        dataset['label'] = dataset['label'].apply(lambda sen: self.LABEL2INDEX[sen])
        return dataset

    def __init__(self, dataset_path, dico, params, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.dico = dico
        self.params = params
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        tweet, label = data['tweet'], data['label']
        tweet = '</s> ' + tweet + ' </s>'
        subwords = torch.LongTensor([self.dico.index(w) for w in tweet])
      
        return subwords, np.array(label), tweet
    
    def __len__(self):
        return len(self.data)
        
class EmotionDetectionDataLoader(DataLoader):
    def __init__(self,params,  max_seq_len=512, *args, **kwargs):
        super(EmotionDetectionDataLoader, self).__init__(*args, **kwargs)
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
    
 #####
# Entailment UI
#####
class EntailmentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'NotEntail': 0, 'Entail_or_Paraphrase': 1}
    INDEX2LABEL = {0: 'NotEntail', 1: 'Entail_or_Paraphrase'}
    NUM_LABELS = 2
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        df['label'] = df['label'].apply(lambda label: self.LABEL2INDEX[label])
        return df
    
    def __init__(self, dataset_path, dico, params, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.dico = dico
        self.params = params
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sent_A, sent_B, label = data['sent_A'], data['sent_B'], data['label']
        
        pair_sentence = '</s> ' + sent_A + ' <pad> ' + sent_B + ' </s>'
        subwords = torch.LongTensor([self.dico.index(w) for w in pair_sentence])
      
        return subwords, np.array(label), pair_sentence
    
    def __len__(self):
        return len(self.data)
    
        
class EntailmentDataLoader(DataLoader):
    def __init__(self,params,  max_seq_len=512, *args, **kwargs):
        super(EntailmentDataLoader, self).__init__(*args, **kwargs)
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