import os
import pandas as pd

codes = "/projectnb/statnlp/gkuwanto/XLM/data/processed/en-id/codes" # path to the codes of the model
fastbpe = os.path.join('/projectnb/statnlp/gik/XLM', 'tools/fastBPE/fast')
print(fastbpe)
def to_bpe(sent):
    # write sentences to tmp file
    with open('/projectnb/statnlp/gik/XLM/sentences.bpe', 'w') as fwrite:
         fwrite.write(sent + '\n')
        
    # apply bpe to tmp file
    os.system('%s applybpe /projectnb/statnlp/gik/XLM/sentences_modified.bpe /projectnb/statnlp/gik/XLM/sentences.bpe %s' % (fastbpe, codes))
    
    # load bpe-ized sentences
    sentences_bpe = []
    with open('/projectnb/statnlp/gik/XLM/sentences_modified.bpe') as f:
        for line in f:
            sentences_bpe.append(line.rstrip())
    return sentences_bpe[0]

indo_collex_path = '/projectnb/statnlp/gik/indo-collex/dict/inforformal-formal-Indonesian-dictionary.tsv'
df_indo_collex = pd.read_csv(indo_collex_path, sep='\t')
dict_informal_to_formal = {}
for i,row in df_indo_collex.iterrows():
    dict_informal_to_formal[row['informal']] = row['formal']
print(len(dict_informal_to_formal))


### EMOT ###
train_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/emot_emotion-twitter/train_preprocess.csv'
valid_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/emot_emotion-twitter/valid_preprocess.csv'
test_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/emot_emotion-twitter/test_preprocess_masked_label.csv'

train = pd.read_csv(train_dataset_path)
for i,row in train.iterrows():
    sent = row['tweet']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['tweet'] = to_bpe(sent)   
train.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/emot_emotion-twitter/train_preprocess_bpe.csv', index=False)

valid = pd.read_csv(valid_dataset_path)
for i,row in valid.iterrows():
    sent = row['tweet']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['tweet'] = to_bpe(sent)  
valid.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/emot_emotion-twitter/valid_preprocess_bpe.csv', index=False)

test = pd.read_csv(test_dataset_path)
for i,row in test.iterrows():
    sent = row['tweet']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['tweet'] = to_bpe(sent)  
test.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/emot_emotion-twitter/test_preprocess_masked_label_bpe.csv', index=False)
print('done preprocess emot data')

### SMSA  ###
train_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv'
valid_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv'
test_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/smsa_doc-sentiment-prosa/test_preprocess_masked_label.tsv'

df = pd.read_csv(train_dataset_path, sep='\t', header=None)
df.columns = ['text','sentiment']
for i,row in df.iterrows():
    sent = row['text']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['text'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/smsa_doc-sentiment-prosa/train_preprocess_bpe.csv', index=False)

df = pd.read_csv(valid_dataset_path, sep='\t', header=None)
df.columns = ['text','sentiment']
for i,row in df.iterrows():
    sent = row['text']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['text'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/smsa_doc-sentiment-prosa/valid_preprocess_bpe.csv', index=False)

df = pd.read_csv(test_dataset_path, sep='\t', header=None)
df.columns = ['text','sentiment']
for i,row in df.iterrows():
    sent = row['text']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['text'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/smsa_doc-sentiment-prosa/test_preprocess_masked_label_bpe.csv', index=False)
print('done preprocess smsa data') 

### Wrete ###
train_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/wrete_entailment-ui/train_preprocess.csv'
valid_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/wrete_entailment-ui/valid_preprocess.csv'
test_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/wrete_entailment-ui/test_preprocess_masked_label.csv'

df = pd.read_csv(train_dataset_path)
for i,row in df.iterrows():
    sent = row['sent_A']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['sent_A'] = to_bpe(sent)   
    sent = row['sent_B']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['sent_B'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/wrete_entailment-ui/train_preprocess_bpe.csv', index=False)

df = pd.read_csv(valid_dataset_path)
for i,row in df.iterrows():
    sent = row['sent_A']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['sent_A'] = to_bpe(sent)   
    sent = row['sent_B']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['sent_B'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/wrete_entailment-ui/valid_preprocess_bpe.csv', index=False)

df = pd.read_csv(test_dataset_path)
for i,row in df.iterrows():
    sent = row['sent_A']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['sent_A'] = to_bpe(sent)   
    sent = row['sent_B']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['sent_B'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/wrete_entailment-ui/test_preprocess_masked_label_bpe.csv', index=False)
print('done preprocess wrete')

### hoasa ###

train_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/hoasa_absa-airy/train_preprocess.csv'
valid_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/hoasa_absa-airy/valid_preprocess.csv'
test_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/hoasa_absa-airy/test_preprocess_masked_label.csv'

df = pd.read_csv(train_dataset_path)
for i,row in df.iterrows():
    sent = row['review']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['review'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/hoasa_absa-airy/train_preprocess_bpe.csv', index=False)

df = pd.read_csv(valid_dataset_path)
for i,row in df.iterrows():
    sent = row['review']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['review'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/hoasa_absa-airy/valid_preprocess_bpe.csv', index=False)

df = pd.read_csv(test_dataset_path)
for i,row in df.iterrows():
    sent = row['review']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['review'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/hoasa_absa-airy/test_preprocess_masked_label_bpe.csv', index=False)
print('done preprocess hoasa')

### CASA ###


train_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/casa_absa-prosa/train_preprocess.csv'
valid_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/casa_absa-prosa/valid_preprocess.csv'
test_dataset_path = '/projectnb/statnlp/gkuwanto/indonlu/dataset/casa_absa-prosa/test_preprocess_masked_label.csv'

df = pd.read_csv(train_dataset_path)
for i,row in df.iterrows():
    sent = row['sentence']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['sentence'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/casa_absa-prosa/train_preprocess_bpe.csv', index=False)

df = pd.read_csv(valid_dataset_path)
for i,row in df.iterrows():
    sent = row['sentence']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['sentence'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/casa_absa-prosa/valid_preprocess_bpe.csv', index=False)

df = pd.read_csv(test_dataset_path)
for i,row in df.iterrows():
    sent = row['sentence']
    if (sent in dict_informal_to_formal):
        sent = dict_informal_to_formal[sent]
    row['sentence'] = to_bpe(sent)   
df.to_csv('/projectnb/statnlp/gik/XLM/IndoNLU/data/casa_absa-prosa/test_preprocess_masked_label_bpe.csv', index=False)
print('done preprocess casa')

    
    
    