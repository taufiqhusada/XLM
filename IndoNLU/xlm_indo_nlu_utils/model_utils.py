import torch
from torch import optim

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def forward_sequence_classification(proj, model, batch_data, i2w, is_test=False, device='cuda', **kwargs):
    (subword_batch, label_batch, lengths, langs) = batch_data
    token_type_batch = None
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
            
    if device == "cuda":
        subword_batch = subword_batch.cuda()
        label_batch = label_batch.cuda()
        proj = proj.cuda()
        model = model.cuda()
        lengths = lengths.cuda()

    # Forward model
    logits = proj(model('fwd', x=subword_batch, lengths=lengths, langs=langs, causal=False).contiguous()[0])

    list_hyp = []
    list_label = []
    hyp = torch.topk(logits, 1)[1]
    for j in range(len(hyp)):
        list_hyp.append(i2w[hyp[j].item()])
        list_label.append(i2w[label_batch[j][0].item()])
        

    loss = F.cross_entropy(logits, label_batch.squeeze(1))
    
    return loss, logits, list_hyp, list_label


def forward_sequence_multi_classification(proj, model, batch_data, i2w, num_labels, is_test=False, device='cuda', **kwargs):
    (subword_batch, label_batch, lengths, langs) = batch_data
    token_type_batch = None
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
            
    if device == "cuda":
        subword_batch = subword_batch.cuda()
        label_batch = label_batch.cuda()
        proj = proj.cuda()
        model = model.cuda()
        lengths = lengths.cuda()

    # Forward model
    logits = proj(model('fwd', x=subword_batch, lengths=lengths, langs=langs, causal=False).contiguous()[0])

    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = [torch.topk(logit, 1)[1] for logit in logits] # list<tensor(bs)>
    batch_size = label_batch.shape[0]
    num_label = len(hyp)
    for i in range(batch_size):
        hyps = []
        labels = label_batch[i,:].cpu().numpy().tolist()
        for j in range(num_label):
            hyps.append(hyp[j][i].item())
        list_hyp.append([i2w[hyp] for hyp in hyps])
        list_label.append([i2w[label] for label in labels])
        
    # count loss
    loss_fct = CrossEntropyLoss()
    total_loss = 0
    for i, (logit, num_label) in enumerate(zip(logits, num_labels)):
        label = label_batch[:,i]
        loss = loss_fct(logit, label.view(-1))
        total_loss += loss
    
    return total_loss, logits, list_hyp, list_label