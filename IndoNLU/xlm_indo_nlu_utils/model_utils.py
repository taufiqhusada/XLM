import torch
from torch import optim

import torch.nn.functional as F

def forward_sequence_classification(proj, model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    (subword_batch, label_batch, lengths, langs) = batch_data
    token_type_batch = None
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
            
    if device == "cuda":
        subword_batch = subword_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # Forward model
    logits = proj(model('fwd', x=subword_batch, lengths=lengths.cuda(), langs=langs, causal=False).contiguous()[0])

    list_hyp = []
    list_label = []
    hyp = torch.topk(logits, 1)[1]
    for j in range(len(hyp)):
        list_hyp.append(i2w[hyp[j].item()])
        list_label.append(i2w[label_batch[j][0].item()])
        

    loss = F.cross_entropy(logits, label_batch.squeeze(1))
    
    return loss, logits, list_hyp, list_label