import torch
import numpy as np
import scipy as sp

def get_teacher_forced_logits(model,input_ids,decoder_input_ids):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids, return_dict=True)
    logits=outputs.logits.detach().cpu().numpy()
    del outputs
    return logits

def cal_conditional_logits(model,input_ids,decoder_input_ids,attention_mask=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_ids=input_ids.to(device)
    conditional_logits = []
    logits=get_teacher_forced_logits(model,input_ids,decoder_input_ids)
    for i in range(1,logits.shape[1]):
        probs = (np.exp(logits[0][i-1]).T / np.exp(logits[0][i-1]).sum(-1)).T
        logit_dist = sp.special.logit(probs)
        conditional_logits.append(logit_dist[decoder_input_ids[0,i].item()])
    return np.array(conditional_logits)