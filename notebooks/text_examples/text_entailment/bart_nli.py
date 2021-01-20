#!/usr/bin/env python
# coding: utf-8

# # Multi-Input Text Explanation: Textual Entailment with Facebook BART

# This notebook demonstrates how to get explanations for the output of BART trained on the mnli dataset and used for textual entailment. 

# In[1]:


import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shap

# ### Load model and tokenizer

# In[2]:

model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli', use_fast=False) # TODO changed use_fast to False
device = 'cpu'

candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
label = candidate_labels[0]
sequence = 'I want to go to Japan.'
premise = sequence
hypothesis = f'This example is {label}.'
# TODO data = load_data('nli') # note: mnli is not loadable with this environment

# run through model pre-trained on MNLI
x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')

print('INPUT')
print(x)

logits = model(x.to(device))[0]
print('LOGITS')
print(logits)

import scipy as sp
import torch

# def f(x, x_opt):
#     tv = torch.tensor([tokenizer.encode(v, v_opt) for v, v_opt in zip(x, x_opt)])
#     outputs = model(tv)[0].detach().cpu().numpy()
#     scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
#     val = sp.special.logit(scores)
#     return val

def f(x): # TODO takes in the already - masked string which is the concatenation of 2 strings
    # tv = torch.tensor([tokenizer.encode(*_x) for _x in zip(*x)])
    tv = torch.tensor([tokenizer.encode(_x) for _x in x])
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores)
    return val

# f_results = f([premise], [hypothesis])
# print('FUNCTION RESULTS')
# print(f_results)

print('EXPLAINER')
explainer = shap.Explainer(f, tokenizer)

encoded = tokenizer(premise, hypothesis)['input_ids'][1:-1]
decoded = tokenizer.decode(encoded)
print(decoded)

# shap_values = explainer([premise], [hypothesis]) # wrap in list, otherwise zip would iterate across letters
shap_values = explainer([decoded]) # wrap in list, otherwise zip would iterate across letters

# print('SHAP VALUES')
# shap_values = explainer([premise], [hypothesis]) # wrap in list, otherwise zip would iterate across letters
print(shap_values)
