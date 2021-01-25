#!/usr/bin/env python
# coding: utf-8

# # Multi-Input Text Explanation: Textual Entailment with Facebook BART

# This notebook demonstrates how to get explanations for the output of BART trained on the mnli dataset and used for textual entailment. 

# In[1]:


import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shap
from datasets import load_dataset

# ### Load model and tokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

# dataset
dataset = load_dataset("snli")
print(dataset)

example_ind = 6
premise, hypothesis, label = dataset['train']['premise'][example_ind], dataset['train']['hypothesis'][example_ind], dataset['train']['label'][example_ind]
print(premise)
print(hypothesis)
print(label)
# p2, h2, l2 = (
#     dataset["train"]["premise"][example_ind + 1],
#     dataset["train"]["hypothesis"][example_ind + 1],
#     dataset["train"]["label"][example_ind + 1]
# )
# print(p2)
# print(h2)
# print(l2)


input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt')
print(input_ids)
logits = model(input_ids)[0]

probs = logits.softmax(dim=1)
p1, p2, p3 = probs[:,0].item() * 100, probs[:,1].item()*100, probs[:,2].item()*100
print('Contradiction Probability: {p1:0.2f}%, Neutral Probability: {p2:0.2f}%, Entailment Probability: {p3:0.2f}%'.format(p1 = p1, p2 = p2, p3=p3))
ind_to_label = {0: 'entailment', 1: 'neural', 2: 'contradiction'}
true_label = ind_to_label[label]
print('The true label is: {true_label}'.format(true_label=true_label))

###


# In[2]:

device = 'cpu'

candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
label = candidate_labels[0]
# premise = 'I want to go to Japan.'
# hypothesis = f'This example is {label}.'
# TODO data = load_data('nli') # note: mnli is not loadable with this environment
premise = 'A boy is jumping on skateboard in the middle of a red bridge.'
hypothesis = 'A boy skates down the sidewalk.'
# run through model pre-trained on MNLI
# x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
#                      truncation_strategy='only_first')

# print('INPUT')
# print(x)

# logits = model(x.to(device))[0]
# print('LOGITS')
# print(logits)

import scipy as sp
import torch

# def f(x, x_opt):
#     tv = torch.tensor([tokenizer.encode(v, v_opt) for v, v_opt in zip(x, x_opt)])
#     outputs = model(tv)[0].detach().cpu().numpy()
#     scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
#     val = sp.special.logit(scores)
#     return val

def f(x): # TODO takes in the already - masked string which is the concatenation of 2 strings
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


# testing tokenizer

s1 = 'I ate some pie.'
s2 = 'You are a dog.'
encode1 = tokenizer(s1)
encode2 = tokenizer(s2) 
print(encode1)
print(encode2)
# 
# 
# shap_values = explainer([premise], [hypothesis]) # wrap in list, otherwise zip would iterate across letters
shap_values = explainer([decoded]) # wrap in list, otherwise zip would iterate across letters

print('SHAP VALUES')
# shap_values = explainer([premise], [hypothesis]) # wrap in list, otherwise zip would iterate across letters
print(shap_values)
