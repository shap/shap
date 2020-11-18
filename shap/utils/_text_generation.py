
import torch
import numpy as np
import scipy as sp

#Dummy class which mimics TextGeneration class.

class TextGeneration:
    def __init__(self,tmodel,max_length): #parameters for generating output text, #transformer model
        self.model=tmodel
        #self.prefix &self.suffix
        #define parameters

    def generate(self,X):
        self.model.eval()
        #if model is text prediction, 
        input_ids = torch.tensor([tokenizer.encode(X)]).cuda()
        with torch.no_grad():
            out=self.model.generate(input_ids)#pass params
        #sentence = [tokenizer.decode(g, skip_special_tokens=True) for g in out][0]
        del input_ids, out
        #out should be sliced and only return set of ids we want to generate log odds for
        return out

    def __call__(self,X): #generate function
        target_sentence_ids=self.generate(X)
        return target_sentence_ids