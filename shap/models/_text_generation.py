import torch
from ._model import Model
from ..utils import get_tokenizer_prefix_suffix

class TextGeneration(Model):
    def __init(self, model, tokenizer=None, text_similarity_tokenizer=None, device='cpu'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 
        self.model = model
        self.tokenizer = tokenizer
        self.text_similarity_tokenizer = text_similarity_tokenizer
        if text_similarity_tokenizer is None:
            self.model_agnostic = False
        else:
            self.model_agnostic = True