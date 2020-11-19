import torch
import numpy as np
import scipy as sp
from ._model import Model

class TeacherForcingLogits(Model):
    def __init__(self, model, tokenizer, generation_function = None, text_similarity_model = None, text_similarity_tokenizer = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.to_device(model, self.device)
        self.tokenizer = tokenizer
        self.generation_function = generation_function
        self.text_similarity_model = text_similarity_model
        self.text_similarity_tokenizer = text_similarity_tokenizer
        self.is_model_function = False
        self.is_model_agnostic = False
        if not self.is_model_agnostic:
            # if not model agnostic, then we use the model itself to generate logits
            self.text_similarity_model = self.model
            self.text_similarity_tokenizer = self.tokenizer
        else:
            self.text_similarity_model = text_similarity_model
            self.text_similarity_tokenizer = text_similarity_tokenizer



    def to_device(self, variables, device):
        if isinstance(variables, list):
            deviced_variables = []
            for variable in variables:
                deviced_variables.append(variable.to(device))
            return deviced_variables
        else:
            return variable.to(device)
