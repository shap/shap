import torch
from ._model import Model
from ..utils.transformers import parse_prefix_suffix_for_tokenizer

class TextGeneration(Model):
    def __init__(self, model, tokenizer=None, text_similarity_tokenizer=None, device='cpu'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 
        self.model = model
        self.tokenizer = tokenizer
        self.text_similarity_tokenizer = text_similarity_tokenizer
        if text_similarity_tokenizer is None:
            self.model_agnostic = False
        else:
            self.model_agnostic = True

    def __call__(self, X):
        target_sentence_ids = None
        if self.model_agnostic:
            target_sentence = self.model(X)
            parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(self.text_similarity_tokenizer)
            keep_prefix, keep_suffix = parsed_tokenizer_dict['keep_prefix'], parsed_tokenizer_dict['keep_suffix']
            if keep_suffix > 0:
                target_sentence_ids = torch.tensor([self.text_similarity_tokenizer.encode(target_sentence)])[:,keep_prefix:-keep_suffix]
            else:
                target_sentence_ids = torch.tensor([self.text_similarity_tokenizer.encode(target_sentence)])[:,keep_prefix:]
        else:
            self.model.eval()
            self.model = self.to_device(self.model, device=self.device)
            input_ids = torch.tensor([self.tokenizer.encode(X)])
            input_ids = self.to_device(input_ids, device=self.device)
            with torch.no_grad():
                output = self.model.generate(input_ids)
            if (hasattr(self.model.config, "is_encoder_decoder") and not self.model.config.is_encoder_decoder) \
                and (hasattr(self.model.config, "is_decoder") and not self.model.config.is_decoder):
                raise ValueError(
                    "Please assign either of is_encoder_decoder or is_decoder to True in model config for extracting target sentence ids"
                )
            if self.model.config.is_encoder_decoder:
                parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(self.tokenizer)
                keep_prefix, keep_suffix = parsed_tokenizer_dict['keep_prefix'], parsed_tokenizer_dict['keep_suffix']
                if keep_suffix > 0:
                    target_sentence_ids = output[:, keep_prefix:-keep_suffix]
                else:
                    target_sentence_ids = output[:, keep_prefix:]
            else:
                # incase of only decoder we slice target ids after the input ids
                target_sentence_ids = output[:,input_ids.shape[1]:]

        return target_sentence_ids

    def to_device(self, variables, device=None):
        if isinstance(variables, list):
            deviced_variables = []
            for variable in variables:
                deviced_variables.append(variable.to(device))
            return deviced_variables
        else:
            return variables.to(device)