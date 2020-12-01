from ._model import Model
from ..utils.transformers import parse_prefix_suffix_for_tokenizer
from ..utils import record_import_error

try:
    import torch
except ImportError as e:
    record_import_error("torch", "Torch could not be imported!", e)

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
            # in non model agnostic case, the model is assumed to be a transformer model and hence we move to_device
            self.model = self.to_device(self.model, device=self.device)
            input_ids = torch.tensor([self.tokenizer.encode(X)])
            input_ids = self.to_device(input_ids, device=self.device)
            text_generation_params = {}
            # check if user assigned any text generation specific kwargs
            if "text_generation_params" in self.model.config.__dict__:
                text_generation_params = self.model.config.text_generation_params
                if not isinstance(text_generation_params, dict):
                    raise ValueError(
                    "Please assign text generation params as a dictionary"
                )
            # generate text
            with torch.no_grad():
                output = self.model.generate(input_ids, **text_generation_params)
            if (hasattr(self.model.config, "is_encoder_decoder") and not self.model.config.is_encoder_decoder) \
                and (hasattr(self.model.config, "is_decoder") and not self.model.config.is_decoder):
                raise ValueError(
                    "Please assign either of is_encoder_decoder or is_decoder to True in model config for extracting target sentence ids"
                )
            if self.model.config.is_encoder_decoder:
                parsed_tokenizer_dict = self.parse_prefix_suffix_for_encoder_decoder(output[0,:].detach().cpu().tolist())
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

    def parse_prefix_suffix_for_encoder_decoder(self, output):
        keep_prefix, keep_suffix = 0, 0
        if self.tokenizer.convert_ids_to_tokens(output[0]) in self.tokenizer.special_tokens_map.values():
            keep_prefix = 1
        if len(output) > 1 and self.tokenizer.convert_ids_to_tokens(output[-1]) in self.tokenizer.special_tokens_map.values():
            keep_suffix = 1
        return {
            'keep_prefix' : keep_prefix,
            'keep_suffix' : keep_suffix
        }