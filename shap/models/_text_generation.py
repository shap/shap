from ._model import Model
from ..utils.transformers import parse_prefix_suffix_for_tokenizer
from ..utils import record_import_error

try:
    import torch
except ImportError as e:
    record_import_error("torch", "Torch could not be imported!", e)

class TextGeneration(Model):
    def __init__(self, model, tokenizer=None, similarity_tokenizer=None, device='cpu'):
        """ Generates target sentence using model and returns tokenized ids.

        It generates target sentence ids for a pretrained transformer model and a function. For a pretrained transformer model, 
        tokenizer should be passed. In case model is a function, then the similarity_tokenizer is used to tokenize generated 
        sentence to ids.

        Parameters
        ----------
        model: object or function
            A object of any pretrained transformer model or function for which target sentence and tokenized ids are to be generated.

        tokenizer: object
            A tokenizer object(PreTrainedTokenizer/PreTrainedTokenizerFast) which is used to tokenize sentence.

        Returns
        -------
        tensor
            A tensor of target sentence ids.
        """
        super(TextGeneration, self).__init__(model)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 
        self.tokenizer = tokenizer
        self.similarity_tokenizer = similarity_tokenizer
        if similarity_tokenizer is None:
            self.model_agnostic = False
        else:
            self.model_agnostic = True

    def __call__(self, X):
        """ Generates target sentence ids from X.

        Parameters
        ----------
        X: str or numpy.array
            Input in the form of text or image.

        Returns
        -------
        tensor
            Tensor of target sentence ids.
        """
        target_sentence_ids = None
        if self.model_agnostic:
            target_sentence = self.model(X)
            parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(self.similarity_tokenizer)
            keep_prefix, keep_suffix = parsed_tokenizer_dict['keep_prefix'], parsed_tokenizer_dict['keep_suffix']
            if keep_suffix > 0:
                target_sentence_ids = torch.tensor([self.similarity_tokenizer.encode(target_sentence)])[:,keep_prefix:-keep_suffix]
            else:
                target_sentence_ids = torch.tensor([self.similarity_tokenizer.encode(target_sentence)])[:,keep_prefix:]
        else:
            self.model.eval()
            # in non model agnostic case, the model is assumed to be a transformer model and hence we move to_device
            self.model = self.to_device(self.model)
            input_ids = torch.tensor([self.tokenizer.encode(X)])
            input_ids = self.to_device(input_ids)
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
            if self.model.config.is_decoder:
                # slice the output ids after the input ids
                output = output[:,input_ids.shape[1]:]
            # parse output ids to find special tokens in prefix and suffix
            parsed_tokenizer_dict = self.parse_prefix_suffix_for_model_generate_output(output[0,:].detach().cpu().tolist())
            keep_prefix, keep_suffix = parsed_tokenizer_dict['keep_prefix'], parsed_tokenizer_dict['keep_suffix']
            # extract target sentence ids by slicing off prefix and suffix
            if keep_suffix > 0:
                target_sentence_ids = output[:, keep_prefix:-keep_suffix]
            else:
                target_sentence_ids = output[:, keep_prefix:]

        return target_sentence_ids

    def to_device(self, variables):
        if isinstance(variables, list):
            deviced_variables = []
            for variable in variables:
                deviced_variables.append(variable.to(self.device))
            return deviced_variables
        else:
            return variables.to(self.device)

    def parse_prefix_suffix_for_model_generate_output(self, output):
        """ Calculates if special tokens are present in the begining/end of the output.
        """
        keep_prefix, keep_suffix = 0, 0
        if self.tokenizer.convert_ids_to_tokens(output[0]) in self.tokenizer.special_tokens_map.values():
            keep_prefix = 1
        if len(output) > 1 and self.tokenizer.convert_ids_to_tokens(output[-1]) in self.tokenizer.special_tokens_map.values():
            keep_suffix = 1
        return {
            'keep_prefix' : keep_prefix,
            'keep_suffix' : keep_suffix
        }