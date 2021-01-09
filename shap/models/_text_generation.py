import numpy as np
from ._model import Model
from .. import models
from ..utils import record_import_error, safe_isinstance

try:
    import torch
except ImportError as e:
    record_import_error("torch", "Torch could not be imported!", e)

try:
    import tensorflow as tf
except ImportError as e:
    record_import_error("tensorflow", "TensorFlow could not be imported!", e)


class TextGeneration(Model):
    def __init__(self, model, tokenizer=None, similarity_tokenizer=None, device=None):
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
        np.ndarray
            Array of target sentence or ids.
        """
        super(TextGeneration, self).__init__(model)

        self.tokenizer = tokenizer
        self.similarity_tokenizer = similarity_tokenizer
        self.device = device
        # X is input used to generate target sentence
        # used for caching
        self.X = None
        # target sentence/ids generated from the model using X 
        self.target_X = None

        if self.__class__ is TextGeneration:
            if safe_isinstance(self.model, "transformers.PreTrainedModel"):
                self.__class__ = models.PTTextGeneration
                models.PTTextGeneration.__init__(self, self.model, self.tokenizer, self.similarity_tokenizer, self.device)
            if safe_isinstance(self.model, "transformers.TFPreTrainedModel"):
                self.__class__ = models.TFTextGeneration
                models.TFTextGeneration.__init__(self, self.model, self.tokenizer, self.similarity_tokenizer, self.device)


    def __call__(self, X):
        """ Generates target sentence ids from X.

        Parameters
        ----------
        X: str or np.ndarray
            Input in the form of text or image.

        Returns
        -------
        np.ndarray
            Array of target sentence.
        """
        # generate target sentence ids in model agnostic scenario
        if (self.X is None) or (isinstance(self.X, np.ndarray) and (self.X != X).all()) or (isinstance(self.X, str) and (self.X != X)):
            self.X = X
            # wrap text input in a numpy array
            if isinstance(X, str):
                X = np.array([X])
            self.target_X = self.model(X)
        return np.array(self.target_X)

    def parse_prefix_suffix_for_model_generate_output(self, output):
        """ Calculates if special tokens are present in the begining/end of the model generated output.
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

class PTTextGeneration(TextGeneration):
    def __init__(self, model, tokenizer=None, similarity_tokenizer=None, device=None):
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
        np.ndarray
            Array of target sentence ids.
        """
        super(PTTextGeneration, self).__init__(model, tokenizer, similarity_tokenizer, device)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 

    def __call__(self, X):
        """ Generates target sentence ids from X.

        Parameters
        ----------
        X: str or np.ndarray
            Input in the form of text or image.

        Returns
        -------
        np.ndarray
            Array of target sentence ids.
        """
        if (self.X is None) or (isinstance(self.X, np.ndarray) and (self.X != X).all()) or (isinstance(self.X, str) and (self.X != X)):
            self.X = X
            # in non model agnostic case, the model is assumed to be a transformer model and hence we move to device
            self.model.eval()
            self.model = self.model.to(self.device)
            # wrap text input in a numpy array
            if isinstance(X, str):
                X = np.array([X])

            # set pad token if not defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # presently supports only text input for hugging face models
            padded_sequences = self.tokenizer(X.tolist(), padding=True)
            input_ids = torch.tensor(padded_sequences["input_ids"]).to(self.device)
            attention_mask = torch.tensor(padded_sequences["attention_mask"]).to(self.device)
            # check if user assigned any text generation specific kwargs
            text_generation_params = {}
            if "text_generation_params" in self.model.config.__dict__:
                text_generation_params = self.model.config.text_generation_params
                if not isinstance(text_generation_params, dict):
                    raise ValueError(
                    "Please assign text generation params as a dictionary"
                )
            # generate text
            with torch.no_grad():
                output = self.model.generate(input_ids, attention_mask=attention_mask, **text_generation_params).detach().cpu()
            if (hasattr(self.model.config, "is_encoder_decoder") and not self.model.config.is_encoder_decoder) \
                and (hasattr(self.model.config, "is_decoder") and not self.model.config.is_decoder):
                raise ValueError(
                    "Please assign either of is_encoder_decoder or is_decoder to True in model config for extracting target sentence ids"
                )
            if self.model.config.is_decoder:
                # slice the output ids after the input ids
                output = output[:,input_ids.shape[1]:]
            # parse output ids to find special tokens in prefix and suffix
            parsed_tokenizer_dict = self.parse_prefix_suffix_for_model_generate_output(output[0,:].tolist())
            keep_prefix, keep_suffix = parsed_tokenizer_dict['keep_prefix'], parsed_tokenizer_dict['keep_suffix']
            # extract target sentence ids by slicing off prefix and suffix
            if keep_suffix > 0:
                self.target_X = output[:, keep_prefix:-keep_suffix]
            else:
                self.target_X = output[:, keep_prefix:]

        return self.target_X.numpy()

class TFTextGeneration(TextGeneration):
    def __init__(self, model, tokenizer=None, similarity_tokenizer=None, device=None):
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
        np.ndarray
            Array of target sentence ids.
        """
        super(TFTextGeneration, self).__init__(model, tokenizer, similarity_tokenizer, device)

    def __call__(self, X):
        """ Generates target sentence ids from X.

        Parameters
        ----------
        X: str or np.ndarray
            Input in the form of text or image.

        Returns
        -------
        np.ndarray
            Array of target sentence ids.
        """
        if (self.X is None) or (isinstance(self.X, np.ndarray) and (self.X != X).all()) or (isinstance(self.X, str) and (self.X != X)):
            self.X = X
            # wrap text input in a numpy array
            if isinstance(X, str):
                X = np.array([X])
            padded_sequences = self.tokenizer(X.tolist(), padding=True)
            input_ids = tf.convert_to_tensor(padded_sequences["input_ids"])
            attention_mask = tf.convert_to_tensor(padded_sequences["attention_mask"])
            text_generation_params = {}
            # check if user assigned any text generation specific kwargs
            if "text_generation_params" in self.model.config.__dict__:
                text_generation_params = self.model.config.text_generation_params
                if not isinstance(text_generation_params, dict):
                    raise ValueError(
                    "Please assign text generation params as a dictionary"
                )
            # generate text
            if self.device is None:
                output = self.model.generate(input_ids, attention_mask=attention_mask, **text_generation_params)
            else:
                try:
                    with tf.device(self.device):
                        output = self.model.generate(input_ids, attention_mask=attention_mask, **text_generation_params)
                except RuntimeError as e:
                    print(e)
            if (hasattr(self.model.config, "is_encoder_decoder") and not self.model.config.is_encoder_decoder) \
                and (hasattr(self.model.config, "is_decoder") and not self.model.config.is_decoder):
                raise ValueError(
                    "Please assign either of is_encoder_decoder or is_decoder to True in model config for extracting target sentence ids"
                )
            if self.model.config.is_decoder:
                # slice the output ids after the input ids
                output = output[:,input_ids.shape[1]:]
            # parse output ids to find special tokens in prefix and suffix
            parsed_tokenizer_dict = self.parse_prefix_suffix_for_model_generate_output(output[0,:].numpy().tolist())
            keep_prefix, keep_suffix = parsed_tokenizer_dict['keep_prefix'], parsed_tokenizer_dict['keep_suffix']
            # extract target sentence ids by slicing off prefix and suffix
            if keep_suffix > 0:
                self.target_X = output[:, keep_prefix:-keep_suffix]
            else:
                self.target_X = output[:, keep_prefix:]

        return self.target_X.numpy()