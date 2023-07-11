import numpy as np
import scipy.special

from .._serializable import Deserializer, Serializer
from ..utils import safe_isinstance
from ..utils.transformers import MODELS_FOR_CAUSAL_LM, getattr_silent
from ._model import Model


class TopKLM(Model):
    """ Generates scores (log odds) for the top-k tokens for Causal/Masked LM.
    """

    def __init__(self, model, tokenizer, k=10, generate_topk_token_ids=None, batch_size=128, device=None):
        """ Take Causal/Masked LM model and tokenizer and build a log odds output model for the top-k tokens.

        Parameters
        ----------
        model: object or function
            A object of any pretrained transformer model which is to be explained.

        tokenizer: object
            A tokenizer object(PreTrainedTokenizer/PreTrainedTokenizerFast).

        generation_function_for_topk_token_ids: function
            A function which is used to generate top-k token ids. Log odds will be generated for these custom token ids.

        batch_size: int
            Batch size for model inferencing and computing logodds (default=128).

        device: str
            By default, it infers if system has a gpu and accordingly sets device. Should be 'cpu' or 'cuda' or pytorch models.

        Returns
        -------
        numpy.ndarray
            The scores (log odds) of generating top-k token ids using the model.
        """
        super().__init__(model)

        self.tokenizer = tokenizer
        # set pad token if not defined
        if getattr_silent(self.tokenizer, "pad_token") is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.k = k
        self._custom_generate_topk_token_ids = generate_topk_token_ids
        self.batch_size = batch_size
        self.device = device

        self.X = None
        self.topk_token_ids = None
        self.output_names = None

        self.model_type = None
        if safe_isinstance(self.inner_model, "transformers.PreTrainedModel"):
            self.model_type = "pt"
            import torch  # pylint: disable=import-outside-toplevel
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if self.device is None else self.device
            self.inner_model = self.inner_model.to(self.device)
        elif safe_isinstance(self.inner_model, "transformers.TFPreTrainedModel"):
            self.model_type = "tf"


    def __call__(self, masked_X, X):
        """ Computes log odds scores for a given batch of masked inputs for the top-k tokens for Causal/Masked LM.

        Parameters
        ----------
        masked_X: numpy.ndarray
            An array containing a list of masked inputs.

        X: numpy.ndarray
            An array containing a list of original inputs

        Returns
        -------
        numpy.ndarray
            A numpy array of log odds scores for top-k tokens for every input pair (masked_X, X)
        """
        output_batch = None
        self.update_cache_X(X[:1])
        start_batch_idx, end_batch_idx = 0, len(masked_X)
        while start_batch_idx < end_batch_idx:
            logits = self.get_lm_logits(masked_X[start_batch_idx:start_batch_idx+self.batch_size])
            logodds = self.get_logodds(logits)
            if output_batch is None:
                output_batch = logodds
            else:
                output_batch = np.concatenate((output_batch, logodds))
            start_batch_idx += self.batch_size
        return output_batch

    def update_cache_X(self, X):
        """ The function updates original input(X) and top-k token ids for the Causal/Masked LM.

        It mimics the caching mechanism to update the original input and topk token ids
        that are to be explained and which updates for every new row of explanation.

        Parameters
        ----------
        X: np.ndarray
            Input(Text) for an explanation row.
        """
        # check if the source sentence has been updated (occurs when explaining a new row)
        if (self.X is None) or (not np.array_equal(self.X, X)):
            self.X = X
            self.output_names = self.get_output_names_and_update_topk_token_ids(self.X)

    def get_output_names_and_update_topk_token_ids(self, X):
        """ Gets the token names for top-k token ids for Causal/Masked LM.

        Parameters
        ----------
        X: np.ndarray
            Input(Text) for an explanation row.

        Returns
        -------
        list
            A list of output tokens.
        """

        # see if the user gave a custom token generator
        if self._custom_generate_topk_token_ids is not None:
            return self._custom_generate_topk_token_ids(X)

        # otherwise we pick the top k tokens from the model
        self.topk_token_ids = self.generate_topk_token_ids(X)
        output_names = [self.tokenizer.decode([x]) for x in self.topk_token_ids]
        return output_names

    def get_logodds(self, logits):
        """ Calculates log odds from logits.

        This function passes the logits through softmax and then computes log odds for the top-k token ids.

        Parameters
        ----------
        logits: numpy.ndarray
            An array of logits generated from the model.

        Returns
        -------
        numpy.ndarray
            Computes log odds for corresponding top-k token ids.
        """
        # pass logits through softmax, get the token corresponding score and convert back to log odds (as one vs all)
        def calc_logodds(arr):
            probs = np.exp(arr) / np.exp(arr).sum(-1)
            logodds = scipy.special.logit(probs)
            return logodds

        # pass logits through softmax, get the token corresponding score and convert back to log odds (as one vs all)
        logodds = np.apply_along_axis(calc_logodds, -1, logits)
        logodds_for_topk_token_ids = np.take(logodds, self.topk_token_ids, axis=-1)
        return logodds_for_topk_token_ids

    def get_inputs(self, X, padding_side='right'):
        """ The function tokenizes source sentence.

        Parameters
        ----------
        X: numpy.ndarray
            X is a batch of text.

        Returns
        -------
        dict
            Dictionary of padded source sentence ids and attention mask as tensors("pt" or "tf" based on similarity_model_type).
        """
        self.tokenizer.padding_side = padding_side
        inputs = self.tokenizer(X.tolist(), return_tensors=self.model_type, padding=True)
        # set tokenizer padding to default
        self.tokenizer.padding_side = 'right'
        return inputs

    def generate_topk_token_ids(self, X):
        """ Generates top-k token ids for Causal/Masked LM.

        Parameters
        ----------
        X: numpy.ndarray
            X is the original input sentence for an explanation row.

        Returns
        -------
        np.ndarray
            An array of top-k token ids.
        """
        logits = self.get_lm_logits(X)
        topk_tokens_ids = (-logits).argsort()[0, :self.k]
        return topk_tokens_ids

    def get_lm_logits(self, X):
        """ Evaluates a Causal/Masked LM model and returns logits corresponding to next word/masked word.

        Parameters
        ----------
        X: numpy.ndarray
            An array containing a list of masked inputs.

        Returns
        -------
        numpy.ndarray
            Logits corresponding to next word/masked word.
        """
        if safe_isinstance(self.inner_model, MODELS_FOR_CAUSAL_LM):
            inputs = self.get_inputs(X, padding_side="left")
            if self.model_type == "pt":
                import torch  # pylint: disable=import-outside-toplevel
                inputs["position_ids"] = (inputs["attention_mask"].long().cumsum(-1) - 1)
                inputs["position_ids"].masked_fill_(inputs["attention_mask"] == 0, 0)
                inputs = inputs.to(self.device)
                # generate outputs and logits
                with torch.no_grad():
                    outputs = self.inner_model(**inputs, return_dict=True)
                # extract only logits corresponding to target sentence ids
                logits = outputs.logits.detach().cpu().numpy().astype('float64')[:, -1, :]
            elif self.model_type == "tf":
                import tensorflow as tf  # pylint: disable=import-outside-toplevel
                inputs["position_ids"] = tf.math.cumsum(inputs["attention_mask"], axis=-1) - 1
                inputs["position_ids"] = tf.where(inputs["attention_mask"] == 0, 0, inputs["position_ids"])
                if self.device is None:
                    outputs = self.inner_model(inputs, return_dict=True)
                else:
                    try:
                        with tf.device(self.device):
                            outputs = self.inner_model(inputs, return_dict=True)
                    except RuntimeError as err:
                        print(err)
                logits = outputs.logits.numpy().astype('float64')[:, -1, :]
        return logits

    def save(self, out_file):
        super().save(out_file)

        # Increment the verison number when the encoding changes!
        with Serializer(out_file, "shap.models.TextGeneration", version=0) as s:
            s.save("tokenizer", self.tokenizer)
            s.save("k", self.k)
            s.save("generate_topk_token_ids", self._custom_generate_topk_token_ids)
            s.save("batch_size", self.batch_size)
            s.save("device", self.device)

    @classmethod
    def load(cls, in_file, instantiate=True):
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.models.TextGeneration", min_version=0, max_version=0) as s:
            kwargs["tokenizer"] = s.load("tokenizer")
            kwargs["k"] = s.load("k")
            kwargs["generate_topk_token_ids"] = s.load("generate_topk_token_ids")
            kwargs["batch_size"] = s.load("batch_size")
            kwargs["device"] = s.load("device")
        return kwargs
