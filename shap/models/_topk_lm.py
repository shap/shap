import numpy as np
import scipy as sp
from ._model import Model
from ..utils import safe_isinstance, record_import_error
from ..utils.transformers import MODELS_FOR_CAUSAL_LM

try:
    import torch
except ImportError as e:
    record_import_error("torch", "Torch could not be imported!", e)

try:
    import tensorflow as tf
except ImportError as e:
    record_import_error("tensorflow", "TensorFlow could not be imported!", e)

class TopKLM(Model):
    def __init__(self, model, tokenizer, k=10, generate_topk_token_ids=None, device=None):
        """ Generates scores (log odds) for the top-k tokens for Causal/Masked LM.

        Parameters
        ----------
        model: object or function
            A object of any pretrained transformer model which is to be explained.

        tokenizer: object
            A tokenizer object(PreTrainedTokenizer/PreTrainedTokenizerFast).

        generation_function_for_topk_token_ids: function
            A function which is used to generate top-k token ids. Log odds will be generated for these custom token ids.

        Returns
        -------
        numpy.array
            The scores (log odds) of generating top-k token ids using the model.
        """
        super(TopKLM, self).__init__(model)

        self.tokenizer = tokenizer
        # set pad token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.k = k
        self.X = None
        self.topk_token_ids = None
        self.output_names = None
        self.device = device
        self.generate_topk_token_ids = generate_topk_token_ids if generate_topk_token_ids is not None else self.generate_topk_token_ids
        self.model_type = None
        if safe_isinstance(self.model,"transformers.PreTrainedModel"):
            self.model_type = "pt"
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if self.device is None else self.device
            self.model = self.model.to(self.device)
        elif safe_isinstance(self.model,"transformers.TFPreTrainedModel"):
            self.model_type = "tf"


    def __call__(self, masked_X, X):
        """ Computes log odds scores for a given batch of masked inputs for the top-k tokens for Causal/Masked LM.

        Parameters
        ----------
        masked_X: numpy.array
            An array containing a list of masked inputs.

        X: numpy.array
            An array containing a list of original inputs

        Returns
        -------
        numpy.array
            A numpy array of log odds scores for topk tokens for every input pair (masked_X, X)
        """
        #output_batch=[]
        self.update_cache_X(X[:1])
        logits = self.get_lm_logits(masked_X)
        logodds = self.get_logodds(logits)
        return np.array(logodds)

    def update_cache_X(self, X):
        """ The function updates original input(X) and top-k token ids for the Causal/Masked LM.

        It mimics the caching mechanism to update the original input and topk token ids
        that are to be explained and which updates for every new row of explanation.

        Parameters
        ----------
        X: string
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
        X: string
            Input(Text) for an explanation row.

        Returns
        -------
        list
            A list of output tokens.
        """
        self.topk_token_ids = self.generate_topk_token_ids(X)
        output_names = [self.tokenizer.decode([x]) for x in self.topk_token_ids]
        return output_names

    def get_logodds(self, logits):
        """ Calculates log odds from logits.

        This function passes the logits through softmax and then computes log odds for the top-k token ids.

        Parameters
        ----------
        logits: numpy.array
            An array of logits generated from the model.

        Returns
        -------
        numpy.array
            Computes log odds for corresponding target sentence ids.
        """
        # pass logits through softmax, get the token corresponding score and convert back to log odds (as one vs all)
        def calc_logodds(arr):
            probs = np.exp(arr) / np.exp(arr).sum(-1)
            logodds = sp.special.logit(probs)
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
        torch.Tensor or tf.Tensor
            Array of padded source sentence ids and attention mask.
        """
        self.tokenizer.padding_side = padding_side
        inputs = self.tokenizer(X.tolist(), return_tensors=self.model_type, padding=True)
        # set tokenizer padding to default
        self.tokenizer.padding_side = 'right'
        return inputs

    def generate_topk_token_ids(self, X):
        """ Implement in subclass. Returns a tensor of top-k token ids.
        """
        logits = self.get_lm_logits(X)
        topk_tokens_ids = (-logits).argsort()[0,:self.k]
        return topk_tokens_ids

    def get_lm_logits(self, X):
        """ Evaluates a Causal/Masked LM model and returns logits corresponding to next word/masked word.
        Parameters
        ----------
        source_sentence_ids: torch.Tensor of shape (batch size, len of sequence)
            Tokenized ids fed to the model.
        Returns
        -------
        torch.Tensor
            Logits corresponding to next word/masked word.
        """
        if safe_isinstance(self.model, MODELS_FOR_CAUSAL_LM):
            inputs = self.get_inputs(X, padding_side="left")
            if self.model_type == "pt":
                inputs["position_ids"] = (inputs["attention_mask"].long().cumsum(-1) - 1)
                inputs["position_ids"].masked_fill_(inputs["attention_mask"] == 0, 0)
                inputs = inputs.to(self.device)
                # generate outputs and logits
                with torch.no_grad():
                    outputs = self.model(**inputs, return_dict=True)
                # extract only logits corresponding to target sentence ids
                logits=outputs.logits.detach().cpu().numpy().astype('float64')[:,-1,:]
            elif self.model_type == "tf":
                inputs["position_ids"] = tf.math.cumsum(inputs["attention_mask"], axis=-1) - 1
                inputs["position_ids"] = tf.where(inputs["attention_mask"] == 0, 0, inputs["position_ids"])
                if self.device is None:
                    outputs = self.model(inputs, return_dict=True)
                else:
                    try:
                        with tf.device(self.device):
                            outputs = self.model(inputs, return_dict=True)
                    except RuntimeError as e:
                        print(e)
                logits=outputs.logits.numpy().astype('float64')[:,-1,:]
        return logits