import numpy as np
import scipy as sp
from ._model import Model
from .. import models
from ..utils import safe_isinstance

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
        self.k = k
        self.X = None
        self.topk_token_ids = None
        self.output_names = None
        self.device = device
        self.generate_topk_token_ids = generate_topk_token_ids if generate_topk_token_ids is not None else self.generate_topk_token_ids

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
        probs = (np.exp(logits).T / np.exp(logits).sum(-1)).T
        logit_dist = sp.special.logit(probs)
        indices = np.tile(self.topk_token_ids, (logits.shape[0],1))
        logodds = np.take(logit_dist, indices)
        return logodds[0]

    def get_sentence_ids(self, X):
        """ Implement in subclass. Returns a tensor of sentence ids.
        """
        pass

    def generate_topk_token_ids(self, X):
        """ Implement in subclass. Returns a tensor of top-k token ids.
        """
        pass

    def get_lm_logits(self, sentence_ids):
        """ Implement in subclass. Returns a tensor of logits.
        """
        pass