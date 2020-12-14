import numpy as np
import scipy as sp
from ._model import Model
from ..utils import record_import_error

try:
    import torch
except ImportError as e:
    record_import_error("torch", "Torch could not be imported!", e)

class PTGenerateTopKLM(Model):
    def __init__(self, model, tokenizer, topk_token_ids=None, device=None):
        """ Generates scores (log odds) for the top-k next word/blank word prediction.

         Parameters
        ----------
        model: object or function
            A object of any pretrained transformer model or function which is to be explained.

        tokenizer: object
            A tokenizer object(PreTrainedTokenizer/PreTrainedTokenizerFast) which is used to tokenize source and target sentence.

        Returns
        -------
        numpy.array
            The scores (log odds) of generating target sentence ids using the model.
        """
        super(PTGenerateTopKLM, self).__init__(model)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 
        self.tokenizer = tokenizer
        self.X = None
        self.topk_token_ids = topk_token_ids
        self.output_names = None

    def __call__(self, masked_X, X):
        """ Computes log odds scores for a given batch of masked inputs for the topk tokens for Causal/Masked LM.

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
        output_batch=[]
        for masked_x, x in zip(masked_X, X):
            # update target sentence ids and original input for a new explanation row
            self.update_cache_X(x)
            # pass the masked input from which to generate source sentence ids
            source_sentence_ids = self.get_source_sentence_ids(masked_x)
            logits = self.get_teacher_forced_logits(source_sentence_ids, self.target_sentence_ids)
            logodds = self.get_logodds(logits)
            output_batch.append(logodds)
        return np.array(output_batch)

    def update_cache_X(self, X):
        """ The function updates original input(X) and topk token ids for the Causal/Masked LM.

        It mimics the caching mechanism to update the original input and topk token ids
        that are to be explained and which updates for every new row of explanation.

        Parameters
        ----------
        X: string
            Input(Text) for an explanation row.
        """
        # check if the source sentence has been updated (occurs when explaining a new row)
        if (self.X is None) or (self.X != X):
            self.X = X
            self.output_names = self.get_output_names_and_update_topk_token_ids(self.X)

    def get_output_names_and_update_topk_token_ids(self, X):
        """ Gets the token names for top k token ids for Causal/Masked LM.
        
        Parameters
        ----------
        X: string or numpy array
            Input(Text/Image) for an explanation row.

        Returns
        -------
        list
            A list of output tokens.
        """
        self.topk_token_ids = self.generate_topk_token_ids(X)
        return self.tokenizer.convert_ids_to_tokens(self.topk_token_ids)