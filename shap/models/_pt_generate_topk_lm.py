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