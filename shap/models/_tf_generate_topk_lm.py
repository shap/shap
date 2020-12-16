import numpy as np
import scipy as sp
from ._generate_topk_lm import GenerateTopKLM
from ..utils import safe_isinstance, record_import_error
from ..utils.transformers import MODELS_FOR_CAUSAL_LM, MODELS_FOR_MASKED_LM

try:
    import tensorflow as tf
except ImportError as e:
    record_import_error("tensorflow", "TensorFlow could not be imported!", e)

class TFGenerateTopKLM(GenerateTopKLM):
    def __init__(self, model, tokenizer, k=10, generation_function_for_topk_token_ids=None, device=None):
        """ Generates scores (log odds) for the top-k tokens for Causal/Masked LM for PyTorch models.

        This model inherits from GenerateTopKLM. Check the superclass documentation for the generic methods the library implements for all its model.

        Parameters
        ----------
        model: object or function
            A object of any pretrained transformer model which is to be explained.

        tokenizer: object
            A tokenizer object(PreTrainedTokenizer/PreTrainedTokenizerFast).

        Returns
        -------
        numpy.array
            The scores (log odds) of generating top-k token ids using the model.
        """
        super(TFGenerateTopKLM, self).__init__(model, tokenizer, k, generation_function_for_topk_token_ids, device)