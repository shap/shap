import numpy as np
import scipy as sp
from ._model import Model
from ..utils import safe_isinstance, record_import_error
from ..utils.transformers import MODELS_FOR_CAUSAL_LM, MODELS_FOR_MASKED_LM

class GenerateTopKLM(Model):
    def __init__(self, model, tokenizer, k=10, generation_function_for_topk_token_ids=None, device=None):
        """ Generates scores (log odds) for the top-k tokens for Causal/Masked LM.

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
        super(GenerateTopKLM, self).__init__(model)

        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 
        #self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.k = k
        self.generate_topk_token_ids = generation_function_for_topk_token_ids if generation_function_for_topk_token_ids is not None else self.generate_topk_token_ids
        self.X = None
        self.topk_token_ids = None
        self.output_names = None