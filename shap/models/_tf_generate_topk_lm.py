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

    def get_sentence_ids(self, X):
        """ The function tokenizes sentence.

        Parameters
        ----------
        X: string
            X is a sentence.

        Returns
        -------
        torch.Tensor
            Tensor of sentence ids.
        """
        sentence_ids = tf.convert_to_tensor([self.tokenizer.encode(X)])
        return sentence_ids

    def generate_topk_token_ids(self, X):
        """ Generates top-k token ids for Causal/Masked LM.

        Parameters
        ----------
        X: string
            Input(Text) for an explanation row.

        Returns
        -------
        list
            A list of top-k token ids.
        """
        
        sentence_ids = self.get_sentence_ids(X)
        logits = self.get_lm_logits(sentence_ids)
        topk_tokens_ids = tf.math.top_k(logits, k=self.k, sorted=True).indices[0]
        return topk_tokens_ids

    def get_lm_logits(self, sentence_ids):
        """ Evaluates a Causal/Masked LM model and returns logits corresponding to next word/masked word.

        Parameters
        ----------
        source_sentence_ids: torch.Tensor of shape (batch size, len of sequence)
            Tokenized ids fed to the model.

        Returns
        -------
        numpy.array
            Logits corresponding to next word/masked word.
        """
        if safe_isinstance(self.model, MODELS_FOR_CAUSAL_LM):
            if sentence_ids.shape[1]==0:
                if hasattr(self.model.config,"bos_token_id") and self.model.config.bos_token_id is not None:
                    sentence_ids = (
                        tf.ones((sentence_ids.shape[0], 1), dtype=tf.int32)
                        * self.model.config.bos_token_id
                    )
                else:
                    raise ValueError(
                    "Context ids (source sentence ids) are null and no bos token defined in model config"
                )
            # generate outputs and logits
            if self.device is None:
                outputs = self.model(sentence_ids, return_dict=True)
            else:
                try:
                    with tf.device(self.device):
                        outputs = self.model(sentence_ids, return_dict=True)
                except RuntimeError as e:
                    print(e)
            # extract only logits corresponding to target sentence ids
            logits=outputs.logits[:,sentence_ids.shape[1]-1,:]
            del outputs    
        return logits