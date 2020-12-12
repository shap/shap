from ._model import Model
from ..utils.transformers import parse_prefix_suffix_for_tokenizer
from ..utils import record_import_error

try:
    import tensorflow
except ImportError as e:
    record_import_error("tensorflow", "TensorFlow could not be imported!", e)

class TFTextGeneration(Model):
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
        super(TFTextGeneration, self).__init__(model)

        self.device = False
        self.tokenizer = tokenizer
        self.similarity_tokenizer = similarity_tokenizer
        if similarity_tokenizer is None:
            self.model_agnostic = False
        else:
            self.model_agnostic = True