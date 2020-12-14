import numpy as np
import scipy as sp
from ._model import Model
from ..utils import safe_isinstance, record_import_error
from ..utils.transformers import MODELS_FOR_CAUSAL_LM, MODELS_FOR_MASKED_LM

try:
    import torch
except ImportError as e:
    record_import_error("torch", "Torch could not be imported!", e)

class PTGenerateTopKLM(Model):
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
        super(PTGenerateTopKLM, self).__init__(model)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.k = k
        self.generate_topk_token_ids = generation_function_for_topk_token_ids if generation_function_for_topk_token_ids is not None else self.generate_topk_token_ids
        self.X = None
        self.topk_token_ids = None
        self.output_names = None

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
        output_batch=[]
        for masked_x, x in zip(masked_X, X):
            # update target sentence ids and original input for a new explanation row
            self.update_cache_X(x)
            # pass the masked input from which to generate source sentence ids
            sentence_ids = self.get_sentence_ids(masked_x)
            logits = self.get_lm_logits(sentence_ids).numpy().astype('float64')
            logodds = self.get_logodds(logits)
            output_batch.append(logodds)
        return np.array(output_batch)

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
        sentence_ids = torch.tensor([self.tokenizer.encode(X)], device=self.device).to(torch.int64)
        return sentence_ids

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
        if (self.X is None) or (self.X != X):
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
        output_names = self.tokenizer.convert_ids_to_tokens(self.topk_token_ids)
        # adding \n to tokens for better asthetic in text-to-text viz
        output_names = [output_name + " &#10;" for output_name in output_names]
        return output_names

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
        topk_tokens_ids = torch.topk(logits, self.k, dim=1).indices[0]
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
        # set model to eval mode
        self.model.eval()
        if safe_isinstance(self.model, MODELS_FOR_CAUSAL_LM):
            if sentence_ids.shape[1]==0:
                if hasattr(self.model.config,"bos_token_id") and self.model.config.bos_token_id is not None:
                    sentence_ids = (
                        torch.ones((sentence_ids.shape[0], 1), dtype=sentence_ids.dtype, device=sentence_ids.device)
                        * self.model.config.bos_token_id
                    )
                else:
                    raise ValueError(
                    "Context ids (source sentence ids) are null and no bos token defined in model config"
                )
            # generate outputs and logits
            with torch.no_grad():
                outputs = self.model(sentence_ids, return_dict=True)
            # extract only logits corresponding to target sentence ids
            logits=outputs.logits.detach().cpu()[:,sentence_ids.shape[1]-1,:]
            del outputs    
        return logits

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