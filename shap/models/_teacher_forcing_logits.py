import sys
import numpy as np
import scipy as sp
from ._model import Model
from ..utils import safe_isinstance, record_import_error
from ._text_generation import TextGeneration

try:
    import torch
except ImportError as e:
    record_import_error("torch", "Torch could not be imported!", e)

class TeacherForcingLogits(Model):
    def __init__(self, model, tokenizer=None, generation_function_for_target_sentence_ids=None, text_similarity_model=None, text_similarity_tokenizer=None, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 
        self.model = model
        self.tokenizer = tokenizer
        # assign text generation function
        if safe_isinstance(model,"transformers.PreTrainedModel"):
            if generation_function_for_target_sentence_ids is None:
                self.generation_function_for_target_sentence_ids = TextGeneration(self.model, tokenizer=self.tokenizer, device=self.device)
            else:
                self.generation_function_for_target_sentence_ids = generation_function_for_target_sentence_ids
            self.model_agnostic = False
            self.model = self.to_device(model, device=self.device)
            self.text_similarity_model = model
            self.text_similarity_tokenizer = tokenizer
        else:
            if generation_function_for_target_sentence_ids is None:
                self.generation_function_for_target_sentence_ids = TextGeneration(self.model, text_similarity_tokenizer=text_similarity_tokenizer, device=self.device)
            else:
                self.generation_function_for_target_sentence_ids = generation_function_for_target_sentence_ids
            self.text_similarity_model = self.to_device(text_similarity_model, device=self.device)
            self.text_similarity_tokenizer = text_similarity_tokenizer
            self.model_agnostic = True
        # initializing X which is the original input for every new row of explanation
        self.X = None
        self.target_sentence_ids = None
        self.output_names = None

    def __call__(self, masked_X, X):
        output_batch=[]
        for masked_x, x in zip(masked_X, X):
            # update target sentence ids and original input for a new explanation row
            self.update_cache_X(x)
            # pass the masked input from which to generate source sentence ids
            source_sentence_ids = self.get_source_sentence_ids(masked_x)
            logits = self.get_teacher_forced_logits(source_sentence_ids, self.target_sentence_ids)
            logodds = self.get_logodds(logits)
            output_batch.append(logodds)
        return output_batch

    def update_cache_X(self, X):
        """ The function updates original input(X) and target sentence ids.
        It mimics the caching mechanism to update the original input and target sentence ids
        that are to be explained and which updates for every new row of explanation.
        Parameters:
        ----------
        X: string or numpy array
            Source sentence for an explanation row.
        """
        # check if the source sentence has been updated (occurs when explaining a new row)
        if self.X != X:
            self.X = X
            self.target_sentence_ids = self.generation_function_for_target_sentence_ids(X)
            self.output_names = self.get_output_names()
            self.target_sentence_ids = self.to_device(self.target_sentence_ids, device=self.device).to(torch.int64)

    def get_output_names(self):
        return self.text_similarity_tokenizer.convert_ids_to_tokens(self.target_sentence_ids[0,:])

    def to_device(self, variables, device=None):
        if isinstance(variables, list):
            deviced_variables = []
            for variable in variables:
                deviced_variables.append(variable.to(device))
            return deviced_variables
        else:
            return variables.to(device)

    def get_source_sentence_ids(self, X):
        """ The function tokenizes source sentence.
        Parameters:
        ----------
        X: string or tensor
            X could be a text or image.
        Returns:
        -------
        tensor
            Tensor of source sentence ids.
        """
        # TODO: batch source_sentence_ids
        if self.model_agnostic:
            # In model agnostic case, we first pass the input through the model and then tokenize output sentence
            source_sentence = self.model(X)
            source_sentence_ids = torch.tensor([self.text_similarity_tokenizer.encode(source_sentence)])
        else:
            # TODO: check if X is text/image cause presently only when X=text is supported to use model decoder
            source_sentence_ids = torch.tensor([self.text_similarity_tokenizer.encode(X)])
        source_sentence_ids = self.to_device(source_sentence_ids, device=self.device).to(torch.int64)
        return source_sentence_ids

    def get_logodds(self, logits):
        logodds = []
        # pass logits through softmax, get the token corresponding score and convert back to log odds (as one vs all)
        for i in range(0,logits.shape[1]-1):
            probs = (np.exp(logits[0][i]).T / np.exp(logits[0][i]).sum(-1)).T
            logit_dist = sp.special.logit(probs)
            logodds.append(logit_dist[self.target_sentence_ids[0,i].item()])
        return np.array(logodds)

    def get_teacher_forced_logits(self,source_sentence_ids,target_sentence_ids):
        """ The function generates logits for transformer models.
        It generates logits for encoder-decoder models as well as decoder only models by using the teacher forcing technique.
        Parameters:
        ----------
        source_sentence_ids: 2D tensor of shape (batch size, len of sequence)
            Tokenized ids fed to the model.
        target_sentence_ids: 2D tensor of shape (batch size, len of sequence)
            Tokenized ids for which logits are generated using the decoder.
        Returns:
        -------
        numpy array
            Decoder output logits for target sentence ids.
        """
        # set model to eval mode
        self.text_similarity_model.eval()
        # check if type of model architecture assigned in model config
        if (hasattr(self.model.config, "is_encoder_decoder") and not self.model.config.is_encoder_decoder) \
            and (hasattr(self.model.config, "is_decoder") and not self.model.config.is_decoder):
            raise ValueError(
                "Please assign either of is_encoder_decoder or is_decoder to True in model config for extracting target sentence ids"
            )
        if self.text_similarity_model.config.is_encoder_decoder:
            # assigning decoder start token id as it is needed for encoder decoder model generation
            decoder_start_token_id = None
            if hasattr(self.text_similarity_model.config, "decoder_start_token_id") and self.text_similarity_model.config.decoder_start_token_id is not None:
                decoder_start_token_id = self.text_similarity_model.config.decoder_start_token_id
            elif hasattr(self.text_similarity_model.config, "bos_token_id") and self.text_similarity_model.config.bos_token_id is not None:
                decoder_start_token_id = self.text_similarity_model.config.bos_token_id
            elif (hasattr(self.text_similarity_model.config, "decoder") and hasattr(self.text_similarity_model.config.decoder, "bos_token_id") and self.text_similarity_model.config.decoder.bos_token_id is not None):
                decoder_start_token_id = self.text_similarity_model.config.decoder.bos_token_id
            else:
                raise ValueError(
                    "No decoder_start_token_id or bos_token_id defined in config for encoder-decoder generation"
                )
            # concat decoder start token id to target sentence ids
            target_sentence_start_id = (
                torch.ones((source_sentence_ids.shape[0], 1), dtype=source_sentence_ids.dtype, device=source_sentence_ids.device)
                * decoder_start_token_id
            )
            target_sentence_ids = torch.cat((target_sentence_start_id,target_sentence_ids),dim=-1)
            # generate outputs and logits
            with torch.no_grad():
                outputs = self.text_similarity_model(input_ids=source_sentence_ids, decoder_input_ids=target_sentence_ids, labels=target_sentence_ids, return_dict=True)
            logits=outputs.logits.detach().cpu().numpy().astype('float64')
        else:
            # check if source sentence ids are null then add bos token id to decoder
            if source_sentence_ids.shape[1]==0:
                if hasattr(self.text_similarity_model.config,"bos_token_id") and self.text_similarity_model.config.bos_token_id is not None:
                    source_sentence_ids = (
                        torch.ones((source_sentence_ids.shape[0], 1), dtype=source_sentence_ids.dtype, device=source_sentence_ids.device)
                        * self.text_similarity_model.config.bos_token_id
                    )
                else:
                    raise ValueError(
                    "Context ids (source sentence ids) are null and no bos token defined in model config"
                )
            # combine source and target sentence ids  to pass into decoder eg: in case of distillgpt2
            combined_sentence_ids = torch.cat((source_sentence_ids,target_sentence_ids),dim=-1)
            # generate outputs and logits
            with torch.no_grad():
                outputs = self.text_similarity_model(input_ids=combined_sentence_ids, return_dict=True)
            # extract only logits corresponding to target sentence ids
            logits=outputs.logits.detach().cpu().numpy()[:,source_sentence_ids.shape[1]-1:,:].astype('float64')
        del outputs
        return logits