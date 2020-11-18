import torch
import numpy as np
import scipy as sp

class TeacherForcingLogits:
    def __init__(self,  model, f, tokenizer, device=None):#model->, f-> make up generation function


        #WARNING: ALL CODE BELOW WAS KEEPING IN MIND "f" WAS THE MODEL PASSED BY THE USER TO EXPLAIN AND "model" WAS THE LANGUAGE MODEL USED TO GENERATE LOG ODDS. 
        # TO DO: UPDATE CODE BASED ON RECENT CHANGES 

        """ Generates scores (log odds) for output text explanation algorithms.

        This class supports generation of log odds for decoder specific transformers (eg: distilgpt2)
        and encoder-decoder transformers (eg: BART).It takes 2 sentences (source,target) in any combination
        of string or token ids and generates log odds of generating target sentence from source sentence.

        Parameters
        ----------
        model: object
            User supplied model object for any transformer model.

        tokenizer: object
            User supplied tokenizer object which is used to tokenize source and target sentences to use
            the model to generate log odds for every tokenized target sentence ids from source sentences ids.

        device: "cpu" or "cuda" or None
            Used to generate scores either using cpu or gpu. By default, it infers if system has a gpu and accordingly sets device.
        """
        #assigning function
        self.f = f

        self.source_sentence = None
        self.target_sentence_ids = None

        if model is None or tokenizer is None:
            raise ValueError(
                "Model or Tokenizer assigned is None."
            )
        # We cannot generate if the model does not have a LM head
        if model.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )
        # assign model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        # set device
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load model onto device
        self.model = self.model.to(self.device)
        null_tokens = self.tokenizer.encode("")
        # set prefix and sufix tokens based on null tokens
        # example for distillgpt2: null_tokens=[], for BART: null_tokens = [0,2] and for MarianMT: null_tokens=[0] 
        # used to slice tokens belonging to sentence after passing through tokenizer.encode()
        if len(null_tokens) == 0:
            self.keep_prefix = 0
            self.keep_suffix = 0
        elif len(null_tokens) == 1:
            null_token = null_tokens[0]
            assert (('eos_token' in self.tokenizer.special_tokens_map) or ('bos_token' in self.tokenizer.special_tokens_map)), "No eos token or bos token found in tokenizer. Cannot assign keep_prefix and keep_suffix for given tokenizer!"
            if ('eos_token' in self.tokenizer.special_tokens_map) and (self.tokenizer.decode(null_token) == self.tokenizer.special_tokens_map['eos_token']):
                self.keep_prefix = 0
                self.keep_suffix = 1
            elif ('bos_token' in self.tokenizer.special_tokens_map) and (self.tokenizer.decode(null_token) == self.tokenizer.special_tokens_map['bos_token']):
                self.keep_prefix = 1
                self.keep_suffix = 0
            else:
                raise ValueError(
                    "Unable to assign null tokens to prefix or suffix of a sentence as null tokens dont map to either of eos_token or bos_token"
                )
        else:
            assert len(null_tokens) % 2 == 0, "An odd number of boundary tokens greater than 2 are added to the null string!"
            self.keep_prefix = len(null_tokens) // 2
            self.keep_suffix = len(null_tokens) // 2

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
        self.model.eval()
        if self.model.config.is_encoder_decoder:
            # assigning decoder start token id as it is needed for encoder decoder model generation
            decoder_start_token_id = None
            if hasattr(self.model.config, "decoder_start_token_id") and self.model.config.decoder_start_token_id is not None:
                decoder_start_token_id = self.model.config.decoder_start_token_id
            elif hasattr(self.model.config, "bos_token_id") and self.model.config.bos_token_id is not None:
                decoder_start_token_id = self.model.config.bos_token_id
            elif (hasattr(self.model.config, "decoder") and hasattr(self.model.config.decoder, "bos_token_id") and self.model.config.decoder.bos_token_id is not None):
                decoder_start_token_id = self.model.config.decoder.bos_token_id
            else:
                raise ValueError(
                    "No decoder_start_token_id or bos_token_id defined in config for encoder-decoder generation"
                )
            # concat decoder start token id to target sentence ids
            target_sentence_start_id = torch.tensor([[decoder_start_token_id]]).to(self.device)
            target_sentence_ids = torch.cat((target_sentence_start_id,target_sentence_ids),dim=-1)
            # generate outputs and logits
            with torch.no_grad():
                outputs = self.model(input_ids=source_sentence_ids, decoder_input_ids=target_sentence_ids, labels=target_sentence_ids, return_dict=True)
            logits=outputs.logits.detach().cpu().numpy().astype('float64')
        else:
            # check if source sentence ids are null then add bos token id to decoder
            if source_sentence_ids.shape[1]==0:
                if hasattr(self.model.config.bos_token_id) and self.model.config.bos_token_id is not None:
                    source_sentence_ids = torch.tensor([[self.model.config.bos_token_id]]).to(self.device)
                else:
                    raise ValueError(
                    "Context ids (source sentence ids) are null and no bos token defined in model config"
                )
            # combine source and target sentence ids  to pass into decoder eg: in case of distillgpt2
            combined_sentence_ids = torch.cat((source_sentence_ids,target_sentence_ids),dim=-1)
            # generate outputs and logits
            with torch.no_grad():
                outputs = self.model(input_ids=combined_sentence_ids, return_dict=True)
            # extract only logits corresponding to target sentence ids
            logits=outputs.logits.detach().cpu().numpy()[:,source_sentence_ids.shape[1]-1:,:].astype('float64')
        del outputs
        return logits

    def get_target_sentence_ids(self, target_sentence):
        """ The function tokenizes target sentence.

        It encodes entire target sentence into ids but only extracts ids not corresponding
        to null tokens for target sentence.

        Parameters:
        ----------
        target_sentence: string or tensor
            Target sentence or target sentence ids for which logits are generated using the decoder.

        Returns:
        -------
        tensor
            Tensor of target sentence ids.
        """
        # only encode target sentence if ids not provided
        if isinstance(target_sentence,str):
            if self.keep_suffix > 0:
                target_sentence_ids = torch.tensor([self.tokenizer.encode(target_sentence)])[:,self.keep_prefix:-self.keep_suffix]
            else:
                target_sentence_ids = torch.tensor([self.tokenizer.encode(target_sentence)])[:,self.keep_prefix:]
        else:
            target_sentence_ids = target_sentence

        return target_sentence_ids.to(self.device)

    def get_source_sentence_ids(self, source_sentence):
        """ The function tokenizes source sentence.

        Parameters:
        ----------
        source_sentence: string or tensor
            Source sentence or source sentence ids fed to the model.

        Returns:
        -------
        tensor
            Tensor of source sentence ids.
        """
        # only encode source sentence if ids not provided
        if isinstance(source_sentence,str):
            source_sentence_ids = torch.tensor([self.tokenizer.encode(source_sentence)])
        else:
            source_sentence_ids = source_sentence

        return source_sentence_ids.to(self.device)

    def get_output_names(self, sentence):
        """ The function returns tokens for sentence or sentence ids.

        Parameters:
        ----------
        sentence: string or tensor
            Sentence or sentence ids to tokenize.

        Returns:
        -------
        list
            List of tokens.
        """
        output_names = None
        if isinstance(sentence,str):
            output_names = self.tokenizer.tokenize(sentence) 
        elif torch.is_tensor(sentence):
            # sentence is a list of ids
                output_names =  self.tokenizer.convert_ids_to_tokens(sentence[0,:])
        else:
            raise ValueError(
                    "Sentence should be of type str or tensor of dim: (1,sentence length)"
                )
        return output_names

    def update_source_sentence_cache(self, source_sentence):
        """ The function updates source sentence and target sentence ids.

        It mimics the caching mechanism to update the source sentence and target sentence ids
        that are to be explained and which updates for every new row of explanation.

        Parameters:
        ----------
        source_sentence: string
            Source sentence for an explanation row.
        """
        # check if the source sentence has been updated (occurs when explaining a new row)
        if self.source_sentence != source_sentence:
            self.source_sentence = source_sentence
            self.target_sentence_ids = self.f(source_sentence)
            self.target_sentence_ids = self.get_target_sentence_ids(self.target_sentence_ids)

    def __call__(self, masked_X, X):
        """ The function generates scores to explain output text explantion algorithms.

        Its produces the log odds of generating f(source_sentence) from f(masked_source sentence)
        using the model when f is an api/outputs text and in case where f outputs decoder ids
        it produces the log odds of generating decode ids from source sentence using model.

        Parameters:
        ----------
        masked_X: string or tensor
            Masked input fed to the model.

        X: string or tensor
            Unmasked input.

        Returns:
        -------
        numpy array
            Numpy array of log odds.
        """
        #WARNING: NEEDS TO BE UPDATED BASED ON NEW VARIABLE NAMING

        # update cache for every new row
        self.update_source_sentence_cache(source_sentence)
        # get source sentence ids
        source_sentence_ids = self.get_source_sentence_ids(source_sentence)
        # generate logits
        logits = self.get_teacher_forced_logits(source_sentence_ids,self.target_sentence_ids)
        conditional_logits = []
        # pass logits through softmax, get the token corresponding score and convert back to log odds (as one vs all)
        for i in range(0,logits.shape[1]-1):
            probs = (np.exp(logits[0][i]).T / np.exp(logits[0][i]).sum(-1)).T
            logit_dist = sp.special.logit(probs)
            conditional_logits.append(logit_dist[self.target_sentence_ids[0,i].item()])
        del source_sentence_ids
        return np.array(conditional_logits)