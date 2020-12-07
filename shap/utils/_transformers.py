import torch
import numpy as np
import scipy as sp

class GenerateLogits:
    def __init__(self,model,tokenizer,device=None):
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
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
            assert len(null_tokens) % 2 == 0, "An odd number of boundary tokens greater than 2 are added to the null string!"
            self.keep_prefix = len(null_tokens) // 2
            self.keep_suffix = len(null_tokens) // 2

    def get_teacher_forced_logits(self,source_sentence_ids,target_sentence_ids):
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

    def get_sentence_ids(self, source_sentence,target_sentence):
        # only encode source sentence if ids not provided
        if isinstance(source_sentence,str):
            source_sentence_ids = torch.tensor([self.tokenizer.encode(source_sentence)])
        else:
            source_sentence_ids = source_sentence
        # only encode target sentence if ids not provided
        if isinstance(target_sentence,str):
            if self.keep_suffix > 0:
                target_sentence_ids = torch.tensor([self.tokenizer.encode(target_sentence)])[:,self.keep_prefix:-self.keep_suffix]
            else:
                target_sentence_ids = torch.tensor([self.tokenizer.encode(target_sentence)])[:,self.keep_prefix:]
        else:
            target_sentence_ids = target_sentence
        return source_sentence_ids.to(self.device), target_sentence_ids.to(self.device)

    def get_output_names(self, sentence):
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

    def generate_logits(self,source_sentence, target_sentence):
        # get sentence ids
        source_sentence_ids, target_sentence_ids = self.get_sentence_ids(source_sentence,target_sentence)
        # generate logits
        logits = self.get_teacher_forced_logits(source_sentence_ids,target_sentence_ids)
        conditional_logits = []
        # pass logits through softmax, get the token corresponding score and convert back to logit (as one vs all)
        for i in range(0,logits.shape[1]-1):
            probs = (np.exp(logits[0][i]).T / np.exp(logits[0][i]).sum(-1)).T
            logit_dist = sp.special.logit(probs)
            conditional_logits.append(logit_dist[target_sentence_ids[0,i].item()])
        del source_sentence_ids
        return np.array(conditional_logits) 