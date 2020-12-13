

MODELS_FOR_SEQ_TO_SEQ_CAUSAL_LM = [
        "transformers.T5ForConditionalGeneration",
        "transformers.PegasusForConditionalGeneration",
        "transformers.MarianMTModel",
        "transformers.MBartForConditionalGeneration",
        "transformers.BlenderbotForConditionalGeneration",
        "transformers.BartForConditionalGeneration",
        "transformers.FSMTForConditionalGeneration",
        "transformers.EncoderDecoderModel",
        "transformers.XLMProphetNetForConditionalGeneration",
        "transformers.ProphetNetForConditionalGeneration",
        "transformers.TFMT5ForConditionalGeneration",
        "transformers.TFT5ForConditionalGeneration",
        "transformers.TFMarianMTModel",
        "transformers.TFMBartForConditionalGeneration",
        "transformers.TFPegasusForConditionalGeneration",
        "transformers.TFBlenderbotForConditionalGeneration",
        "transformers.TFBartForConditionalGeneration"
    ]

MODELS_FOR_CAUSAL_LM = [
        "transformers.CamembertForCausalLM",
        "transformers.XLMRobertaForCausalLM",
        "transformers.RobertaForCausalLM",
        "transformers.BertLMHeadModel",
        "transformers.OpenAIGPTLMHeadModel",
        "transformers.GPT2LMHeadModel",
        "transformers.TransfoXLLMHeadModel",
        "transformers.XLNetLMHeadModel",
        "transformers.XLMWithLMHeadModel",
        "transformers.CTRLLMHeadModel",
        "transformers.ReformerModelWithLMHead",
        "transformers.BertGenerationDecoder",
        "transformers.XLMProphetNetForCausalLM",
        "transformers.ProphetNetForCausalLM",
        "transformers.TFBertLMHeadModel",
        "transformers.TFOpenAIGPTLMHeadModel",
        "transformers.TFGPT2LMHeadModel",
        "transformers.TFTransfoXLLMHeadModel",
        "transformers.TFXLNetLMHeadModel",
        "transformers.TFXLMWithLMHeadModel",
        "transformers.TFCTRLLMHeadModel",
    ]

SENTENCEPIECE_TOKENIZERS = [
    "transformers.MarianTokenizer",
    "transformers.T5Tokenizer",
    "transformers.XLNetTokenizer",
    "transformers.AlbertTokenizer"
]

def parse_prefix_suffix_for_tokenizer(tokenizer):
    null_tokens = tokenizer.encode("")
    keep_prefix, keep_suffix, prefix_strlen, suffix_strlen = None, None, None, None
    # set prefix and suffix tokens based on null tokens
    # example for distillgpt2: null_tokens=[], for BART: null_tokens = [0,2] and for MarianMT: null_tokens=[0] 
    # used to slice tokens belonging to sentence after passing through tokenizer.encode()
    if len(null_tokens)==1:
        null_token = null_tokens[0]
        assert (('eos_token' in tokenizer.special_tokens_map) or ('bos_token' in tokenizer.special_tokens_map)), "No eos token or bos token found in tokenizer!"
        if ('eos_token' in tokenizer.special_tokens_map) and (tokenizer.decode(null_token) == tokenizer.special_tokens_map['eos_token']):
            keep_prefix = 0
            keep_suffix = 1
            prefix_strlen = 0
            suffix_strlen = len(tokenizer.decode(null_tokens[-keep_suffix:]))
        elif ('bos_token' in tokenizer.special_tokens_map) and (tokenizer.decode(null_token) == tokenizer.special_tokens_map['bos_token']):
            keep_prefix = 1
            keep_suffix = 0
            prefix_strlen = len(tokenizer.decode(null_tokens[:keep_prefix]))
            suffix_strlen = 0
    else:
        assert len(null_tokens) % 2 == 0, "An odd number of boundary tokens are added to the null string!"
        keep_prefix = len(null_tokens) // 2
        keep_suffix = len(null_tokens) // 2
        prefix_strlen = len(tokenizer.decode(null_tokens[:keep_prefix]))
        suffix_strlen = len(tokenizer.decode(null_tokens[-keep_suffix:]))

    return {
        'keep_prefix' : keep_prefix, 
        'keep_suffix' : keep_suffix,
        'prefix_strlen' : prefix_strlen,
        'suffix_strlen' : suffix_strlen,
        'null_tokens' : null_tokens
    }