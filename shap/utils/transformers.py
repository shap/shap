from ._general import safe_isinstance

SENTENCEPIECE_TOKENIZERS = [
    "transformers.MarianTokenizer",
    "transformers.T5Tokenizer",
    "transformers.XLNetTokenizer",
    "transformers.AlbertTokenizer",
]


def is_transformers_lm(model):
    """Check if the given model object is a huggingface transformers language model."""
    if safe_isinstance(model, "transformers.PreTrainedModel") or safe_isinstance(
        model, "transformers.TFPreTrainedModel"
    ):
        from transformers import (
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        )

        return (
            type(model) in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values()
            or type(model) in MODEL_FOR_CAUSAL_LM_MAPPING.values()
        )
    return False


def parse_prefix_suffix_for_tokenizer(tokenizer):
    """Set prefix and suffix tokens based on null tokens.

    Example for distillgpt2: null_tokens=[], for BART: null_tokens = [0,2] and for MarianMT: null_tokens=[0]
    used to slice tokens belonging to sentence after passing through tokenizer.encode().
    """
    null_tokens = tokenizer("")["input_ids"]
    keep_prefix, keep_suffix = None, None

    if len(null_tokens) == 1:
        null_token = null_tokens[0]
        if hasattr(tokenizer, "special_tokens_map") and hasattr(tokenizer, "decode"):
            st_map = tokenizer.special_tokens_map
            assert ("eos_token" in st_map) or ("bos_token" in st_map), "No eos token or bos token found in tokenizer!"
            if ("eos_token" in st_map) and (tokenizer.decode(null_token) == st_map["eos_token"]):
                keep_prefix = 0
                keep_suffix = 1
                # prefix_strlen = 0
                # suffix_strlen = len(tokenizer.decode(null_tokens[-keep_suffix:]))
            elif ("bos_token" in st_map) and (tokenizer.decode(null_token) == st_map["bos_token"]):
                keep_prefix = 1
                keep_suffix = 0
                # prefix_strlen = len(tokenizer.decode(null_tokens[:keep_prefix]))
                # suffix_strlen = 0
        else:
            raise Exception(
                "The given tokenizer produces one token when applied to the empty string, but "
                "does not have a .special_tokens_map['eos_token'] or .special_tokens_map['bos_token'] "
                "property (and .decode) to specify if it is an eos (end) of bos (beginning) token!"
            )
    else:
        assert len(null_tokens) % 2 == 0, "An odd number of boundary tokens are added to the null string!"
        keep_prefix = len(null_tokens) // 2
        keep_suffix = len(null_tokens) // 2
        # prefix_strlen = len(tokenizer.decode(null_tokens[:keep_prefix]))
        # suffix_strlen = len(tokenizer.decode(null_tokens[-keep_suffix:]))

    return {
        "keep_prefix": keep_prefix,
        "keep_suffix": keep_suffix,
        # 'prefix_strlen' : prefix_strlen,
        # 'suffix_strlen' : suffix_strlen,
        "null_tokens": null_tokens,
    }


def getattr_silent(obj, attr):
    """This turns of verbose logging of missing attributes for huggingface transformers.

    This is motivated by huggingface transformers objects that print error warnings
    when we access unset properties.
    """
    reset_verbose = False
    if getattr(obj, "verbose", False):
        reset_verbose = True
        obj.verbose = False

    val = getattr(obj, attr, None)

    if reset_verbose:
        obj.verbose = True

    # fix strange huggingface bug where `obj.verbose = False` causes val to change from None to "None"
    if val == "None":
        val = None

    return val
