import math
import re

import numpy as np

from .._serializable import Deserializer, Serializer
from ..utils import safe_isinstance
from ..utils.transformers import (
    SENTENCEPIECE_TOKENIZERS,
    getattr_silent,
    parse_prefix_suffix_for_tokenizer,
)
from ._masker import Masker


class Text(Masker):
    """ This masks out tokens according to the given tokenizer.

    The masked variables are

    output_type : "string" (default) or "token_ids"

    """
    def __init__(self, tokenizer=None, mask_token=None, collapse_mask_token="auto", output_type="string"):
        """ Build a new Text masker given an optional passed tokenizer.

        Parameters
        ----------
        tokenizer : callable or None
            The tokenizer used to break apart strings during masking. The passed tokenizer must support a minimal
            subset of the HuggingFace Transformers PreTrainedTokenizerBase API. This minimal subset means the
            tokenizer must return a dictionary with 'input_ids' and then either include
            an 'offset_mapping' entry in the same dictionary or provide a .convert_ids_to_tokens or .decode method.

        mask_token : string, int, or None
            The sub-string or integer token id used to mask out portions of a string. If None it will use the
            tokenizer's .mask_token attribute, if defined, or "..." if the tokenizer does not have a .mask_token
            attribute.

        collapse_mask_token : True, False, or "auto"
            If True, when several consecutive tokens are masked only one mask token is used to replace the entire
            series of original tokens.
        """

        if tokenizer is None:
            self.tokenizer = SimpleTokenizer()
        elif callable(tokenizer):
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = SimpleTokenizer(tokenizer)
            except Exception:
                raise Exception( # pylint: disable=raise-missing-from
                    "The passed tokenizer cannot be wrapped as a masker because it does not have a __call__ " + \
                    "method, not can it be interpreted as a splitting regexp!"
                )

        self.output_type = output_type
        self.collapse_mask_token = collapse_mask_token
        self.input_mask_token = mask_token
        self.mask_token = mask_token # could be recomputed later in this function
        self.mask_token_id = mask_token if isinstance(mask_token, int) else None
        parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(self.tokenizer)

        self.keep_prefix = parsed_tokenizer_dict['keep_prefix']
        self.keep_suffix = parsed_tokenizer_dict['keep_suffix']
        # self.prefix_strlen = parsed_tokenizer_dict['prefix_strlen']
        # self.suffix_strlen = parsed_tokenizer_dict['suffix_strlen']
        #null_tokens = parsed_tokenizer_dict['null_tokens']

        self.text_data = True

        if mask_token is None:
            if getattr_silent(self.tokenizer, "mask_token") is not None:
                self.mask_token = self.tokenizer.mask_token
                self.mask_token_id = getattr_silent(self.tokenizer, "mask_token_id")
                if self.collapse_mask_token == "auto":
                    self.collapse_mask_token = False
            else:
                self.mask_token = "..."
        else:
            self.mask_token = mask_token

        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer(self.mask_token)["input_ids"][self.keep_prefix]

        if self.collapse_mask_token == "auto":
            self.collapse_mask_token = True

        # assign mask token segment
        # if self.keep_suffix > 0:
        #     self.mask_token_segment = self.token_segments(self.mask_token)[self.keep_prefix:-self.keep_suffix]
        # else:
        #     self.mask_token_segment = self.token_segments(self.mask_token)[self.keep_prefix:]

        # note if this masker can use a different background for different samples
        self.fixed_background = self.mask_token_id is None

        self.default_batch_size = 5

        # cache variables
        self._s = None
        self._tokenized_s_full = None
        self._tokenized_s = None
        self._segments_s = None

        # flag that we return outputs that will not get changed by later masking calls
        self.immutable_outputs = True

    def __call__(self, mask, s):
        mask = self._standardize_mask(mask, s)
        self._update_s_cache(s)

        # if we have a fixed prefix or suffix then we need to grow the mask to account for that
        if self.keep_prefix > 0 or self.keep_suffix > 0:
            mask = mask.copy()
            mask[:self.keep_prefix] = True
            mask[-self.keep_suffix:] = True

        if self.output_type == "string":
            # if self.mask_token == "":
            #     out = self._segments_s[mask]
            # else:
            #     #out = np.array([self._segments_s[i] if mask[i] else self.mask_token for i in range(len(mask))])
            out_parts = []
            is_previous_appended_token_mask_token = False
            sep_token = getattr_silent(self.tokenizer, "sep_token")
            for i, v in enumerate(mask):
                # mask ignores separator tokens and keeps them unmasked
                if v or sep_token == self._segments_s[i]:
                    out_parts.append(self._segments_s[i])
                    is_previous_appended_token_mask_token = False
                else:
                    if not self.collapse_mask_token or (self.collapse_mask_token and not is_previous_appended_token_mask_token):
                        out_parts.append(" " + self.mask_token)
                        is_previous_appended_token_mask_token = True
            out = "".join(out_parts)

            # tokenizers which treat spaces like parts of the tokens and dont replace the special token while decoding need further postprocessing
            # by replacing whitespace encoded as '_' for sentencepiece tokenizer or 'Ġ' for sentencepiece like encoding (GPT2TokenizerFast)
            # with ' '
            if safe_isinstance(self.tokenizer, SENTENCEPIECE_TOKENIZERS):
                out = out.replace('▁', ' ')

            # replace sequence of spaces with a single space and strip beginning and end spaces
            out = re.sub(r"[\s]+", " ", out).strip() # TODOmaybe: should do strip?? (originally because of fast vs. slow tokenizer differences)

        else:
            if self.mask_token_id is None:
                out = self._tokenized_s[mask]
            else:
                out = np.array([self._tokenized_s[i] if mask[i] else self.mask_token_id for i in range(len(mask))])
                # print("mask len", len(out))
                # # crop the output if needed
                # if self.max_length is not None and len(out) > self.max_length:
                #     new_out = np.zeros(self.max_length)
                #     new_out[:] = out[:self.max_length]
                #     new_out[-self.keep_suffix:] = out[-self.keep_suffix:]
                #     out = new_out

        # for some sentences with strange configurations around the separator tokens, tokenizer encoding/decoding may contain
        # extra unnecessary tokens, for example ''. you may want to strip out spaces adjacent to separator tokens. Refer to PR
        # for more details.
        return (np.array([out]),)

    def data_transform(self, s):
        """ Called by explainers to allow us to convert data to better match masking (here this means tokenizing).
        """
        return (self.token_segments(s)[0],)

    def token_segments(self, s):
        """ Returns the substrings associated with each token in the given string.
        """

        try:
            token_data = self.tokenizer(s, return_offsets_mapping=True)
            offsets = token_data["offset_mapping"]
            offsets = [(0, 0) if o is None else o for o in offsets]
            parts = [s[offsets[i][0]:max(offsets[i][1], offsets[i+1][0])] for i in range(len(offsets)-1)]
            parts.append(s[offsets[len(offsets)-1][0]:offsets[len(offsets)-1][1]])
            return parts, token_data["input_ids"]
        except (NotImplementedError, TypeError): # catch lack of support for return_offsets_mapping
            token_ids = self.tokenizer(s)['input_ids']
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            else:
                tokens = [self.tokenizer.decode([id]) for id in token_ids]
            if hasattr(self.tokenizer, "get_special_tokens_mask"):
                special_tokens_mask = self.tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True)
                # avoid masking separator tokens, but still mask beginning of sentence and end of sentence tokens
                special_keep = [getattr_silent(self.tokenizer, 'sep_token'), getattr_silent(self.tokenizer, 'mask_token')]
                for i, v in enumerate(special_tokens_mask):
                    if v == 1 and (tokens[i] not in special_keep or i + 1 == len(special_tokens_mask)):
                        tokens[i] = ""

            # add spaces to separate the tokens (since we want segments not tokens)
            if safe_isinstance(self.tokenizer, SENTENCEPIECE_TOKENIZERS):
                for i, v in enumerate(tokens):
                    if v.startswith("_"):
                        tokens[i] = " " + tokens[i][1:]
            else:
                for i, v in enumerate(tokens):
                    if v.startswith("##"):
                        tokens[i] = tokens[i][2:]
                    elif v != "" and i != 0:
                        tokens[i] = " " + tokens[i]

            return tokens, token_ids

    def clustering(self, s):
        """ Compute the clustering of tokens for the given string.
        """
        self._update_s_cache(s)
        special_tokens = []
        sep_token = getattr_silent(self.tokenizer, "sep_token")
        if sep_token is None:
            special_tokens = []
        else:
            special_tokens = [sep_token]

        # convert the text segments to tokens that the partition tree function expects
        tokens = []
        space_end = re.compile(r"^.*\W$")
        letter_start = re.compile(r"^[A-Za-z]")
        for i, v in enumerate(self._segments_s):
            if i > 0 and space_end.match(self._segments_s[i-1]) is None and letter_start.match(v) is not None and tokens[i-1] != "":
                tokens.append("##" + v.strip())
            else:
                tokens.append(v.strip())

        pt = partition_tree(tokens, special_tokens)

        # use the rescaled size of the clusters as their height since the merge scores are just a
        # heuristic and not scaled well
        pt[:, 2] = pt[:, 3]
        pt[:, 2] /= pt[:, 2].max()

        return pt

    # unused because restricts meaningful perturbations
    # def _mark_uninvertable(self, clustering):
    #     """ This marks which clusters have non-invertable mappings through the tokenizer when masked.

    #     It seems like a bug that you can decode and then encode a set of token ids and not get what
    #     you started with...but this is possible with word endings in the transformers implementation
    #     of BERT for example. So here we mark such uninvertable clusters with negative values.
    #     """

    #     M = len(self._tokenized_s)
    #     assert len(clustering)+1 == M

    #     def recursive_mark(ind):
    #         if ind < M:
    #             return list(self._tokenized_s[ind:ind+1])

    #         lind = int(clustering[ind-M, 0])
    #         rind = int(clustering[ind-M, 1])
    #         ltokens = recursive_mark(lind)
    #         rtokens = recursive_mark(rind)

    #         tmp = ltokens + [self.mask_token_id]
    #         s2 = self.tokenizer.decode(tmp)
    #         e2 = self.tokenizer.encode(s2)
    #         if not np.all(e2[1:-1] == tmp):
    #             clustering[ind-M, 2] = -1 # set the distance of this cluster negative so it can't be split

    #         tmp = [self.mask_token_id] + rtokens
    #         s2 = self.tokenizer.decode(tmp)
    #         e2 = self.tokenizer.encode(s2)
    #         if not np.all(e2[1:-1] == tmp):
    #             clustering[ind-M, 2] = -1 # set the distance of this cluster negative so it can't be split

    #         return ltokens + rtokens

    #     recursive_mark(M+len(clustering)-1)

    def _update_s_cache(self, s):
        if self._s != s:
            self._s = s
            tokens, token_ids = self.token_segments(s)
            self._tokenized_s = np.array(token_ids)
            self._segments_s = np.array(tokens)

    def shape(self, s):
        """ The shape of what we return as a masker.

        Note we only return a single sample, so there is no expectation averaging.
        """
        self._update_s_cache(s)
        return (1, len(self._tokenized_s))

    def mask_shapes(self, s):
        """ The shape of the masks we expect.
        """
        self._update_s_cache(s)
        return [(len(self._tokenized_s),)]

    def invariants(self, s):
        """ The names of the features for each mask position for the given input string.
        """
        self._update_s_cache(s)

        invariants = np.zeros(len(self._tokenized_s), dtype=bool)
        if self.keep_prefix > 0:
            invariants[:self.keep_prefix] = True
        if self.keep_suffix > 0:
            invariants[-self.keep_suffix:] = True
        # mark separator tokens as invariant
        for i, v in enumerate(self._tokenized_s):
            if v == getattr_silent(self.tokenizer, "sep_token_id"):
                invariants[i] = True
        return invariants.reshape(1, -1)

    def feature_names(self, s):
        """ The names of the features for each mask position for the given input string.
        """
        self._update_s_cache(s)
        return [[v.strip() for v in self._segments_s]]

    def save(self, out_file):
        """ Save a Text masker to a file stream.
        """
        super().save(out_file)
        with Serializer(out_file, "shap.maskers.Text", version=0) as s:
            s.save("tokenizer", self.tokenizer)
            s.save("mask_token", self.input_mask_token)
            s.save("collapse_mask_token", self.collapse_mask_token)
            s.save("output_type", self.output_type)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """ Load a Text masker from a file stream.
        """
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.maskers.Text", min_version=0, max_version=0) as s:
            kwargs["tokenizer"] = s.load("tokenizer")
            kwargs["mask_token"] = s.load("mask_token")
            kwargs["collapse_mask_token"] = s.load("collapse_mask_token")
            kwargs["output_type"] = s.load("output_type")
        return kwargs


class SimpleTokenizer: # pylint: disable=too-few-public-methods
    """ A basic model agnostic tokenizer.
    """
    def __init__(self, split_pattern=r"\W+"):
        """ Create a tokenizer based on a simple splitting pattern.
        """
        self.split_pattern = re.compile(split_pattern)

    def __call__(self, s, return_offsets_mapping=True):
        """ Tokenize the passed string, optionally returning the offsets of each token in the original string.
        """
        pos = 0
        offset_ranges = []
        input_ids = []
        for m in re.finditer(self.split_pattern, s):
            start, end = m.span(0)
            offset_ranges.append((pos, start))
            input_ids.append(s[pos:start])
            pos = end
        if pos != len(s):
            offset_ranges.append((pos, len(s)))
            input_ids.append(s[pos:])

        out = {}
        out["input_ids"] = input_ids
        if return_offsets_mapping:
            out["offset_mapping"] = offset_ranges
        return out


def post_process_sentencepiece_tokenizer_output(s):
    """ replaces whitespace encoded as '_' with ' ' for sentencepiece tokenizers.
    """
    s = s.replace('▁', ' ')
    return s

openers = {
    "(": ")"
}
closers = {
    ")": "("
}
enders = [".", ","]
connectors = ["but", "and", "or"]

class Token:
    """ A token representation used for token clustering.
    """
    def __init__(self, value):
        self.s = value
        if value in openers or value in closers:
            self.balanced = False
        else:
            self.balanced = True

    def __str__(self):
        return self.s

    def __repr__(self):
        if not self.balanced:
            return self.s + "!"
        return self.s

class TokenGroup:
    """ A token group (substring) representation used for token clustering.
    """
    def __init__(self, group, index=None):
        self.g = group
        self.index = index

    def __repr__(self):
        return self.g.__repr__()

    def __getitem__(self, index):
        return self.g[index]

    def __add__(self, o):
        return TokenGroup(self.g + o.g)

    def __len__(self):
        return len(self.g)

def merge_score(group1, group2, special_tokens):
    """ Compute the score of merging two token groups.

    special_tokens: tokens (such as separator tokens) that should be grouped last
    """
    score = 0
    # ensures special tokens are combined last, so 1st subtree is 1st sentence and 2nd subtree is 2nd sentence
    if len(special_tokens) > 0:
        if group1[-1].s in special_tokens and group2[0].s in special_tokens:
            score -= math.inf # subtracting infinity to create lowest score and ensure combining these groups last

    # merge broken-up parts of words first
    if group2[0].s.startswith("##"):
        score += 20

    # merge apostrophe endings next
    if group2[0].s == "'" and (len(group2) == 1 or (len(group2) == 2 and group2[1].s in ["t", "s"])):
        score += 15
    if group1[-1].s == "'" and group2[0].s in ["t", "s"]:
        score += 15

    start_ctrl = group1[0].s.startswith("[") and group1[0].s.endswith("]")
    end_ctrl = group2[-1].s.startswith("[") and group2[-1].s.endswith("]")

    if (start_ctrl and not end_ctrl) or (end_ctrl and not start_ctrl):
        score -= 1000
    if group2[0].s in openers and not group2[0].balanced:
        score -= 100
    if group1[-1].s in closers and not group1[-1].balanced:
        score -= 100

    # attach surrounding an openers and closers a bit later
    if group1[0].s in openers and group2[-1] not in closers:
        score -= 2

    # reach across connectors later
    if group1[-1].s in connectors or group2[0].s in connectors:
        score -= 2

    # reach across commas later
    if group1[-1].s == ",":
        score -= 10
    if group2[0].s == ",":
        if len(group2) > 1: # reach across
            score -= 10
        else:
            score -= 1

    # reach across sentence endings later
    if group1[-1].s in [".", "?", "!"]:
        score -= 20
    if group2[0].s in [".", "?", "!"]:
        if len(group2) > 1: # reach across
            score -= 20
        else:
            score -= 1

    score -= len(group1) + len(group2)
    #print(group1, group2, score)
    return score

def merge_closest_groups(groups, special_tokens):
    """ Finds the two token groups with the best merge score and merges them.
    """
    scores = [merge_score(groups[i], groups[i+1], special_tokens) for i in range(len(groups)-1)]
    #print(scores)
    ind = np.argmax(scores)
    groups[ind] = groups[ind] + groups[ind+1]
    #print(groups[ind][0].s in openers, groups[ind][0])
    if groups[ind][0].s in openers and groups[ind+1][-1].s == openers[groups[ind][0].s]:
        groups[ind][0].balanced = True
        groups[ind+1][-1].balanced = True


    groups.pop(ind+1)

def partition_tree(decoded_tokens, special_tokens):
    """ Build a heriarchial clustering of tokens that align with sentence structure.

    Note that this is fast and heuristic right now.
    TODO: Build this using a real constituency parser.
    """
    token_groups = [TokenGroup([Token(t)], i) for i, t in enumerate(decoded_tokens)]
#     print(token_groups)
    M = len(decoded_tokens)
    new_index = M
    clustm = np.zeros((M-1, 4))
    for i in range(len(token_groups)-1):
        scores = [merge_score(token_groups[i], token_groups[i+1], special_tokens) for i in range(len(token_groups)-1)]
#         print(scores)
        ind = np.argmax(scores)

        lind = token_groups[ind].index
        rind = token_groups[ind+1].index
        clustm[new_index-M, 0] = token_groups[ind].index
        clustm[new_index-M, 1] = token_groups[ind+1].index
        clustm[new_index-M, 2] = -scores[ind]
        clustm[new_index-M, 3] = (clustm[lind-M, 3] if lind >= M else 1) + (clustm[rind-M, 3] if rind >= M else 1)

        token_groups[ind] = token_groups[ind] + token_groups[ind+1]
        token_groups[ind].index = new_index

        # track balancing of openers/closers
        if token_groups[ind][0].s in openers and token_groups[ind+1][-1].s == openers[token_groups[ind][0].s]:
            token_groups[ind][0].balanced = True
            token_groups[ind+1][-1].balanced = True

        token_groups.pop(ind+1)
        new_index += 1

    # negative means we should never split a group, so we add 10 to ensure these are very tight groups
    # (such as parts of the same word)
    clustm[:, 2] = clustm[:, 2] + 10

    return clustm
