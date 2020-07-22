import numpy as np
from ._masker import Masker

class Text(Masker):
    """ This masks out tokens according to the given tokenizer.

    The masked variables are 
    
    output_type : "string" (default) or "token_ids"
        
    """
    def __init__(self, tokenizer, mask_token="auto", output_type="string"):
        self.mask_history = {}
        self.tokenizer = tokenizer
        null_tokens = tokenizer.encode("")
        self.output_type = output_type
        
        if mask_token == "auto":
            if hasattr(self.tokenizer, "mask_token_id"):
                self.mask_token = self.tokenizer.mask_token_id
            else:
                self.mask_token = None
        assert len(null_tokens) % 2 == 0, "An odd number of boundary tokens are added to the null string!"
        self.keep_prefix = len(null_tokens) // 2
        self.keep_suffix = len(null_tokens) // 2
        self.prefix_strlen = len(tokenizer.decode(null_tokens[:self.keep_prefix]))
        self.suffix_strlen = len(tokenizer.decode(null_tokens[-self.keep_suffix:]))

        # note if this masker can use different background for different samples
        self.variable_background = self.mask_token is not None
    
    def __call__(self, x, mask):
        mask = mask.copy()
        mask[:self.keep_prefix] = True
        mask[-self.keep_suffix:] = True
        if self.mask_token is None:
            out = x[mask]
        else:
#             out = []
#             need_mask_token = True
#             for i in range(len(mask)):
#                 if mask[i]:
#                     out.append(x[i])
#                     need_mask_token = True
#                 elif need_mask_token:
#                     out.append(self.mask_token)
#                     need_mask_token = False
#             out = np.array(out)
            out = np.array([x[i] if mask[i] else self.mask_token for i in range(len(mask))])
        #print(out)
        if self.output_type == "string":
            return np.array([self.tokenizer.decode(out)[self.prefix_strlen:][:-self.suffix_strlen].strip()])
        else:
            return np.array([out])
    
    def tokenize(self, s):
        return self.tokenizer.encode_plus(s, return_offsets_mapping=True)
    
    def token_segments(self, s):
        offsets = self.tokenizer.encode_plus(s, return_offsets_mapping=True)["offset_mapping"]
        offsets = [(0,0) if o is None else o for o in offsets]
        parts = [s[offsets[i][0]:max(offsets[i][1], offsets[i+1][0])] for i in range(len(offsets)-1)] 
        parts.append(s[offsets[len(offsets)-1][0]:offsets[len(offsets)-1][1]])
        return parts

    def partition_tree(self, x):
        decoded_x = [self.tokenizer.decode([v]) for v in x]
        return partition_tree(decoded_x)

openers = {
    "(": ")"
}
closers = {
    ")": "("
}
enders = [".", ","]
connectors = ["but", "and", "or"]

class Token():
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
    
class TokenGroup():
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

def merge_score(group1, group2):
    score = 0
    
    
    # merge broken-up parts of words first
    if group2[0].s.startswith("##"):
        score += 20
        
    # merge apostrophe endings next
    if group2[0].s == "'" and (len(group2) == 1 or (len(group2) == 2 and group2[1].s in ["t", "s"])):
        score += 15
    if group1[-1].s == "'" and group2[0].s in ["t", "s"]:
        score += 15
    
    start_ctrl = group1[0].s[0] == "[" and group1[0].s[-1] == "]"
    end_ctrl = group2[-1].s[0] == "[" and group2[-1].s[-1] == "]"
    if (start_ctrl and not end_ctrl) or (end_ctrl and not start_ctrl):
        score -= 1000
    if group2[0].s in openers and not group2[0].balanced:
        score -= 100
    if group1[-1].s in closers and not group1[-1].balanced:
        score -= 100
    
    # attach surrounding an openers and closers a bit later
    if group1[0].s in openers and not group2[-1] in closers:
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
    
def merge_closest_groups(groups):
    scores = [merge_score(groups[i], groups[i+1]) for i in range(len(groups)-1)]
    #print(scores)
    ind = np.argmax(scores)
    groups[ind] = groups[ind] + groups[ind+1]
    #print(groups[ind][0].s in openers, groups[ind][0])
    if groups[ind][0].s in openers and groups[ind+1][-1].s == openers[groups[ind][0].s]:
        groups[ind][0].balanced = True
        groups[ind+1][-1].balanced = True
        
    
    groups.pop(ind+1)    
    
def partition_tree(decoded_tokens):
    token_groups = [TokenGroup([Token(t)], i) for i,t in enumerate(decoded_tokens)]
#     print(token_groups)
    M = len(decoded_tokens)
    new_index = M
    clustm = np.zeros((M-1, 2), dtype=np.int)
    for i in range(len(token_groups)-1):
        scores = [merge_score(token_groups[i], token_groups[i+1]) for i in range(len(token_groups)-1)]
#         print(scores)
        ind = np.argmax(scores)

        clustm[new_index-M,0] = token_groups[ind].index
        clustm[new_index-M,1] = token_groups[ind+1].index

        token_groups[ind] = token_groups[ind] + token_groups[ind+1]
        token_groups[ind].index = new_index

        # track balancing of openers/closers
        if token_groups[ind][0].s in openers and token_groups[ind+1][-1].s == openers[token_groups[ind][0].s]:
            token_groups[ind][0].balanced = True
            token_groups[ind+1][-1].balanced = True

        token_groups.pop(ind+1)
        new_index += 1
#         print(token_groups)
    return clustm