import numpy as np
import scipy.sparse
from numba import jit


class MaskedModel():
    """ This is a utility class that combines a model, a masker object, and a current input.

    The combination of a model, a masker object, and a current input produces a binary set
    function that can be called to mask out any set of inputs. This class attempts to be smart
    about only evaluating the model for background samples when the inputs changed (note this
    requires the masker object to have a .invariants method).
    """

    delta_mask_noop_value = 2147483647 # used to encode a noop for delta masking

    def __init__(self, model, masker, link, *args):
        self.model = model
        self.masker = masker
        self.link = link
        self.args = args
        # if the masker supports it, save what positions vary from the background
        if callable(getattr(self.masker, "invariants", None)):
            self._variants = ~self.masker.invariants(*args)
            self._variants_column_sums = self._variants.sum(0)
            self._variants_row_inds = [
                self._variants[:,i] for i in range(self._variants.shape[1])
            ]
        else:
            self._variants = None
        
        # compute the length of the mask (and hence our length)
        if hasattr(self.masker, "shape"):
            if callable(self.masker.shape):
                mshape = self.masker.shape(*self.args)
                self._masker_rows = mshape[0]
                self._masker_cols = mshape[1]
            else:
                mshape = self.masker.shape
                self._masker_rows = mshape[0]
                self._masker_cols = mshape[1]
        else:
            self._masker_rows = None# # just assuming...
            self._masker_cols = sum(np.prod(a.shape) for a in self.args)

    def __call__(self, masks, batch_size=None):

        # if we are passed a 1D array of indexes then we are delta masking and have a special implementation
        if len(masks.shape) == 1:
            if getattr(self.masker, "supports_delta_masking", False):
                return self._delta_masking_call(masks, batch_size=batch_size)

            # we need to convert from delta masking to a full masking call because we were given a delta masking
            # input but the masker does not support delta masking
            else: 
                full_masks = np.zeros((int(np.sum(masks >= 0)), self._masker_cols), dtype=bool)
                _convert_delta_mask_to_full(masks, full_masks)
                return self._full_masking_call(full_masks, batch_size=batch_size)

        else:
            return self._full_masking_call(masks, batch_size=batch_size)

    def _full_masking_call(self, masks, batch_size=None):

        # # TODO: we need to do batching here
        # else:
        #out = []
        do_delta_masking = getattr(self.masker, "reset_delta_masking", None) is not None
        last_mask = np.zeros(masks.shape[1], dtype=bool)
        batch_positions = np.zeros(len(masks)+1, dtype=int)
        #masked_inputs = np.zeros((len(masks) * self.masker.max_output_samples, masks.shape[1]))
        all_masked_inputs = []
        #batch_masked_inputs = []
        num_mask_samples = np.zeros(len(masks), dtype=int)
        num_varying_rows = np.zeros(len(masks), dtype=int)
        varying_rows = []
        if self._variants is not None:
            delta_tmp = self._variants.copy().astype(int)
        for i, mask in enumerate(masks):

            # mask the inputs
            delta_mask = mask ^ last_mask
            if do_delta_masking and delta_mask.sum() == 1:
                delta_ind = np.nonzero(delta_mask)[0][0]
                masked_inputs = self.masker(delta_ind, *self.args).copy()
            else:
                masked_inputs = self.masker(mask, *self.args)

            # wrap the masked inputs if they are not already in a tuple
            if not isinstance(masked_inputs, tuple):
                masked_inputs = (masked_inputs.copy(),)
 
            # masked_inputs = self.masker(mask, *self.args)
            num_mask_samples[i] = len(masked_inputs[0])

            # see which rows have been updated, so we can only evaluate the model on the rows we need to
            if i == 0 or self._variants is None:
                varying_rows.append(np.ones(num_mask_samples[i], dtype=bool))
                num_varying_rows[i] = num_mask_samples[i]
            else:
                # a = np.any(self._variants & delta_mask, axis=1)
                # a = np.any(self._variants & delta_mask, axis=1)
                # a = np.any(self._variants & delta_mask, axis=1)
                # (self._variants & delta_mask).sum(1) > 0

                np.bitwise_and(self._variants, delta_mask, out=delta_tmp)
                varying_rows.append(np.any(delta_tmp, axis=1))#np.any(self._variants & delta_mask, axis=1))
                num_varying_rows[i] = varying_rows[-1].sum()
                # for i in range(20):
                #     varying_rows[-1].sum()
            last_mask[:] = mask
            
            batch_positions[i+1] = batch_positions[i] + num_varying_rows[i]

            # subset the masked input to only the rows that vary
            if num_varying_rows[i] != num_mask_samples[i]:
                if len(self.args) == 1:
                    # _ = masked_inputs[varying_rows[-1]]
                    # _ = masked_inputs[varying_rows[-1]]
                    # _ = masked_inputs[varying_rows[-1]]
                    masked_inputs_subset = masked_inputs[0][varying_rows[-1]]
                else:
                    masked_inputs_subset = [v[varying_rows[-1]] for v in zip(*masked_inputs[0])]
                masked_inputs = (masked_inputs_subset,) + masked_inputs[1:]

            # define no. of list based on output of masked_inputs
            if len(all_masked_inputs) != len(masked_inputs):
                all_masked_inputs = [[] for m in range(len(masked_inputs))]

            for i in range(len(masked_inputs)):
                all_masked_inputs[i].append(masked_inputs[i])
        
        joined_masked_inputs = self._stack_inputs(*all_masked_inputs)
        outputs = self.model(*joined_masked_inputs)
        _assert_output_input_match(joined_masked_inputs, outputs)

        averaged_outs = np.zeros((len(batch_positions)-1,) + outputs.shape[1:])
        max_outs = self._masker_rows if self._masker_rows is not None else max(len(r) for r in varying_rows) 
        last_outs = np.zeros((max_outs,) + outputs.shape[1:])
        varying_rows = np.array(varying_rows)
        
        _build_fixed_output(averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, self.link)

        return averaged_outs

        # return self._build_output(outputs, batch_positions, varying_rows)

    # def _build_varying_delta_mask_rows(self, masks):
    #     """ This builds the _varying_delta_mask_rows property which is a list of rows that
    #     could change for each delta set.
    #     """

    #     self._varying_delta_mask_rows = []
    #     i = -1
    #     masks_pos = 0
    #     while masks_pos < len(masks):
    #         i += 1

    #         delta_index = masks[masks_pos]
    #         masks_pos += 1

    #         # update the masked inputs
    #         varying_rows_set = []
    #         while delta_index < 0: # negative values mean keep going
    #             original_index = -delta_index + 1
    #             varying_rows_set.append(self._variants_row_inds[original_index])
    #             delta_index = masks[masks_pos]
    #             masks_pos += 1
    #         self._varying_delta_mask_rows.append(np.unique(np.concatenate(varying_rows_set)))

    
    def _delta_masking_call(self, masks, batch_size=None):
        # TODO: we need to do batching here

        assert getattr(self.masker, "supports_delta_masking", None) is not None, "Masker must support delta masking!"
        
        masked_inputs, varying_rows = self.masker(masks, *self.args)
        num_varying_rows = varying_rows.sum(1)

        subset_masked_inputs = [arg[varying_rows.reshape(-1)] for arg in masked_inputs]

        batch_positions = np.zeros(len(varying_rows)+1, dtype=int)
        for i in range(len(varying_rows)):
            batch_positions[i+1] = batch_positions[i] + num_varying_rows[i]
        
        # joined_masked_inputs = self._stack_inputs(all_masked_inputs)
        outputs = self.model(*subset_masked_inputs)
        _assert_output_input_match(subset_masked_inputs, outputs)

        averaged_outs = np.zeros((varying_rows.shape[0],) + outputs.shape[1:])
        last_outs = np.zeros((varying_rows.shape[1],) + outputs.shape[1:])
        #print("link", self.link)
        _build_fixed_output(averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, self.link)

        return averaged_outs
    
    def _stack_inputs(self, *inputs):
        return tuple([np.concatenate(v) for v in inputs])

    @property
    def mask_shapes(self):
        if hasattr(self.masker, "mask_shapes") and callable(self.masker.mask_shapes):
            return self.masker.mask_shapes(*self.args)
        else:
            return [a.shape for a in self.args] # TODO: this will need to get more flexible

    def __len__(self):
        """ How many binary inputs there are to toggle.

        By default we just match what the masker tells us. But if the masker doesn't help us
        out by giving a length then we assume is the number of data inputs.
        """
        return self._masker_cols

    def varying_inputs(self):
        if self._variants is None:
            return np.arange(self._masker_cols)
        else:
            return np.where(np.any(self._variants, axis=0))[0]

    def main_effects(self, inds=None):
        """ Compute the main effects for this model.
        """

        # if no indexes are given then we assume all indexes could be non-zero
        if inds is None:
            inds = np.arange(len(self))

        # mask each potentially nonzero input in isolation
        masks = np.zeros(2*len(inds), dtype=int)
        masks[0] = MaskedModel.delta_mask_noop_value
        last_ind = -1
        for i in range(len(inds)):
            if i > 0:
                masks[2*i] = -last_ind - 1 # turn off the last input
            masks[2*i+1] = inds[i] # turn on this input
            last_ind = inds[i]

        # compute the main effects for the given indexes
        outputs = self(masks)
        main_effects = outputs[1:] - outputs[0]
        
        # expand the vector to the full input size
        expanded_main_effects = np.zeros((len(self),) + outputs.shape[1:])
        for i,ind in enumerate(inds):
            expanded_main_effects[ind] = main_effects[i]
        
        return expanded_main_effects

def _assert_output_input_match(inputs, outputs):
    assert len(outputs) == len(inputs[0]), \
        f"The model produced {len(outputs)} output rows when given {len(inputs[0])} input rows! Check the implementation of the model you provided for errors."

def _convert_delta_mask_to_full(masks, full_masks):
    """ This converts a delta masking array to a full bool masking array.
    """
    
    i = -1
    masks_pos = 0
    while masks_pos < len(masks):
        i += 1

        if i > 0:
            full_masks[i] = full_masks[i-1]
        
        while masks[masks_pos] < 0:
            full_masks[i,-masks[masks_pos]-1] = ~full_masks[i,-masks[masks_pos]-1] # -value - 1 is the original index that needs flipped
            masks_pos += 1
        
        if masks[masks_pos] != MaskedModel.delta_mask_noop_value:
            full_masks[i,masks[masks_pos]] = ~full_masks[i,masks[masks_pos]]
        masks_pos += 1

#@jit # TODO: figure out how to jit this function, or most of it
def _build_delta_masked_inputs(masks, batch_positions, num_mask_samples, num_varying_rows, delta_indexes,
                               varying_rows, args, masker, variants, variants_column_sums):
    all_masked_inputs = [[] for a in args]
    dpos = 0
    i = -1
    masks_pos = 0
    while masks_pos < len(masks):
        i += 1
        
        dpos = 0
        delta_indexes[0] = masks[masks_pos]

        # update the masked inputs
        while delta_indexes[dpos] < 0: # negative values mean keep going
            delta_indexes[dpos] = -delta_indexes[dpos] - 1 # -value + 1 is the original index that needs flipped
            masker(delta_indexes[dpos], *args) 
            dpos += 1
            delta_indexes[dpos] = masks[masks_pos + dpos]
        masked_inputs = masker(delta_indexes[dpos], *args).copy()

        masks_pos += dpos + 1

        num_mask_samples[i] = len(masked_inputs)
        #print(i, dpos, delta_indexes[dpos])
        # see which rows have been updated, so we can only evaluate the model on the rows we need to
        if i == 0: 
            varying_rows[i,:] = True
            #varying_rows.append(np.arange(num_mask_samples[i]))
            num_varying_rows[i] = num_mask_samples[i]
            
        else: 
            # only one column was changed
            if dpos == 0: 
                
                varying_rows[i,:] = variants[:,delta_indexes[dpos]]
                #varying_rows.append(_variants_row_inds[delta_indexes[dpos]])
                num_varying_rows[i] = variants_column_sums[delta_indexes[dpos]]
                

            # more than one column was changed
            else: 
                varying_rows[i,:] = np.any(variants[:,delta_indexes[:dpos+1]], axis=1)
                #varying_rows.append(np.any(variants[:,delta_indexes[:dpos+1]], axis=1))
                num_varying_rows[i] = varying_rows[i,:].sum()

        batch_positions[i+1] = batch_positions[i] + num_varying_rows[i]

        # subset the masked input to only the rows that vary
        if num_varying_rows[i] != num_mask_samples[i]:
            if len(args) == 1:
                masked_inputs = masked_inputs[varying_rows[i,:]]
            else:
                masked_inputs = [v[varying_rows[i,:]] for v in zip(*masked_inputs)]

        # wrap the masked inputs if they are not already in a tuple
        if len(args) == 1:
            masked_inputs = (masked_inputs,)
        
        for j in range(len(masked_inputs)):
            all_masked_inputs[j].append(masked_inputs[j])

    return all_masked_inputs, i + 1 # i + 1 is the number of output rows after averaging


def _build_fixed_output(averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, link):
    if len(last_outs.shape) == 1:
        _build_fixed_single_output(averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, link)
    else:
        _build_fixed_multi_output(averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, link)

@jit # we can't use this when using a custom link function...
def _build_fixed_single_output(averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, link):
    # here we can assume that the outputs will always be the same size, and we need
    # to carry over evaluation outputs
    last_outs[:] = outputs[batch_positions[0]:batch_positions[1]]
    sample_count = last_outs.shape[0]
    multi_output = len(last_outs.shape) > 1
    averaged_outs[0] = np.mean(last_outs)
    for i in range(1, len(averaged_outs)):
        if batch_positions[i] < batch_positions[i+1]:
            if num_varying_rows[i] == sample_count:
                last_outs[:] = outputs[batch_positions[i]:batch_positions[i+1]]
            else:
                last_outs[varying_rows[i]] = outputs[batch_positions[i]:batch_positions[i+1]]
            averaged_outs[i] = link(np.mean(last_outs))
            averaged_outs[i] = np.mean(last_outs)
        else:
            averaged_outs[i] = averaged_outs[i-1]

@jit
def _build_fixed_multi_output(averaged_outs, last_outs, outputs, batch_positions, varying_rows, num_varying_rows, link):
    # here we can assume that the outputs will always be the same size, and we need
    # to carry over evaluation outputs
    last_outs[:] = outputs[batch_positions[0]:batch_positions[1]]
    sample_count = last_outs.shape[0]
    multi_output = len(last_outs.shape) > 1
    for j in range(last_outs.shape[-1]): # using -1 is important so 
        averaged_outs[0,j] = np.mean(last_outs[:,j]) # we can't just do np.mean(last_outs, 0) because that fails to numba compile
    for i in range(1, len(averaged_outs)):
        if batch_positions[i] < batch_positions[i+1]:
            if num_varying_rows[i] == sample_count:
                last_outs[:] = outputs[batch_positions[i]:batch_positions[i+1]]
            else:
                last_outs[varying_rows[i]] = outputs[batch_positions[i]:batch_positions[i+1]]
            averaged_outs[i] = link(np.mean(last_outs))
            for j in range(last_outs.shape[1]):
                averaged_outs[i,j] = np.mean(last_outs[:,j])
        else:
            averaged_outs[i] = averaged_outs[i-1]


def make_masks(cluster_matrix):

    # build the mask matrix recursively as an array of index lists
    global count
    count = 0
    M = cluster_matrix.shape[0] + 1
    mask_matrix_inds = np.zeros(2 * M - 1, dtype=np.object)
    rec_fill_masks(mask_matrix_inds, cluster_matrix, M)
    
    # convert the array of index lists into CSR format
    indptr = np.zeros(len(mask_matrix_inds) + 1, dtype=int)
    indices = np.zeros(np.sum([len(v) for v in mask_matrix_inds]), dtype=int)
    pos = 0
    for i in range(len(mask_matrix_inds)):
        inds = mask_matrix_inds[i]
        indices[pos:pos+len(inds)] = inds
        pos += len(inds)
        indptr[i+1] = pos
    mask_matrix = scipy.sparse.csr_matrix(
        (np.ones(len(indices), dtype=bool), indices, indptr),
        shape=(len(mask_matrix_inds), M)
    )

    return mask_matrix

def rec_fill_masks(mask_matrix, cluster_matrix, M, ind=None):
    if ind is None:
        ind = cluster_matrix.shape[0] - 1 + M
        
    if ind < M:
        mask_matrix[ind] = np.array([ind])
        return

    lind = int(cluster_matrix[ind-M,0])
    rind = int(cluster_matrix[ind-M,1])
    
    rec_fill_masks(mask_matrix, cluster_matrix, M, lind)
    mask_matrix[ind] = mask_matrix[lind]

    rec_fill_masks(mask_matrix, cluster_matrix, M, rind)
    mask_matrix[ind] = np.concatenate((mask_matrix[ind], mask_matrix[rind]))
