import types

from ..utils._exceptions import InvalidMaskerError
from ._masker import Masker


class Composite(Masker):
    """ This merges several maskers for different inputs together into a single composite masker.

    This is not yet implemented.
    """

    def __init__(self, *maskers):

        self.maskers = maskers

        self.arg_counts = []
        self.total_args = 0
        self.text_data = False
        self.image_data = False
        all_have_clustering = True
        for masker in self.maskers:
            all_args = masker.__call__.__code__.co_argcount

            if masker.__call__.__defaults__ is not None: # in case there are no kwargs
                kwargs = len(masker.__call__.__defaults__)
            else:
                kwargs = 0
            num_args = all_args - kwargs - 2
            self.arg_counts.append(num_args) # -2 is for the self and mask arg
            self.total_args += num_args

            if not hasattr(masker, "clustering"):
                all_have_clustering = False

            self.text_data = self.text_data or getattr(masker, "text_data", False)
            self.image_data = self.image_data or getattr(masker, "image_data", False)

        if all_have_clustering:
            self.clustering = types.MethodType(joint_clustering, self)

    def shape(self, *args):
        """ Compute the shape of this masker as the sum of all the sub masker shapes.
        """
        assert len(args) == self.total_args, "The number of passed args is incorrect!"

        rows = None
        cols = 0
        pos = 0
        for i, masker in enumerate(self.maskers):
            if callable(masker.shape):
                shape = masker.shape(*args[pos:pos+self.arg_counts[i]])
            else:
                shape = masker.shape
            if rows is None:
                rows = shape[0]
            else:
                assert shape[1] == 0 or rows == shape[0], "All submaskers of a Composite masker must return the same number of rows!"
            cols += shape[1]
            pos += self.arg_counts[i]
        return rows, cols

    def mask_shapes(self, *args):
        """ The shape of the masks we expect.
        """
        out = []
        pos = 0
        for i, masker in enumerate(self.maskers):
            out.extend(masker.mask_shapes(*args[pos:pos+self.arg_counts[i]]))
        return out

    def data_transform(self, *args):
        """ Transform the argument
        """
        arg_pos = 0
        out = []
        for i, masker in enumerate(self.maskers):
            masker_args = args[arg_pos:arg_pos+self.arg_counts[i]]
            if hasattr(masker, "data_transform"):
                out.extend(masker.data_transform(*masker_args))
            else:
                out.extend(masker_args)
            arg_pos += self.arg_counts[i]

        return out

    def __call__(self, mask, *args):
        mask = self._standardize_mask(mask, *args)
        assert len(args) == self.total_args, "The number of passed args is incorrect!"

        # compute all the shapes and confirm they align
        arg_pos = 0
        shapes = []
        num_rows = None
        for i, masker in enumerate(self.maskers):
            masker_args = args[arg_pos:arg_pos+self.arg_counts[i]]
            if callable(masker.shape):
                shapes.append(masker.shape(*masker_args))
            else:
                shapes.append(masker.shape)

            if num_rows is None:
                num_rows = shapes[-1][0]
            elif num_rows == 1 and shapes[-1][0] is not None:
                num_rows = shapes[-1][0]

            if shapes[-1][0] != num_rows and shapes[-1][0] != 1 and shapes[-1][0] is not None:
                raise InvalidMaskerError("The composite masker can only join together maskers with a compatible number of background rows!")
            arg_pos += self.arg_counts[i]

        # call all the submaskers and combine their outputs
        arg_pos = 0
        mask_pos = 0
        masked = []
        for i, masker in enumerate(self.maskers):
            masker_args = args[arg_pos:arg_pos+self.arg_counts[i]]
            masked_out = masker(mask[mask_pos:mask_pos+shapes[i][1]], *masker_args)
            if num_rows > 1 and (shapes[i][0] == 1 or shapes[i][0] is None):
                masked_out = tuple([m[0] for _ in range(num_rows)] for m in masked_out)
            masked.extend(masked_out)

            mask_pos += shapes[i][1]
            arg_pos += self.arg_counts[i]

        return tuple(masked)

def joint_clustering(self, *args):
    """ Return a joint clustering that merges the clusterings of all the submaskers.
    """

    single_clustering = []
    arg_pos = 0
    for i, masker in enumerate(self.maskers):
        masker_args = args[arg_pos:arg_pos+self.arg_counts[i]]
        if callable(masker.clustering):
            clustering = masker.clustering(*masker_args)
        else:
            clustering = masker.clustering

        if len(single_clustering) == 0:
            single_clustering = clustering
        elif len(clustering) != 0:
            raise NotImplementedError("Joining two non-trivial clusterings is not yet implemented in the Composite masker!")
    return single_clustering
