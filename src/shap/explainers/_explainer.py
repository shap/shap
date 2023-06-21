import copy
import time

import numpy as np
import scipy.sparse

from .. import explainers, links, maskers, models
from .._explanation import Explanation
from .._serializable import Deserializer, Serializable, Serializer
from ..maskers import Masker
from ..models import Model
from ..utils import safe_isinstance, show_progress
from ..utils._exceptions import InvalidAlgorithmError
from ..utils.transformers import is_transformers_lm


class Explainer(Serializable):
    """ Uses Shapley values to explain any machine learning model or python function.

    This is the primary explainer interface for the SHAP library. It takes any combination
    of a model and masker and returns a callable subclass object that implements
    the particular estimation algorithm that was chosen.
    """

    def __init__(self, model, masker=None, link=links.identity, algorithm="auto", output_names=None, feature_names=None, linearize_link=True,
                 seed=None, **kwargs):
        """ Build a new explainer for the passed model.

        Parameters
        ----------
        model : object or function
            User supplied function or model object that takes a dataset of samples and
            computes the output of the model for those samples.

        masker : function, numpy.array, pandas.DataFrame, tokenizer, None, or a list of these for each model input
            The function used to "mask" out hidden features of the form `masked_args = masker(*model_args, mask=mask)`.
            It takes input in the same form as the model, but for just a single sample with a binary
            mask, then returns an iterable of masked samples. These
            masked samples will then be evaluated using the model function and the outputs averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. Domain specific masking
            functions are available in shap such as shap.ImageMasker for images and shap.TokenMasker
            for text. In addition to determining how to replace hidden features, the masker can also
            constrain the rules of the cooperative game used to explain the model. For example
            shap.TabularMasker(data, hclustering="correlation") will enforce a hierarchical clustering
            of coalitions for the game (in this special case the attributions are known as the Owen values).

        link : function
            The link function used to map between the output units of the model and the SHAP value units. By
            default it is shap.links.identity, but shap.links.logit can be useful so that expectations are
            computed in probability units while explanations remain in the (more naturally additive) log-odds
            units. For more details on how link functions work see any overview of link functions for generalized
            linear models.

        algorithm : "auto", "permutation", "partition", "tree", or "linear"
            The algorithm used to estimate the Shapley values. There are many different algorithms that
            can be used to estimate the Shapley values (and the related value for constrained games), each
            of these algorithms have various tradeoffs and are preferable in different situations. By
            default the "auto" options attempts to make the best choice given the passed model and masker,
            but this choice can always be overridden by passing the name of a specific algorithm. The type of
            algorithm used will determine what type of subclass object is returned by this constructor, and
            you can also build those subclasses directly if you prefer or need more fine grained control over
            their options.

        output_names : None or list of strings
            The names of the model outputs. For example if the model is an image classifier, then output_names would
            be the names of all the output classes. This parameter is optional. When output_names is None then
            the Explanation objects produced by this explainer will not have any output_names, which could effect
            downstream plots.

        seed: None or int
            seed for reproducibility

        """

        self.model = model
        self.output_names = output_names
        self.feature_names = feature_names

        # wrap the incoming masker object as a shap.Masker object
        if safe_isinstance(masker, "pandas.core.frame.DataFrame") or \
                ((safe_isinstance(masker, "numpy.ndarray") or scipy.sparse.issparse(masker)) and len(masker.shape) == 2):
            if algorithm == "partition":
                self.masker = maskers.Partition(masker)
            else:
                self.masker = maskers.Independent(masker)
        elif safe_isinstance(masker, ["transformers.PreTrainedTokenizer", "transformers.tokenization_utils_base.PreTrainedTokenizerBase"]):
            if is_transformers_lm(self.model):
                # auto assign text infilling if model is a transformer model with lm head
                self.masker = maskers.Text(masker, mask_token="...", collapse_mask_token=True)
            else:
                self.masker = maskers.Text(masker)
        elif (masker is list or masker is tuple) and masker[0] is not str:
            self.masker = maskers.Composite(*masker)
        elif (masker is dict) and ("mean" in masker):
            self.masker = maskers.Independent(masker)
        elif masker is None and isinstance(self.model, models.TransformersPipeline):
            return self.__init__( # pylint: disable=non-parent-init-called
                self.model, self.model.inner_model.tokenizer,
                link=link, algorithm=algorithm, output_names=output_names, feature_names=feature_names, linearize_link=linearize_link, **kwargs
            )
        else:
            self.masker = masker

        # Check for transformer pipeline objects and wrap them
        if safe_isinstance(self.model, "transformers.pipelines.Pipeline"):
            if is_transformers_lm(self.model.model):
                return self.__init__( # pylint: disable=non-parent-init-called
                    self.model.model, self.model.tokenizer if self.masker is None else self.masker,
                    link=link, algorithm=algorithm, output_names=output_names, feature_names=feature_names, linearize_link=linearize_link, **kwargs
                )
            else:
                return self.__init__( # pylint: disable=non-parent-init-called
                    models.TransformersPipeline(self.model), self.masker,
                    link=link, algorithm=algorithm, output_names=output_names, feature_names=feature_names, linearize_link=linearize_link, **kwargs
                )

        # wrap self.masker and self.model for output text explanation algorithm
        if is_transformers_lm(self.model):
            self.model = models.TeacherForcing(self.model, self.masker.tokenizer)
            self.masker = maskers.OutputComposite(self.masker, self.model.text_generate)
        elif safe_isinstance(self.model, "shap.models.TeacherForcing") and safe_isinstance(self.masker, ["shap.maskers.Text", "shap.maskers.Image"]):
            self.masker = maskers.OutputComposite(self.masker, self.model.text_generate)
        elif safe_isinstance(self.model, "shap.models.TopKLM") and safe_isinstance(self.masker, "shap.maskers.Text"):
            self.masker = maskers.FixedComposite(self.masker)

        #self._brute_force_fallback = explainers.BruteForce(self.model, self.masker)

        # validate and save the link function
        if callable(link):
            self.link = link
        else:
            raise TypeError("The passed link function needs to be callable!")
        self.linearize_link = linearize_link

        # if we are called directly (as opposed to through super()) then we convert ourselves to the subclass
        # that implements the specific algorithm that was chosen
        if self.__class__ is Explainer:

            # do automatic algorithm selection
            #from .. import explainers
            if algorithm == "auto":

                # use implementation-aware methods if possible
                if explainers.Linear.supports_model_with_masker(model, self.masker):
                    algorithm = "linear"
                elif explainers.Tree.supports_model_with_masker(model, self.masker): # TODO: check for Partition?
                    algorithm = "tree"
                elif explainers.Additive.supports_model_with_masker(model, self.masker):
                    algorithm = "additive"

                # otherwise use a model agnostic method
                elif callable(self.model):
                    if issubclass(type(self.masker), maskers.Independent):
                        if self.masker.shape[1] <= 10:
                            algorithm = "exact"
                        else:
                            algorithm = "permutation"
                    elif issubclass(type(self.masker), maskers.Partition):
                        if self.masker.shape[1] <= 32:
                            algorithm = "exact"
                        else:
                            algorithm = "permutation"
                    elif (getattr(self.masker, "text_data", False) or getattr(self.masker, "image_data", False)) and hasattr(self.masker, "clustering"):
                        algorithm = "partition"
                    else:
                        algorithm = "permutation"

                # if we get here then we don't know how to handle what was given to us
                else:
                    raise TypeError("The passed model is not callable and cannot be analyzed directly with the given masker! Model: " + str(model))

            # build the right subclass
            if algorithm == "exact":
                self.__class__ = explainers.Exact
                explainers.Exact.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, linearize_link=linearize_link, **kwargs)
            elif algorithm == "permutation":
                self.__class__ = explainers.Permutation
                explainers.Permutation.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, linearize_link=linearize_link, seed=seed, **kwargs)
            elif algorithm == "partition":
                self.__class__ = explainers.Partition
                explainers.Partition.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, linearize_link=linearize_link, output_names=self.output_names, **kwargs)
            elif algorithm == "tree":
                self.__class__ = explainers.Tree
                explainers.Tree.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, linearize_link=linearize_link, **kwargs)
            elif algorithm == "additive":
                self.__class__ = explainers.Additive
                explainers.Additive.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, linearize_link=linearize_link, **kwargs)
            elif algorithm == "linear":
                self.__class__ = explainers.Linear
                explainers.Linear.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, linearize_link=linearize_link, **kwargs)
            elif algorithm == "deep":
                self.__class__ = explainers.Deep
                explainers.Deep.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, linearize_link=linearize_link, **kwargs)
            else:
                raise InvalidAlgorithmError("Unknown algorithm type passed: %s!" % algorithm)


    def __call__(self, *args, max_evals="auto", main_effects=False, error_bounds=False, batch_size="auto",
                 outputs=None, silent=False, **kwargs):
        """ Explains the output of model(*args), where args is a list of parallel iteratable datasets.

        Note this default version could be an abstract method that is implemented by each algorithm-specific
        subclass of Explainer. Descriptions of each subclasses' __call__ arguments
        are available in their respective doc-strings.
        """

        # if max_evals == "auto":
        #     self._brute_force_fallback

        start_time = time.time()

        if issubclass(type(self.masker), maskers.OutputComposite) and len(args)==2:
            self.masker.model = models.TextGeneration(target_sentences=args[1])
            args = args[:1]
        # parse our incoming arguments
        num_rows = None
        args = list(args)
        if self.feature_names is None:
            feature_names = [None for _ in range(len(args))]
        elif issubclass(type(self.feature_names[0]), (list, tuple)):
            feature_names = copy.deepcopy(self.feature_names)
        else:
            feature_names = [copy.deepcopy(self.feature_names)]
        for i in range(len(args)):

            # try and see if we can get a length from any of the for our progress bar
            if num_rows is None:
                try:
                    num_rows = len(args[i])
                except Exception: # pylint: disable=broad-except
                    pass

            # convert DataFrames to numpy arrays
            if safe_isinstance(args[i], "pandas.core.frame.DataFrame"):
                feature_names[i] = list(args[i].columns)
                args[i] = args[i].to_numpy()

            # convert nlp Dataset objects to lists
            if safe_isinstance(args[i], "nlp.arrow_dataset.Dataset"):
                args[i] = args[i]["text"]
            elif issubclass(type(args[i]), dict) and "text" in args[i]:
                args[i] = args[i]["text"]

        if batch_size == "auto":
            if hasattr(self.masker, "default_batch_size"):
                batch_size = self.masker.default_batch_size
            else:
                batch_size = 10

        # loop over each sample, filling in the values array
        values = []
        output_indices = []
        expected_values = []
        mask_shapes = []
        main_effects = []
        hierarchical_values = []
        clustering = []
        output_names = []
        error_std = []
        if callable(getattr(self.masker, "feature_names", None)):
            feature_names = [[] for _ in range(len(args))]
        for row_args in show_progress(zip(*args), num_rows, self.__class__.__name__+" explainer", silent):
            row_result = self.explain_row(
                *row_args, max_evals=max_evals, main_effects=main_effects, error_bounds=error_bounds,
                batch_size=batch_size, outputs=outputs, silent=silent, **kwargs
            )
            values.append(row_result.get("values", None))
            output_indices.append(row_result.get("output_indices", None))
            expected_values.append(row_result.get("expected_values", None))
            mask_shapes.append(row_result["mask_shapes"])
            main_effects.append(row_result.get("main_effects", None))
            clustering.append(row_result.get("clustering", None))
            hierarchical_values.append(row_result.get("hierarchical_values", None))
            tmp = row_result.get("output_names", None)
            output_names.append(tmp(*row_args) if callable(tmp) else tmp)
            error_std.append(row_result.get("error_std", None))
            if callable(getattr(self.masker, "feature_names", None)):
                row_feature_names = self.masker.feature_names(*row_args)
                for i in range(len(row_args)):
                    feature_names[i].append(row_feature_names[i])

        # split the values up according to each input
        arg_values = [[] for a in args]
        for i, v in enumerate(values):
            pos = 0
            for j in range(len(args)):
                mask_length = np.prod(mask_shapes[i][j])
                arg_values[j].append(values[i][pos:pos+mask_length])
                pos += mask_length

        # collapse the arrays as possible
        expected_values = pack_values(expected_values)
        main_effects = pack_values(main_effects)
        output_indices = pack_values(output_indices)
        main_effects = pack_values(main_effects)
        hierarchical_values = pack_values(hierarchical_values)
        error_std = pack_values(error_std)
        clustering = pack_values(clustering)

        # getting output labels
        ragged_outputs = False
        if output_indices is not None:
            ragged_outputs = not all(len(x) == len(output_indices[0]) for x in output_indices)
        if self.output_names is None:
            if None not in output_names:
                if not ragged_outputs:
                    sliced_labels = np.array(output_names)
                else:
                    sliced_labels = [np.array(output_names[i])[index_list] for i,index_list in enumerate(output_indices)]
            else:
                sliced_labels = None
        else:
            assert output_indices is not None, "You have passed a list for output_names but the model seems to not have multiple outputs!"
            labels = np.array(self.output_names)
            sliced_labels = [labels[index_list] for index_list in output_indices]
            if not ragged_outputs:
                sliced_labels = np.array(sliced_labels)

        if isinstance(sliced_labels, np.ndarray) and len(sliced_labels.shape) == 2:
            if np.all(sliced_labels[0,:] == sliced_labels):
                sliced_labels = sliced_labels[0]

        # allow the masker to transform the input data to better match the masking pattern
        # (such as breaking text into token segments)
        if hasattr(self.masker, "data_transform"):
            new_args = []
            for row_args in zip(*args):
                new_args.append([pack_values(v) for v in self.masker.data_transform(*row_args)])
            args = list(zip(*new_args))

        # build the explanation objects
        out = []
        for j, data in enumerate(args):

            # reshape the attribution values using the mask_shapes
            tmp = []
            for i, v in enumerate(arg_values[j]):
                if np.prod(mask_shapes[i][j]) != np.prod(v.shape): # see if we have multiple outputs
                    tmp.append(v.reshape(*mask_shapes[i][j], -1))
                else:
                    tmp.append(v.reshape(*mask_shapes[i][j]))
            arg_values[j] = pack_values(tmp)

            if feature_names[j] is None:
                feature_names[j] = ["Feature " + str(i) for i in range(data.shape[1])]


            # build an explanation object for this input argument
            out.append(Explanation(
                arg_values[j], expected_values, data,
                feature_names=feature_names[j], main_effects=main_effects,
                clustering=clustering,
                hierarchical_values=hierarchical_values,
                output_names=sliced_labels, # self.output_names
                error_std=error_std,
                compute_time=time.time() - start_time
                # output_shape=output_shape,
                #lower_bounds=v_min, upper_bounds=v_max
            ))
        return out[0] if len(out) == 1 else out

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, outputs, silent, **kwargs):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes, main_effects).

        This is an abstract method meant to be implemented by each subclass.

        Returns
        -------
        tuple
            A tuple of (row_values, row_expected_values, row_mask_shapes), where row_values is an array of the
            attribution values for each sample, row_expected_values is an array (or single value) representing
            the expected value of the model for each sample (which is the same for all samples unless there
            are fixed inputs present, like labels when explaining the loss), and row_mask_shapes is a list
            of all the input shapes (since the row_values is always flattened),
        """

        return {}

    @staticmethod
    def supports_model_with_masker(model, masker):
        """ Determines if this explainer can handle the given model.

        This is an abstract static method meant to be implemented by each subclass.
        """
        return False

    @staticmethod
    def _compute_main_effects(fm, expected_value, inds):
        """ A utility method to compute the main effects from a MaskedModel.
        """

        # mask each input on in isolation
        masks = np.zeros(2*len(inds)-1, dtype=int)
        last_ind = -1
        for i in range(len(inds)):
            if i > 0:
                masks[2*i - 1] = -last_ind - 1 # turn off the last input
            masks[2*i] = inds[i] # turn on this input
            last_ind = inds[i]

        # compute the main effects for the given indexes
        main_effects = fm(masks) - expected_value

        # expand the vector to the full input size
        expanded_main_effects = np.zeros(len(fm))
        for i, ind in enumerate(inds):
            expanded_main_effects[ind] = main_effects[i]

        return expanded_main_effects

    def save(self, out_file, model_saver=".save", masker_saver=".save"):
        """ Write the explainer to the given file stream.
        """
        super().save(out_file)
        with Serializer(out_file, "shap.Explainer", version=0) as s:
            s.save("model", self.model, model_saver)
            s.save("masker", self.masker, masker_saver)
            s.save("link", self.link)

    @classmethod
    def load(cls, in_file, model_loader=Model.load, masker_loader=Masker.load, instantiate=True):
        """ Load an Explainer from the given file stream.

        Parameters
        ----------
        in_file : The file stream to load objects from.
        """
        if instantiate:
            return cls._instantiated_load(in_file, model_loader=model_loader, masker_loader=masker_loader)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.Explainer", min_version=0, max_version=0) as s:
            kwargs["model"] = s.load("model", model_loader)
            kwargs["masker"] = s.load("masker", masker_loader)
            kwargs["link"] = s.load("link")
        return kwargs

def pack_values(values):
    """ Used the clean up arrays before putting them into an Explanation object.
    """

    if not hasattr(values, "__len__"):
        return values

    # collapse the values if we didn't compute them
    if values is None or values[0] is None:
        return None

    # convert to a single numpy matrix when the array is not ragged
    elif np.issubdtype(type(values[0]), np.number) or len(np.unique([len(v) for v in values])) == 1:
        return np.array(values)
    else:
        return np.array(values, dtype=object)
