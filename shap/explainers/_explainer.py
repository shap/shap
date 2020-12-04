from .. import maskers
from .. import links
from ..utils import safe_isinstance, show_progress, Model
from ..utils.transformers import MODELS_FOR_CAUSAL_LM, MODELS_FOR_SEQ_TO_SEQ_CAUSAL_LM
from .. import models
from .._explanation import Explanation
import numpy as np
import scipy as sp
import copy
import pickle
from .. import explainers


class Explainer():
    def __init__(self, model, masker=None, link=links.identity, algorithm="auto", output_names=None, feature_names=None, **kwargs):
        """ Uses Shapley values to explain any machine learning model or python function.

        This is the primary explainer interface for the SHAP library. It takes any combination
        of a model and masker and returns a callable subclass object that implements
        the particular estimation algorithm that was chosen.

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
            slice()
            model(*masker(*args, mask=mask)).mean()
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. Domain specific masking
            functions are available in shap such as shap.ImageMasker for images and shap.TokenMasker
            for text. In addition to determining how to replace hidden features, the masker can also
            constrain the rules of the cooperative game used to explain the model. For example
            shap.TabularMasker(data, hclustering="correlation") will enforce a hierarchial clustering
            of coalitions for the game (in this special case the attributions are known as the Owen values).

        link : function 
            The link function used to map between the output units of the model and the SHAP value units. By
            default it is shap.links.identity, but shap.links.logit can be useful so that expectations are
            computed in probability units while explanations remain in the (more naturally additive) log-odds
            units. For more details on how link functions work see any overview of link functions for generalized
            linear models.

        algorithm : "auto", "permutation", "partition", "tree", "kernel", "sampling", "linear", "deep", or "gradient"
            The algorithm used to estimate the Shapley values. There are many different algorithms that
            can be used to estimate the Shapley values (and the related value for constrained games), each
            of these algorithms have various tradeoffs and are preferrable in different situations. By 
            default the "auto" options attempts to make the best choice given the passed model and masker,
            but this choice can always be overriden by passing the name of a specific algorithm. The type of
            algorithm used will determine what type of subclass object is returned by this constructor, and
            you can also build those subclasses directly if you prefer or need more fine grained control over
            their options.

        output_names : None or list of strings
            The names of the model outputs. For example if the model is an image classifier, then output_names would
            be the names of all the output classes. This parameter is optional. When output_names is None then
            the Explanation objects produced by this explainer will not have any output_names, which could effect
            downstream plots.
        """

        self.model = Model(model)
        self.output_names = output_names
        self.feature_names = feature_names
        
        # wrap the incoming masker object as a shap.Masker object
        if safe_isinstance(masker, "pandas.core.frame.DataFrame") or ((safe_isinstance(masker, "numpy.ndarray") or sp.sparse.issparse(masker)) and len(masker.shape) == 2):
            if algorithm == "partition":
                self.masker = maskers.Partition(masker)
            else:
                self.masker = maskers.Independent(masker)
        elif safe_isinstance(masker, ["transformers.PreTrainedTokenizer", "transformers.tokenization_utils_base.PreTrainedTokenizerBase"]):
            if safe_isinstance(self.model, "transformers.PreTrainedModel") and safe_isinstance(self.model, MODELS_FOR_SEQ_TO_SEQ_CAUSAL_LM + MODELS_FOR_CAUSAL_LM):
                # auto assign text infilling if model is a transformer model with lm head
                self.masker = maskers.Text(masker, mask_token="...", collapse_mask_token=True)
            else:
                self.masker = maskers.Text(masker)
        elif (masker is list or masker is tuple) and masker[0] is not str:
            self.masker = maskers.Composite(*masker)
        elif (masker is dict) and ("mean" in masker):
            self.masker = maskers.Independent(masker)
        else:
            self.masker = masker
        
        # wrap self.masker and self.model for output text explanation algorithm
        if safe_isinstance(self.model, "transformers.PreTrainedModel") and safe_isinstance(self.model, MODELS_FOR_SEQ_TO_SEQ_CAUSAL_LM + MODELS_FOR_CAUSAL_LM):
            self.model = models.TeacherForcingLogits(self.model, self.masker.tokenizer)
            self.masker = maskers.FixedComposite(self.masker)
        elif safe_isinstance(self.model, "shap.models.TeacherForcingLogits") and safe_isinstance(self.masker, ["shap.maskers.Text", "shap.maskers.Image"]):
            self.masker = maskers.FixedComposite(self.masker)

        #self._brute_force_fallback = explainers.BruteForce(self.model, self.masker)

        # validate and save the link function
        if callable(link) and callable(getattr(link, "inverse", None)):
            self.link = link
        else:
            raise Exception("The passed link function needs to be callable and have a callable .inverse property!")

        # if we are called directly (as opposed to through super()) then we convert ourselves to the subclass
        # that implements the specific algorithm that was chosen
        if self.__class__ is Explainer:

            # do automatic algorithm selection
            from .. import explainers
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
                    elif issubclass(type(self.masker), maskers.Composite):
                        if getattr(self.masker, "partition_tree", None) is None:
                            algorithm = "permutation"
                        else:
                            algorithm = "partition" # TODO: should really only do this if there is more than just tab
                    elif issubclass(type(self.masker), maskers.Image) or issubclass(type(self.masker), maskers.Text) or issubclass(type(self.masker), maskers.FixedComposite):
                        algorithm = "partition"
                    else:
                        algorithm = "permutation"
                
                # if we get here then we don't know how to handle what was given to us
                else:
                    raise Exception("The passed model is not callable and cannot be analyzed directly with the given masker! Model: " + str(model))

            # build the right subclass
            if algorithm == "exact":
                self.__class__ = explainers.Exact
                explainers.Exact.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, **kwargs)
            elif algorithm == "permutation":
                self.__class__ = explainers.Permutation
                explainers.Permutation.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, **kwargs)
            elif algorithm == "partition":
                self.__class__ = explainers.Partition
                explainers.Partition.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, output_names=self.output_names, **kwargs)
            elif algorithm == "tree":
                self.__class__ = explainers.Tree
                explainers.Tree.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, **kwargs)
            elif algorithm == "additive":
                self.__class__ = explainers.Additive
                explainers.Additive.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, **kwargs)
            elif algorithm == "linear":
                self.__class__ = explainers.Linear
                explainers.Linear.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, **kwargs)
            else:
                raise Exception("Unknown algorithm type passed: %s!" % algorithm)


    def __call__(self, *args, max_evals="auto", main_effects=False, error_bounds=False, batch_size="auto",
                 outputs=None, silent=False, **kwargs):
        """ Explains the output of model(*args), where args is a list of parallel iteratable datasets.

        Note this default version could be ois an abstract method that is implemented by each algorithm-specific
        subclass of Explainer. Descriptions of each subclasses' __call__ arguments
        are available in their respective doc-strings.
        """

        # if max_evals == "auto":
        #     self._brute_force_fallback

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
                except:
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
            output_names.append(row_result.get("output_names", None))
            
            if callable(getattr(self.masker, "feature_names", None)):
                row_feature_names = self.masker.feature_names(*row_args)
                for i in range(len(row_args)):
                    feature_names[i].append(row_feature_names[i])

        # split the values up according to each input
        arg_values = [[] for a in args]
        for i in range(len(values)):
            pos = 0
            for j in range(len(args)):
                mask_length = np.prod(mask_shapes[i][j])
                arg_values[j].append(values[i][pos:pos+mask_length])
                pos += mask_length

        # collapse the expected values if they are the same for each sample
        expected_values = np.array(expected_values)
        # if np.allclose(expected_values, expected_values[0]):
        #     expected_values = expected_values[0]

        # collapse the output_indices if they are the same for each sample
        output_indices = np.array(output_indices)
        
        # collapse the main effects if we didn't compute them
        if main_effects[0] is None:
            main_effects = None
        else:
            main_effects = np.array(main_effects)

        # collapse the hierarchical values if we didn't compute them
        if hierarchical_values[0] is None:
            hierarchical_values = None
        else:
            hierarchical_values = np.array(hierarchical_values)

        # collapse the hierarchical values if we didn't compute them
        if clustering[0] is None:
            clustering = None
        else:
            clustering = np.array(clustering)

            # collapse across all the sample if we have just one clustering
            # if len(clustering.shape) == 3 and clustering.std(0).sum() < 1e-8:
            #     clustering = clustering[0]

        # getting output labels 
        if self.output_names is None:
            if None not in output_names:
                labels = np.array(output_names)
                sliced_labels = np.array([np.array(labels[i])[index_list] for i,index_list in enumerate(output_indices)])
            else:
                sliced_labels = None
        else:
            labels = np.array(self.output_names)
            sliced_labels = np.array([labels[index_list] for index_list in output_indices])

        # build the explanation objects
        out = []
        for j in range(len(args)):

            # reshape the attribution values using the mask_shapes
            tmp = []
            for i,v in enumerate(arg_values[j]):
                if np.prod(mask_shapes[i][j]) != np.prod(v.shape): # see if we have multiple outputs
                    tmp.append(v.reshape(*mask_shapes[i][j], -1))
                else:
                    tmp.append(v.reshape(*mask_shapes[i][j]))
            arg_values[j] = np.array(tmp)
            
            # allow the masker to transform the input data to better match the masking pattern
            # (such as breaking text into token segments)
            if hasattr(self.masker, "data_transform"):
                data = np.array([self.masker.data_transform(v) for v in args[j]])
            else:
                data = args[j]
            
            # build an explanation object for this input argument
            out.append(Explanation(
                arg_values[j], expected_values, data,
                feature_names=feature_names[j], main_effects=main_effects,
                clustering=clustering,
                hierarchical_values=hierarchical_values,
                output_names= sliced_labels # self.output_names
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
        masks = np.zeros(2*len(inds)-1, dtype=np.int)
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
        for i,ind in enumerate(inds):
            expanded_main_effects[ind] = main_effects[i]
        
        return expanded_main_effects

    def save(self, out_file):
        """ Serializes the type of subclass of explainer used, this will be used during deserialization
        """
        pickle.dump(type(self), out_file)
    
    @classmethod
    def load(cls, in_file, custom_model_loader = None, custom_masker_loader = None):
        """ Deserializes the explainer subtype, and calls respective load function
        """
        explainer_type = pickle.load(in_file)
        return explainer_type.load(in_file)
