import numpy as np
import scipy.special

from ._model import Model


class TransformersPipeline(Model):
    """This wraps a transformers pipeline object for easy explanations.

    By default transformers pipeline object output lists of dictionaries, not standard
    tensors as expected by SHAP. This class wraps pipelines to make them output nice
    tensor formats.
    """

    def __init__(self, pipeline, rescale_to_logits=False):
        """Build a new model by wrapping the given pipeline object."""
        super().__init__(pipeline) # the pipeline becomes our inner_model
        self.rescale_to_logits = rescale_to_logits

        #self.tokenizer = self.inner_model.model.tokenizer
        self.label2id = self.inner_model.model.config.label2id
        self.id2label = self.inner_model.model.config.id2label
        self.output_shape = (max(self.label2id.values())+1,)
        if len(self.output_shape) == 1:
            self.output_names = [self.id2label.get(i, "Unknown") for i in range(self.output_shape[0])]

    def __call__(self, strings):
        assert not isinstance(strings, str), "shap.models.TransformersPipeline expects a list of strings not a single string!"
        output = np.zeros([len(strings)] + list(self.output_shape))
        pipeline_dicts = self.inner_model(list(strings))
        for i, val in enumerate(pipeline_dicts):
            if not isinstance(val, list):
                val = [val]
            for obj in val:
                output[i, self.label2id[obj["label"]]] = scipy.special.logit(obj["score"]) if self.rescale_to_logits else obj["score"]
        return output
