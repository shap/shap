import numpy as np
from ..common import shaparray
import warnings

try:
    import tensorflow as tf
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import nn_grad
except ImportError:
    pass
except:
    print("tensorflow is installed...but failed to load!")
    pass


class DeepExplainer(object):
    """ Meant to approximate SHAP values for deep learning models.

    This is an enhanced version of the DeepLIFT algorithm where, similar to Kernel SHAP, we
    approximate the conditional expectations of SHAP values using a selection of background samples.
    Lundberg and Lee, NIPS 2017 showed that DeepLIFT (Shrikumar, Greenside, and Kundaje, arXiv 2017)
    can be viewed as an approximation of Shapley values. By integrating over many backgound samples
    DeepExplainer estimates approximate SHAP values such that they sum up to the difference between
    the expected model output on the passed background samples and the current model output
    (f(x) - E[f(x)]). This implementation also supports max pooling layers, unlike the original DeepLIFT.

    There are variety of potential improvements to better approximate SHAP values, but for broad and
    easy initial use we have started with an implementation inspired by the excellent gradient based
    implementation approach proposed by Ancona et al, ICLR 2018. As additional approximation options
    are added these docs will be updated.
    """

    # these are the supported non-linear components
    nonlinearities = [
        "Relu", "Elu", "Sigmoid", "Tanh", "Softplus", "MaxPool"
    ]

    # these are the components that are linear no matter how they are used. All linear
    # components are supported, this list just enumerates which components are always
    # linear in terms of the model inputs.
    guaranteed_linearities = [
        "Identity", "Reshape", "Shape", "StridedSlice", "Squeeze", "Pack", "ExpandDims",
        "BiasAdd", "Unpack", "Add", "Merge"
    ]

    # these involve products and so are linear if only one of the terms in the product depends
    # on the model inputs
    single_input_linearities = [
        "MatMul", "Prod", "Conv2D", "Mul", "RealDiv"
    ]

    # these operations may be connected above input data values in the graph but their outputs
    # don't depend on the input values
    dependence_breakers = [
        "Shape", "RandomUniform"
    ]

    def __init__(self, model, data, session=None):
        """ An explainer object for a deep model using a given background dataset.

        Note that the complexity of the method scales linearly with the number of background data
        samples. Passing the entire training dataset as `data` will give very accurate expected
        values, but be unreasonably expensive. The variance of the expectation estimates scale by
        roughly 1/sqrt(N) for N background data samples. So 100 samples will give a good estimate,
        and 1000 samples a very good estimate of the expected values.

        Parameters
        ----------
        model : (input : [tf.Operation], output : tf.Operation)
            A pair of TensorFlow operations (or a pair of lists) that specifies the input and
            output of the model to be explained. Note that SHAP values are specific to a single
            output value, so the output tf.Operation should be a single dimensional output (,1).

        data : [numpy.array] or [pandas.DataFrame]
            The background dataset to use for integrating out features. DeepExplainer integrates
            over all these samples for each explanation. The data passed here must match the input
            operations given in the first argument.
        """

        warnings.warn("shap.DeepExplainer is in an alpha state, use at your own risk!")

        self.model_inputs = model[0]
        self.model_output = model[1]
        assert type(self.model_output) != list, "The model output to be explained must be a single tensor!"
        assert len(self.model_output.shape) < 3, "The model output must be a vector or a single value!"
        self.multi_output = True
        if len(self.model_output.shape) == 1:
            self.multi_output = False

        # check if we have multiple inputs
        self.multi_input = True
        if type(self.model_inputs) != list:
            self.multi_input = False
            self.model_inputs = [self.model_inputs]
        if type(data) != list:
            data = [data]

        self.data = data
        self.session = tf.get_default_session() if session is None else session
        self.session = tf.keras.backend.get_session() if self.session is None else self.session

        # see if there is a keras operation we need to save
        self.keras_phase_placeholder = None
        for op in self.session.graph.get_operations():
            if 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]

        self.expected_value = self.run(self.model_output, self.model_inputs, self.data).mean(0)
        if not self.multi_output:
            self.expected_value = np.array([self.expected_value])

        # check to make sure we have no unsupported operations in the graph between our
        # inputs and outputs, and save all the non-linearities
        self.nonlinear_ops = []
        back_ops = tf.contrib.graph_editor.get_backward_walk_ops([self.model_output])
        between_ops = tf.contrib.graph_editor.get_forward_walk_ops(
            self.model_inputs, within_ops=back_ops,
            within_ops_fn=lambda op: op.type not in DeepExplainer.dependence_breakers
        )
        for op in between_ops:
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in DeepExplainer.nonlinearities:
                    self.nonlinear_ops.append(op)
                elif op.type in DeepExplainer.guaranteed_linearities:
                    pass
                elif op.type in DeepExplainer.single_input_linearities:
                    num_model_inputs = 0
                    for t in op.inputs:
                        if t.op in between_ops:
                            num_model_inputs += 1
                    assert num_model_inputs <= 1, op.name + " is not linear in terms of the model inputs!"
                elif op.type == "Switch":
                    # this first check is because I don't see in the API that the second input is always the flag
                    assert len(op.inputs[1].shape) == 0, "The second switch input does seem to be the flag?"
                    assert op.inputs[1].op not in between_ops, "A Switch control depending on the input is not supported!"
                else:
                    assert False, op.type + " is not known to be either linear or a supported non-linearity!"

        # replace the gradients for all the non-linear activations
        self.orig_grads = {}
        reg = ops._gradient_registry._registry # hack our way in to the registry (TODO: find an API for this)
        for n in DeepExplainer.nonlinearities:
            self.orig_grads[n] = reg[n]["type"]
            reg[n]["type"] = self.custom_grad

        # define the computation graph for the attribution values using custom a gradient-like computation
        try:
            if not self.multi_output:
                self.phi_symbolics = [tf.gradients(self.model_output, self.model_inputs)]
            else:
                self.phi_symbolics = [tf.gradients(self.model_output[:,i], self.model_inputs) for i in range(self.model_output.shape[1])]

        # restore the original gradient definitions
        finally:
            for n in DeepExplainer.nonlinearities:
                reg[n]["type"] = self.orig_grads[n]

    def learning_phase_only(self, op):
        self.only_learning_phase

    def shap_values(self, X):

        # check if we have multiple inputs
        if not self.multi_input:
            assert type(X) != list, "Expected a single tensor model input!"
            X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"
        assert len(self.model_inputs) == len(X), "Number of model inputs does not match the number given!"

        # compute the attributions
        output_phis = []
        for i in range(len(self.phi_symbolics)):
            phis = []
            for k in range(len(X)):
                phis.append(shaparray(self.expected_value[i], X[k].shape))
                phis[-1].fill(0)
            for j in range(X[0].shape[0]):

                # tile the inputs to line up with the background data samples
                tiled_X = [np.tile(X[l][j:j+1], (self.data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape)-1)])) for l in range(len(X))]

                # we use the first sample for the current sample and the rest for the references
                joint_input = [np.concatenate([tiled_X[l], self.data[l]], 0) for l in range(len(X))]

                # run attribution computation graph
                sample_phis = self.run(self.phi_symbolics[i], self.model_inputs, joint_input)

                # assign the attributions to the right part of the output arrays
                for l in range(len(X)):
                    phis[l][j] = (sample_phis[l][self.data[l].shape[0]:] * (X[l][j] - self.data[l])).mean(0)

            output_phis.append(phis[0] if not self.multi_input else phis)
        return output_phis[0] if not self.multi_output else output_phis

    def run(self, out, model_inputs, X):
        feed_dict = dict(zip(model_inputs, X))
        if self.keras_phase_placeholder is not None:
            feed_dict[self.keras_phase_placeholder] = 0
        return self.session.run(out, feed_dict)

    def custom_grad(self, op, grad):
        xinput,rinput = tf.split(op.inputs[0], 2)
        xoutput,routput = tf.split(op.outputs[0], 2)

        delta_in = xinput - rinput
        multiples2 = [2] + [1 for i in delta_in.shape[1:]]
        delta_in2 = tf.tile(delta_in, multiples2)

        # The max pool operation sends credit to both the r max element and the x max element
        if op.type == "MaxPool":
            cross_max = tf.maximum(xoutput, routput)
            diffs = tf.concat([cross_max - routput, xoutput - cross_max], 0)
            xmax_pos,rmax_pos = tf.split(self.orig_grads[op.type](op, grad * diffs), 2)

            return tf.tile(tf.where(
                tf.abs(delta_in) < 1e-7,
                delta_in * 0,
                (xmax_pos + rmax_pos) / delta_in
            ), multiples2)

        # all non-linear one-to-one mappings (like activation functions)
        else:
            return tf.where(
                tf.abs(delta_in2) < 1e-6,
                self.orig_grads[op.type](op, grad),
                grad * tf.tile((xoutput - routput) / delta_in, multiples2)
            )

        return orig_grad
