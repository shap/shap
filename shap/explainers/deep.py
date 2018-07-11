import numpy as np
import warnings
from distutils.version import LooseVersion
keras = None
tf = None


class DeepExplainer(object):
    """ Meant to approximate SHAP values for deep learning models.

    This is an enhanced version of the DeepLIFT algorithm (Deep SHAP) where, similar to Kernel SHAP, we
    approximate the conditional expectations of SHAP values using a selection of background samples.
    Lundberg and Lee, NIPS 2017 showed that the per node attribution rules in DeepLIFT (Shrikumar,
    Greenside, and Kundaje, arXiv 2017) can be chosen to approximate Shapley values. By integrating
    over many backgound samples DeepExplainer estimates approximate SHAP values such that they sum
    up to the difference between the expected model output on the passed background samples and the
    current model output (f(x) - E[f(x)]). Using tf.gradients to implement the backgropagation was
    inspired by the gradient based implementation approach proposed by Ancona et al, ICLR 2018.
    """

    # these are the supported non-linear components
    nonlinearities = [
        "Relu", "Elu", "Sigmoid", "Tanh", "Softplus", "MaxPool", "Exp", "RealDiv", "Softmax",
        "Mul", "ClipByValue"
    ]

    # these are the components that are linear no matter how they are used. All linear
    # components are supported, this list just enumerates which components are always
    # linear in terms of the model inputs.
    guaranteed_linearities = [
        "Identity", "Reshape", "Shape", "StridedSlice", "Squeeze", "Pack", "ExpandDims",
        "BiasAdd", "Unpack", "Add", "Merge", "Sub", "Sum", "Cast", "GatherV2", "Transpose", 
        "TensorArrayScatterV3", "Enter", "Tile", "TensorArrayReadV3", "NextIteration",
        "TensorArrayWriteV3", "Exit"
    ]

    # these involve products and so are linear if only one of the terms in the product depends
    # on the model inputs
    single_input_linearities = [
        "MatMul", "Prod", "Conv2D", "Mul", "RealDiv"
    ]

    # these operations may be connected above input data values in the graph but their outputs
    # don't depend on the input values (for example they just depend on the shape).
    # We include StopGradient to allow attributions to stop when gradients also stop.
    dependence_breakers = [
        "Shape", "RandomUniform", "StopGradient", "ZerosLike"
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
            A pair of TensorFlow operations (or a list and an op) that specifies the input and
            output of the model to be explained. Note that SHAP values are specific to a single
            output value, so the output tf.Operation should be a single dimensional output (,1).

        data : [numpy.array] or [pandas.DataFrame]
            The background dataset to use for integrating out features. DeepExplainer integrates
            over all these samples for each explanation. The data passed here must match the input
            operations given in the first argument.
        """

        # try and import keras and tensorflow
        global tf, tf_ops
        if tf is None:
            import tensorflow as tf
            from tensorflow.python.framework import ops as tf_ops
            if LooseVersion(tf.__version__) < LooseVersion("1.8.0"):
                warnings.warn("Your TensorFlow version is older than 1.8.0 and not supported.")
        global keras
        if keras is None:
            try:
                import keras
                if LooseVersion(keras.__version__) < LooseVersion("2.2.0"):
                    warnings.warn("Your Keras version is older than 2.2.0 and not supported.")
            except:
                pass

        # determine the model inputs and outputs
        if str(type(model)).endswith("keras.engine.sequential.Sequential'>"):
            self.model_inputs = model.layers[0].input
            self.model_output = model.layers[-1].output
        elif str(type(model)).endswith("keras.models.Sequential'>"):
            self.model_inputs = model.layers[0].input
            self.model_output = model.layers[-1].output
        elif str(type(model)).endswith("keras.engine.training.Model"):
            self.model_inputs = model.layers[0].input
            self.model_output = model.layers[-1].output
        elif str(type(model)).endswith("tuple'>"):
            self.model_inputs = model[0]
            self.model_output = model[1]
        else:
            assert False, str(type(model)) + " is not currently a supported model type!"
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
        self._num_vinputs = {}

        # if we are not given a session find a default session
        if session is None:
            # if keras is installed and already has a session then use it
            if keras is not None and keras.backend.tensorflow_backend._SESSION is not None:
                session = keras.backend.get_session()
            else:
                session = tf.keras.backend.get_session()
        self.session = tf.get_default_session() if session is None else session

        # see if there is a keras operation we need to save
        self.keras_phase_placeholder = None
        for op in self.session.graph.get_operations():
            if 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]

        # save the expected output of the model
        self.expected_value = self.run(self.model_output, self.model_inputs, self.data).mean(0)

        # check to make sure we have no unsupported operations in the graph between our
        # inputs and outputs, and save all the non-linearities
        self.nonlinear_ops = []
        back_ops = tf.contrib.graph_editor.get_backward_walk_ops(
            [self.model_output],
            within_ops_fn=lambda op: op.type not in DeepExplainer.dependence_breakers
        )
        self.between_ops = tf.contrib.graph_editor.get_forward_walk_ops(
            self.model_inputs, within_ops=back_ops,
            within_ops_fn=lambda op: op.type not in DeepExplainer.dependence_breakers
        )
        for op in self.between_ops:
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in DeepExplainer.nonlinearities:
                    self.nonlinear_ops.append(op)
                elif op.type in DeepExplainer.guaranteed_linearities:
                    pass
                elif op.type in DeepExplainer.single_input_linearities:
                    assert self.num_variable_inputs(op) <= 1, op.name + " is not linear in terms of the model inputs!"
                elif op.type == "Switch":
                    # this first check is because I don't see in the API that the second input is always the flag
                    assert len(op.inputs[1].shape) == 0, "The second switch input does seem to be the flag?"
                    assert op.inputs[1].op not in self.between_ops, "A Switch control depending on the input is not supported!"
                elif op.type == "ClipByValue":
                    # Check if our thresholds (clip_value_min and clip_value_max respectively) 
                    # are in the correct position in the input 
                    assert len(op.inputs[1].shape) == 0 and len(op.inputs[2].shape) == 0
                    # Thresholds that depend on input are not supported!
                    assert op.inputs[1].op not in self.between_ops and op.inputs[2].op not in self.between_ops
                else:
                    assert False, op.type + " is not known to be either linear or a supported non-linearity!"

        if not self.multi_output:
            self.phi_symbolics = [None]
        else:
            self.phi_symbolics = [None for i in range(self.model_output.shape[1])]

    def num_variable_inputs(self, op):
        if op.name not in self._num_vinputs:
            num_variable_inputs = 0
            for t in op.inputs:
                if t.op in self.between_ops:
                    num_variable_inputs += 1
            self._num_vinputs[op.name] = num_variable_inputs
        return self._num_vinputs[op.name]

    def phi_symbolic(self, i):

        if self.phi_symbolics[i] is None:

            # replace the gradients for all the non-linear activations
            self.orig_grads = {}
            reg = tf_ops._gradient_registry._registry # hack our way in to the registry (TODO: find an API for this)
            for n in DeepExplainer.nonlinearities:
                self.orig_grads[n] = reg[n]["type"]
                reg[n]["type"] = self.custom_grad

            # define the computation graph for the attribution values using custom a gradient-like computation
            try:
                out = self.model_output[:,i] if self.multi_output else self.model_output
                self.phi_symbolics[i] = tf.gradients(out, self.model_inputs)

            # restore the original gradient definitions
            finally:
                for n in DeepExplainer.nonlinearities:
                    reg[n]["type"] = self.orig_grads[n]

        return self.phi_symbolics[i]

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max"):
        """ Return the values for the model applied to X.

        Parameters
        ----------
        X : list, numpy.array, or pandas.DataFrame
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        ranked_outputs : None or int
            If ranked_outputs is None then we explain all the outputs in a multi-output model. If
            ranked_outputs is a positive integer then we only explain that many of the top model
            outputs (where "top" is determined by output_rank_order). Note that this causes a pair
            of values to be returned (shap_values, indexes), where phi is a list of numpy arrays for each of
            the output ranks, and indexes is a matrix that tells for each sample which output indexes
            were choses as "top".

        output_rank_order : "max", "min", or "max_abs"
            How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
            maximum absolute value.

        Returns
        -------
        For a models with a single output this returns a tensor of SHAP values with the same shape
        as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
        which are the same shape as X. If ranked_outputs is None then this list of tensors matches
        the number of model outputs. If ranked_outputs is a positive integer a pair is returned
        (shap_values, indexes), where shap_values is a list of tensors with a length of
        ranked_outputs, and indexes is a matrix that tells for each sample which output indexes
        were chosen as "top".
        """

        # check if we have multiple inputs
        if not self.multi_input:
            assert type(X) != list, "Expected a single tensor model input!"
            X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"
        assert len(self.model_inputs) == len(X), "Number of model inputs does not match the number given!"

        # rank and determine the model outputs that we will explain
        if ranked_outputs is not None and self.multi_output:
            model_output_values = self.run(self.model_output, self.model_inputs, X)
            if output_rank_order == "max":
                model_output_ranks = np.argsort(-model_output_values)
            elif output_rank_order == "min":
                model_output_ranks = np.argsort(model_output_values)
            elif output_rank_order == "max_abs":
                model_output_ranks = np.argsort(np.abs(model_output_values))
            else:
                assert False, "output_rank_order must be max, min, or max_abs!"
            model_output_ranks = model_output_ranks[:,:ranked_outputs]
        else:
            model_output_ranks = np.tile(np.arange(len(self.phi_symbolics)), (X[0].shape[0], 1))

        # compute the attributions
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            for k in range(len(X)):
                phis.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):

                # tile the inputs to line up with the background data samples
                tiled_X = [np.tile(X[l][j:j+1], (self.data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape)-1)])) for l in range(len(X))]

                # we use the first sample for the current sample and the rest for the references
                joint_input = [np.concatenate([tiled_X[l], self.data[l]], 0) for l in range(len(X))]

                # run attribution computation graph
                feature_ind = model_output_ranks[j,i]
                sample_phis = self.run(self.phi_symbolic(feature_ind), self.model_inputs, joint_input)

                # assign the attributions to the right part of the output arrays
                for l in range(len(X)):
                    phis[l][j] = (sample_phis[l][self.data[l].shape[0]:] * (X[l][j] - self.data[l])).mean(0)

            output_phis.append(phis[0] if not self.multi_input else phis)
        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis

    def run(self, out, model_inputs, X):
        feed_dict = dict(zip(model_inputs, X))
        if self.keras_phase_placeholder is not None:
            feed_dict[self.keras_phase_placeholder] = 0
        return self.session.run(out, feed_dict)

    def custom_grad(self, op, grad):
        if not op.type == "RealDiv" and not op.type == "Mul":
            xin0,rin0 = tf.split(op.inputs[0], 2)
            xout,rout = tf.split(op.outputs[0], 2)
            delta_in0 = xin0 - rin0
            dup0 = [2] + [1 for i in delta_in0.shape[1:]]

        # Division with two varying inputs can be handled by using the direct SHAP values
        if op.type == "RealDiv":
            if self.num_variable_inputs(op) == 1:
                return self.orig_grads[op.type](op, grad)
            else:
                xout,rout = tf.split(op.outputs[0], 2)
                xin0,rin0 = tf.split(op.inputs[0], 2)
                xin1,rin1 = tf.split(op.inputs[1], 2)
                delta_in0 = xin0 - rin0
                delta_in1 = xin1 - rin1
                dup0 = [2] + [1 for i in delta_in0.shape[1:]]
                out10 = xin0 / rin1
                out01 = rin0 / xin1
                out11,out00 = xout,rout
                out0 = 0.5 * (out11 - out01 + out10 - out00)
                out0 = grad * tf.tile(out0 / delta_in0, dup0)
                out1 = 0.5 * (out11 - out10 + out01 - out00)
                out1 = grad * tf.tile(out1 / delta_in1, dup0)
                                
                # see if due to broadcasting our gradient shapes don't match our input shapes
#                 if out1.shape != delta_in1.shape:
                if (np.any(np.array(out1.shape) != np.array(delta_in1.shape))):
                    broadcast_index = np.where(np.array(out1.shape) != np.array(delta_in1.shape))[0][0]
                    out1 = tf.reduce_sum(out1, axis=broadcast_index, keepdims=True)
#                 elif out0.shape != delta_in0.shape:
                elif (np.any(np.array(out0.shape) != np.array(delta_in0.shape))):
                    broadcast_index = np.where(np.array(out0.shape) != np.array(delta_in0.shape))[0][0]
                    out0 = tf.reduce_sum(out0, axis=broadcast_index, keepdims=True)

                # Avoid divide by zero nans
                out0 = tf.where(tf.abs(tf.tile(delta_in0,dup0)) < 1e-7, 
                                0 * tf.tile(delta_in0,dup0), out0)
                out1 = tf.where(tf.abs(tf.tile(delta_in1,dup0))<1e-7, 
                                0 * tf.tile(delta_in1,dup0), out1)
                
                return [out0, out1]

        elif op.type == "Mul":
            if self.num_variable_inputs(op) == 1:
                return self.orig_grads[op.type](op, grad)
            else:
                xout,rout = tf.split(op.outputs[0], 2)
                xin0,rin0 = tf.split(op.inputs[0], 2)
                xin1,rin1 = tf.split(op.inputs[1], 2)
                delta_in0 = xin0 - rin0
                delta_in1 = xin1 - rin1
                dup0 = [2] + [1 for i in delta_in0.shape[1:]]
                out10 = xin0 * rin1
                out01 = rin0 * xin1
                out11,out00 = xout,rout
                out0 = 0.5 * (out11 - out01 + out10 - out00)
                out0 = grad * tf.tile(out0 / delta_in0, dup0)
                out1 = 0.5 * (out11 - out10 + out01 - out00)
                out1 = grad * tf.tile(out1 / delta_in1, dup0)

                # see if due to broadcasting our gradient shapes don't match our input shapes
#                 if out1.shape != delta_in1.shape:
                if (np.any(np.array(out1.shape) != np.array(delta_in1.shape))):
                    broadcast_index = np.where(np.array(out1.shape) != np.array(delta_in1.shape))[0][0]
                    out1 = tf.reduce_sum(out1, axis=broadcast_index, keepdims=True)
#                 elif out0.shape != delta_in0.shape:
                elif (np.any(np.array(out0.shape) != np.array(delta_in0.shape))):
                    broadcast_index = np.where(np.array(out0.shape) != np.array(delta_in0.shape))[0][0]
                    out0 = tf.reduce_sum(out0, axis=broadcast_index, keepdims=True)

                # Avoid divide by zero nans
                out0 = tf.where(tf.abs(tf.tile(delta_in0,dup0)) < 1e-7, 
                                0 * tf.tile(delta_in0,dup0), out0)
                out1 = tf.where(tf.abs(tf.tile(delta_in1,dup0))<1e-7, 
                                0 * tf.tile(delta_in1,dup0), out1)
                
                return [out0, out1]

        # The max pool operation sends credit to both the r max element and the x max element
        elif op.type == "MaxPool":
            cross_max = tf.maximum(xout, rout)
            diffs = tf.concat([cross_max - rout, xout - cross_max], 0)
            xmax_pos,rmax_pos = tf.split(self.orig_grads[op.type](op, grad * diffs), 2)

            return tf.tile(tf.where(
                tf.abs(delta_in0) < 1e-7,
                delta_in0 * 0,
                (xmax_pos + rmax_pos) / delta_in0
            ), dup0)

        # Just decompose softmax into its components and recurse, we can handle all of them :)
        # we assume the 'axis' is the last dimension because the TF codebase swaps the 'axis' to
        # the last dim before the softmax op if 'axis' is not already the last dimself.
        # We also don't subtract the max before tf.exp for numerical stability since that might
        # mess up the attributions and it seems like TensorFlow doesn't define softmax that way
        # (according to the docs)
        elif op.type == "Softmax":
            offset_in = op.inputs[0]
            evals = tf.exp(offset_in, name="custom_exp")
            return tf.gradients(evals / tf.reduce_sum(evals, axis=-1, keepdims=True), offset_in, grad_ys=grad)[0]

        # Same as other non-linear mappings (aside from additional inputs)
        elif op.type == "ClipByValue": 

            orig_grad = self.orig_grads[op.type](op, grad)

            return [tf.where(
                tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
                orig_grad[0], grad * tf.tile((xout - rout) / delta_in0, dup0)
            ), orig_grad[1], orig_grad[2]]


        # all non-linear one-to-one mappings (like activation functions)
        else:
            return tf.where(
                tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
                self.orig_grads[op.type](op, grad),
                grad * tf.tile((xout - rout) / delta_in0, dup0)
            )

        #return self.orig_grads[op.type](op, grad)
