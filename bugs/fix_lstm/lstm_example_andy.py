"""
This is a working lstm shap example with random input using the following versions/settings:
tf 2.3.0
shap 0.41.0
np 1.19.5
python 3.7.16 (default, Jan 17 2023, 16:06:28) [MSC v.1916 64 bit (AMD64)]
### disable tensorflow 2 behaviour tf.compat.v1.disable_v2_behavior()

with tf > 2.4.1 it will not work
"""

import datetime
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import shap

project_name = 'lstm_test'

embedding_layer = True
use_event_attributes = True
dense_layer = False
additional_dense_layer = False

n = 1000
n_num_event_attributes = 5
voc_size = 100
sequence_length = 80
n_features = 10

X_events = np.random.randint(voc_size, size=(n, sequence_length))
X_numerical_attributes = np.random.randint(100, size=(n, sequence_length, n_num_event_attributes))
X_features = np.random.randint(100, size=(n, n_features))
y = np.random.randint(2, size=n)

# pip install numpy==1.19.5
# pip install tensorflow==2.3.0

def define_callbacks():
    dirpath = os.path.dirname(__file__)
    model_file_name = os.path.join(dirpath, project_name + '.h5')
    print("model file name: ", model_file_name)
    tensorboard_dict = os.path.expanduser('~') + '/' + datetime.date.today().strftime('%Y-%m-%d')
    experiment_log_dir = dirpath + 'tensorboard_logs/'
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=True, mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
    callbacks = [early_stopping, model_checkpoint]
    return callbacks, model_file_name


def _convert_numpy_or_python_types(x):
    if isinstance(x, (tf.Tensor, np.ndarray, float, int)):
        return tf.convert_to_tensor(x)
    return x

class OperationCaptureModel(tf.keras.Model):
    def __init__(self, layers, model):
        super().__init__()
        self.model = model
        self._layers = layers
        self.ops = []

    # unfortunately this cannot be a simple function since tf.function must return tensors or None
    @tf.function
    def __call__(self, inputs):
        # for layer in self._layers:
        #     inputs = layer(inputs)
        #     if isinstance(inputs, list):
        #         for input in inputs:
        #             if hasattr(input, "op"):
        #                 self.ops.append(input.op)
        #     else:
        #         self.ops.append(inputs.op)
        inputs = tf.nest.map_structure(
                    _convert_numpy_or_python_types, inputs
                )
        inputs = self.model._flatten_to_reference_inputs(inputs)
        # if mask is None:
        masks = [None] * len(inputs)
        # else:
        #     masks = self.model._flatten_to_reference_inputs(mask)
        for input_t, mask in zip(inputs, masks):
            input_t._keras_mask = mask

        # Dictionary mapping reference tensors to computed tensors.
        tensor_dict = {}
        tensor_usage_count = self.model._tensor_usage_count
        for x, y in zip(self.model.inputs, inputs):
            y = self.model._conform_to_reference_input(y, ref_input=x)
            x_id = str(id(x))
            tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

        nodes_by_depth = self.model._nodes_by_depth
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        for depth in depth_keys:
            nodes = nodes_by_depth[depth]
            for node in nodes:
                if node.is_input:
                    continue  # Input tensors already exist.

                if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
                    continue  # Node is not computable, try skipping.

                args, kwargs = node.map_arguments(tensor_dict)
                outputs = node.layer(*args, **kwargs)
                if hasattr(outputs, "op"):
                    self.ops.append(outputs.op)

                # Update tensor_dict.
                for x_id, y in zip(
                    node.flat_output_ids, tf.nest.flatten(outputs)
                ):
                    tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

        output_tensors = []
        for x in self.model.outputs:
            x_id = str(id(x))
            assert x_id in tensor_dict, 'Could not compute output ' + str(x)
            output_tensors.append(tensor_dict[x_id].pop())
            if hasattr(outputs, "op"):
                self.ops.append(outputs.op)


# we basically need to adapt this function and capture the ops
# This is a copy of keras.functional._run_internal_graph
# def _run_internal_graph(self, inputs, training=None, mask=None):
#     """Computes output tensors for new inputs.
#
#     # Note:
#         - Can be run on non-Keras tensors.
#
#     Args:
#         inputs: Tensor or nested structure of Tensors.
#         training: Boolean learning phase.
#         mask: (Optional) Tensor or nested structure of Tensors.
#
#     Returns:
#         output_tensors
#     """
#     inputs = self._flatten_to_reference_inputs(inputs)
#     if mask is None:
#         masks = [None] * len(inputs)
#     else:
#         masks = self._flatten_to_reference_inputs(mask)
#     for input_t, mask in zip(inputs, masks):
#         input_t._keras_mask = mask
#
#     # Dictionary mapping reference tensors to computed tensors.
#     tensor_dict = {}
#     tensor_usage_count = self._tensor_usage_count
#     for x, y in zip(self.inputs, inputs):
#         y = self._conform_to_reference_input(y, ref_input=x)
#         x_id = str(id(x))
#         tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
#
#     nodes_by_depth = self._nodes_by_depth
#     depth_keys = list(nodes_by_depth.keys())
#     depth_keys.sort(reverse=True)
#
#     for depth in depth_keys:
#         nodes = nodes_by_depth[depth]
#         for node in nodes:
#             if node.is_input:
#                 continue  # Input tensors already exist.
#
#             if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
#                 continue  # Node is not computable, try skipping.
#
#             args, kwargs = node.map_arguments(tensor_dict)
#             outputs = node.layer(*args, **kwargs)
#
#             # Update tensor_dict.
#             for x_id, y in zip(
#                 node.flat_output_ids, tf.nest.flatten(outputs)
#             ):
#                 tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
#
#     output_tensors = []
#     for x in self.outputs:
#         x_id = str(id(x))
#         assert x_id in tensor_dict, "Could not compute output " + str(x)
#         output_tensors.append(tensor_dict[x_id].pop())
#
#     return tf.nest.pack_sequence_as(self._nested_outputs, output_tensors)

def main():
    print("tf", tf.__version__)
    print("shap", shap.__version__)
    print("np", np.__version__)
    print("python", sys.version)
    print("### disable tensorflow 2 behaviour", "tf.compat.v1.disable_v2_behavior()")
    # tf.compat.v1.disable_v2_behavior()
    X = create_input()
    model = create_model(X)
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy', tf.keras.metrics.AUC()])
    # checke /home/tobias/programming/github/shap/.venv/lib/python3.11/site-packages/keras/src/utils/layer_utils.py(462)
#     import ipdb; ipdb.set_trace(context=20)
    model.summary()

    callbacks, model_file_name = define_callbacks()

    model.fit(X, y, epochs=1, batch_size=16, validation_split=0.2, verbose=2, shuffle=True, callbacks=callbacks)
    # model = tf.keras.models.load_model(model_file_name)  # load best model

    # capture operations
    # capture_model = OperationCaptureModel(model.layers, model)

    # import ipdb; ipdb.set_trace(context=20)
    # debug the model call, to see how the graph structure of the layers is executed
    # _keras_inputs_ids_and_indices
    # _single_positional_tensor_passed
    # model(X)
    # capture_model(X)
    # import ipdb; ipdb.set_trace(context=20)

    n_example = 100
    n_explainer = 100

    arr_rand_choice_explainer = np.random.choice(np.arange(0, n), size=n_explainer, replace=False)
    arr_rand_choice = np.random.choice(np.arange(0, n), size=n_example, replace=False)

    X_shap_explainer = [X[0][arr_rand_choice], X[1][arr_rand_choice]]
    X_shap = [X[0][arr_rand_choice_explainer], X[1][arr_rand_choice_explainer]]

    explainer = shap.DeepExplainer(model, X_shap_explainer[:n_explainer])
    print(f'explainer - expected value: {explainer.expected_value[0]}')

    print('calculate shap values')
    shap_values = explainer.shap_values(X_shap, check_additivity=True)[0]

    print(shap_values)


def create_input():
    if embedding_layer:
        if use_event_attributes:
            if dense_layer:
                X = [X_events, X_numerical_attributes, X_features]
            else:
                X = [X_events, X_numerical_attributes]
        else:
            if dense_layer:
                X = [X_events, X_features]
            else:
                X = [X_events]
    else:
        event_input = to_categorical(X_events, num_classes=voc_size, dtype='int32')
        if use_event_attributes:
            event_input = np.concatenate((event_input, X_numerical_attributes), 2)
        if dense_layer:
            X = [event_input, X_features]
        else:
            X = [event_input]
    return X


def create_model(X):
    tf.keras.backend.clear_session()
    neurons_dense = 27 * 3
    metric: str = "accuracy"
    emb_rate_of_dim_reductions = 1
    dropout = 0.4
    activation_setting = "tanh"  # 'relu'
    neurons = 64

    if embedding_layer:
        input_shape = [sequence_length, ]
        input = Input(input_shape)
        hidden = Embedding(voc_size, int(voc_size * emb_rate_of_dim_reductions))(input)
        if use_event_attributes:
            input_shape_2 = [sequence_length, n_num_event_attributes]
            input_2 = Input(input_shape_2)
            hidden = Concatenate(axis=2)([hidden, input_2])
        hidden = BatchNormalization(axis=-1)(hidden)
    else:
        x_shape = X[0].shape
        input_shape = [x_shape[1], x_shape[2]]
        input = Input(input_shape)
        hidden = BatchNormalization(axis=-1)(input)

    hidden = LSTM(neurons)(hidden)

    if dense_layer:
        input_3 = Input([n_features, ])
        hidden_dense = Dense(neurons_dense, activation=activation_setting)(input_3)
        hidden = Concatenate(axis=1)([hidden, hidden_dense])

    if additional_dense_layer:
        hidden = Dense(neurons)(hidden)
        hidden = BatchNormalization(axis=-1)(hidden)
        hidden = Activation(activation_setting)(hidden)
        hidden = Dropout(dropout)(hidden)

    output = Dense(1, activation='sigmoid')(hidden)

    # define model & input dimensions
    if embedding_layer:
        if use_event_attributes:
            if dense_layer:
                model = Model([input, input_2, input_3], output)
            else:
                model = Model([input, input_2], output)
        else:
            if dense_layer:
                model = Model(input, input_3, output)
            else:
                model = Model(input, output)
    elif dense_layer:
        model = Model([input, input_3], output)
    else:
        model = Model(input, output)
    return model


if __name__ == '__main__':
    main()
