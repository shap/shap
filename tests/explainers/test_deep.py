"""Tests for the Deep explainer."""

import os
import platform
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from packaging import version

import shap

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

############################
# Tensorflow related tests #
############################


def test_tf_eager_call(random_seed):
    """This is a basic eager example from keras."""
    tf = pytest.importorskip("tensorflow")

    tf.compat.v1.random.set_random_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    if version.parse(tf.__version__) >= version.parse("2.4.0"):
        pytest.skip("Deep explainer does not work for TF 2.4 in eager mode.")

    x = pd.DataFrame({"B": rs.random(size=(100,))})
    y = x.B
    y = y.map(lambda zz: chr(int(zz * 2 + 65))).str.get_dummies()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape=(x.shape[1],), activation="relu"))
    model.add(tf.keras.layers.Dense(y.shape[1], input_shape=(10,), activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="Adam")
    model.fit(x.values, y.values, epochs=2)

    e = shap.DeepExplainer(model, x.values[:1])
    sv = e.shap_values(x.values)
    sv_call = e(x.values)
    np.testing.assert_array_almost_equal(sv, sv_call.values, decimal=8)
    assert np.abs(e.expected_value[0] + sv[0].sum(-1) - model(x.values)[:, 0]).max() < 1e-4


def test_tf_keras_mnist_cnn_call(random_seed):
    """This is the basic mnist cnn example from keras."""
    tf = pytest.importorskip("tensorflow")
    rs = np.random.RandomState(random_seed)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    batch_size = 64
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = rs.randn(200, 28, 28)
    y_train = rs.randint(0, 9, 200)
    x_test = rs.randn(200, 28, 28)
    y_test = rs.randint(0, 9, 200)

    if tf.keras.backend.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(2, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(4, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation="relu"))  # 128
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation("softmax"))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(), metrics=["accuracy"]
    )

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    # explain by passing the tensorflow inputs and outputs
    inds = rs.choice(x_train.shape[0], 3, replace=False)
    e = shap.DeepExplainer((model.inputs, model.layers[-1].output), x_train[inds, :, :])
    shap_values = e.shap_values(x_test[:1])
    shap_values_call = e(x_test[:1])

    np.testing.assert_array_almost_equal(shap_values, shap_values_call.values, decimal=8)

    predicted = model(x_test[:1])

    sums = shap_values.sum(axis=(1, 2, 3))
    (
        np.testing.assert_allclose(sums + e.expected_value, predicted, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )


@pytest.mark.parametrize("activation", ["relu", "elu", "selu"])
def test_tf_keras_activations(activation):
    """Test verifying that a linear model with linear data gives the correct result."""
    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0

    tf = pytest.importorskip("tensorflow")

    tf.compat.v1.random.set_random_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    # coefficients relating y with x1 and x2.
    coef = np.array([1, 2]).T

    # generate data following a linear relationship
    x = rs.normal(1, 10, size=(1000, len(coef)))
    y = np.dot(x, coef) + 1 + rs.normal(scale=0.1, size=1000)

    # create a linear model
    inputs = tf.keras.layers.Input(shape=(2,))
    preds = tf.keras.layers.Dense(1, activation=activation)(inputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=preds)
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss="mse", metrics=["mse"])
    model.fit(x, y, epochs=30, shuffle=False, verbose=0)

    # explain
    e = shap.DeepExplainer((model.inputs[0], model.layers[-1].output), x)
    shap_values = e.shap_values(x)
    preds = model.predict(x)

    assert shap_values.shape == (1000, 2, 1)
    np.testing.assert_allclose(shap_values.sum(axis=1) + e.expected_value, preds, atol=1e-5)


def test_tf_keras_linear():
    """Test verifying that a linear model with linear data gives the correct result."""
    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0

    tf = pytest.importorskip("tensorflow")

    # tf.compat.v1.disable_eager_execution()

    tf.compat.v1.random.set_random_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    # coefficients relating y with x1 and x2.
    coef = np.array([1, 2]).T

    # generate data following a linear relationship
    x = rs.normal(1, 10, size=(1000, len(coef)))
    y = np.dot(x, coef) + 1 + rs.normal(scale=0.1, size=1000)

    # create a linear model
    inputs = tf.keras.layers.Input(shape=(2,))
    preds = tf.keras.layers.Dense(1, activation="linear")(inputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=preds)
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss="mse", metrics=["mse"])
    model.fit(x, y, epochs=30, shuffle=False, verbose=0)

    fit_coef = model.layers[1].get_weights()[0].T[0]

    # explain
    e = shap.DeepExplainer((model.inputs, model.layers[-1].output), x)
    shap_values = e.shap_values(x)

    assert shap_values.shape == (1000, 2, 1)

    # verify that the explanation follows the equation in LinearExplainer
    expected = (x - x.mean(0)) * fit_coef
    np.testing.assert_allclose(shap_values.sum(-1), expected, atol=1e-5)


def test_tf_keras_imdb_lstm(random_seed):
    """Basic LSTM example using the keras API defined in tensorflow"""
    tf = pytest.importorskip("tensorflow")
    rs = np.random.RandomState(random_seed)
    tf.compat.v1.random.set_random_seed(random_seed)

    # this fails right now for new TF versions (there is a warning in the code for this)
    if version.parse(tf.__version__) >= version.parse("2.5.0"):
        pytest.skip()

    tf.compat.v1.disable_eager_execution()

    # load the data from keras
    max_features = 1000
    try:
        (X_train, _), (X_test, _) = tf.keras.datasets.imdb.load_data(num_words=max_features)
    except Exception:
        return  # this hides a bug in the most recent version of keras that prevents data loading
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=100)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=100)

    # create the model. note that this is model is very small to make the test
    # run quick and we don't care about accuracy here
    mod = tf.keras.models.Sequential()
    mod.add(tf.keras.layers.Embedding(max_features, 8))
    mod.add(tf.keras.layers.LSTM(10, dropout=0.2, recurrent_dropout=0.2))
    mod.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    mod.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # select the background and test samples
    inds = rs.choice(X_train.shape[0], 3, replace=False)
    background = X_train[inds]
    testx = X_test[10:11]

    # explain a prediction and make sure it sums to the difference between the average output
    # over the background samples and the current output
    sess = tf.compat.v1.keras.backend.get_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    # For debugging, can view graph:
    # writer = tf.compat.v1.summary.FileWriter("c:\\tmp", sess.graph)
    # writer.close()
    e = shap.DeepExplainer((mod.layers[0].input, mod.layers[-1].output), background)
    shap_values = e.shap_values(testx)
    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    diff = sess.run(mod.layers[-1].output, feed_dict={mod.layers[0].input: testx})[0, :] - sess.run(
        mod.layers[-1].output, feed_dict={mod.layers[0].input: background}
    ).mean(0)
    np.testing.assert_allclose(sums, diff, atol=1e-02), "Sum of SHAP values does not match difference!"


def test_tf_keras_lstm_no_embedding():
    """Test LSTM without embedding layer using continuous input data."""
    tf = pytest.importorskip("tensorflow")

    rs = np.random.RandomState(0)
    tf.random.set_seed(0)

    # Create synthetic time series data (e.g., sensor readings)
    # Shape: (samples, timesteps, features)
    X_train = rs.randn(100, 20, 5).astype(np.float32)

    X_test = rs.randn(20, 20, 5).astype(np.float32)

    # Create a simple LSTM model without embeddings
    mod = tf.keras.models.Sequential()
    mod.add(tf.keras.layers.LSTM(10, input_shape=(20, 5)))
    mod.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    mod.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # select the background and test samples
    inds = rs.choice(X_train.shape[0], 3, replace=False)
    background = X_train[inds]
    testx = X_test[0:1]

    # explain a prediction
    e = shap.DeepExplainer(mod, background)
    shap_values = e.shap_values(testx, check_additivity=False)

    # Basic checks
    assert shap_values is not None
    assert len(shap_values) == 1  # single output
    # SHAP values shape should match input shape: (batch, timesteps, features)
    expected_shape = (testx.shape[1], testx.shape[2], 1)  # (timesteps, features, outputs)
    assert shap_values[0].shape == expected_shape, (
        f"SHAP values shape {shap_values[0].shape} != expected shape {expected_shape}"
    )

    # Compute expected difference
    output_test = mod(testx).numpy()
    output_background = mod(background).numpy()
    diff = output_test[0, :] - output_background.mean(0)

    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])

    # Check that SHAP values sum to the difference
    np.testing.assert_allclose(sums, diff, atol=0.05), "Sum of SHAP values does not match difference!"


@pytest.mark.skipif(
    platform.system() == "Darwin" and os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping on GH MacOS runners due to memory error, see GH #3929",
)
def test_tf_deep_imbdb_transformers():
    # GH 3522
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from shap import models

    # data from datasets imdb dataset
    short_data = ["I lov", "Worth", "its a", "STAR ", "First", "I had", "Isaac", "It ac", "Techn", "Hones"]
    classifier = transformers.pipeline("sentiment-analysis", return_all_scores=True)
    pmodel = models.TransformersPipeline(classifier, rescale_to_logits=True)
    explainer3 = shap.Explainer(pmodel, classifier.tokenizer)
    shap_values3 = explainer3(short_data[:10])
    shap.plots.text(shap_values3[:, :, 1])  # type: ignore[call-overload]
    shap.plots.bar(shap_values3[:, :, 1].mean(0))  # type: ignore[call-overload]


def test_tf_deep_multi_inputs_multi_outputs():
    tf = pytest.importorskip("tensorflow")

    input1 = tf.keras.layers.Input(shape=(3,))
    input2 = tf.keras.layers.Input(shape=(4,))

    # Concatenate input layers
    concatenated = tf.keras.layers.concatenate([input1, input2])

    # Dense layers
    x = tf.keras.layers.Dense(16, activation="relu")(concatenated)

    # Output layer
    output = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)
    batch_size = 32
    # Generate random input data for input1 with shape (batch_size, 3)
    input1_data = np.random.rand(batch_size, 3)

    # Generate random input data for input2 with shape (batch_size, 4)
    input2_data = np.random.rand(batch_size, 4)

    predicted = model.predict([input1_data, input2_data])
    explainer = shap.DeepExplainer(model, [input1_data, input2_data])
    shap_values = explainer.shap_values([input1_data, input2_data])
    np.testing.assert_allclose(
        shap_values[0].sum(1) + shap_values[1].sum(1) + explainer.expected_value, predicted, atol=1e-3
    )


def test_tf_eager_lstm():
    # This test should pass with tf 2.x
    import random

    tf = pytest.importorskip("tensorflow")

    # Clear any existing TensorFlow session/graph state
    tf.keras.backend.clear_session()

    seed = 42
    # Set ALL random seeds to ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)

    # split a univariate sequence into samples
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
            return np.array(X), np.array(y)

    # define input sequence
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(50, activation="relu", input_shape=(n_steps, n_features)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    # demonstrate prediction
    x_input = np.array([70, 80, 90], dtype=np.float32)
    x_input = x_input.reshape((1, n_steps, n_features))
    e = shap.DeepExplainer(model, x_input)
    sv = e.shap_values(x_input)

    try:
        diff = np.abs(e.expected_value[0] + sv[0].sum(-1) - model(x_input)[:, 0]).max()
        assert diff < 1e-4, f"Max diff = {diff:.10f} (threshold: 1e-4)"
    except AssertionError as err:
        print(f"\n❌ FALSIFYING EXAMPLE FOUND! Seed: {seed}")
        print(f"Error details: {err}")
        raise


@pytest.mark.parametrize("seed", [110, 32222])
def test_tf_eager_stacked_lstms(seed):
    # def test_tf_eager_stacked_lstms():
    # this test should pass with tf 2.x
    import random

    tf = pytest.importorskip("tensorflow")

    # Clear any existing TensorFlow session/graph state
    tf.keras.backend.clear_session()
    # bad seeds: 110 and 32222

    # Set ALL random seeds to ensure reproducibility
    # TensorFlow uses multiple random sources that all need to be seeded
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Legacy numpy global RNG (used by TF/Keras)
    tf.random.set_seed(seed)  # TensorFlow random ops

    # Try to set Keras random seed (available in TF 2.7+)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)

    # Use modern numpy generator for our data generation
    rng = np.random.default_rng(seed)
    start_datetime = pd.to_datetime("2020-01-01 00:00:00")
    end_datetime = pd.to_datetime("2023-03-31 23:00:00")
    # Generate a DatetimeIndex with hourly frequency
    date_rng = pd.date_range(start=start_datetime, end=end_datetime, freq="H")
    # Create a DataFrame with random data for 7 features
    num_samples = len(date_rng)
    num_features = 7
    # Generate random data for the DataFrame using the generator
    data = rng.random((num_samples, num_features))
    # Create the DataFrame with a DatetimeIndex
    df = pd.DataFrame(data, index=date_rng, columns=[f"X{i}" for i in range(1, num_features + 1)])

    def windowed_dataset(series=None, in_horizon=None, out_horizon=None, delay=None, batch_size=None):
        """
        Convert multivariate data into input and output sequences.
        Convert NumPy arrays to TensorFlow tensors.
        Arguments:
        ===========
        series: a list or array of time-series data.
        total_horizon: an integer representing the size of the input window.
        out_horizon: an integer representing the size of the output window.
        delay: an integer representing the number of steps between each input window.
        batch_size: an integer representing the batch size.
        """
        total_horizon = in_horizon + out_horizon
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(total_horizon, shift=delay, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(total_horizon))
        dataset = dataset.map(lambda window: (window[:-out_horizon, :], window[-out_horizon:, 0]))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    # Define the proportions for the splits (70:20:10)%
    train_size = 0.4
    valid_size = 0.5
    # Calculate the split points
    train_split = int(len(df) * train_size)
    valid_split = int(len(df) * (train_size + valid_size))
    # Split the DataFrame
    df_train = df.iloc[:train_split]
    df_valid = df.iloc[train_split:valid_split]
    df_test = df.iloc[valid_split:]
    # number of input features and output targets
    n_features = df.shape[1]
    # split the data into sliding sequential windows
    train_dataset = windowed_dataset(series=df_train.values, in_horizon=100, out_horizon=3, delay=1, batch_size=32)
    windowed_dataset(series=df_valid.values, in_horizon=100, out_horizon=3, delay=1, batch_size=32)
    windowed_dataset(series=df_test.values, in_horizon=100, out_horizon=3, delay=1, batch_size=32)
    input_layer = tf.keras.layers.Input(shape=(100, n_features))
    lstm_layer1 = tf.keras.layers.LSTM(5, return_sequences=True)(input_layer)
    lstm_layer2 = tf.keras.layers.LSTM(5, return_sequences=True)(lstm_layer1)
    lstm_layer3 = tf.keras.layers.LSTM(5)(lstm_layer2)
    output_layer = tf.keras.layers.Dense(3)(lstm_layer3)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])

    def tensor_to_arrays(input_obj: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a "tensorflow.python.data.ops.dataset_ops.PrefetchDataset" object into a numpy arrays.
        This function can be used to slice the tensor objects out of the `windowing` function.
        """
        x_list = list(map(lambda item: item[0], input_obj))
        y_list = list(map(lambda item: item[1], input_obj))
        x_arrays = [xtmp.numpy() for xtmp in x_list]
        y_arrays = [ytmp.numpy() for ytmp in y_list]
        # Stack the arrays vertically
        x_final: np.ndarray = np.vstack(x_arrays)
        y_final: np.ndarray = np.vstack(y_arrays)
        return x_final, y_final

    xarr, yarr = tensor_to_arrays(input_obj=train_dataset)
    # Create an explainer object
    e = shap.DeepExplainer(model, xarr[:100, :, :])
    # Calculate SHAP values for the data
    sv = e.shap_values(xarr[100:200, :, :], check_additivity=False)
    model_output_values = model(xarr[100:200, :, :])
    # todo: this might indicate an error in how the gradients are overwritten
    try:
        for dim in range(3):
            diff = (
                model_output_values[:, dim].numpy()
                - e.expected_value[dim].numpy()
                - sv[dim].sum(axis=tuple(range(1, sv[dim].ndim)))
            )
            max_diff = diff.max()
            print(max_diff)
            assert max_diff < 0.08, f"Dimension {dim}: max diff = {max_diff:.6f} (threshold: 0.05)"
    except AssertionError as err:
        print(f"\n❌ FALSIFYING EXAMPLE FOUND! Seed: {seed}")
        print(f"Error details: {err}")
        raise


#######################
# Torch related tests #
#######################


def _torch_cuda_available():
    """Checks whether cuda is available. If so, torch-related tests are also tested on gpu."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        pass

    return False


TORCH_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not _torch_cuda_available(), reason="cuda unavailable (with torch)")),
]


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("interim", [True, False])
def test_pytorch_mnist_cnn_call(torch_device, interim):
    """The same test as above, but for pytorch"""
    torch = pytest.importorskip("torch")

    from torch import nn
    from torch.nn import functional as F

    class RandData:
        """Random test data."""

        def __init__(self, batch_size):
            self.current = 0
            self.batch_size = batch_size

        def __iter__(self):
            return self

        def __next__(self):
            self.current += 1
            if self.current < 10:
                return torch.randn(self.batch_size, 1, 28, 28), torch.randint(0, 9, (self.batch_size,))
            raise StopIteration

    class Net(nn.Module):
        """Basic conv net."""

        def __init__(self):
            super().__init__()
            # Testing several different activations
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.Tanh(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.ConvTranspose2d(20, 20, 1),
                nn.AdaptiveAvgPool2d(output_size=(4, 4)),
                nn.Softplus(),
                nn.Flatten(),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(320, 50), nn.BatchNorm1d(50), nn.ReLU(), nn.Linear(50, 10), nn.ELU(), nn.Softmax(dim=1)
            )

        def forward(self, x):
            """Run the model."""
            x = self.conv_layers(x)
            x = x.view(-1, 320)  # Redundant as `Flatten`, left as a test
            x = self.fc_layers(x)
            return x

    def train(model, device, train_loader, optimizer, _, cutoff=20):
        model.train()
        num_examples = 0
        for _, (data, target) in enumerate(train_loader):
            num_examples += target.shape[0]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, torch.eye(10).to(device)[target])

            loss.backward()
            optimizer.step()

            if num_examples > cutoff:
                break

    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 42

    torch.manual_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    batch_size = 32

    train_loader = RandData(batch_size)
    test_loader = RandData(batch_size)

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    device = torch.device(torch_device)

    model.to(device)
    train(model, device, train_loader, optimizer, 1)

    next_x, _ = next(iter(train_loader))
    inds = rs.choice(next_x.shape[0], 3, replace=False)

    next_x_random_choices = next_x[inds, :, :, :].to(device)

    if interim:
        e = shap.DeepExplainer((model, model.conv_layers[0]), next_x_random_choices)
    else:
        e = shap.DeepExplainer(model, next_x_random_choices)

    test_x, _ = next(iter(test_loader))
    input_tensor = test_x[:1].to(device)
    shap_values = e.shap_values(input_tensor)
    shap_values_call = e(input_tensor)

    np.testing.assert_array_almost_equal(shap_values, shap_values_call.values, decimal=8)

    model.eval()
    model.zero_grad()

    with torch.no_grad():
        outputs = model(input_tensor).detach().cpu().numpy()

    sums = shap_values.sum((1, 2, 3))
    (
        np.testing.assert_allclose(sums + e.expected_value, outputs, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_pytorch_custom_nested_models(torch_device):
    """Testing single outputs"""
    torch = pytest.importorskip("torch")

    from sklearn.datasets import fetch_california_housing
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    class CustomNet1(nn.Module):
        """Model 1."""

        def __init__(self, num_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Sequential(
                    nn.Identity(),
                    nn.Conv1d(1, 1, 1),
                    nn.ConvTranspose1d(1, 1, 1),
                ),
                nn.AdaptiveAvgPool1d(output_size=num_features // 2),
            )

        def forward(self, X):
            """Run the model."""
            return self.net(X.unsqueeze(1)).squeeze(1)

    class CustomNet2(nn.Module):
        """Model 2."""

        def __init__(self, num_features):
            super().__init__()
            self.net = nn.Sequential(nn.LeakyReLU(), nn.Linear(num_features // 2, 2))

        def forward(self, X):
            """Run the model."""
            return self.net(X).unsqueeze(1)

    class CustomNet(nn.Module):
        """Model 3."""

        def __init__(self, num_features):
            super().__init__()
            self.net1 = CustomNet1(num_features)
            self.net2 = CustomNet2(num_features)
            self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        def forward(self, X):
            """Run the model."""
            x = self.net1(X)
            return self.maxpool2(self.net2(x)).squeeze(1)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        num_examples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            num_examples += target.shape[0]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                    f"\tLoss: {loss.item():.6f}"
                )

    random_seed = 777  # TODO: #2960

    torch.manual_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    X, y = fetch_california_housing(return_X_y=True)

    num_features = X.shape[1]

    data = TensorDataset(
        torch.tensor(X).float(),
        torch.tensor(y).float(),
    )

    loader = DataLoader(data, batch_size=128)

    model = CustomNet(num_features)
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device(torch_device)

    model.to(device)

    train(model, device, loader, optimizer, 1)

    next_x, _ = next(iter(loader))

    inds = rs.choice(next_x.shape[0], 20, replace=False)

    next_x_random_choices = next_x[inds, :].to(device)
    e = shap.DeepExplainer(model, next_x_random_choices)

    test_x_tmp, _ = next(iter(loader))
    test_x = test_x_tmp[:1].to(device)

    shap_values = e.shap_values(test_x)

    model.eval()
    model.zero_grad()

    with torch.no_grad():
        diff = model(test_x).detach().cpu().numpy()

    sums = shap_values.sum(axis=(1))
    (
        np.testing.assert_allclose(sums + e.expected_value, diff, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_pytorch_single_output(torch_device):
    """Testing single outputs"""
    torch = pytest.importorskip("torch")

    from sklearn.datasets import fetch_california_housing
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    class Net(nn.Module):
        """Test model."""

        def __init__(self, num_features):
            super().__init__()
            self.linear = nn.Linear(num_features // 2, 2)
            self.conv1d = nn.Conv1d(1, 1, 1)
            self.convt1d = nn.ConvTranspose1d(1, 1, 1)
            self.leaky_relu = nn.LeakyReLU()
            self.aapool1d = nn.AdaptiveAvgPool1d(output_size=num_features // 2)
            self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        def forward(self, X):
            """Run the model."""
            x = self.aapool1d(self.convt1d(self.conv1d(X.unsqueeze(1)))).squeeze(1)
            return self.maxpool2(self.linear(self.leaky_relu(x)).unsqueeze(1)).squeeze(1)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        num_examples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            num_examples += target.shape[0]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                    f"\tLoss: {loss.item():.6f}"
                )

    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0
    torch.manual_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    X, y = fetch_california_housing(return_X_y=True)

    num_features = X.shape[1]

    data = TensorDataset(
        torch.tensor(X).float(),
        torch.tensor(y).float(),
    )

    loader = DataLoader(data, batch_size=128)

    model = Net(num_features)
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device(torch_device)

    model.to(device)

    train(model, device, loader, optimizer, 1)

    next_x, _ = next(iter(loader))
    inds = rs.choice(next_x.shape[0], 20, replace=False)

    next_x_random_choices = next_x[inds, :].to(device)

    e = shap.DeepExplainer(model, next_x_random_choices)
    test_x_tmp, _ = next(iter(loader))
    test_x = test_x_tmp[:1].to(device)

    shap_values = e.shap_values(test_x)

    model.eval()
    model.zero_grad()

    with torch.no_grad():
        outputs = model(test_x).detach().cpu().numpy()

    sums = shap_values.sum(axis=(1))
    (
        np.testing.assert_allclose(sums + e.expected_value, outputs, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping on MacOS due to torch segmentation error, see GH #4075.",
)
@pytest.mark.parametrize("activation", ["relu", "selu", "gelu"])
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("disconnected", [True, False])
def test_pytorch_multiple_inputs(torch_device, disconnected, activation):
    """Check a multi-input scenario."""
    torch = pytest.importorskip("torch")

    from sklearn.datasets import fetch_california_housing
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    activation_func = {"relu": nn.ReLU(), "selu": nn.SELU(), "gelu": nn.GELU()}[activation]

    class Net(nn.Module):
        """Testing model."""

        def __init__(self, num_features, disconnected):
            super().__init__()
            self.disconnected = disconnected
            if disconnected:
                num_features = num_features // 2
            self.linear = nn.Linear(num_features, 2)
            self.output = nn.Sequential(nn.MaxPool1d(2), activation_func)

        def forward(self, x1, x2):
            """Run the model."""
            if self.disconnected:
                x = self.linear(x1).unsqueeze(1)
            else:
                x = self.linear(torch.cat((x1, x2), dim=-1)).unsqueeze(1)
            return self.output(x).squeeze(1)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        num_examples = 0
        for batch_idx, (data1, data2, target) in enumerate(train_loader):
            num_examples += target.shape[0]
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = F.mse_loss(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                    f"\tLoss: {loss.item():.6f}"
                )

    random_seed = 42  # TODO: 2960
    torch.manual_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    X, y = fetch_california_housing(return_X_y=True)

    num_features = X.shape[1]
    x1 = X[:, num_features // 2 :]
    x2 = X[:, : num_features // 2]

    data = TensorDataset(
        torch.tensor(x1).float(),
        torch.tensor(x2).float(),
        torch.tensor(y).float(),
    )

    loader = DataLoader(data, batch_size=128)

    model = Net(num_features, disconnected)
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device(torch_device)

    model.to(device)

    train(model, device, loader, optimizer, 1)

    next_x1, next_x2, _ = next(iter(loader))
    inds = rs.choice(next_x1.shape[0], 20, replace=False)
    background = [next_x1[inds, :].to(device), next_x2[inds, :].to(device)]
    e = shap.DeepExplainer(model, background)

    test_x1_tmp, test_x2_tmp, _ = next(iter(loader))
    test_x1 = test_x1_tmp[:1].to(device)
    test_x2 = test_x2_tmp[:1].to(device)

    shap_values = e.shap_values([test_x1[:1], test_x2[:1]])

    model.eval()
    model.zero_grad()

    with torch.no_grad():
        outputs = model(test_x1, test_x2[:1]).detach().cpu().numpy()

    # the shap values have the shape (num_samples, num_features, num_inputs, num_outputs)
    # so since we have just one output, we slice it out
    sums = shap_values[0].sum(1) + shap_values[1].sum(1)
    (
        np.testing.assert_allclose(sums + e.expected_value, outputs, atol=1e-3),
        "Sum of SHAP values does not match difference!",
    )
