from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest
from packaging import version

import shap


def test_tf_keras_mnist_cnn_tf216_and_above(random_seed):
    """This is the basic mnist cnn example from keras."""
    tf = pytest.importorskip("tensorflow")
    if version.parse(tf.__version__) < version.parse("2.16.0"):
        pytest.skip(
            "This test only works with tensorflow==2.16.1 and and above, see the test test_tf_keras_mnist_cnn_tf215_and_lower for lower tensorflow versions."
        )

    rs = np.random.RandomState(random_seed)
    tf.compat.v1.random.set_random_seed(random_seed)

    from tensorflow.compat.v1 import ConfigProto, InteractiveSession
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (
        Activation,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        MaxPooling2D,
    )
    from tensorflow.keras.models import Sequential

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

    batch_size = 128
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = rs.randn(200, 28, 28)
    y_train = rs.randint(0, 9, 200)
    x_test = rs.randn(200, 28, 28)
    y_test = rs.randint(0, 9, 200)

    if K.image_data_format() == "channels_first":
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

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))  # 128
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(), metrics=["accuracy"]
    )

    model.fit(
        x_train[:1000, :],
        y_train[:1000, :],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test[:1000, :], y_test[:1000, :]),
    )

    # explain by passing the tensorflow inputs and outputs
    inds = rs.choice(x_train.shape[0], 20, replace=False)
    e = shap.GradientExplainer((model.inputs, model.layers[-1].input), x_train[inds, :, :])
    shap_values = e.shap_values(x_test[:1], nsamples=2000)

    model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-1].input)
    outputs = model(x_test[:1]).numpy()
    background = model(x_train[inds, :, :]).numpy()
    # outputs = sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_test[:1]})
    # background = sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_train[inds, :, :]})
    expected_value = background.mean(0)

    sums = shap_values.sum((1, 2, 3))
    np.testing.assert_allclose(sums + expected_value, outputs, atol=1e-4)
    sess.close()


def test_tf_keras_mnist_cnn_tf215_and_lower(random_seed):
    """This is the basic mnist cnn example from keras."""
    tf = pytest.importorskip("tensorflow")
    if version.parse(tf.__version__) >= version.parse("2.16.0"):
        pytest.skip(
            "This test only works with tensorflow==2.15.1 and lower, see the test test_tf_keras_mnist_cnn_tf216_and_above for higher tensorflow versions."
        )

    rs = np.random.RandomState(random_seed)
    tf.compat.v1.random.set_random_seed(random_seed)

    from tensorflow.compat.v1 import ConfigProto, InteractiveSession
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (
        Activation,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        MaxPooling2D,
    )
    from tensorflow.keras.models import Sequential

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

    tf.compat.v1.disable_eager_execution()

    batch_size = 128
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = rs.randn(200, 28, 28)
    y_train = rs.randint(0, 9, 200)
    x_test = rs.randn(200, 28, 28)
    y_test = rs.randint(0, 9, 200)

    if K.image_data_format() == "channels_first":
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

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))  # 128
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.legacy.Adadelta(),
        metrics=["accuracy"],
    )

    model.fit(
        x_train[:1000, :],
        y_train[:1000, :],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test[:1000, :], y_test[:1000, :]),
    )

    # explain by passing the tensorflow inputs and outputs
    inds = rs.choice(x_train.shape[0], 20, replace=False)
    e = shap.GradientExplainer((model.layers[0].input, model.layers[-1].input), x_train[inds, :, :])
    shap_values = e.shap_values(x_test[:1], nsamples=2000)

    outputs = sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_test[:1]})
    background = sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_train[inds, :, :]})
    expected_value = background.mean(0)

    sums = shap_values.sum((1, 2, 3))
    np.testing.assert_allclose(sums + expected_value, outputs, atol=1e-4)
    sess.close()


def test_tf_multi_inputs_multi_outputs():
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
    explainer = shap.GradientExplainer(model, [input1_data, input2_data])
    shap_values = explainer.shap_values([input1_data, input2_data])
    np.testing.assert_allclose(shap_values[0].sum(1) + shap_values[1].sum(1) + predicted.mean(0), predicted, atol=1e-1)


def test_pytorch_mnist_cnn():
    """The same test as above, but for pytorch"""
    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0

    torch = pytest.importorskip("torch")
    torch.manual_seed(random_seed)
    rs = np.random.RandomState(random_seed)

    from torch import nn
    from torch.nn import functional as F

    batch_size = 128

    class RandData:
        """Ranomd data for testing."""

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

    try:
        # train_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST(tmpdir, train=True, download=True,
        #                 transform=transforms.Compose([
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.1307,), (0.3081,))
        #                 ])),
        #     batch_size=batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST(tmpdir, train=False, download=True,
        #                 transform=transforms.Compose([
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.1307,), (0.3081,))
        #                 ])),
        #     batch_size=batch_size, shuffle=True)
        train_loader = RandData(batch_size)
        test_loader = RandData(batch_size)
    except HTTPError:
        pytest.skip()

    def run_test(train_loader, test_loader, interim):
        class Net(nn.Module):
            """A test model."""

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
                self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(160, 20)
                self.fc2 = nn.Linear(20, 10)

            def forward(self, x):
                """Run the model."""
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, 160)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)

        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        def train(model, device, train_loader, optimizer, _, cutoff=20):
            model.train()
            num_examples = 0
            for _, (data, target) in enumerate(train_loader):
                num_examples += target.shape[0]
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                # if batch_idx % 10 == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(train_loader.dataset),
                #         100. * batch_idx / len(train_loader), loss.item()
                #     ))
                if num_examples > cutoff:
                    break

        device = torch.device("cpu")
        train(model, device, train_loader, optimizer, 1)

        next_x, _ = next(iter(train_loader))
        inds = rs.choice(next_x.shape[0], 3, replace=False)
        if interim:
            e = shap.GradientExplainer((model, model.conv1), next_x[inds, :, :, :])
        else:
            e = shap.GradientExplainer(model, next_x[inds, :, :, :])
        test_x, _ = next(iter(test_loader))
        shap_values = e.shap_values(test_x[:1], nsamples=1000)

        if not interim:
            # unlike deepLIFT, Integrated Gradients aren't necessarily consistent for interim layers
            model.eval()
            model.zero_grad()
            with torch.no_grad():
                outputs = model(test_x[:1]).detach().numpy()
                expected_value = model(next_x[inds, :, :, :]).detach().numpy().mean(0)
            sums = shap_values.sum(axis=(1, 2, 3))
            np.testing.assert_allclose(sums + expected_value, outputs, atol=1e-2)

    print("Running test from interim layer")
    run_test(train_loader, test_loader, True)
    print("Running test on whole model")
    run_test(train_loader, test_loader, False)


def test_pytorch_multiple_inputs(random_seed):
    """Test multi-input scenarios."""
    torch = pytest.importorskip("torch")
    from torch import nn

    torch.manual_seed(random_seed)
    batch_size = 10
    x1 = torch.ones(batch_size, 3)
    x2 = torch.ones(batch_size, 4)

    background = [torch.zeros(batch_size, 3), torch.zeros(batch_size, 4)]

    class Net(nn.Module):
        """A test model."""

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(7, 1)

        def forward(self, x1, x2):
            """Run the model."""
            return self.linear(torch.cat((x1, x2), dim=-1))

    model = Net()

    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values([x1, x2])

    model.eval()
    model.zero_grad()
    with torch.no_grad():
        outputs = model(x1, x2).detach().numpy()
        expected_value = model(*background).detach().numpy().mean(0)

    sums = np.sum([shap_values[i].sum(axis=1) for i in range(len(shap_values))], axis=0)
    np.testing.assert_allclose(sums + expected_value, outputs, atol=1e-2)


def test_pytorch_multiple_inputs_multiple_outputs(random_seed):
    """Test multi-input scenarios."""
    torch = pytest.importorskip("torch")
    from torch import nn

    torch.manual_seed(random_seed)
    batch_size = 10

    background = [torch.zeros(batch_size, 3), torch.zeros(batch_size, 4)]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(7, 6)  # Combined fully connected layer for both inputs

        def forward(self, input1, input2):
            x = torch.cat((input1, input2), dim=1)  # Concatenate both inputs
            x1 = self.fc(x)  # Final processing
            return x1

    model = Net()
    batch_size = 10
    input1 = torch.randn(batch_size, 3)
    input2 = torch.randn(batch_size, 4)
    model = Net()

    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values([input1, input2])

    model.eval()
    model.zero_grad()
    with torch.no_grad():
        outputs = model(input1, input2).detach().numpy()
        expected_value = model(*background).detach().numpy().mean(0)

    sums = np.sum([shap_values[i].sum(axis=1) for i in range(len(shap_values))], axis=0)
    np.testing.assert_allclose(sums + expected_value, outputs, atol=1e-5)


@pytest.mark.parametrize("input_type", ["numpy", "dataframe"])
def test_tf_input(random_seed, input_type):
    """Test tabular (batch_size, features) pd.DataFrame and numpy input."""
    tf = pytest.importorskip("tensorflow")
    tf.random.set_seed(random_seed)

    batch_size = 10
    num_features = 5
    feature_names = [f"TF_pd_test_feature_{i}" for i in range(num_features)]

    background = np.zeros((batch_size, num_features))
    if input_type == "dataframe":
        background = pd.DataFrame(background, columns=feature_names)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, input_shape=(num_features,), activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    explainer = shap.GradientExplainer(model, background)
    example = np.ones((1, num_features))
    explanation = explainer(example)

    diff = (model.predict(example) - model.predict(background)).mean(0)
    sums = np.array([values.sum() for values in explanation.values])
    d = np.abs(sums - diff).sum()
    assert d / (np.abs(diff).sum() + 0.01) < 0.1, "Sum of SHAP values does not match difference! %f" % (
        d / np.abs(diff).sum()
    )
