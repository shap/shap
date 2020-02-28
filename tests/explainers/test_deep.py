import shutil

import numpy as np
import nose
import os

from tests.fixtures import set_seed
from nose.tools import with_setup


# force us to not use any GPUs since running many tests may cause trouble
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def _skip_if_no_tensorflow():
    try:
        import tensorflow
    except ImportError:
        raise nose.SkipTest('Tensorflow not installed.')


def _skip_if_no_pytorch():
    try:
        import torch
    except ImportError:
        raise nose.SkipTest('Pytorch not installed.')


def test_tf_eager():
    """ This is a basic eager example from keras.
    """
    _skip_if_no_tensorflow()

    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from shap import DeepExplainer
    import datetime

    x = pd.DataFrame({ "B": np.random.random(size=(100,)) })
    y = x.B
    y = y.map(lambda zz: chr(int(zz * 2 + 65))).str.get_dummies()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape=(x.shape[1],), activation="relu"))
    model.add(tf.keras.layers.Dense(y.shape[1], input_shape=(10,), activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="Adam")
    model.fit(x.values, y.values, epochs=2)

    e = DeepExplainer(model, x.values[:1])
    sv = e.shap_values(x.values)
    assert np.abs(e.expected_value[0] + sv[0].sum(-1) - model(x.values)[:,0]).max() < 1e-4

def test_tf_keras_mnist_cnn():
    """ This is the basic mnist cnn example from keras.
    """
    _skip_if_no_tensorflow()

    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras import backend as K
    import tensorflow as tf
    import shap

    tf.compat.v1.disable_eager_execution()

    batch_size = 128
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu')) # 128
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train[:1000,:], y_train[:1000,:],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test[:1000,:], y_test[:1000,:]))

    # explain by passing the tensorflow inputs and outputs
    np.random.seed(0)
    inds = np.random.choice(x_train.shape[0], 10, replace=False)
    e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].input), x_train[inds,:,:])
    shap_values = e.shap_values(x_test[:1])

    sess = tf.compat.v1.keras.backend.get_session()
    diff = sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_test[:1]}) - \
    sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_train[inds,:,:]}).mean(0)

    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    d = np.abs(sums - diff).sum()
    assert d / np.abs(diff).sum() < 0.001, "Sum of SHAP values does not match difference! %f" % d

def test_tf_keras_linear():
    """Test verifying that a linear model with linear data gives the correct result.
    """
    _skip_if_no_tensorflow()

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import SGD
    import tensorflow as tf
    import shap

    tf.compat.v1.disable_eager_execution()

    np.random.seed(0)

    # coefficients relating y with x1 and x2.
    coef = np.array([1, 2]).T

    # generate data following a linear relationship
    x = np.random.normal(1, 10, size=(1000, len(coef)))
    y = np.dot(x, coef) + 1 + np.random.normal(scale=0.1, size=1000)

    # create a linear model
    inputs = Input(shape=(2,))
    preds = Dense(1, activation='linear')(inputs)

    model = Model(inputs=inputs, outputs=preds)
    model.compile(optimizer=SGD(), loss='mse', metrics=['mse'])
    model.fit(x, y, epochs=30, shuffle=False, verbose=0)

    fit_coef = model.layers[1].get_weights()[0].T[0]

    # explain
    e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), x)
    shap_values = e.shap_values(x)

    # verify that the explanation follows the equation in LinearExplainer
    values = shap_values[0] # since this is a "multi-output" model with one output

    assert values.shape == (1000, 2)

    expected = (x - x.mean(0)) * fit_coef
    np.testing.assert_allclose(expected - values, 0, atol=1e-5)

def test_tf_keras_imdb_lstm():
    """ Basic LSTM example using the keras API defined in tensorflow
    """
    _skip_if_no_tensorflow()

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.preprocessing import sequence
    import shap

    tf.compat.v1.disable_eager_execution()

    # load the data from keras
    np.random.seed(7)
    max_features = 1000
    try:
        (X_train, y_train), (X_test, _) = imdb.load_data(num_words=max_features)
    except:
        return # this hides a bug in the most recent version of keras that prevents data loading
    X_train = sequence.pad_sequences(X_train, maxlen=100)
    X_test = sequence.pad_sequences(X_test, maxlen=100)

    # create the model. note that this is model is very small to make the test
    # run quick and we don't care about accuracy here
    mod = Sequential()
    mod.add(Embedding(max_features, 8))
    mod.add(LSTM(10, dropout=0.2, recurrent_dropout=0.2))
    mod.add(Dense(1, activation='sigmoid'))
    mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # select the background and test samples
    inds = np.random.choice(X_train.shape[0], 3, replace=False)
    background = X_train[inds]
    testx = X_test[10:11]

    # Question for Scott: we can explain without fitting?
    # mod.fit(X_train, y_train, epochs=1, shuffle=False, verbose=1)

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
    diff = sess.run(mod.layers[-1].output, feed_dict={mod.layers[0].input: testx})[0,:] - \
        sess.run(mod.layers[-1].output, feed_dict={mod.layers[0].input: background}).mean(0)
    assert np.allclose(sums, diff, atol=1e-02), "Sum of SHAP values does not match difference!"


@with_setup(set_seed)
def test_pytorch_mnist_cnn():
    """The same test as above, but for pytorch
    """
    _skip_if_no_pytorch()

    import torch, torchvision
    from torchvision import datasets, transforms
    from torch import nn
    from torch.nn import functional as F
    import shap

    def run_test(train_loader, test_loader, interim, device):

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Testing several different activations
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 10, kernel_size=5),
                    nn.MaxPool2d(2),
                    nn.Tanh(),
                    nn.Conv2d(10, 20, kernel_size=5),
                    nn.ConvTranspose2d(20, 20, 1),
                    nn.AdaptiveAvgPool2d(output_size=(4, 4)),
                    nn.Softplus(),
                )
                self.fc_layers = nn.Sequential(
                    nn.Linear(320, 50),
                    nn.BatchNorm1d(50),
                    nn.ReLU(),
                    nn.Linear(50, 10),
                    nn.ELU(),
                    nn.Softmax(dim=1)
                )

            def forward(self, x):
                x = self.conv_layers(x)
                x = x.view(-1, 320)
                x = self.fc_layers(x)
                return x

        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        def train(model, device, train_loader, optimizer, epoch, cutoff=2000):
            model.train()
            num_examples = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                num_examples += target.shape[0]
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.mse_loss(output, torch.eye(10)[target].to(device))
                # loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
                if num_examples > cutoff:
                    break

        train(model.to(device), device, train_loader, optimizer, 1)

        next_x, next_y = next(iter(train_loader))
        np.random.seed(0)
        inds = np.random.choice(next_x.shape[0], 20, replace=False)
        if interim:
            e = shap.DeepExplainer(
                (model.to(device), model.conv_layers[0].to(device)), next_x[inds, :, :, :].to(device)
            )
        else:
            e = shap.DeepExplainer(
                model.to(device), next_x[inds, :, :, :].to(device)
            )
        test_x, test_y = next(iter(test_loader))
        input_tensor = test_x[:1]
        input_tensor.requires_grad = True
        shap_values = e.shap_values(input_tensor.to(device))

        model = model.to(device)
        model.eval()
        model.zero_grad()
        with torch.no_grad():
            test_outputs = model(test_x[:1].to(device))
            next_outputs = model(next_x[inds, :].to(device))
            diff = (test_outputs - next_outputs).detach().cpu().numpy().mean(0)
        sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
        d = np.abs(sums - diff).sum()
        assert d / np.abs(diff).sum() < 0.001, "Sum of SHAP values does not match difference! %f" % (
                d / np.abs(diff).sum())

    batch_size = 128
    root_dir = 'mnist_data'

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root_dir, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    print('Running test on interim layer')
    run_test(train_loader, test_loader, interim=True, device="cpu")
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = 0
        run_test(train_loader, test_loader, interim=True, device="cuda:0")
    print('Running test on whole model')
    run_test(train_loader, test_loader, interim=False, device="cpu")
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = 0
        run_test(train_loader, test_loader, interim=True, device="cuda:0")
    # clean up
    shutil.rmtree(root_dir)

@with_setup(set_seed)
def test_pytorch_single_output():
    """Testing single outputs
    """
    _skip_if_no_pytorch()

    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.datasets import load_boston
    import shap

    def run_test(loader, device):
        class Net(nn.Module):
            def __init__(self, num_features):
                super(Net, self).__init__()
                self.linear = nn.Linear(num_features // 2, 2)
                self.conv1d = nn.Conv1d(1, 1, 1)
                self.convt1d = nn.ConvTranspose1d(1, 1, 1)
                self.leaky_relu = nn.LeakyReLU()
                self.aapool1d = nn.AdaptiveAvgPool1d(output_size=6)
                self.maxpool2 = nn.MaxPool1d(kernel_size=2)

            def forward(self, X):
                x = self.aapool1d(self.convt1d(self.conv1d(X.unsqueeze(1)))).squeeze(1)
                return self.maxpool2(self.linear(self.leaky_relu(x)).unsqueeze(1)).squeeze(1)
        model = Net(num_features)
        optimizer = torch.optim.Adam(model.parameters())

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
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))

        train(model.to(device), device, loader, optimizer, 1)

        next_x, next_y = next(iter(loader))
        np.random.seed(0)
        inds = np.random.choice(next_x.shape[0], 20, replace=False)
        e = shap.DeepExplainer(model.to(device), next_x[inds, :].to(device))
        test_x, test_y = next(iter(loader))
        shap_values = e.shap_values(test_x[:1].to(device))

        model = model.to(device)
        model.eval()
        model.zero_grad()
        with torch.no_grad():
            test_outputs = model(test_x[:1].to(device))
            next_outputs = model(next_x[inds, :].to(device))
            diff = (test_outputs - next_outputs).detach().cpu().numpy().mean(0)
        sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
        d = np.abs(sums - diff).sum()
        assert d / np.abs(diff).sum() < 0.001, "Sum of SHAP values does not match difference! %f" % (
                d / np.abs(diff).sum())

    X, y = load_boston(return_X_y=True)
    num_features = X.shape[1]
    data = TensorDataset(torch.tensor(X).float(),
                         torch.tensor(y).float())
    loader = DataLoader(data, batch_size=128)

    run_test(loader, device="cpu")
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = 0
        run_test(loader, device="cuda:0")

@with_setup(set_seed)
def test_pytorch_multiple_inputs():
    _skip_if_no_pytorch()

    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.datasets import load_boston
    import shap

    def _run_pytorch_multiple_inputs_test(disconnected, device):
        """Testing multiple inputs
        """

        X, y = load_boston(return_X_y=True)
        num_features = X.shape[1]
        x1 = X[:, num_features // 2:]
        x2 = X[:, :num_features // 2]
        data = TensorDataset(torch.tensor(x1).float(),
                             torch.tensor(x2).float(),
                             torch.tensor(y).float())
        loader = DataLoader(data, batch_size=128)

        class Net(nn.Module):
            def __init__(self, num_features, disconnected):
                super(Net, self).__init__()
                self.disconnected = disconnected
                if disconnected:
                    num_features = num_features // 2 + 1
                self.linear = nn.Linear(num_features, 2)
                self.output = nn.Sequential(
                    nn.MaxPool1d(2),
                    nn.ReLU()
                )

            def forward(self, x1, x2):
                if self.disconnected:
                    x = self.linear(x1).unsqueeze(1)
                else:
                    x = self.linear(torch.cat((x1, x2), dim=-1)).unsqueeze(1)
                return self.output(x).squeeze(1)

        model = Net(num_features, disconnected)
        optimizer = torch.optim.Adam(model.parameters())

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
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))

        train(model.to(device), device, loader, optimizer, 1)

        next_x1, next_x2, next_y = next(iter(loader))
        np.random.seed(0)
        inds = np.random.choice(next_x1.shape[0], 20, replace=False)
        background = [next_x1[inds, :].to(device), next_x2[inds, :].to(device)]
        e = shap.DeepExplainer(model.to(device), background)
        test_x1, test_x2, test_y = next(iter(loader))
        shap_x1, shap_x2 = e.shap_values([test_x1[:1].to(device), test_x2[:1].to(device)])

        model = model.to(device)
        model.eval()
        model.zero_grad()
        with torch.no_grad():
            test_outputs = model(test_x1[:1].to(device), test_x2[:1].to(device))
            next_outputs = model(*[b.to(device) for b in background])
            diff = (test_outputs - next_outputs).detach().cpu().numpy().mean(0)
        sums = np.array([shap_x1[i].sum() + shap_x2[i].sum() for i in range(len(shap_x1))])
        d = np.abs(sums - diff).sum()
        assert d / np.abs(diff).sum() < 0.001, "Sum of SHAP values does not match difference! %f" % (
                d / np.abs(diff).sum())

    _run_pytorch_multiple_inputs_test(disconnected=True, device="cpu")
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = 0
        _run_pytorch_multiple_inputs_test(disconnected=False, device="cuda:0")
