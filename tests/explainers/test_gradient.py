from urllib.error import HTTPError
import numpy as np
import pytest
import shap


# pylint: disable=import-error, import-outside-toplevel, no-name-in-module, import-error

def test_tf_keras_mnist_cnn():
    """ This is the basic mnist cnn example from keras.
    """
    tf = pytest.importorskip('tensorflow')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras import backend as K
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)

    tf.compat.v1.disable_eager_execution()

    batch_size = 128
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.random.randn(200, 28, 28)
    y_train = np.random.randint(0, 9, 200)
    x_test = np.random.randn(200, 28, 28)
    y_test = np.random.randint(0, 9, 200)

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
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu')) # 128
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(
        x_train[:1000, :],
        y_train[:1000, :],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test[:1000, :], y_test[:1000, :])
    )

    # explain by passing the tensorflow inputs and outputs
    np.random.seed(0)
    inds = np.random.choice(x_train.shape[0], 20, replace=False)
    e = shap.GradientExplainer((model.layers[0].input, model.layers[-1].input), x_train[inds, :, :])
    shap_values = e.shap_values(x_test[:1], nsamples=2000)

    sess = tf.compat.v1.keras.backend.get_session()
    diff = sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_test[:1]}) - \
    sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_train[inds, :, :]}).mean(0)

    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    d = np.abs(sums - diff).sum()
    assert d / (np.abs(diff).sum() + 0.01) < 0.1, "Sum of SHAP values does not match difference! %f" % (d / np.abs(diff).sum())


def test_pytorch_mnist_cnn():
    """The same test as above, but for pytorch
    """
    torch = pytest.importorskip('torch')

    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(0)

    batch_size = 128

    class RandData:
        """ Ranomd data for testing.
        """
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
            """ A test model.
            """
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
                self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(160, 20)
                self.fc2 = nn.Linear(20, 10)

            def forward(self, x):
                """ Run the model.
                """
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

        device = torch.device('cpu') # pylint: disable=no-member
        train(model, device, train_loader, optimizer, 1)

        next_x, _ = next(iter(train_loader))
        np.random.seed(0)
        inds = np.random.choice(next_x.shape[0], 3, replace=False)
        if interim:
            e = shap.GradientExplainer((model, model.conv1), next_x[inds, :, :, :])
        else:
            e = shap.GradientExplainer(model, next_x[inds, :, :, :])
        test_x, _ = next(iter(test_loader))
        shap_values = e.shap_values(test_x[:1], nsamples=5000)

        if not interim:
            # unlike deepLIFT, Integrated Gradients aren't necessarily consistent for interim layers
            model.eval()
            model.zero_grad()
            with torch.no_grad():
                diff = (model(test_x[:1]) - model(next_x[inds, :, :, :])).detach().numpy().mean(0)
            sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
            d = np.abs(sums - diff).sum()
            assert d / (np.abs(diff).sum() + 0.01) < 0.1, "Sum of SHAP values " \
                                                 "does not match difference! %f" % (d / np.abs(diff).sum())

    print('Running test from interim layer')
    run_test(train_loader, test_loader, True)
    print('Running test on whole model')
    run_test(train_loader, test_loader, False)


def test_pytorch_multiple_inputs():
    """ Test multi-input scenarios.
    """
    # pylint: disable=no-member
    torch = pytest.importorskip('torch')
    from torch import nn
    torch.manual_seed(1)
    batch_size = 10
    x1 = torch.ones(batch_size, 3)
    x2 = torch.ones(batch_size, 4)

    background = [torch.zeros(batch_size, 3), torch.zeros(batch_size, 4)]

    class Net(nn.Module):
        """ A test model.
        """
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(7, 1)

        def forward(self, x1, x2):
            """ Run the model.
            """
            return self.linear(torch.cat((x1, x2), dim=-1))

    model = Net()

    e = shap.GradientExplainer(model, background)
    shap_x1, shap_x2 = e.shap_values([x1, x2])

    model.eval()
    model.zero_grad()
    with torch.no_grad():
        diff = (model(x1, x2) - model(*background)).detach().numpy().mean(0)

    sums = np.array([shap_x1[i].sum() + shap_x2[i].sum() for i in range(len(shap_x1))])
    d = np.abs(sums - diff).sum()
    assert d / (np.abs(diff).sum()+0.01) < 0.1, "Sum of SHAP values does not match difference! %f" % (d / np.abs(diff).sum())
