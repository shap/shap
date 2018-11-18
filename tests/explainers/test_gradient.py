import matplotlib
import numpy as np
import shutil
matplotlib.use('Agg')
import shap

def test_tf_keras_mnist_cnn():
    """ This is the basic mnist cnn example from keras.
    """

    try:
        import tensorflow as tf
        from tensorflow.python import keras
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation
        from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
        from tensorflow.python.keras import backend as K
    except Exception as e:
        print("Skipping test_tf_keras_mnist_cnn!")
        return
    import shap

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
    inds = np.random.choice(x_train.shape[0], 20, replace=False)
    e = shap.GradientExplainer((model.layers[0].input, model.layers[-1].input), x_train[inds,:,:])
    shap_values = e.shap_values(x_test[:1], nsamples=2000)

    sess = tf.keras.backend.get_session()
    diff = sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_test[:1]}) - \
    sess.run(model.layers[-1].input, feed_dict={model.layers[0].input: x_train[inds,:,:]}).mean(0)

    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    d = np.abs(sums - diff).sum()
    assert d / np.abs(diff).sum() < 0.05, "Sum of SHAP values does not match difference! %f" % (d / np.abs(diff).sum())


def test_pytorch_mnist_cnn():
    """The same test as above, but for pytorch
    """
    try:
        import torch, torchvision
        from torchvision import datasets, transforms
        from torch import nn
        from torch.nn import functional as F
    except Exception as e:
        print("Skipping test_pytorch_mnist_cnn!")
        return
    import shap

    batch_size=128
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

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

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
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            if num_examples > cutoff:
                break

    device = torch.device('cpu')
    train(model, device, train_loader, optimizer, 1)

    next_x, next_y = next(iter(train_loader))
    np.random.seed(0)
    inds = np.random.choice(next_x.shape[0], 20, replace=False)
    e = shap.GradientExplainer(model, next_x[inds, :, :, :])
    test_x, test_y = next(iter(test_loader))
    shap_values = e.shap_values(test_x[:1], nsamples=1000)

    model.eval()
    model.zero_grad()
    with torch.no_grad():
        diff = (model(test_x[:1]) - model(next_x[inds, :, :, :])).detach().numpy().mean(0)
    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    d = np.abs(sums - diff).sum()
    assert d / np.abs(diff).sum() < 0.05, "Sum of SHAP values does not match difference! %f" % (d / np.abs(diff).sum())

    # clean up
    shutil.rmtree(root_dir)
