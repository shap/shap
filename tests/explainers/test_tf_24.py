import numpy as np
import tensorflow as tf

import shap


# tf-estimator-nightly   2.4.0.dev2020092901
# tf-nightly             2.4.0.dev20200929


def test_dense():
    x = np.random.random((200, 10))
    w_ = np.random.random((10,))
    y = x.dot(w_)

    m = tf.keras.models.Sequential()
    m.add(
        tf.keras.layers.Dense(1, activation="linear", use_bias=None, input_shape=(10,))
    )
    m.compile(loss="MSE", optimizer=tf.keras.optimizers.Adam(lr=2))
    m.fit(x, y, epochs=20)
    pred = m.predict(x)[:, 0]

    w = m.get_weights()[0][:, 0]
    assert np.allclose(w, w_, atol=0.01, rtol=0.05)

    exp = shap.DeepExplainer(m, data=x[15:30])
    vals = exp.shap_values(x)
    rec = vals[0].sum(axis=1) + exp.expected_value.numpy()

    assert np.allclose(rec, pred)


def test_lstm():
    x = np.random.random((200, 10, 5))
    w_ = np.random.random((10,))
    y = x.sum(axis=-1).dot(w_)

    m = tf.keras.models.Sequential()
    m.add(
        tf.keras.layers.LSTM(
            1, input_shape=(10, 5), return_sequences=True, activation="linear"
        )
    )
    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(1))
    m.summary()
    m.compile(loss="MSE", optimizer=tf.keras.optimizers.Adam(lr=0.1))
    m.fit(x, y, epochs=50)
    pred = m.predict(x)[:, 0]

    exp = shap.DeepExplainer(m, data=x[15:30])
    vals = exp.shap_values(x, check_additivity=False)
    rec = vals[0].sum(axis=(1, 2)) + exp.expected_value.numpy()
    # print(pred)
    # print(rec)

    print(np.abs((pred - rec)/pred).mean())
    assert not np.allclose(vals, 0)
    assert np.allclose(rec, pred)


def test_cnn():
    x = np.random.random((200, 28, 28, 3))
    y = x.sum(axis=(1, 2, 3))

    m = tf.keras.models.Sequential()
    m.add(tf.keras.layers.Conv2D(3, (3, 3), input_shape=(28, 28, 3)))
    m.add(tf.keras.layers.Conv2D(3, (3, 3)))
    m.add(tf.keras.layers.MaxPool2D(2))
    m.add(tf.keras.layers.Conv2D(3, (3, 3)))
    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(1))
    m.summary()
    m.compile(loss="MSE", optimizer=tf.keras.optimizers.Adam(lr=0.1))
    m.fit(x, y, epochs=50)
    pred = m.predict(x)[:, 0]

    exp = shap.DeepExplainer(m, data=x[15:30])
    vals = exp.shap_values(x, check_additivity=False)
    rec = vals[0].sum(axis=(1, 2, 3)) + exp.expected_value.numpy()
    print(pred)
    print(rec)
    print(vals)
    assert not np.allclose(vals, 0)
    assert np.allclose(rec, pred)


test_lstm()
