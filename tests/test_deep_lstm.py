import matplotlib
import numpy as np

matplotlib.use('Agg')
import shap

def test_multiply():
    """ Basic LSTM example from keras
    """

    try:
        import keras
        import numpy as np
        import tensorflow as tf
        from keras.datasets import imdb
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers.embeddings import Embedding
        from keras.preprocessing import sequence
    except Exception as e:
        print("Skipping test_tf_keras_mnist_cnn!")
        return
    import shap

    np.random.seed(7)
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000)
    X_train = np.expand_dims(sequence.pad_sequences(X_train, maxlen=100),axis=2)
    X_test = np.expand_dims(sequence.pad_sequences(X_test, maxlen=100),axis=2)
    # create the model
    embedding_vector_length = 32
    mod = Sequential()
    mod.add(LSTM(100, input_shape=(100,1)))
    mod.add(Dense(1, activation='sigmoid'))
    mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    inds = np.random.choice(X_train.shape[0], 3, replace=False)
    data = X_train[inds,:]
    test_in = X_test[10:11,:,:]
    
    e = shap.DeepExplainer((mod.layers[0].input, mod.layers[-1].input), data)
    shap_values = e.shap_values(test_in)
    sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
    sess = tf.keras.backend.get_session()
    diff = sess.run(mod.layers[-1].input, feed_dict={mod.layers[0].input: test_in})[0,:] - \
        sess.run(mod.layers[-1].input, feed_dict={mod.layers[0].input: data}).mean(0)
    assert np.allclose(sums, diff, atol=1e-06), "Sum of SHAP values does not match difference!"
