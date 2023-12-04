# univariate lstm example
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from shap.explainers._deep import DeepExplainer
import tensorflow as tf

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
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
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = np.array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

class OperationCaptureModel(tf.keras.Model):
    def __init__(self, layers):
        super().__init__()
        self._layers = layers
        self.ops = []

    @tf.function
    def __call__(self, inputs):
        layer_outputs = []
        for layer in self._layers:
            inputs = layer(inputs)
            self.ops.append(inputs.op)
            layer_outputs.append(inputs)
        return layer_outputs

capture_model = OperationCaptureModel(model.layers)

# breakpoint()
input_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)

# this works but the same thing does not work when calling the deep explainer
layer_outputs = capture_model(input_tensor)


breakpoint()
explainer = DeepExplainer(model, x_input)

# Calculate SHAP values for the data
#shap_values = explainer(xarr)
breakpoint()
shap_values = explainer.shap_values(x_input)
import pdb; pdb.set_trace()
print('hi')