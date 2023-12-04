import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Sequential

from shap.explainers._deep import DeepExplainer


# Generate some sample time series data
def generate_time_series(num_data_points):
    time = np.arange(0, num_data_points)
    values = np.sin(0.1 * time) + 0.2 * np.random.randn(num_data_points)
    return values

# Create a time series dataset
num_data_points = 1000
time_series = generate_time_series(num_data_points)

# Function to create input sequences and corresponding targets
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Define sequence length and split data into sequences and targets
sequence_length = 10
X, y = create_sequences(time_series, sequence_length)

# Reshape data for RNN input (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build the RNN model using SimpleRNN layer
model = Sequential()
model.add(GRU(50, input_shape=(sequence_length, 1)))
model.add(Dense(1))  # Output layer with one neuron for regression task

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=2, batch_size=32, validation_split=0.1)

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
capture_model(X)
gru_ops = capture_model.ops
# ListWrapper([<tf.Operation 'gru/PartitionedCall' type=PartitionedCall>, <tf.Operation 'dense/BiasAdd' type=BiasAdd>])
# Wo werden die weiteren Ops eingefügt?
explainer = DeepExplainer(model, X)

# Calculate SHAP values for the data
#shap_values = explainer(xarr)
shap_values = explainer.shap_values(X[:10])
