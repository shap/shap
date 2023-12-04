import numpy as np
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

# Generate predictions on new data
new_data = generate_time_series(20)
new_sequences, _ = create_sequences(new_data, sequence_length)
new_sequences = new_sequences.reshape((new_sequences.shape[0], new_sequences.shape[1], 1))
predictions = model.predict(new_sequences)

# Print the predictions
print("Predictions:")
print(predictions.flatten())

explainer = DeepExplainer(model, new_sequences)

# Calculate SHAP values for the data
#shap_values = explainer(xarr)
breakpoint()
shap_values = explainer.shap_values(new_sequences[:10])
