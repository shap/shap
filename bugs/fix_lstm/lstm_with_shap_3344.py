import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import shap
from tqdm import tqdm
shap.initjs()

# does not work with this kind of data creation
# disable_eager_execution()


# Define the start and end datetime
start_datetime = pd.to_datetime('2020-01-01 00:00:00')
end_datetime = pd.to_datetime('2023-03-31 23:00:00')

# Generate a DatetimeIndex with hourly frequency
date_rng = pd.date_range(start=start_datetime, end=end_datetime, freq='H')

# Create a DataFrame with random data for 7 features
num_samples = len(date_rng)
num_features = 7

# Generate random data for the DataFrame
data = np.random.rand(num_samples, num_features)

# Create the DataFrame with a DatetimeIndex
df = pd.DataFrame(data, index=date_rng, columns=[f'X{i}' for i in range(1, num_features+1)])


def windowed_dataset(series=None, in_horizon=None, out_horizon=None, delay=None, batch_size=None):
    '''
    Convert multivariate data into input and output sequences.
    Convert NumPy arrays to TensorFlow tensors.
    Arguments:
    ===========
    series: a list or array of time-series data.
    total_horizon: an integer representing the size of the input window.
    out_horizon: an integer representing the size of the output window.
    delay: an integer representing the number of steps between each input window.
    batch_size: an integer representing the batch size. 
    '''
    total_horizon = in_horizon + out_horizon
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(total_horizon, shift=delay, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(total_horizon))
    dataset = dataset.map(lambda window: (window[:-out_horizon,:], window[-out_horizon:,0]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset



# Define the proportions for the splits (70:20:10)%
train_size = 0.4
valid_size = 0.5
test_size = 0.1

# Calculate the split points
train_split = int(len(df)*train_size)
valid_split = int(len(df)*(train_size + valid_size))

# Split the DataFrame
df_train = df.iloc[:train_split]
df_valid = df.iloc[train_split:valid_split]
df_test = df.iloc[valid_split:]

# number of input features and output targets
n_features = df.shape[1]

# split the data into sliding sequential windows
train_dataset = windowed_dataset(series=df_train.values, 
                                in_horizon=100, 
                                out_horizon=3, 
                                delay=1, 
                                batch_size=32)

valid_dataset = windowed_dataset(series=df_valid.values, 
                                in_horizon=100, 
                                out_horizon=3, 
                                delay=1, 
                                batch_size=32)

test_dataset = windowed_dataset(series=df_test.values, 
                            in_horizon=100, 
                            out_horizon=3, 
                            delay=1, 
                            batch_size=32)

input_layer = tf.keras.layers.Input(shape=(100, n_features))
lstm_layer1 = tf.keras.layers.LSTM(5, return_sequences=True)(input_layer)
lstm_layer2 = tf.keras.layers.LSTM(5, return_sequences=True)(lstm_layer1)
lstm_layer3 = tf.keras.layers.LSTM(5)(lstm_layer2)
output_layer = tf.keras.layers.Dense(3)(lstm_layer3)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mae'])

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

breakpoint()
tt = list(train_dataset.as_numpy_iterator())
ss = [k[0] for k in tt]
input_tensor = tf.convert_to_tensor(ss[3], dtype=tf.float32)

layer_outputs = capture_model(input_tensor)
# 
# @tf.function
# def get_ops(input_data):
#     return model(input_data)
# 
# breakpoint()
# history = model.fit(train_dataset, epochs=1, validation_data=valid_dataset, verbose=1)
# output = get_ops(input_tensor)


# history2 = model_lazy().fit(train_dataset, epochs=1, validation_data=valid_dataset, verbose=1)

def tensor_to_arrays(input_obj=None):
    '''
    Convert a "tensorflow.python.data.ops.dataset_ops.PrefetchDataset" object into a numpy arrays.
    This function can be used to slice the tensor objects out of the `windowing` function.
    '''
    x = list(map(lambda x: x[0], input_obj))
    y = list(map(lambda x: x[1], input_obj))
    
    x_ = [xtmp.numpy() for xtmp in x]
    y_ = [ytmp.numpy() for ytmp in y]
    
    # Stack the arrays vertically
    x = np.vstack(x_)
    y = np.vstack(y_)
    
    return x, y


xarr, yarr = tensor_to_arrays(input_obj=train_dataset)

# Create an explainer object
import pdb; pdb.set_trace()
explainer = shap.DeepExplainer(model, xarr[:100, :, :])

# Calculate SHAP values for the data
#shap_values = explainer(xarr)
shap_values = explainer.shap_values(xarr[:1000, :, :], check_additivity=True)
import pdb; pdb.set_trace()


