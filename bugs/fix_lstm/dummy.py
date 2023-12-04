import tensorflow as tf
import numpy as np

# Define your model architecture
input_data = tf.keras.layers.Input(shape=(4,))
x = tf.keras.layers.Dense(128, activation='relu')(input_data)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

# Create a Keras Model
model = tf.keras.Model(inputs=input_data, outputs=output)

# Sample input data
sample_input = tf.constant([[1.0, 2.0, 3.0, 4.0]])

# Define a function to convert the Keras tensor to a TensorFlow tensor
def keras_tensor_to_tf_tensor(keras_tensor):
    return keras_tensor

# Create a function using tf.keras.backend.function
tf_function = tf.keras.backend.function([model.input], [keras_tensor_to_tf_tensor(model.output)])

# Get the TensorFlow tensor
tf_output = tf_function([sample_input])
breakpoint()

# Now you can convert the TensorFlow tensor to a NumPy array
numpy_array = np.array(tf_output[0])

# Perform operations on the NumPy array if needed
print(numpy_array)