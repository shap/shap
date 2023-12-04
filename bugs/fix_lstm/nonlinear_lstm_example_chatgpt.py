import tensorflow as tf

# Example model
input_1 = tf.keras.layers.Input(shape=(80,))
embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=100)(input_1)

input_2 = tf.keras.layers.Input(shape=(80, 5))
concatenate_layer = tf.keras.layers.Concatenate()([embedding, input_2])

batch_norm = tf.keras.layers.BatchNormalization()(concatenate_layer)
lstm = tf.keras.layers.LSTM(64)(batch_norm)
output = tf.keras.layers.Dense(1)(lstm)

model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)

# Custom model for capturing operations
class OperationCaptureModel(tf.keras.Model):
    def __init__(self, base_model):
        super(OperationCaptureModel, self).__init__()
        self.base_model = base_model

    def call(self, inputs, training=None, mask=None):
        layer_outputs = []
        current_output = inputs

        for layer in self.base_model.layers:
            if isinstance(current_output, list):
                current_output = [current_output]
            current_output = layer(current_output, training=training, mask=mask)
            layer_outputs.append(current_output)

        return layer_outputs

# Sample input data
sample_input_1 = tf.constant([[1] * 80])
sample_input_2 = tf.constant([[[1] * 5] * 80])

# Create OperationCaptureModel
capture_model = OperationCaptureModel(model)

# Execute the custom model with the sample input
layer_outputs = capture_model([sample_input_1, sample_input_2])

# Print the operations in each layer
for layer, output in zip(model.layers, layer_outputs):
    print(f"Layer: {layer.name}, Operations: {output.op if hasattr(output, 'op') else None}")