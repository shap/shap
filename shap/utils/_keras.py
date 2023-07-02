""" This file contains various utility functions that are useful but not core to SHAP.
"""


def clone_keras_layers(model, start_layer, stop_layer):
        """ Clones the keras layers between the start and stop layer as a new model.
        """

        import tensorflow as tf

        if type(start_layer) is int:
            start_layer = model.layers[start_layer]
        if type(stop_layer) is int:
            stop_layer = model.layers[stop_layer]

        input_shape = start_layer.get_input_shape_at(0) # get the input shape of desired layer
        layer_input = tf.keras.Input(shape=input_shape[1:]) # a new input tensor to be able to feed the desired layer

        new_layers = {start_layer.input.name: layer_input}
        layers_to_process = [l for l in model.layers]
        last_len = 0
        dup_try = 0
        while len(layers_to_process) > 0:
            layer = layers_to_process.pop(0)
            if len(layers_to_process) == last_len:
                dup_try += 1
            else:
                dup_try = 0
            last_len = len(layers_to_process)
            if dup_try > len(layers_to_process):
                raise Exception("Failed to find a complete graph starting at the given layer!")
            try:
                if type(layer.input) is list:
                    layer_inputs = [new_layers[v.name] for v in layer.input]
                else:
                    layer_inputs = new_layers[layer.input.name]
            except KeyError:
                # we don't have all the inputs ready for us read so put us back on the list
                # behind the next one in line
                layers_to_process.append(layer)
                continue
            if layer.output.name not in new_layers:
                new_layers[layer.output.name] = layer(layer_inputs)
            if layer.output.name == stop_layer.output.name:
                break
        return tf.keras.Model(layer_input, new_layers[stop_layer.output.name])

def split_keras_model(model, layer):
    """ Splits the keras model around layer into two models.

    This is done such that model2(model1(X)) = model(X)
    and mode11(X) == layer(X)
    """

    if type(layer) is str:
        layer = model.get_layer(layer)
    elif type(layer) is int:
        layer = model.layers[layer]

    prev_layer = model.get_layer(layer.get_input_at(0).name.split("/")[0])

    model1 = clone_keras_layers(model, model.layers[1], prev_layer)
    model2 = clone_keras_layers(model, layer, model.layers[-1])

    return model1, model2
