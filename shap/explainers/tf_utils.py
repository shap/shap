tf = None
import warnings

def _import_tf():
    """ Tries to import tensorflow.
    """
    global tf
    if tf is None:
        import tensorflow as tf

def _get_session(session):
    """ Common utility to get the session for the tensorflow-based explainer.

    Parameters
    ----------
    explainer : Explainer

        One of the tensorflow-based explainers.

    session : tf.compat.v1.Session

        An optional existing session.
    """
    _import_tf()
    # if we are not given a session find a default session
    if session is None:
        try:
            session = tf.compat.v1.keras.backend.get_session()
        except:
            session = tf.keras.backend.get_session()
    return tf.get_default_session() if session is None else session

def _get_graph(explainer):
    """ Common utility to get the graph for the tensorflow-based explainer.

    Parameters
    ----------
    explainer : Explainer

        One of the tensorflow-based explainers.
    """
    _import_tf()
    if not tf.executing_eagerly():
        return explainer.session.graph
    else:
        return getattr(explainer.model_output, "graph", None)

def _get_model_inputs(model):
    """ Common utility to determine the model inputs.

    Parameters
    ----------
    model : Tensorflow Keras model or tuple

        The tensorflow model or tuple.
    """
    _import_tf()
    if str(type(model)).endswith("keras.engine.sequential.Sequential'>") or \
        str(type(model)).endswith("keras.models.Sequential'>") or \
        str(type(model)).endswith("keras.engine.training.Model'>") or \
        isinstance(model, tf.keras.Model):
        return model.inputs
    elif str(type(model)).endswith("tuple'>"):
        return model[0]
    else:
        assert False, str(type(model)) + " is not currently a supported model type!"

def _get_model_output(model):
    """ Common utility to determine the model output.

    Parameters
    ----------
    model : Tensorflow Keras model or tuple

        The tensorflow model or tuple.
    """
    _import_tf()
    if str(type(model)).endswith("keras.engine.sequential.Sequential'>") or \
        str(type(model)).endswith("keras.models.Sequential'>") or \
        str(type(model)).endswith("keras.engine.training.Model'>") or \
        isinstance(model, tf.keras.Model):
        if len(model.layers[-1]._inbound_nodes) == 0:
            if len(model.outputs) > 1:
                warnings.warn("Only one model output supported.")
            return model.outputs[0]
        else:
            return model.layers[-1].output
    elif str(type(model)).endswith("tuple'>"):
        return model[1]
    else:
        assert False, str(type(model)) + " is not currently a supported model type!"
