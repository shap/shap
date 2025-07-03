import warnings

import lazy_loader as lazy  # type: ignore[import-untyped]

tf = lazy.load("tensorflow", error_on_import=False)


def _get_session(session):
    """Common utility to get the session for the tensorflow-based explainer.

    Parameters
    ----------
    explainer : Explainer

        One of the tensorflow-based explainers.

    session : tf.compat.v1.Session

        An optional existing session.

    """
    # if we are not given a session find a default session
    if session is None:
        try:
            session = tf.compat.v1.keras.backend.get_session()
        except Exception:
            session = tf.keras.backend.get_session()
    return tf.get_default_session() if session is None else session


def _get_graph(explainer):
    """Common utility to get the graph for the tensorflow-based explainer.

    Parameters
    ----------
    explainer : Explainer

        One of the tensorflow-based explainers.

    """
    if not tf.executing_eagerly():
        return explainer.session.graph
    else:
        from tensorflow.python.keras import backend

        graph = backend.get_graph()
        return graph


def _get_model_inputs(model):
    """Common utility to determine the model inputs.

    Parameters
    ----------
    model : Tensorflow Keras model or tuple

        The tensorflow model or tuple.

    """
    if (
        str(type(model)).endswith("keras.engine.sequential.Sequential'>")
        or str(type(model)).endswith("keras.models.Sequential'>")
        or str(type(model)).endswith("keras.engine.training.Model'>")
        or isinstance(model, tf.keras.Model)
    ):
        return model.inputs
    if str(type(model)).endswith("tuple'>"):
        return model[0]

    emsg = f"{type(model)} is not currently a supported model type!"
    raise ValueError(emsg)


def _get_model_output(model):
    """Common utility to determine the model output.

    Parameters
    ----------
    model : Tensorflow Keras model or tuple

        The tensorflow model or tuple.

    """
    if (
        str(type(model)).endswith("keras.engine.sequential.Sequential'>")
        or str(type(model)).endswith("keras.models.Sequential'>")
        or str(type(model)).endswith("keras.engine.training.Model'>")
        or isinstance(model, tf.keras.Model)
    ):
        if len(model.layers[-1]._inbound_nodes) == 0:
            if len(model.outputs) > 1:
                warnings.warn("Only one model output supported.")
            return model.outputs[0]
        else:
            return model.layers[-1].output
    if str(type(model)).endswith("tuple'>"):
        return model[1]

    emsg = f"{type(model)} is not currently a supported model type!"
    raise ValueError(emsg)
