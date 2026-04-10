from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

tf: ModuleType | None = None


def _import_tf() -> None:
    """Tries to import tensorflow."""
    global tf  # noqa: PLW0603
    if tf is None:
        import tensorflow as tf  # type: ignore[no-redef]  # noqa: PLW0603


def _get_session(session: Any | None) -> Any:
    """Common utility to get the session for the tensorflow-based explainer.

    Parameters
    ----------
    session : tf.compat.v1.Session
        An optional existing session.

    """
    _import_tf()
    # if we are not given a session find a default session
    if session is None:
        try:
            session = tf.compat.v1.keras.backend.get_session()  # type: ignore[union-attr]
        except Exception:
            session = tf.keras.backend.get_session()  # type: ignore[union-attr]
    return tf.get_default_session() if session is None else session  # type: ignore[union-attr]


def _get_graph(explainer: Any) -> Any:
    """Common utility to get the graph for the tensorflow-based explainer.

    Parameters
    ----------
    explainer : Explainer
        One of the tensorflow-based explainers.

    """
    _import_tf()
    if not tf.executing_eagerly():  # type: ignore[union-attr]
        return explainer.session.graph
    else:
        from tensorflow.python.keras import backend

        graph = backend.get_graph()
        return graph


def _get_model_inputs(model: Any) -> Any:
    """Common utility to determine the model inputs.

    Parameters
    ----------
    model : Tensorflow Keras model or tuple
        The tensorflow model or tuple.

    """
    _import_tf()
    if (
        str(type(model)).endswith("keras.engine.sequential.Sequential'>")
        or str(type(model)).endswith("keras.models.Sequential'>")
        or str(type(model)).endswith("keras.engine.training.Model'>")
        or isinstance(model, tf.keras.Model)  # type: ignore[union-attr]
    ):
        return model.inputs
    if str(type(model)).endswith("tuple'>"):
        return model[0]

    emsg = f"{type(model)} is not currently a supported model type!"
    raise ValueError(emsg)


def _get_model_output(model: Any) -> Any:
    """Common utility to determine the model output.

    Parameters
    ----------
    model : Tensorflow Keras model or tuple
        The tensorflow model or tuple.

    """
    _import_tf()
    if (
        str(type(model)).endswith("keras.engine.sequential.Sequential'>")
        or str(type(model)).endswith("keras.models.Sequential'>")
        or str(type(model)).endswith("keras.engine.training.Model'>")
        or isinstance(model, tf.keras.Model)  # type: ignore[union-attr]
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
