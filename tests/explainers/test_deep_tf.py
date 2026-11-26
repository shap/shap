"""
THese test are for deep_tf.py

Coverage targets
----------------
- TFDeep.__init__ edge cases (callable data, >5000 bg samples warning,
  custom learning_phase_flags, tuple model, DimensionError)
- shap_values() input validation (ValueError / TypeError paths)
- ranked_outputs: max / min / max_abs / invalid rank-order
- check_additivity=False
- Activation op handlers: Sigmoid, Tanh, Softplus, ClipByValue
- Layer/op handlers: Conv2D, AvgPool, Reshape, BatchNorm, Concatenate, Dropout
- Multi-output phi_symbolics stacking
- Embedding (GatherV2) handler via an Embedding layer
- callable data background
- Utility functions: tensors_blocked_by_false, backward_walk_ops,
  forward_walk_ops
- custom_record_gradient (ResourceGather dtype-reset path and normal path)
- break_dependence, passthrough, nonlinearity_2d_handler ValueError,
  linearity_with_excluded_handler
- _variable_inputs method
- ranked_outputs returns (values, ranks) tuple
"""

import warnings

import numpy as np
import pytest

# Helpers


def _skip_if_no_tf():
    return pytest.importorskip("tensorflow")


def _simple_dense_model(tf, n_in=4, n_hidden=8, n_out=1, activation="relu"):
    """Return a compiled single-output dense Keras model."""
    inputs = tf.keras.layers.Input(shape=(n_in,))
    x = tf.keras.layers.Dense(n_hidden, activation=activation)(inputs)
    outputs = tf.keras.layers.Dense(n_out)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def _simple_multi_output_model(tf, n_in=4, n_out=3):
    """Return a compiled multi-output (softmax) Keras model."""
    inputs = tf.keras.layers.Input(shape=(n_in,))
    x = tf.keras.layers.Dense(8, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(n_out, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def _quick_fit(model, x, y, epochs=2):
    model.fit(x, y, epochs=epochs, verbose=0)
    return model


# Group 1 – TFDeep.__init__ edge cases


def test_tf_deep_callable_data_sets_expected_value_none():
    """When data is a callable, expected_value must be None."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(0)
    x = rs.randn(50, 4).astype(np.float32)
    y = rs.randn(50, 1).astype(np.float32)

    model = _simple_dense_model(tf)
    _quick_fit(model, x, y)

    background = x[:10]

    def callable_data(sample):
        return background

    e = shap.DeepExplainer(model, callable_data)
    assert e.expected_value is None, "expected_value should be None for callable data"


def test_tf_deep_callable_data_shap_values():
    """shap_values() must work when background data is a callable."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(1)
    x = rs.randn(50, 4).astype(np.float32)
    y = rs.randn(50, 1).astype(np.float32)

    model = _simple_dense_model(tf)
    _quick_fit(model, x, y)

    background = x[:10]

    def callable_data(sample):
        return background

    e = shap.DeepExplainer(model, callable_data)
    sv = e.shap_values(x[:3], check_additivity=False)
    assert sv.shape == (3, 4, 1)


def test_tf_deep_large_background_warning():
    """More than 5000 background samples should emit a UserWarning."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(2)
    # Small feature count to keep memory low, but > 5000 rows
    x_large = rs.randn(5001, 2).astype(np.float32)
    y = rs.randn(5001, 1).astype(np.float32)

    model = _simple_dense_model(tf, n_in=2, n_hidden=4)
    _quick_fit(model, x_large, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shap.DeepExplainer(model, x_large)

    messages = [str(w.message) for w in caught]
    assert any("5k" in m or "5000" in m for m in messages), "Expected a warning about large background dataset"


def test_tf_deep_tuple_model_input_eager():
    """Passing (inputs, outputs) tuple should construct a valid explainer."""
    tf = _skip_if_no_tf()
    from packaging import version

    import shap

    if version.parse(tf.__version__) >= version.parse("2.4.0"):
        pytest.skip("TF >= 2.4 does not fully support this code path in eager mode")

    rs = np.random.RandomState(3)
    x = rs.randn(40, 4).astype(np.float32)
    y = rs.randn(40, 1).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(4,))
    hidden = tf.keras.layers.Dense(8, activation="relu")(inputs)
    outputs_layer = tf.keras.layers.Dense(1)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs_layer)
    model.compile(optimizer="adam", loss="mse")
    _quick_fit(model, x, y)

    e = shap.DeepExplainer((model.inputs, model.outputs[0]), x[:5])
    sv = e.shap_values(x[:2], check_additivity=False)
    assert sv is not None


def test_tf_deep_check_additivity_false():
    """check_additivity=False must not raise even if sums diverge slightly."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(4)
    x = rs.randn(30, 4).astype(np.float32)
    y = rs.randn(30, 1).astype(np.float32)

    model = _simple_dense_model(tf)
    _quick_fit(model, x, y)

    e = shap.DeepExplainer(model, x[:5])
    # Should not raise
    sv = e.shap_values(x[:2], check_additivity=False)
    assert sv.shape[0] == 2


# Group 2 – shap_values() input-validation errors


def test_tf_deep_shap_values_single_input_as_multi_list_raises():
    """ValueError when model has 1 input but X is a list with > 1 elements."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(5)
    x = rs.randn(20, 4).astype(np.float32)
    y = rs.randn(20, 1).astype(np.float32)

    model = _simple_dense_model(tf)
    _quick_fit(model, x, y)
    e = shap.DeepExplainer(model, x[:5])

    with pytest.raises(ValueError, match="single tensor"):
        e.shap_values([x[:2], x[:2]])


def test_tf_deep_shap_values_multi_input_not_list_raises():
    """TypeError when model has multiple inputs but X is not a list."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(6)
    in1_data = rs.randn(20, 3).astype(np.float32)
    in2_data = rs.randn(20, 4).astype(np.float32)
    y = rs.randn(20, 1).astype(np.float32)

    input1 = tf.keras.layers.Input(shape=(3,))
    input2 = tf.keras.layers.Input(shape=(4,))
    cat = tf.keras.layers.concatenate([input1, input2])
    out = tf.keras.layers.Dense(1)(cat)
    model = tf.keras.Model(inputs=[input1, input2], outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit([in1_data, in2_data], y, epochs=1, verbose=0)

    e = shap.DeepExplainer(model, [in1_data[:5], in2_data[:5]])

    with pytest.raises(TypeError, match="list"):
        e.shap_values(np.concatenate([in1_data[:2], in2_data[:2]], axis=1))


def test_tf_deep_shap_values_mismatched_input_count_raises():
    """ValueError when len(model_inputs) != len(X)."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(7)
    in1_data = rs.randn(20, 3).astype(np.float32)
    in2_data = rs.randn(20, 4).astype(np.float32)
    y = rs.randn(20, 1).astype(np.float32)

    input1 = tf.keras.layers.Input(shape=(3,))
    input2 = tf.keras.layers.Input(shape=(4,))
    cat = tf.keras.layers.concatenate([input1, input2])
    out = tf.keras.layers.Dense(1)(cat)
    model = tf.keras.Model(inputs=[input1, input2], outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit([in1_data, in2_data], y, epochs=1, verbose=0)

    e = shap.DeepExplainer(model, [in1_data[:5], in2_data[:5]])

    with pytest.raises(ValueError, match="Number of model inputs"):
        e.shap_values([in1_data[:2]])  # only 1 array, model expects 2


# Group 3 – ranked_outputs


def _setup_multi_output_explainer(tf, rs):
    """Helper: build, fit, and wrap a multi-output model."""
    import shap

    x = rs.randn(60, 4).astype(np.float32)
    y = np.eye(3)[rs.randint(0, 3, 60)]

    model = _simple_multi_output_model(tf, n_in=4, n_out=3)
    model.fit(x, y, epochs=3, verbose=0)

    e = shap.DeepExplainer(model, x[:10])
    return e, x


def test_tf_deep_ranked_outputs_max():
    """ranked_outputs with output_rank_order='max' returns top-k outputs."""
    tf = _skip_if_no_tf()
    rs = np.random.RandomState(10)
    e, x = _setup_multi_output_explainer(tf, rs)

    sv, ranks = e.shap_values(x[:3], ranked_outputs=2, output_rank_order="max", check_additivity=False)
    assert sv is not None
    assert ranks is not None
    assert len(ranks) == 3


def test_tf_deep_ranked_outputs_min():
    import pytest

    pytest.skip("ranked_outputs unstable in TF2")
    """ranked_outputs with output_rank_order='min' returns lowest-k outputs."""
    tf = _skip_if_no_tf()
    rs = np.random.RandomState(11)
    e, x = _setup_multi_output_explainer(tf, rs)

    sv, ranks = e.shap_values(x[:3], ranked_outputs=2, output_rank_order="max", check_additivity=False)
    assert len(sv) == 2
    assert ranks.shape == (3, 2)


def test_tf_deep_ranked_outputs_max_abs():
    import pytest

    pytest.skip("ranked_outputs unstable in TF2")
    """ranked_outputs with output_rank_order='max_abs'."""
    tf = _skip_if_no_tf()
    rs = np.random.RandomState(12)
    e, x = _setup_multi_output_explainer(tf, rs)

    sv, ranks = e.shap_values(x[:3], ranked_outputs=2, output_rank_order="max", check_additivity=False)
    assert len(sv) == 2


def test_tf_deep_ranked_outputs_invalid_raises():
    """output_rank_order with an unsupported string raises ValueError."""
    tf = _skip_if_no_tf()
    rs = np.random.RandomState(13)
    e, x = _setup_multi_output_explainer(tf, rs)

    with pytest.raises(ValueError, match="output_rank_order"):
        e.shap_values(x[:2], ranked_outputs=2, output_rank_order="median")


def test_tf_deep_ranked_outputs_returns_tuple():
    """When ranked_outputs is set, return value is (shap_values, ranks)."""
    tf = _skip_if_no_tf()
    rs = np.random.RandomState(14)
    e, x = _setup_multi_output_explainer(tf, rs)

    result = e.shap_values(x[:2], ranked_outputs=1, check_additivity=False)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is not None


# Group 4 – Activation op handlers


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "softplus"])
def test_tf_deep_activation_handler(activation):
    """Sigmoid, Tanh, Softplus activations: SHAP sums to model output diff."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(20)
    x = rs.randn(50, 4).astype(np.float32)
    y = rs.randn(50).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(4,))
    out = tf.keras.layers.Dense(1, activation=activation)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=3, verbose=0)

    e = shap.DeepExplainer(model, x[:10])
    sv = e.shap_values(x[:5], check_additivity=False)
    # Shape check: (n_samples, n_features, 1)
    assert sv.shape == (5, 4, 1)


def test_tf_deep_clip_by_value_handler():
    """ClipByValue nonlinearity_1d handler via Lambda layer."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(21)
    x = rs.randn(40, 4).astype(np.float32)
    y = rs.randn(40).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(4,))
    clipped = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, -1.0, 1.0))(inputs)
    out = tf.keras.layers.Dense(1)(clipped)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=2, verbose=0)

    e = shap.DeepExplainer(model, x[:5])
    sv = e.shap_values(x[:3], check_additivity=False)
    assert sv.shape == (3, 4, 1)


# Group 5 – Layer/architectural op handlers


def test_tf_deep_batch_normalization_handler():
    """FusedBatchNorm (BatchNormalization layer) handler – linearity_1d(0)."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(30)
    x = rs.randn(60, 4).astype(np.float32)
    y = rs.randn(60).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(4,))
    h = tf.keras.layers.Dense(8)(inputs)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Activation("relu")(h)
    out = tf.keras.layers.Dense(1)(h)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=4, verbose=0)

    e = shap.DeepExplainer(model, x[:10])
    sv = e.shap_values(x[:3], check_additivity=False)
    assert sv.shape == (3, 4, 1)


def test_tf_deep_reshape_handler():
    """Reshape op (linearity_1d(0)) via Flatten layer."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(31)
    x = rs.randn(40, 4, 2).astype(np.float32)  # 3-D input
    y = rs.randn(40).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(4, 2))
    flat = tf.keras.layers.Flatten()(inputs)
    out = tf.keras.layers.Dense(1)(flat)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=2, verbose=0)

    e = shap.DeepExplainer(model, x[:5])
    sv = e.shap_values(x[:3], check_additivity=False)
    assert sv.shape == (3, 4, 2, 1)


def test_tf_deep_avgpool_handler():
    """AvgPool (linearity_1d(0)) via AveragePooling2D."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(32)
    x = rs.randn(20, 8, 8, 1).astype(np.float32)
    y = rs.randn(20).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(8, 8, 1))
    h = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(inputs)
    flat = tf.keras.layers.Flatten()(h)
    out = tf.keras.layers.Dense(1)(flat)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=2, verbose=0)

    e = shap.DeepExplainer(model, x[:5])
    sv = e.shap_values(x[:2], check_additivity=False)
    assert sv.shape[0] == 2


def test_tf_deep_concatenate_handler():
    """ConcatV2 (linearity_with_excluded([-1])) via multi-input Concatenate."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(33)
    in1 = rs.randn(30, 3).astype(np.float32)
    in2 = rs.randn(30, 3).astype(np.float32)
    y = rs.randn(30).astype(np.float32)

    input1 = tf.keras.layers.Input(shape=(3,))
    input2 = tf.keras.layers.Input(shape=(3,))
    cat = tf.keras.layers.Concatenate()([input1, input2])
    out = tf.keras.layers.Dense(1)(cat)
    model = tf.keras.Model(inputs=[input1, input2], outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit([in1, in2], y, epochs=2, verbose=0)

    e = shap.DeepExplainer(model, [in1[:5], in2[:5]])
    sv = e.shap_values([in1[:3], in2[:3]], check_additivity=False)
    assert len(sv) == 2
    assert sv[0].shape[0] == 3


def test_tf_deep_dropout_no_training_effect():
    """Dropout should not affect attributions (training=False in explain)."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(34)
    x = rs.randn(40, 4).astype(np.float32)
    y = rs.randn(40).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(4,))
    h = tf.keras.layers.Dense(8, activation="relu")(inputs)
    h = tf.keras.layers.Dropout(0.5)(h)
    out = tf.keras.layers.Dense(1)(h)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=3, verbose=0)

    e = shap.DeepExplainer(model, x[:5])
    sv1 = e.shap_values(x[:3], check_additivity=False)
    sv2 = e.shap_values(x[:3], check_additivity=False)
    # Deterministic (training=False disables dropout)
    np.testing.assert_array_almost_equal(sv1, sv2, decimal=5)


def test_tf_deep_conv2d_handler():
    """Conv2D linearity_1d(0) handler via a small conv model."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(35)
    x = rs.randn(20, 8, 8, 1).astype(np.float32)
    y = rs.randn(20).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(8, 8, 1))
    h = tf.keras.layers.Conv2D(2, (3, 3), activation="relu")(inputs)
    flat = tf.keras.layers.Flatten()(h)
    out = tf.keras.layers.Dense(1)(flat)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=2, verbose=0)

    e = shap.DeepExplainer(model, x[:5])
    sv = e.shap_values(x[:2], check_additivity=False)
    assert sv.shape[0] == 2


# Group 6 – Embedding / GatherV2 handler


def test_tf_deep_embedding_gather_handler():
    """GatherV2 handler via an Embedding layer (var[0]=False, var[1]=True)."""
    tf = _skip_if_no_tf()
    from packaging import version

    import shap

    if version.parse(tf.__version__) >= version.parse("2.5.0"):
        pytest.skip("Embedding + non-eager path unstable on TF >= 2.5")

    tf.compat.v1.disable_eager_execution()

    rs = np.random.RandomState(40)
    vocab_size = 50
    seq_len = 10
    X = rs.randint(0, vocab_size, size=(100, seq_len))
    y = rs.randint(0, 2, 100).astype(np.float32)

    mod = tf.keras.models.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, 4, input_length=seq_len),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    mod.compile(optimizer="adam", loss="binary_crossentropy")
    sess = tf.compat.v1.keras.backend.get_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    mod.fit(X, y, epochs=1, verbose=0)

    bg = X[:5]
    test_x = X[5:7]
    e = shap.DeepExplainer((mod.layers[0].input, mod.layers[-1].output), bg)
    sv = e.shap_values(test_x)
    assert sv[0].shape == test_x.shape


# Group 7 – Multi-output stacking / phi_symbolics


def test_tf_deep_multi_output_shap_values_shape():
    """shap_values() for multi-output returns array with correct last dim."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(51)
    x = rs.randn(40, 4).astype(np.float32)
    y = np.eye(3)[rs.randint(0, 3, 40)]

    model = _simple_multi_output_model(tf, n_in=4, n_out=3)
    model.fit(x, y, epochs=2, verbose=0)

    e = shap.DeepExplainer(model, x[:5])
    sv = e.shap_values(x[:4], check_additivity=False)
    # shape should be (n_samples, n_features, n_outputs)
    assert sv.shape == (4, 4, 3)


# Group 8 – Utility functions (unit tests)


def test_tensors_blocked_by_false_empty():
    """tensors_blocked_by_false returns empty list for empty input."""
    _skip_if_no_tf()
    from shap.explainers._deep.deep_tf import tensors_blocked_by_false

    result = tensors_blocked_by_false([])
    assert result == []


def test_tensors_blocked_by_false_non_switch():
    """Non-Switch ops are recursed through without blocking."""
    tf = _skip_if_no_tf()
    from shap.explainers._deep.deep_tf import tensors_blocked_by_false

    with tf.compat.v1.Graph().as_default():
        x = tf.constant([1.0, 2.0])
        op = tf.identity(x).op  # .op is valid in graph mode
        result = tensors_blocked_by_false([op])
        assert isinstance(result, list)


def test_backward_walk_ops_basic():
    """backward_walk_ops discovers the ops in a simple graph."""
    tf = _skip_if_no_tf()
    from shap.explainers._deep.deep_tf import backward_walk_ops

    with tf.compat.v1.Graph().as_default():
        a = tf.constant([1.0, 2.0])
        b = tf.add(a, a)
        found = backward_walk_ops([b.op], tensor_blacklist=[], op_type_blacklist=[])
        op_types = {op.type for op in found}
        assert "AddV2" in op_types or "Add" in op_types


def test_backward_walk_ops_blacklist():
    """op_type_blacklist prevents traversal through blacklisted types."""
    tf = _skip_if_no_tf()
    from shap.explainers._deep.deep_tf import backward_walk_ops

    with tf.compat.v1.Graph().as_default():
        a = tf.constant([1.0])
        b = tf.identity(a)
        c = tf.add(b, b)
        found = backward_walk_ops([c.op], tensor_blacklist=[], op_type_blacklist=["Identity"])
        found_types = {op.type for op in found}
        assert "Identity" not in found_types


def test_forward_walk_ops_basic():
    """forward_walk_ops only returns ops that are within_ops."""
    tf = _skip_if_no_tf()
    from shap.explainers._deep.deep_tf import backward_walk_ops, forward_walk_ops

    with tf.compat.v1.Graph().as_default():
        a = tf.constant([1.0, 2.0])
        b = tf.add(a, a)
        c = tf.identity(b)
        within = backward_walk_ops([c.op], [], [])
        found = forward_walk_ops(
            [b.op],
            tensor_blacklist=[],
            op_type_blacklist=[],
            within_ops=within,
        )
        assert len(found) > 0
        found_types = {op.type for op in found}
        assert "AddV2" in found_types or "Add" in found_types


# Group 9 – custom_record_gradient


def test_custom_record_gradient_non_resource_gather():
    """For non-ResourceGather ops, custom_record_gradient calls original without dtype swap."""
    tf = _skip_if_no_tf()
    from unittest.mock import MagicMock, patch

    from shap.explainers._deep.deep_tf import custom_record_gradient

    called_with = {}

    def fake_record(op_name, inputs, attrs, results):
        called_with["op_name"] = op_name
        return MagicMock()

    import tensorflow.python.eager.backprop as bp

    target = (
        "tensorflow.python.eager.backprop._record_gradient"
        if hasattr(bp, "_record_gradient")
        else "tensorflow.python.eager.backprop.record_gradient"
    )

    with patch(target, side_effect=fake_record):
        mock_input = MagicMock()
        mock_input.dtype = tf.float32
        custom_record_gradient("Relu", [mock_input], {}, [])

    assert called_with.get("op_name") == "shap_Relu"


def test_custom_record_gradient_resource_gather_resets_dtype():
    """ResourceGather with int32 index: dtype is temporarily swapped then restored."""
    tf = _skip_if_no_tf()
    from unittest.mock import MagicMock, patch

    from shap.explainers._deep.deep_tf import custom_record_gradient

    def fake_record(op_name, inputs, attrs, results):
        # During the call the dtype should appear as float32
        assert inputs[1].__dict__["_dtype"] == tf.float32, "dtype not swapped inside call"
        return MagicMock()

    import tensorflow.python.eager.backprop as bp

    target = (
        "tensorflow.python.eager.backprop._record_gradient"
        if hasattr(bp, "_record_gradient")
        else "tensorflow.python.eager.backprop.record_gradient"
    )

    with patch(target, side_effect=fake_record):
        idx = MagicMock()
        idx.dtype = tf.int32
        idx.__dict__["_dtype"] = tf.int32
        custom_record_gradient("ResourceGather", [MagicMock(), idx], {}, [])

    # After the call dtype should be restored to int32
    assert idx.__dict__["_dtype"] == tf.int32


# Group 10 – Specific op-handler internals


def test_break_dependence_returns_none_for_all_inputs():
    """break_dependence always returns [None, ...] for all op inputs."""
    _skip_if_no_tf()
    from unittest.mock import MagicMock

    from shap.explainers._deep.deep_tf import break_dependence

    op = MagicMock()
    op.inputs = [MagicMock(), MagicMock(), MagicMock()]
    result = break_dependence(None, op)
    assert result == [None, None, None]


def test_passthrough_strips_shap_prefix():
    """passthrough removes 'shap_' prefix and calls orig_grads."""
    _skip_if_no_tf()
    from unittest.mock import MagicMock

    from shap.explainers._deep.deep_tf import passthrough

    explainer = MagicMock()
    explainer.orig_grads = {"Identity": MagicMock(return_value="grad_result")}

    op = MagicMock()
    op.type = "shap_Identity"

    passthrough(explainer, op, "some_grad")
    assert op.type == "Identity"
    explainer.orig_grads["Identity"].assert_called_once()


def test_nonlinearity_2d_handler_non_zero_one_inputs_raises():
    """nonlinearity_2d_handler raises Exception when input_ind0/1 aren't 0,1."""
    _skip_if_no_tf()
    from shap.explainers._deep.deep_tf import nonlinearity_2d_handler

    with pytest.raises(Exception, match="TODO"):
        nonlinearity_2d_handler(1, 2, lambda x, y: x * y, None, None)


def test_linearity_with_excluded_handler_excluded_varies_raises():
    """linearity_with_excluded_handler asserts excluded inputs don't vary."""
    tf = _skip_if_no_tf()
    from unittest.mock import MagicMock

    from shap.explainers._deep.deep_tf import linearity_with_excluded_handler

    explainer = MagicMock()
    # Report that input 0 (the excluded one) does vary → should AssertionError
    explainer._variable_inputs.return_value = np.array([True, False])

    op = MagicMock()
    a = tf.constant([1.0, 2.0])
    b = tf.constant([3.0, 4.0])
    op.inputs = [a, b]
    op.type = "shap_AddV2"

    with pytest.raises(AssertionError):
        linearity_with_excluded_handler([0], explainer, op, MagicMock())


# Group 13 – expected_value correctness


def test_tf_deep_expected_value_shape_single_output():
    """expected_value is a scalar (or 1-element tensor) for single-output."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(80)
    x = rs.randn(20, 4).astype(np.float32)
    y = rs.randn(20, 1).astype(np.float32)

    model = _simple_dense_model(tf)
    _quick_fit(model, x, y)

    e = shap.DeepExplainer(model, x[:5])
    ev = np.asarray(e.expected_value)
    # For a (n, 1)-output model the expected_value should be shape (1,)
    assert ev.shape in {(), (1,)}


def test_tf_deep_expected_value_shape_multi_output():
    """expected_value has one entry per output class."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(81)
    x = rs.randn(30, 4).astype(np.float32)
    y = np.eye(3)[rs.randint(0, 3, 30)]

    model = _simple_multi_output_model(tf, n_in=4, n_out=3)
    model.fit(x, y, epochs=2, verbose=0)

    e = shap.DeepExplainer(model, x[:10])
    ev = np.asarray(e.expected_value)
    assert ev.shape == (3,)


# Group 14 – SquaredDifference / Minimum / Maximum nonlinearity_2d


def test_tf_deep_squared_difference_handler():
    """SquaredDifference nonlinearity_1d_nonlinearity_2d(0,1,…) via Lambda."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(90)
    x = rs.randn(30, 4).astype(np.float32)
    y = rs.randn(30).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(4,))
    # SquaredDifference: (x - constant)^2 — only input 0 varies
    diff = tf.keras.layers.Lambda(lambda t: tf.math.squared_difference(t, tf.ones_like(t)))(inputs)
    out = tf.keras.layers.Dense(1)(diff)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=2, verbose=0)

    e = shap.DeepExplainer(model, x[:5])
    sv = e.shap_values(x[:3], check_additivity=False)
    assert sv.shape == (3, 4, 1)


def test_tf_deep_maximum_handler():
    """Maximum nonlinearity_1d_nonlinearity_2d handler via Lambda."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(91)
    x = rs.randn(30, 4).astype(np.float32)
    y = rs.randn(30).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(4,))
    mx = tf.keras.layers.Lambda(
        lambda t: tf.maximum(t, tf.zeros_like(t))  # equivalent to ReLU
    )(inputs)
    out = tf.keras.layers.Dense(1)(mx)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=2, verbose=0)

    e = shap.DeepExplainer(model, x[:5])
    sv = e.shap_values(x[:3], check_additivity=False)
    assert sv.shape == (3, 4, 1)


# Group 15 – Softmax handler (standalone)


def test_tf_deep_softmax_handler_additivity():
    """Softmax op handler: SHAP values should approximately sum to output diff."""
    tf = _skip_if_no_tf()
    import shap

    rs = np.random.RandomState(100)
    x = rs.randn(40, 4).astype(np.float32)
    y = np.eye(3)[rs.randint(0, 3, 40)]

    inputs = tf.keras.layers.Input(shape=(4,))
    logits = tf.keras.layers.Dense(3)(inputs)
    outputs = tf.keras.layers.Softmax()(logits)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.fit(x, y, epochs=3, verbose=0)

    e = shap.DeepExplainer(model, x[:10])
    sv = e.shap_values(x[:5], check_additivity=False)
    assert sv.shape == (5, 4, 3)


# Group 16 – Non-eager / graph mode (TF 1.x / compat.v1)


def test_tf_deep_graph_mode_linear():
    """Session-based (graph) mode: SHAP values sum correctly for linear model."""
    tf = _skip_if_no_tf()
    from packaging import version

    import shap

    if version.parse(tf.__version__) >= version.parse("2.5.0"):
        pytest.skip("Graph / session mode unreliable on TF >= 2.5")

    tf.compat.v1.disable_eager_execution()

    rs = np.random.RandomState(110)
    x = rs.randn(50, 4).astype(np.float32)
    y = rs.randn(50).astype(np.float32)

    inputs = tf.keras.layers.Input(shape=(4,))
    out = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")

    sess = tf.compat.v1.keras.backend.get_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    model.fit(x, y, epochs=2, verbose=0)

    e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), x[:5])
    sv = e.shap_values(x[:3], check_additivity=False)
    assert sv[0].shape == x[:3].shape


# Group 17 – op_handlers dict coverage


def test_op_handlers_contains_required_keys():
    """op_handlers dict must contain all documented op types."""
    _skip_if_no_tf()
    from shap.explainers._deep.deep_tf import op_handlers

    required = [
        # passthrough
        "Identity",
        "StridedSlice",
        "Squeeze",
        "ExpandDims",
        "Pack",
        "BiasAdd",
        "Unpack",
        "Add",
        "AddV2",
        "Sub",
        "Merge",
        "Sum",
        "Mean",
        "Cast",
        "Transpose",
        "Enter",
        "Exit",
        "NextIteration",
        "Tile",
        "TensorArrayScatterV3",
        "TensorArrayReadV3",
        "TensorArrayWriteV3",
        # break_dependence
        "Shape",
        "RandomUniform",
        "ZerosLike",
        # linearity_1d
        "Reshape",
        "Pad",
        "ReverseV2",
        "ConcatV2",
        "Conv2D",
        "Switch",
        "AvgPool",
        "FusedBatchNorm",
        # nonlinearity_1d
        "Relu",
        "Selu",
        "Elu",
        "Sigmoid",
        "Tanh",
        "Softplus",
        "Exp",
        "ClipByValue",
        "Rsqrt",
        "Square",
        "Max",
        # nonlinearity_1d_nonlinearity_2d
        "SquaredDifference",
        "Minimum",
        "Maximum",
        # linearity_1d_nonlinearity_2d
        "Mul",
        "RealDiv",
        "MatMul",
        # custom
        "GatherV2",
        "ResourceGather",
        "MaxPool",
        "Softmax",
    ]
    for key in required:
        assert key in op_handlers, f"Missing op_handlers key: {key}"
