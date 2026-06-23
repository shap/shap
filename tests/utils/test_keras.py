import sys
import types

import pytest

import shap.utils._keras as keras_utils


class _FakeTensor:
    def __init__(self, name):
        self.name = name


class _FakeLayer:
    def __init__(self, name, input_tensor, output_name=None, input_shape=(None, 4)):
        self.name = name
        self.input = input_tensor
        self.output = _FakeTensor(output_name or f"{name}_out")
        self._input_shape = input_shape
        self.calls = []

    def get_input_shape_at(self, index):
        assert index == 0
        return self._input_shape

    def get_input_at(self, index):
        if isinstance(self.input, list):
            return self.input[index]
        return self.input

    def __call__(self, layer_inputs):
        self.calls.append(layer_inputs)
        return _FakeTensor(self.output.name)


class _FakeModel:
    def __init__(self, layers):
        self.layers = layers

    def get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(name)


def _install_fake_tensorflow(monkeypatch):
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.keras = types.SimpleNamespace(
        Input=lambda shape: _FakeTensor(f"generated_input_{shape}"),
        Model=lambda layer_input, layer_output: {"input": layer_input, "output": layer_output},
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)


def test_clone_keras_layers_with_indices_and_multi_input(monkeypatch):
    _install_fake_tensorflow(monkeypatch)

    start = _FakeLayer("start", _FakeTensor("input_1"), "start_out")
    mid = _FakeLayer("mid", _FakeTensor("start_out"), "mid_out")
    merge = _FakeLayer("merge", [_FakeTensor("start_out"), _FakeTensor("mid_out")], "merge_out")
    orphan = _FakeLayer("orphan", _FakeTensor("never_ready"), "orphan_out")

    model = _FakeModel([orphan, merge, start, mid])

    cloned = keras_utils.clone_keras_layers(model, 2, 1)

    assert cloned["output"].name == "merge_out"
    assert len(start.calls) == 1
    assert len(mid.calls) == 1
    assert len(merge.calls) == 1
    assert len(orphan.calls) == 0


def test_clone_keras_layers_raises_for_incomplete_graph(monkeypatch):
    _install_fake_tensorflow(monkeypatch)

    start = _FakeLayer("start", _FakeTensor("start_input"), "start_out")
    stuck = _FakeLayer("stuck", _FakeTensor("never_ready"), "stuck_out")
    model = _FakeModel([stuck])

    with pytest.raises(Exception, match="complete graph"):
        keras_utils.clone_keras_layers(model, start, stuck)


@pytest.mark.parametrize("split_layer", ["target", 2])
def test_split_keras_model_calls_clone_for_both_halves(monkeypatch, split_layer):
    input_layer = _FakeLayer("input", _FakeTensor("seed/out:0"), "input/out:0")
    prev_layer = _FakeLayer("prev", _FakeTensor("input/out:0"), "prev/out:0")
    target_layer = _FakeLayer("target", _FakeTensor("prev/out:0"), "target/out:0")
    end_layer = _FakeLayer("end", _FakeTensor("target/out:0"), "end/out:0")
    model = _FakeModel([input_layer, prev_layer, target_layer, end_layer])

    calls = []

    def fake_clone(model_arg, start_arg, stop_arg):
        calls.append((model_arg, start_arg.name, stop_arg.name))
        return f"{start_arg.name}->{stop_arg.name}"

    monkeypatch.setattr(keras_utils, "clone_keras_layers", fake_clone)

    model1, model2 = keras_utils.split_keras_model(model, split_layer)

    assert model1 == "prev->prev"
    assert model2 == "target->end"
    assert calls == [(model, "prev", "prev"), (model, "target", "end")]
