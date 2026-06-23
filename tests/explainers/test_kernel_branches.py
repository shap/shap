import sys
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import shap.explainers._kernel as kernel


class _IdentityLink:
    @staticmethod
    def f(x):
        return x


def _build_solver_explainer(l1_reg):
    explainer = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer.M = 2
    explainer.maskMatrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    explainer.kernelWeights = np.array([1.0, 1.0, 1.0])
    explainer.ey = np.array([[1.0], [2.0], [3.0]])
    explainer.linkfv = np.vectorize(_IdentityLink.f)
    explainer.link = _IdentityLink()
    explainer.fnull = np.array([0.0])
    explainer.fx = np.array([3.0])
    explainer.l1_reg = l1_reg
    return explainer


def test_kernel_init_uses_feature_names_and_dataframe_model_output():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    explainer = kernel.KernelExplainer(
        lambda x: pd.DataFrame(np.sum(x, axis=1), columns=["out"]),
        data,
        feature_names=["f0", "f1"],
    )

    assert explainer.data_feature_names == ["f0", "f1"]


def test_kernel_init_rejects_non_dense_sparse_data(monkeypatch):
    class _BadData:
        transposed = False
        weights = np.array([1.0])
        data = np.array([[0.0]])

    monkeypatch.setattr(kernel, "convert_to_link", lambda _link: _IdentityLink())
    monkeypatch.setattr(kernel, "convert_to_model", lambda model, keep_index=False: types.SimpleNamespace(f=model))
    monkeypatch.setattr(kernel, "convert_to_data", lambda data, keep_index=False: _BadData())
    monkeypatch.setattr(kernel, "match_model_to_data", lambda _model, _data: np.array([0.0]))

    with pytest.raises(TypeError, match="DenseData and SparseData"):
        kernel.KernelExplainer(lambda x: x, np.array([[0.0]]))


def test_kernel_init_rejects_transposed_data(monkeypatch):
    transposed = kernel.DenseData(np.array([[1.0], [2.0]]), ["f0", "f1"])
    assert transposed.transposed

    monkeypatch.setattr(kernel, "convert_to_link", lambda _link: _IdentityLink())
    monkeypatch.setattr(kernel, "convert_to_model", lambda model, keep_index=False: types.SimpleNamespace(f=model))
    monkeypatch.setattr(kernel, "convert_to_data", lambda data, keep_index=False: transposed)
    monkeypatch.setattr(kernel, "match_model_to_data", lambda _model, _data: np.array([0.0]))

    with pytest.raises(kernel.DimensionError, match="transposed"):
        kernel.KernelExplainer(lambda x: x, np.array([[0.0]]))


def test_kernel_init_handles_eager_tensor_model_null(monkeypatch):
    class _FakeEager:
        def __init__(self, values):
            self._values = values
            self.shape = values.shape

        def numpy(self):
            return self._values

    monkeypatch.setattr(
        kernel,
        "safe_isinstance",
        lambda obj, class_str: isinstance(obj, _FakeEager) and "EagerTensor" in class_str,
    )

    explainer = kernel.KernelExplainer(lambda x: _FakeEager(np.array([1.0, 3.0])), np.array([[0.0], [1.0]]))

    assert np.isclose(explainer.fnull[0], 2.0)


def test_kernel_init_handles_symbolic_tensor_model_null(monkeypatch):
    class _FakeSymbolic:
        shape = (2,)

    monkeypatch.setattr(
        kernel,
        "safe_isinstance",
        lambda obj, class_str: isinstance(obj, _FakeSymbolic) and "SymbolicTensor" in class_str,
    )
    monkeypatch.setattr(
        kernel.KernelExplainer,
        "_convert_symbolic_tensor",
        staticmethod(lambda _symbolic: np.array([2.0, 4.0])),
    )

    explainer = kernel.KernelExplainer(lambda x: _FakeSymbolic(), np.array([[0.0], [1.0]]))

    assert np.isclose(explainer.fnull[0], 3.0)


def test_convert_symbolic_tensor_tf2_branch(monkeypatch):
    calls = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, value):
            calls.append(value)
            if value == "initializer":
                return None
            return np.array([1.0, 2.0])

    fake_tf = types.ModuleType("tensorflow")
    fake_tf.__version__ = "2.15.0"
    fake_tf.compat = types.SimpleNamespace(
        v1=types.SimpleNamespace(Session=_Session, global_variables_initializer=lambda: "initializer")
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    result = kernel.KernelExplainer._convert_symbolic_tensor("symbolic")

    np.testing.assert_allclose(result, np.array([1.0, 2.0]))
    assert calls == ["initializer", "symbolic"]


def test_convert_symbolic_tensor_tf1_branch(monkeypatch):
    calls = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, value):
            calls.append(value)
            if value == "initializer":
                return None
            return np.array([3.0, 4.0])

    fake_tf = types.ModuleType("tensorflow")
    fake_tf.__version__ = "1.15.0"
    fake_tf.Session = _Session
    fake_tf.global_variables_initializer = lambda: "initializer"
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    result = kernel.KernelExplainer._convert_symbolic_tensor("symbolic")

    np.testing.assert_allclose(result, np.array([3.0, 4.0]))
    assert calls == ["initializer", "symbolic"]


def test_kernel_call_stacks_list_outputs():
    explainer = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer.expected_value = np.array([0.2, 0.8])
    explainer.shap_values = lambda X, **kwargs: [
        np.ones((X.shape[0], X.shape[1])),
        np.zeros((X.shape[0], X.shape[1])),
    ]

    result = kernel.KernelExplainer.__call__(explainer, np.array([[1.0, 2.0], [3.0, 4.0]]), silent=True)

    assert result.values.shape == (2, 2, 2)


def test_shap_values_single_instance_keep_index_branch(monkeypatch):
    class _OneDimValuesDataFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _OneDimValuesDataFrame

        @property
        def values(self):
            return np.array([1.0, 2.0])

    explainer = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer.keep_index = True
    explainer.explain = lambda data, **kwargs: np.array([5.0, 6.0])

    captured = {}

    def fake_convert(data, *args):
        captured["args"] = args
        return data

    monkeypatch.setattr(kernel, "convert_to_instance_with_index", fake_convert)

    df = _OneDimValuesDataFrame([[1.0, 2.0]], columns=["a", "b"], index=pd.Index([10], name="idx"))
    out = kernel.KernelExplainer.shap_values(explainer, df)

    np.testing.assert_allclose(out, np.array([5.0, 6.0]))
    assert len(captured["args"]) == 3


def test_shap_values_gc_collect_and_invalid_rank(monkeypatch):
    explainer = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer.keep_index = False
    explainer.explain = lambda data, **kwargs: np.array([1.0, 2.0])

    calls = []
    monkeypatch.setattr(kernel.gc, "collect", lambda: calls.append(1))

    out = kernel.KernelExplainer.shap_values(
        explainer,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        gc_collect=True,
        silent=True,
    )
    assert out.shape == (2, 2)
    assert len(calls) == 2

    with pytest.raises(kernel.DimensionError, match="1 or 2 dimensions"):
        kernel.KernelExplainer.shap_values(explainer, np.zeros((1, 2, 3)))


def test_explain_model_out_dataframe_and_single_varying_feature():
    explainer = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer.data = kernel.DenseData(np.array([[0.0, 0.0], [1.0, 1.0]]), ["a", "b"])
    explainer.keep_index = False
    explainer.model = types.SimpleNamespace(f=lambda x: pd.DataFrame([[5.0]], columns=["out"]))
    explainer.vector_out = True
    explainer.D = 1
    explainer.link = _IdentityLink()
    explainer.fnull = np.array([1.0])
    explainer.varying_groups = lambda x: np.array([1], dtype=np.int64)

    phi = kernel.KernelExplainer.explain(explainer, np.array([[2.0, 3.0]]))

    assert phi.shape == (2, 1)
    assert phi[0, 0] == 0
    assert phi[1, 0] == 4


def test_explain_model_out_symbolic_tensor(monkeypatch):
    class _FakeSymbolic:
        pass

    symbolic = _FakeSymbolic()
    explainer = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer.data = kernel.DenseData(np.array([[0.0, 0.0], [1.0, 1.0]]), ["a", "b"])
    explainer.keep_index = False
    explainer.model = types.SimpleNamespace(f=lambda x: symbolic)
    explainer.vector_out = True
    explainer.D = 1
    explainer.link = _IdentityLink()
    explainer.fnull = np.array([1.0])
    explainer.varying_groups = lambda x: np.array([], dtype=np.int64)
    explainer._convert_symbolic_tensor = lambda x: np.array([[7.0]])

    monkeypatch.setattr(
        kernel,
        "safe_isinstance",
        lambda obj, class_str: isinstance(obj, _FakeSymbolic) and "SymbolicTensor" in class_str,
    )

    phi = kernel.KernelExplainer.explain(explainer, np.array([[2.0, 3.0]]))

    assert phi.shape == (2, 1)
    assert np.all(phi == 0)


def test_varying_groups_dense_branch_with_sparse_group_values():
    class _DenseLike:
        def __getitem__(self, key):
            _, inds = key
            idx = int(np.array(inds).ravel()[0])
            if idx == 0:
                return scipy.sparse.csr_matrix([[0.0]])
            return scipy.sparse.csr_matrix([[5.0]])

        def nonzero(self):
            return (np.array([0]), np.array([1]))

    explainer = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer.data = types.SimpleNamespace(
        groups_size=2,
        groups=[np.array([0]), np.array([1])],
        data=np.array([[0.0, 5.0], [0.0, 5.0]]),
    )

    varying = kernel.KernelExplainer.varying_groups(explainer, _DenseLike())

    assert varying.size == 0


def test_addsample_list_and_matrix_group_branches():
    x = np.array([[10.0, 20.0, 30.0, 40.0]])

    explainer_list = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer_list.nsamplesAdded = 0
    explainer_list.N = 2
    explainer_list.M = 2
    explainer_list.varyingFeatureGroups = [np.array([0]), np.array([2])]
    explainer_list.synth_data = np.zeros((2, 4))
    explainer_list.maskMatrix = np.zeros((1, 2))
    explainer_list.kernelWeights = np.zeros(1)

    kernel.KernelExplainer.addsample(explainer_list, x, np.array([1.0, 0.0]), 0.5)

    np.testing.assert_allclose(explainer_list.synth_data[:, 0], np.array([10.0, 10.0]))

    explainer_matrix = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer_matrix.nsamplesAdded = 0
    explainer_matrix.N = 2
    explainer_matrix.M = 2
    explainer_matrix.varyingFeatureGroups = np.array([[0, 1], [2, 3]])
    explainer_matrix.synth_data = np.zeros((2, 4))
    explainer_matrix.maskMatrix = np.zeros((1, 2))
    explainer_matrix.kernelWeights = np.zeros(1)

    kernel.KernelExplainer.addsample(explainer_matrix, x, np.array([1.0, 1.0]), 0.5)

    np.testing.assert_allclose(explainer_matrix.synth_data, np.tile(x, (2, 1)))


def test_run_keep_index_ordered_and_dataframe_model_output(monkeypatch):
    explainer = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer.nsamplesAdded = 1
    explainer.nsamplesRun = 0
    explainer.N = 2
    explainer.D = 1
    explainer.keep_index = True
    explainer.keep_index_ordered = True
    explainer.synth_data = np.array([[10.0, 1.0], [20.0, 2.0]])
    explainer.synth_data_index = np.array([2, 1])
    explainer.data = types.SimpleNamespace(
        index_name="idx",
        group_names=["a", "b"],
        weights=np.array([0.5, 0.5]),
    )
    explainer.y = np.zeros((2, 1))
    explainer.ey = np.zeros((1, 1))

    def fake_model(df):
        assert list(df.index) == [1, 2]
        return pd.DataFrame({"out": [1.0, 3.0]}, index=df.index)

    explainer.model = types.SimpleNamespace(f=fake_model)

    monkeypatch.setattr(kernel, "_exp_val", lambda ns_run, ns_add, D, N, w, y, ey: (np.array([[2.0]]), ns_add))

    kernel.KernelExplainer.run(explainer)

    assert explainer.nsamplesRun == 1
    np.testing.assert_allclose(explainer.y[:, 0], np.array([1.0, 3.0]))


def test_run_symbolic_model_output_branch(monkeypatch):
    class _FakeSymbolic:
        pass

    symbolic = _FakeSymbolic()

    explainer = kernel.KernelExplainer.__new__(kernel.KernelExplainer)
    explainer.nsamplesAdded = 1
    explainer.nsamplesRun = 0
    explainer.N = 2
    explainer.D = 1
    explainer.keep_index = False
    explainer.synth_data = np.array([[10.0], [20.0]])
    explainer.model = types.SimpleNamespace(f=lambda data: symbolic)
    explainer.data = types.SimpleNamespace(weights=np.array([0.5, 0.5]))
    explainer.y = np.zeros((2, 1))
    explainer.ey = np.zeros((1, 1))
    explainer._convert_symbolic_tensor = lambda x: np.array([[4.0], [6.0]])

    monkeypatch.setattr(
        kernel,
        "safe_isinstance",
        lambda obj, class_str: isinstance(obj, _FakeSymbolic) and "SymbolicTensor" in class_str,
    )
    monkeypatch.setattr(kernel, "_exp_val", lambda ns_run, ns_add, D, N, w, y, ey: (np.array([[5.0]]), ns_add))

    kernel.KernelExplainer.run(explainer)

    np.testing.assert_allclose(explainer.y[:, 0], np.array([4.0, 6.0]))


def test_solve_warns_for_deprecated_auto_l1_reg():
    explainer = _build_solver_explainer("auto")

    with pytest.warns(DeprecationWarning, match="deprecated"):
        phi, phi_var = kernel.KernelExplainer.solve(explainer, 0.5, 0)

    assert phi.shape == (2,)
    assert phi_var.shape == (2,)


@pytest.mark.parametrize("use_lt_version", [True, False])
def test_solve_adaptive_l1_reg_branch(monkeypatch, use_lt_version):
    class _Pipeline:
        def fit(self, X, y):
            return [None, types.SimpleNamespace(coef_=np.array([1.0, 1.0]))]

    explainer = _build_solver_explainer("bic")

    monkeypatch.setattr(kernel, "make_pipeline", lambda *args, **kwargs: _Pipeline())
    monkeypatch.setattr(kernel, "LassoLarsIC", lambda criterion, **kwargs: (criterion, kwargs))

    if use_lt_version:
        monkeypatch.setattr(kernel.version, "parse", lambda v: (1, 1, 0) if v != "1.2.0" else (1, 2, 0))
    else:
        monkeypatch.setattr(kernel.version, "parse", lambda v: (1, 3, 0) if v != "1.2.0" else (1, 2, 0))

    phi, phi_var = kernel.KernelExplainer.solve(explainer, 0.1, 0)

    assert phi.shape == (2,)
    assert phi_var.shape == (2,)


def test_solve_fixed_l1_reg_returns_zero_when_no_features_selected(monkeypatch):
    class _Lasso:
        def __init__(self, alpha):
            self.alpha = alpha

        def fit(self, X, y):
            self.coef_ = np.array([0.0, 0.0])
            return self

    explainer = _build_solver_explainer(0.5)
    monkeypatch.setattr(kernel, "Lasso", _Lasso)

    phi, phi_var = kernel.KernelExplainer.solve(explainer, 1.0, 0)

    np.testing.assert_allclose(phi, np.zeros(2))
    np.testing.assert_allclose(phi_var, np.ones(2))


def test_solve_singular_matrix_falls_back_to_lstsq(monkeypatch):
    explainer = _build_solver_explainer(False)

    def raise_singular(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(kernel.np.linalg, "solve", raise_singular)

    with pytest.warns(UserWarning, match="singular"):
        phi, phi_var = kernel.KernelExplainer.solve(explainer, 1.0, 0)

    assert phi.shape == (2,)
    assert phi_var.shape == (2,)
