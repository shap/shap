"""Basic tests for slicer.
An unholy balance of use cases and test coverage.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, lil_matrix

from shap.utils._slicer import Alias as A
from shap.utils._slicer import AtomicSlicer, _handle_newaxis_ellipses
from shap.utils._slicer import Obj as O
from shap.utils._slicer import Slicer as S


def coerced(o: Any):
    if isinstance(o, (csc_matrix, csr_matrix, dok_matrix, lil_matrix)):
        o = o.toarray()

    to_list_collections = tuple([np.ndarray, torch.Tensor, pd.core.series.Series])
    if isinstance(o, (list, tuple)):
        return o
    elif isinstance(o, to_list_collections):
        return o.tolist()
    elif isinstance(o, pd.core.frame.DataFrame):
        return o.values.tolist()
    elif isinstance(o, dict):
        li = [np.nan] * len(o)
        for k, v in o.items():
            li[k] = v
        return li
    else:
        raise ValueError(f"Object {o} of {type(o)} is not a list, tuple nor array.")


def is_close(a: int | float, b: int | float, rel_tol: float = 1e-09, abs_tol: float = 0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def ctr_eq(c1: Any, c2: Any):
    if isinstance(c1, torch.Tensor) and c1.shape == torch.Size([]):
        c1 = c1.item()
    if isinstance(c2, torch.Tensor) and c2.shape == torch.Size([]):
        c2 = c2.item()

    if isinstance(c1, (int, float, np.number)) and isinstance(c2, (int, float, np.number)):
        return is_close(c1, c2)  # type: ignore[arg-type]

    c1 = coerced(c1)
    c2 = coerced(c2)

    return all([ctr_eq(c1[i], c2[i]) for i in range(max(len(c1), len(c2)))])


def test_slicer_ragged_numpy():
    values = np.array([np.array([0, 1]), np.array([2, 3, 4])], dtype=object)
    data = np.array(
        [
            np.array([5, 6, 7]),
        ]
    )

    slicer = S(values=values, data=data)
    sliced = slicer[0, 1]

    assert ctr_eq(sliced.data, data[0][1])
    assert ctr_eq(sliced.values, values[0][1])


def test_slicer_basic():
    data = [[1, 2], [3, 4]]
    values = [[5, 6], [7, 8]]
    identifiers = ["id1", "id1"]
    instance_names = ["r1", "r2"]
    feature_names = ["f1", "f2"]
    full_name = "A"

    slicer = S(
        data=data,
        values=values,
        identifiers=A(identifiers, 0),
        instance_names=A(instance_names, 0),
        feature_names=A(feature_names, 1),
        full_name=full_name,
    )

    colon_actual = slicer[:, 1]
    assert colon_actual.data == [2, 4]
    assert colon_actual.instance_names == ["r1", "r2"]
    assert colon_actual.feature_names == "f2"
    assert colon_actual.values == [6, 8]

    ellipses_actual = slicer[..., 1]
    assert ellipses_actual.data == [2, 4]
    assert ellipses_actual.instance_names == ["r1", "r2"]
    assert ellipses_actual.feature_names == "f2"
    assert ellipses_actual.values == [6, 8]

    array_index_actual = slicer[[0, 1], 1]
    assert array_index_actual.data == [2, 4]
    assert array_index_actual.feature_names == "f2"
    assert array_index_actual.instance_names == ["r1", "r2"]
    assert array_index_actual.values == [6, 8]

    alias_actual = slicer["r1", "f2"]
    assert alias_actual.data == 2
    assert alias_actual.feature_names == "f2"
    assert alias_actual.instance_names == "r1"
    assert alias_actual.values == 6

    alias_actual = slicer["id1", "f2"]
    assert alias_actual.data == [2, 4]
    assert alias_actual.feature_names == "f2"
    assert alias_actual.instance_names == ["r1", "r2"]
    assert alias_actual.values == [6, 8]

    chained_actual = slicer[:][:, 1]
    assert chained_actual.data == [2, 4]
    assert chained_actual.feature_names == "f2"
    assert chained_actual.instance_names == ["r1", "r2"]
    assert chained_actual.values == [6, 8]

    alias_actual = slicer["id1"][:, "f2"]
    assert alias_actual.data == [2, 4]
    assert alias_actual.feature_names == "f2"
    assert alias_actual.instance_names == ["r1", "r2"]
    assert alias_actual.values == [6, 8]

    alias_actual = slicer["r1"]
    alias_actual = alias_actual["f2"]
    assert alias_actual.data == 2
    assert alias_actual.feature_names == "f2"
    assert alias_actual.instance_names == "r1"
    assert alias_actual.values == 6


def test_slicer_unnamed():
    a = [1, 2, 3]
    b = [4, 5, 6]

    slicer = S(a, b)
    actual_a, actual_b = slicer[1].o
    assert actual_a == 2
    assert actual_b == 5

    df1 = pd.DataFrame([[1, 2], [3, 4]])
    df2 = pd.DataFrame([[5, 6], [7, 8]])
    slicer = S(df1, df2)
    actual_1, actual_2 = slicer[:, 0].o

    assert ctr_eq(actual_1.values, [1, 3])
    assert ctr_eq(actual_2.values, [5, 7])


def test_slicer_crud():
    data = [[1, 2], [3, 4]]
    values = [[5, 6], [7, 8]]
    extra = [[9, 10], [11, 12]]
    overridden = [[13, 14], [15, 16]]

    slicer = S(data=data, values=values)
    slicer.extra = extra  # Create
    slicer.data = overridden  # Update
    del slicer.values  # Delete

    sliced = slicer[0, 1]  # Read
    assert sliced.data == 14
    with pytest.raises(Exception):
        _ = sliced.values

    assert sliced.extra == 10

    del slicer.o
    assert slicer.o == []


def test_slicer_default_alias():

    df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
    slicer = S(df)
    assert getattr(slicer, "index", None)
    assert getattr(slicer, "columns", None)
    actual = slicer[:, "A"].o
    assert ctr_eq(actual, [1, 3])


def test_slicer_anon_dict():
    di = {"a": [1, 2, 3], "b": [4, 5, 6]}
    slicer = S(di)

    result = slicer["a", 1].o
    assert result == 2


def test_slicer_3d():
    data_2d = [[1, 2], [3, 4], [5, 6]]
    values_3d = [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18]],
    ]
    names = ["a", "b", "c"]

    slicer = S(data=data_2d, values=values_3d, names=A(names, 2))
    actual = slicer[..., 1]
    assert ctr_eq(actual.data, data_2d)
    assert actual.names == "b"

    actual = slicer[0, :, 1]
    assert ctr_eq(actual.data, data_2d[0])
    assert actual.names == "b"

    actual = slicer[0, :][:, 1]
    assert ctr_eq(actual.data, data_2d[0])
    assert actual.names == "b"


def test_untracked():
    data = [1, 2, 3, 4]
    primitive = 1
    collection = [[8, 9]]
    slicer = S(data=data, primitive=O(primitive, None), collection=O(collection, None))
    actual = slicer[:2]
    assert actual.data == data[:2]
    assert actual.primitive == primitive
    assert ctr_eq(actual.collection, collection)


def test_partial_untracked():
    s = S(a=np.zeros((4, 5, 6)), b=O(np.ones((4, 2, 2)), [0]))
    assert s[:, :, 1].b.shape == (4, 2, 2)


def test_numpy_subkeys():
    data = [1, 2, 3, 4]
    slicer = S(data=data)

    subkey = np.int64(1)
    assert slicer[subkey].data == 2

    subkey_arr_1d = np.array([0, 1])
    assert ctr_eq(slicer[subkey_arr_1d].data, [1, 2])

    subkey_arr_2d = np.array([[0, 1], [3, 4]])
    with pytest.raises(ValueError):
        _ = slicer[subkey_arr_2d]


def test_repr_smoke():
    slicer = S([1, 2], ["a", "b"], named=[3, 4])
    print(slicer)

    atomic = AtomicSlicer([1, 2, 3, 4])
    print(atomic)


def test_slicer_simple_di():
    di = {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
    slicer = S(di)
    actual = slicer["B", 0]
    actual = actual.o
    assert ctr_eq(actual, 3)

    nested_di = {"X": di, "Y": di}
    actual = S(nested_di)["X", "B", 0].o
    assert ctr_eq(actual, 3)


def test_slicer_sparse():
    array = np.array([[1, 0, 4], [0, 0, 5], [2, 3, 6]])
    csc_array = csc_matrix(array)
    csr_array = csr_matrix(array)
    dok_array = dok_matrix(array)
    lil_array = lil_matrix(array)

    candidates = [csc_array, csr_array, dok_array, lil_array]
    for candidate in candidates:
        print("testing:", type(candidate))
        slicer = S(candidate)
        actual = slicer[0, 0]
        assert ctr_eq(actual.o, 1)
        actual = slicer[1, 1]
        assert ctr_eq(actual.o, 0)

        actual = slicer[0]
        expected = np.array([1, 0, 4])
        assert ctr_eq(actual.o, expected)

        actual = slicer[:, 1]
        expected = np.array([0, 0, 3])
        assert ctr_eq(actual.o, expected)

        actual = slicer[:, :]
        expected = np.array([[1, 0, 4], [0, 0, 5], [2, 3, 6]])
        assert ctr_eq(actual.o, expected)

        actual = slicer[0, :]
        expected = np.array([1, 0, 4])
        assert ctr_eq(actual.o, expected)


def test_slicer_torch():
    import torch

    data = torch.tensor([[1, 2], [3, 4]])
    values = torch.tensor([[5, 6], [7, 8]])
    alias = ["f1", "f2"]

    slicer = S(data=data, values=values, alias=A(alias, 1))
    sliced = slicer[0, "f2"]
    assert sliced.data == 2
    assert sliced.values == 6


def test_slicer_pandas():
    di = {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
    df = pd.DataFrame(di)

    slicer = S(df)
    assert slicer[0, "A"].o == 1
    assert ctr_eq(slicer[:, "A"].o, [1, 2])
    assert ctr_eq(slicer[0, :].o, [1, 3, 5])

    df = pd.DataFrame(di, index=["X", "Y"])
    slicer = S(df)
    assert slicer["X", "A"].o == 1
    assert slicer[0, "A"].o == 1
    assert slicer[0, 0].o == 1
    slicer = S(df["A"])
    assert slicer["X"].o == 1
    assert slicer[0].o == 1
    assert ctr_eq(slicer[:].o, [1, 2])


def test_handle_newaxis_ellipses():

    index_tup = (1,)
    max_dim = 3

    expanded_index_tup = _handle_newaxis_ellipses(index_tup, max_dim)
    assert expanded_index_tup == (1, slice(None), slice(None))


def test_tracked_dim_arg_smoke():
    li = ["A", "B"]
    _ = A(li, dim=0)
    _ = A(li, dim=[0])
    _ = A(li, dim=(0,))

    # Aliases must have a single dim
    with pytest.raises(Exception):
        _ = A(li, dim=None)

    with pytest.raises(Exception):
        _ = A(li, dim=[0, 1])

    _ = O(li, dim=0)
    _ = O(li, dim=[0])
    _ = O(li, dim=(0,))

    assert True


def test_operations_1d():
    elements = [1, 2, 3, 4]
    li = elements
    tup = tuple(elements)
    di = {i: x for i, x in enumerate(elements)}
    series = pd.Series(elements)
    array = np.array(elements)
    torch_array = torch.tensor(elements)
    containers = [li, tup, array, torch_array, di, series]
    for ctr in containers:
        print("testing:", type(ctr))
        slicer = AtomicSlicer(ctr)

        assert ctr_eq(slicer[0], elements[0])

        # Array
        assert ctr_eq(slicer[[0, 1, 2, 3]], elements)
        assert ctr_eq(slicer[[0, 1, 2]], elements[:-1])

        # All
        assert ctr_eq(slicer[:], elements[:])
        assert ctr_eq(slicer[tuple()], elements)

        # Ranged slicing
        if not isinstance(ctr, dict):  # Do not test on dictionaries.
            assert ctr_eq(slicer[-1], elements[-1])
            assert ctr_eq(slicer[0:3:2], elements[0:3:2])


def test_operations_2d():
    elements = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    li = elements
    df = pd.DataFrame(elements, columns=["A", "B", "C"])

    sparse_csc = csc_matrix(elements)
    sparse_csr = csr_matrix(elements)
    sparse_dok = dok_matrix(elements)
    sparse_lil = lil_matrix(elements)

    containers = [li, df, sparse_csc, sparse_csr, sparse_dok, sparse_lil]
    for ctr in containers:
        print("testing:", type(ctr))
        slicer = AtomicSlicer(ctr)

        assert ctr_eq(slicer[0], elements[0])

        # Ranged slicing
        if not isinstance(ctr, dict):
            assert ctr_eq(slicer[-1], elements[-1])
            assert ctr_eq(slicer[0, 0:3:2], elements[0][0:3:2])

        # Array
        assert ctr_eq(slicer[[0, 1, 2], :], elements)

        # All
        assert ctr_eq(slicer[:], elements)
        assert ctr_eq(slicer[tuple()], elements)

        assert ctr_eq(slicer[:, 0], [elements[i][0] for i, _ in enumerate(elements)])
        assert ctr_eq(slicer[[0, 1], 0], [elements[i][0] for i in [0, 1]])
        assert ctr_eq(slicer[[0, 1], 1], [elements[i][1] for i in [0, 1]])
        assert ctr_eq(slicer[0, :], elements[0])
        assert ctr_eq(slicer[0, 1], elements[0][1])

        assert ctr_eq(slicer[..., 0], [elements[i][0] for i, _ in enumerate(elements)])


def test_operations_3d():
    # 3-dimensional fixed dimension case
    elements = [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18]],
    ]
    tuple_elements = (
        ((1, 2, 3), (4, 5, 6)),
        ((7, 8, 9), (10, 11, 12)),
        ((13, 14, 15), (16, 17, 18)),
    )
    torch_array = torch.tensor(elements)
    multi_array = np.array(elements)
    list_of_lists = elements
    tuples_of_tuples = tuple_elements
    list_of_multi_arrays = [
        np.array(elements[0]),
        np.array(elements[1]),
        np.array(elements[2]),
    ]
    di_of_multi_arrays = {
        0: np.array(elements[0]),
        1: np.array(elements[1]),
        2: np.array(elements[2]),
    }

    containers = [
        torch_array,
        multi_array,
        tuples_of_tuples,
        list_of_lists,
        list_of_multi_arrays,
        di_of_multi_arrays,
    ]
    for ctr in containers:
        print("testing:", type(ctr))
        slicer = AtomicSlicer(ctr)

        assert ctr_eq(slicer[0], elements[0])

        # Ranged slicing
        if not isinstance(ctr, dict):
            assert ctr_eq(slicer[-1], elements[-1])
            assert ctr_eq(slicer[0, 0:3:2], elements[0][0:3:2])

        # Array
        assert ctr_eq(slicer[[0, 1, 2], :], elements)

        # All
        assert ctr_eq(slicer[:], elements)
        assert ctr_eq(slicer[tuple()], elements)

        assert ctr_eq(slicer[:, 0], [elements[i][0] for i, _ in enumerate(elements)])
        assert ctr_eq(slicer[[0, 1], 0], [elements[i][0] for i in [0, 1]])
        assert ctr_eq(slicer[[0, 1], 1], [elements[i][1] for i in [0, 1]])
        assert ctr_eq(slicer[0, :], elements[0])
        assert ctr_eq(slicer[0, 1], elements[0][1])

        rows = []
        for i, _ in enumerate(elements):
            cols = []
            for j, _ in enumerate(elements[i]):
                cols.append(elements[i][j][1])
            rows.append(cols)
        assert ctr_eq(slicer[..., 1], rows)
        assert ctr_eq(slicer[0, ..., 1], [elements[0][i][1] for i in range(len(elements[0]))])


def test_attribute_assignment():
    data = [[1, 2], [3, 4]]
    values = [[5, 6], [7, 8]]
    identifiers = ["id1", "id1"]
    instance_names = ["r1", "r2"]
    feature_names = ["f1", "f2"]
    full_name = "A"

    exp = S(
        data=data,
        values=values,
        identifiers=A(identifiers, 0),
        instance_names=A(instance_names, 0),
        feature_names=A(feature_names, 1),
        full_name=full_name,
    )

    exp.feature_names = ["f3", "f4"]

    assert exp.feature_names == ["f3", "f4"]
    assert exp[:, 0].feature_names == "f3"

    with pytest.raises(Exception):
        _ = exp[:, "f1"]  # f1 should no longer exist as valid alias

    exp.feature_names = A(["f5", "f6"], dim=0)

    assert exp.feature_names == ["f5", "f6"]
    assert exp[1, :].feature_names == "f6"  # feature_names now tracks dim 0
