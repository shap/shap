''' This file contains tests for the Tabular maskers.
'''

def test_serialization_independent_masker_dataframe():
    import shap
    import numpy as np
    import tempfile

    X, y = shap.datasets.boston()

    # initialize independent masker
    original_independent_masker = shap.maskers.Independent(X)

    temp_serialization_file = tempfile.TemporaryFile()

    # serialize independent masker
    original_independent_masker.save(temp_serialization_file)

    temp_serialization_file.close()

    # deserialize masker
    new_independent_masker = shap.maskers.Independent.load(temp_serialization_file)

    temp_serialization_file.close()

    mask = np.ones(X.shape[1]).astype(np.int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_independent_masker(mask, X[:1].values[0])[1], new_independent_masker(mask, X[:1].values[0])[1])

def test_serialization_independent_masker_numpy():
    import shap
    import numpy as np
    import tempfile

    X, y = shap.datasets.boston()
    X = X.values

    # initialize independent masker
    original_independent_masker = shap.maskers.Independent(X)

    temp_serialization_file = tempfile.TemporaryFile()

    # serialize independent masker
    original_independent_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # deserialize masker
    new_independent_masker = shap.maskers.Masker.load(temp_serialization_file)

    temp_serialization_file.close()

    mask = np.ones(X.shape[1]).astype(np.int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_independent_masker(mask, X[0])[0], new_independent_masker(mask, X[0])[0])


def test_serialization_partion_masker_dataframe():
    import shap
    import numpy as np
    import tempfile

    X, y = shap.datasets.boston()

    # initialize partition masker
    original_partition_masker = shap.maskers.Partition(X)

    temp_serialization_file = tempfile.TemporaryFile()

    # serialize partition masker
    original_partition_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # deserialize masker
    new_partition_masker = shap.maskers.Partition.load(temp_serialization_file)

    temp_serialization_file.close()

    mask = np.ones(X.shape[1]).astype(np.int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[:1].values[0])[1], new_partition_masker(mask, X[:1].values[0])[1])

def test_serialization_partion_masker_numpy():
    import shap
    import numpy as np
    import tempfile

    X, y = shap.datasets.boston()
    X = X.values

    # initialize partition masker
    original_partition_masker = shap.maskers.Partition(X)

    temp_serialization_file = tempfile.TemporaryFile()

    # serialize partition masker
    original_partition_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # deserialize masker
    new_partition_masker = shap.maskers.Masker.load(temp_serialization_file)

    temp_serialization_file.close()

    mask = np.ones(X.shape[1]).astype(np.int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[0])[0], new_partition_masker(mask, X[0])[0])