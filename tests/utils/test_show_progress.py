from shap.utils._show_progress import show_progress


def test_show_progress_basic():
    data = [1, 2, 3, 4, 5]

    result = list(show_progress(data, start_delay=0))

    assert result == data


def test_show_progress_with_total():
    data = range(10)

    result = list(show_progress(data, total=10, start_delay=0))

    assert len(result) == 10