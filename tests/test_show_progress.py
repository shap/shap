from shap.utils._show_progress import show_progress


def test_basic_iteration():
    data = [1, 2, 3]
    result = list(show_progress(data, start_delay=100))
    assert result == data


def test_empty_iterable():
    data = []
    result = list(show_progress(data, start_delay=100))
    assert result == []


def test_iterator_protocol():
    data = [1, 2]
    sp = show_progress(data, start_delay=100)
    assert iter(sp) is sp


def test_generator_input():
    data = (i for i in range(3))
    result = list(show_progress(data, start_delay=100))
    assert result == [0, 1, 2]


def test_progress_bar_trigger(monkeypatch):
    import time

    # force time to trigger progress bar logic
    monkeypatch.setattr(time, "time", lambda: 999999)

    data = [1, 2, 3]
    result = list(show_progress(data, start_delay=0))

    assert result == data
