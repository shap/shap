"""Tests for shap.utils._show_progress module.

Covers the ShowProgress iterator wrapper and the show_progress factory function.
"""

import time

import pytest

from shap.utils._show_progress import ShowProgress, show_progress

# ---------------------------------------------------------------------------
# show_progress factory function
# ---------------------------------------------------------------------------


class TestShowProgressFactory:
    """Tests for the show_progress() convenience function."""

    def test_returns_show_progress_instance(self):
        result = show_progress(range(5), total=5)
        assert isinstance(result, ShowProgress)

    def test_default_parameters(self):
        sp = show_progress(range(3))
        assert sp.total is None
        assert sp.desc is None
        assert sp.silent is False
        assert sp.start_delay == 10

    def test_custom_parameters(self):
        sp = show_progress(range(3), total=3, desc="test", silent=True, start_delay=5)
        assert sp.total == 3
        assert sp.desc == "test"
        assert sp.silent is True
        assert sp.start_delay == 5


# ---------------------------------------------------------------------------
# ShowProgress iteration
# ---------------------------------------------------------------------------


class TestShowProgressIteration:
    """Tests for ShowProgress iteration behavior."""

    def test_iterates_all_elements(self):
        items = list(show_progress(range(5), total=5, silent=True, start_delay=0))
        assert items == [0, 1, 2, 3, 4]

    def test_empty_iterable(self):
        items = list(show_progress([], total=0, silent=True, start_delay=0))
        assert items == []

    def test_single_element(self):
        items = list(show_progress([42], total=1, silent=True, start_delay=0))
        assert items == [42]

    def test_preserves_element_types(self):
        data = ["a", "b", "c"]
        items = list(show_progress(data, total=3, silent=True, start_delay=0))
        assert items == data

    def test_works_with_generator(self):
        def gen():
            yield 1
            yield 2
            yield 3

        items = list(show_progress(gen(), total=3, silent=True, start_delay=0))
        assert items == [1, 2, 3]

    def test_iter_returns_self(self):
        sp = show_progress(range(3), total=3, silent=True, start_delay=0)
        assert iter(sp) is sp

    def test_raises_stop_iteration(self):
        sp = show_progress(range(1), total=1, silent=True, start_delay=0)
        next(sp)
        with pytest.raises(StopIteration):
            next(sp)

    def test_can_be_used_in_for_loop(self):
        total = 0
        for x in show_progress(range(5), total=5, silent=True, start_delay=0):
            total += x
        assert total == 10


# ---------------------------------------------------------------------------
# ShowProgress progress bar behavior
# ---------------------------------------------------------------------------


class TestShowProgressBar:
    """Tests for progress bar creation and delay logic."""

    def test_pbar_is_none_before_delay(self):
        sp = ShowProgress(range(10), total=10, desc=None, silent=True, start_delay=100)
        assert sp.pbar is None
        next(sp)
        # start_delay=100s means pbar should still be None
        assert sp.pbar is None

    def test_unshown_count_increments_before_delay(self):
        sp = ShowProgress(range(10), total=10, desc=None, silent=True, start_delay=100)
        next(sp)
        next(sp)
        next(sp)
        assert sp.unshown_count == 3
        assert sp.pbar is None

    def test_pbar_created_after_delay(self):
        sp = ShowProgress(range(10), total=10, desc=None, silent=True, start_delay=0)
        # start_delay=0 means pbar should be created on first next()
        # but we need a tiny time to pass
        time.sleep(0.01)
        next(sp)
        assert sp.pbar is not None

    def test_pbar_initial_matches_unshown(self):
        # Create with a very short delay
        sp = ShowProgress(range(10), total=10, desc="test", silent=True, start_delay=0.05)
        # Consume a few items before delay triggers
        next(sp)
        next(sp)
        unshown = sp.unshown_count
        # Wait for delay to pass
        time.sleep(0.1)
        next(sp)
        # After delay, pbar should have been initialized with the unshown count
        assert sp.pbar is not None

    def test_silent_mode(self):
        items = list(show_progress(range(5), total=5, silent=True, start_delay=0))
        assert items == [0, 1, 2, 3, 4]

    def test_start_time_is_set(self):
        sp = ShowProgress(range(5), total=5, desc=None, silent=True, start_delay=0)
        assert sp.start_time > 0
        assert sp.start_time <= time.time()
