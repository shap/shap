from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

import tqdm

T = TypeVar("T")


class ShowProgress(Iterator[T]):
    """This is a simple wrapper around tqdm that includes a starting delay before printing."""

    def __init__(
        self,
        iterable: Iterable[T],
        total: int | None,
        desc: str | None,
        silent: bool,
        start_delay: float,
    ) -> None:
        self.iter: Iterator[T] = iter(iterable)
        self.start_time: float = time.time()
        self.pbar: tqdm.tqdm[Any] | None = None
        self.total: int | None = total
        self.desc: str | None = desc
        self.start_delay: float = start_delay
        self.silent: bool = silent
        self.unshown_count: int = 0

    def __next__(self) -> T:
        if self.pbar is None and time.time() - self.start_time > self.start_delay:
            self.pbar = tqdm.tqdm(total=self.total, initial=self.unshown_count, desc=self.desc, disable=self.silent)
            self.pbar.start_t = self.start_time  # type: ignore[attr-defined]
        if self.pbar is not None:
            self.pbar.update(1)
        else:
            self.unshown_count += 1
        try:
            return next(self.iter)
        except StopIteration:
            if self.pbar is not None:
                self.pbar.close()
            raise

    def __iter__(self) -> ShowProgress[T]:
        return self


def show_progress(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str | None = None,
    silent: bool = False,
    start_delay: float = 10,
) -> ShowProgress[T]:
    return ShowProgress(iterable, total, desc, silent, start_delay)
