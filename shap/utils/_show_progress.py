from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

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
        self.iter = iter(iterable)
        self.start_time = time.time()
        self.pbar = None
        self.total = total
        self.desc = desc
        self.start_delay = start_delay
        self.silent = silent
        self.unshown_count = 0

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
