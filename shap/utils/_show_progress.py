import time

import tqdm


class ShowProgress:
    """This is a simple wrapper around tqdm that includes a starting delay before printing."""

    def __init__(self, iterable, total, desc, silent, start_delay):
        self.iter = iter(iterable)
        self.start_time = time.time()
        self.pbar = None
        self.total = total
        self.desc = desc
        self.start_delay = start_delay
        self.silent = silent
        self.unshown_count = 0

    def __next__(self):
        if self.pbar is None and time.time() - self.start_time > self.start_delay:
            self.pbar = tqdm.tqdm(total=self.total, initial=self.unshown_count, desc=self.desc, disable=self.silent)
            self.pbar.start_t = self.start_time
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

    def __iter__(self):
        return self


def show_progress(iterable, total=None, desc=None, silent=False, start_delay=10):
    return ShowProgress(iterable, total, desc, silent, start_delay)
