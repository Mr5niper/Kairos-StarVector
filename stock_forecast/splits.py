# stock_forecast/splits.py
"""
Walk-forward window generation.

Leakage safety: the training block always ends before the validation
block, which always ends before the test block, and no scaler or model is
ever fitted on data from a later block. That ordering is the whole point
of walk-forward evaluation and is easy to break accidentally.
"""
from typing import Iterator, List, Tuple

Window = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def rolling_windows(
    n: int,
    seq_len: int,
    train_len: int,
    val_len: int,
    test_len: int,
    step: int,
) -> Iterator[Window]:
    """
    Expanding training set, fixed validation and test blocks, stepping
    forward by `step` bars.

    Yields ((train_start, train_end), (val_start, val_end),
            (test_start, test_end)) as half-open index ranges.
    """
    if min(n, train_len, val_len, test_len) <= 0 or step <= 0:
        return
    start = int(train_len)
    while start + val_len + test_len <= n:
        yield ((0, start),
               (start, start + val_len),
               (start + val_len, start + val_len + test_len))
        start += int(step)


def count_windows(n: int, seq_len: int, train_len: int, val_len: int,
                  test_len: int, step: int) -> int:
    return sum(1 for _ in rolling_windows(n, seq_len, train_len, val_len,
                                          test_len, step))


def fit_split_sizes(n: int, seq_len: int,
                    min_train: int = 40, min_val: int = 10,
                    min_test: int = 10,
                    max_val: int = 126, max_test: int = 126) -> Tuple[int, int, int, int]:
    """
    Derive split sizes that actually fit `n` bars.

    Two problems being solved here.

    The original config hard-coded train_len=756 (about three years). Any
    shorter range produced zero windows, and the GUI then took a fallback
    branch that referenced a dictionary key it had never set, so a short
    date range crashed rather than simply using smaller splits.

    The caps on validation and test matter just as much. Sizing them as a
    fixed fraction of the data means a ten-year history gets a two-year
    test block, which leaves room for exactly one window. One window is
    not walk-forward evaluation at all: there is no distribution of scores
    to average and no way to tell a robust model from a lucky one. Capping
    both at roughly six months of bars gives several independent windows
    on any decent history.
    """
    seq = int(max(5, min(seq_len, max(5, n // 4))))
    free = max(n - seq, 0)
    need = min_train + min_val + min_test
    if free < need:
        seq = max(5, n - need)
        free = max(n - seq, 0)

    val = int(min(max_val, max(min_val, free * 0.15)))
    test = int(min(max_test, max(min_test, free * 0.15)))
    train = int(max(min_train, free - val - test))

    # On a long history, cap the initial training block so later windows
    # still have data left to step through.
    train = int(min(train, max(min_train, free - val - test)))
    if free >= need * 3:
        train = int(min(train, max(min_train, int(free * 0.55))))

    while train + val + test > free and val > min_val:
        val -= 1
    while train + val + test > free and test > min_test:
        test -= 1
    while train + val + test > free and train > min_train:
        train -= 1
    return seq, int(train), int(val), int(test)
