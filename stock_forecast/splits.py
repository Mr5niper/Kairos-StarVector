# stock_forecast/splits.py
from typing import Iterator, Tuple

def rolling_windows(N: int, seq_len: int, train_len: int, val_len: int, test_len: int, step: int
                   ) -> Iterator[Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]]:
    start = train_len
    while start + val_len + test_len <= N:
        tr = (0, start)
        va = (start, start + val_len)
        te = (start + val_len, start + val_len + test_len)
        yield tr, va, te
        start += step