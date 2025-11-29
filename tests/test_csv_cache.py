# test_csv_cache.py

import pandas as pd
from unittest.mock import MagicMock
from src.utils.cache import csv_cache  # adjust import as needed

def test_cache_miss_calls_function_and_saves(tmp_path):
    # path generator uses tmp directory
    def path_gen(*args, **kwargs):
        return tmp_path / "cache.csv"

    # mock expensive function
    func = MagicMock(return_value=12)

    @csv_cache(path_gen)
    def wrapped(g, k):
        return func(g, k)

    result = wrapped(g=0.5, k=1.2)

    # execution and persistence checks
    func.assert_called_once_with(0.5, 1.2)
    assert result == 12
    assert (tmp_path / "cache.csv").exists()

    df = pd.read_csv(tmp_path / "cache.csv")
    assert len(df) == 1



def test_cache_hit_skips_function_call(tmp_path):
    def path_gen(*args, **kwargs):
        return tmp_path / "cache.csv"

    func = MagicMock(return_value=12)

    @csv_cache(path_gen)
    def wrapped(g, k):
        return func(g, k)

    # first call writes
    wrapped(g=0.2, k=0.9)

    # second call should not call the function again
    func.reset_mock()
    result = wrapped(g=0.2, k=0.9)

    func.assert_not_called()
    assert result == 12
