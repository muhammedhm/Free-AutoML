"""Data loading, task detection and column typing."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


@dataclass
class ColumnInfo:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # column -> reason


def load_csv(path: str, **read_kwargs) -> pd.DataFrame:
    """Read a CSV file and tidy up the column names."""
    try:
        df = pd.read_csv(path, **read_kwargs)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", **read_kwargs)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    """Drop rows whose target is missing. Returns (df, number_of_rows_dropped)."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    before = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if df.empty:
        raise ValueError("All rows have a missing target.")
    return df, before - len(df)


def detect_task(y: pd.Series, max_classes: int = 20, max_unique_ratio: float = 0.1) -> str:
    """Guess 'classification' or 'regression' from the target column."""
    if not is_numeric_dtype(y) or is_bool_dtype(y):
        return "classification"

    n_unique = y.nunique()
    if n_unique <= 2:
        return "classification"

    integer_like = bool((y.dropna() % 1 == 0).all())
    few_values = n_unique <= max_classes and (n_unique / len(y)) <= max_unique_ratio
    return "classification" if (integer_like and few_values) else "regression"


def infer_columns(
    X: pd.DataFrame,
    id_unique_ratio: float = 0.95,
    max_missing_ratio: float = 0.6,
) -> ColumnInfo:
    """Split feature columns into numeric / categorical and drop useless ones."""
    info = ColumnInfo()
    n_rows = len(X)

    for col in X.columns:
        s = X[col]

        if s.nunique(dropna=False) <= 1:
            info.dropped[col] = "constant"
        elif s.isna().mean() > max_missing_ratio:
            info.dropped[col] = f"more than {max_missing_ratio:.0%} missing"
        elif is_numeric_dtype(s) and not is_bool_dtype(s):
            info.numeric.append(col)
        elif n_rows > 20 and s.nunique() / n_rows > id_unique_ratio:
            info.dropped[col] = "looks like an ID / free text (almost all values unique)"
        else:
            info.categorical.append(col)

    if not (info.numeric or info.categorical):
        raise ValueError("No usable feature columns left after cleaning.")
    return info
