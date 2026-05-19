"""Feature engineering for PLC telemetry time series.

Pure functions: no hardware, SQLite, or filesystem access.

Imputation policy
-----------------
Training stays NaN-aware (no imputation in build_features). The two helpers
``impute_raw_columns`` and ``latest_feature_row_imputed`` are *runtime-only*
fallbacks for the inference path on the Pi, where one or more sensors may be
physically absent (e.g. BMP280 has no humidity, DS18B20 may be unplugged).

The order is:
    raw_df -> impute_raw_columns -> build_features -> latest_feature_row_imputed
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Constant fallbacks for raw sensor columns whose physical sensor is absent.
_RAW_IMPUTATION_DEFAULTS: dict[str, float] = {
    "humidity_pct": 50.0,  # typical indoor humidity
}

# Cross-type imputation: if every column in the group except some is NaN,
# fill the all-NaN columns with the median of the non-NaN siblings. Used
# when one of the temperature sensors is missing.
_TYPE_GROUPS: list[list[str]] = [
    ["temp_c", "ds18b20_c"],
]

# Columns from the CSV schema in CLAUDE.md for which lag/rolling/delta are built.
_NUMERIC_COLS = [
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
    "lux",
    "current_ma",
    "mq2_raw",
]

# Lags in rows. MVP assumption: collector writes 1 row/sec.
# t1=1min(60), t5=5min(300), t15=15min(900), t60=60min(3600).
_LAG_ROWS: dict[str, int] = {
    "t1": 60,
    "t5": 300,
    "t15": 900,
    "t60": 3600,
}

# Rolling-window sizes (rows) for mean/std/min/max.
_ROLLING_WINDOWS: dict[str, int] = {
    "5m": 300,
    "15m": 900,
    "60m": 3600,
}

# Pairs (delta-suffix -> lag-suffix in _LAG_ROWS) for gradient features.
_DELTA_LAGS: dict[str, str] = {
    "delta_t1": "t1",
    "delta_t5": "t5",
}


def build_features(
    df: pd.DataFrame,
    feature_spec: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add temporal, lag, rolling and delta features to a telemetry DataFrame.

    Required input columns (subset):
        ts_iso, ts_unix, temp_c, humidity_pct, pressure_hpa,
        lux, current_ma, mq2_raw, pir_state

    Returns a new DataFrame (the original is not mutated).
    NaN values in lag/rolling columns due to insufficient history are normal
    and are not imputed.
    """
    if df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "ts_unix" in out.columns:
        out = out.sort_values("ts_unix").reset_index(drop=True)

    # --- Temporal features -------------------------------------------------
    if "ts_iso" in out.columns:
        ts = pd.to_datetime(out["ts_iso"], utc=True, errors="coerce")
        out["hour_of_day"] = ts.dt.hour
        out["minute_of_hour"] = ts.dt.minute
        out["day_of_week"] = ts.dt.dayofweek
        out["is_weekend"] = (ts.dt.dayofweek >= 5).astype("Int8")
    else:
        logger.warning("build_features: ts_iso column missing - temporal features skipped")

    # --- Lag, rolling, delta for numeric metrics ---------------------------
    available = [c for c in _NUMERIC_COLS if c in out.columns]
    if not available:
        logger.warning("build_features: no numeric metric found in df")
        return out

    # Collect all new columns into a dict, then a single pd.concat.
    # This avoids the PerformanceWarning about fragmentation in pandas 3.x.
    new_cols: dict[str, pd.Series] = {}

    for col in available:
        series = out[col]

        # Lag features (leading NaN is normal - not imputed)
        lag_series: dict[str, pd.Series] = {}
        for lag_name, n_rows in _LAG_ROWS.items():
            lag_series[lag_name] = series.shift(n_rows)
            new_cols[f"{col}_lag_{lag_name}"] = lag_series[lag_name]

        # Rolling mean/std/min/max (std needs >=2 observations -> NaN when window<2)
        for win_name, win_rows in _ROLLING_WINDOWS.items():
            roll = series.rolling(window=win_rows)
            new_cols[f"{col}_rmean_{win_name}"] = roll.mean()
            new_cols[f"{col}_rstd_{win_name}"] = roll.std()
            new_cols[f"{col}_rmin_{win_name}"] = roll.min()
            new_cols[f"{col}_rmax_{win_name}"] = roll.max()

        # Gradient (delta) features
        for delta_name, lag_key in _DELTA_LAGS.items():
            new_cols[f"{col}_{delta_name}"] = series - lag_series[lag_key]

    new_df = pd.DataFrame(new_cols, index=out.index)
    return pd.concat([out, new_df], axis=1)


def latest_feature_row(
    features_df: pd.DataFrame,
    required_features: list[str],
) -> dict[str, float] | None:
    """Return the last row of features_df restricted to required_features.

    Returns None when:
      - df is empty
      - at least one required feature is missing from the columns
      - at least one required feature is NaN in the last row
    """
    if features_df.empty:
        logger.debug("latest_feature_row: empty df")
        return None

    missing_cols = [f for f in required_features if f not in features_df.columns]
    if missing_cols:
        logger.warning("latest_feature_row: missing columns %s", missing_cols)
        return None

    last = features_df.iloc[-1]
    row = last[required_features]

    if row.isna().any():
        nan_cols = row.index[row.isna()].tolist()
        logger.debug("latest_feature_row: NaN in %s - insufficient history", nan_cols)
        return None

    return row.to_dict()


def impute_raw_columns(
    raw_df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fill fully-NULL raw sensor columns with sensible runtime defaults.

    Apply this BEFORE ``build_features`` so the lag/rolling/delta features
    inherit the imputed values automatically.

    Strategy
    --------
    - ``humidity_pct`` (or any other column listed in ``_RAW_IMPUTATION_DEFAULTS``)
      that is fully NaN is filled with the configured constant (50.0 for humidity).
    - For each cross-type group in ``_TYPE_GROUPS``, if at least one column is
      fully populated and at least one is fully NaN, the all-NaN siblings are
      filled with the median of the non-NaN siblings.
    - Partial-NaN columns are NOT touched here - that is handled later, in the
      predictor, after ``build_features``.

    Returns ``(imputed_df, imputed_column_names)``. The original frame is not
    mutated.
    """
    out = raw_df.copy()
    imputed: list[str] = []
    candidates = set(out.columns) if columns is None else set(columns)

    for col, default in _RAW_IMPUTATION_DEFAULTS.items():
        if col in out.columns and col in candidates and out[col].isna().all():
            out[col] = default
            imputed.append(col)

    for group in _TYPE_GROUPS:
        present = [c for c in group if c in out.columns]
        if not present:
            continue
        non_null = [c for c in present if not out[c].isna().all()]
        all_null = [
            c for c in present
            if out[c].isna().all() and c in candidates and c not in imputed
        ]
        if non_null and all_null:
            stacked = pd.concat([out[c].dropna() for c in non_null])
            if stacked.empty:
                continue
            median = float(stacked.median())
            for col in all_null:
                out[col] = median
                imputed.append(col)

    if imputed:
        logger.info(
            "impute_raw_columns: imputed %d columns: %s",
            len(imputed),
            imputed,
        )
    return out, imputed


def latest_feature_row_imputed(
    features_df: pd.DataFrame,
    required_features: list[str],
    *,
    fallback: float = 0.0,
) -> tuple[dict[str, float] | None, list[str]]:
    """Like ``latest_feature_row`` but fills NaN cells with ``fallback``.

    Returns ``(row_dict, imputed_column_names)``. ``row_dict`` is None only
    when the frame is empty or one of the required columns is missing - both
    are real errors, not gaps. NaN within an existing column is treated as a
    "lag/rolling/delta on an absent sensor" condition and replaced with the
    fallback (0.0 by default).
    """
    if features_df.empty:
        logger.debug("latest_feature_row_imputed: empty df")
        return None, []

    missing_cols = [f for f in required_features if f not in features_df.columns]
    if missing_cols:
        logger.warning("latest_feature_row_imputed: missing columns %s", missing_cols)
        return None, []

    last = features_df.iloc[-1]
    row = last[required_features]
    imputed = [c for c in required_features if pd.isna(last[c])]
    if imputed:
        row = row.fillna(fallback)
        logger.info(
            "latest_feature_row_imputed: filled %d NaN features with %s: %s",
            len(imputed),
            fallback,
            imputed,
        )
    return row.to_dict(), imputed


def fill_window_nans(
    window: pd.DataFrame,
    *,
    fallback: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Fill any remaining NaN cells in a sequence-model input window.

    Used by the LSTM path. Returns ``(filled_window, imputed_column_names)``.
    The list contains every column that had at least one NaN before filling.
    """
    nan_cols = [c for c in window.columns if window[c].isna().any()]
    if not nan_cols:
        return window, []
    out = window.fillna(fallback)
    logger.info(
        "fill_window_nans: filled NaN in %d columns with %s: %s",
        len(nan_cols),
        fallback,
        nan_cols,
    )
    return out, nan_cols
