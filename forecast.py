import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import pm25_features as pf

ART = joblib.load(HERE / "model.pkl")
RIDGE, TREES, W_RIDGE = ART["ridge"], ART["trees"], ART["blend_weight"]
GROUPS, RIDGE_COLS, TREE_COLS = ART["groups"], ART["ridge_cols"], ART["tree_cols"]
ORDER, CLIM = ART["station_order"], ART["clim"]
STATIONS = pd.read_csv(HERE / "bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id").reindex(ORDER)
M = pf.neighbour_matrices(pf.station_xy(STATIONS["latitude"], STATIONS["longitude"]))
KEEP_H = 800


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex 'timestamp', one string column per sensor_id),
    every hour from 1 Jan 2024 00:00 up to and including issue_time.
    issue_time: 12:00 on some day (IST).
    Returns predicted PM2.5 for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    issue_time = pd.Timestamp(issue_time)
    h = history.copy()
    h.columns = h.columns.astype(str)
    h = h.loc[:issue_time, ORDER].tail(KEEP_H)
    h = h.reindex(pd.date_range(h.index[0], issue_time, freq="h"))
    h = h.interpolate(limit_direction="both", axis=0).ffill().bfill()
    if h.isna().any().any():
        h = h.apply(lambda col: col.fillna(h.mean(axis=1))).fillna(0.0)
    V = h.to_numpy(dtype=float)
    X = pf.build_features(V, np.array([len(V) - 1]), [issue_time], GROUPS, M, clim=CLIM)
    today = V[-1]
    p_ridge = np.maximum(today + RIDGE.predict(X[:, RIDGE_COLS]), 0.0)
    p_trees = np.maximum(today + np.mean([m.predict(X[:, TREE_COLS]) for m in TREES], axis=0), 0.0)
    pred = W_RIDGE * p_ridge + (1 - W_RIDGE) * p_trees
    pred = np.where(np.isfinite(pred), pred, today)
    pred = np.nan_to_num(pred, nan=float(np.nanmean(today)))
    return pd.Series(np.clip(pred, 0, None), index=pd.Index(ORDER, name="sensor_id"), name="pm25_forecast")
