import numpy as np
import pandas as pd

KM_X, KM_Y = 100.3, 110.6
RADII_KM = (15, 30, 60)

GROUP_FEATURES = {
    "lags24":       [f"lag{k}h" for k in range(24)],
    "history_days": [f"sameHour_d-{d}" for d in range(1, 7)] + ["mean_72h", "mean_168h", "std_168h", "min_168h"],
    "neigh_knn":    ["knn10_now", "knn10_24h", "knn10_168h"],
    "neigh_radius": [f"r{r}_{w}" for r in RADII_KM for w in ("now", "24h")],
    "network":      ["net_now", "net_6h", "net_24h", "net_168h"],
    "trend":        ["d_sameHour", "d_24hmean", "d_net24h"],
    "seasonal":     ["clim_train", "mean_30d", "rel_30d", "sin_doy", "cos_doy"],
    "dow":          ["sin_dow", "cos_dow"],
}
ALL_GROUPS = list(GROUP_FEATURES)
MAX_HISTORY_H = 168


def station_xy(lat, lon):
    return np.column_stack([(np.asarray(lon, float) - 85.5) * KM_X, (np.asarray(lat, float) - 25.7) * KM_Y])


def neighbour_matrices(xy, k=10, radii=RADII_KM):
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    N = len(xy)
    order = np.argsort(D, axis=1, kind="stable")
    M = {}
    A = np.zeros((N, N)); A[np.arange(N)[:, None], order[:, :k]] = 1.0 / k
    M["knn10"] = A
    for r in radii:
        A = (D <= r).astype(float)
        empty = np.where(A.sum(1) == 0)[0]
        if len(empty):
            A[empty[:, None], order[empty, :3]] = 1.0
        M[f"r{r}"] = A / A.sum(1, keepdims=True)
    return M


def _win_mean(C, idx, w):
    lo = np.maximum(idx - w + 1, 0)
    return (C[idx + 1] - C[lo]) / (idx + 1 - lo)[:, None] if C.ndim == 2 else (C[idx + 1] - C[lo]) / (idx + 1 - lo)


def build_features(V, issue_idx, issue_times, groups, M, clim=None):
    issue_idx = np.asarray(issue_idx); n_i, N = len(issue_idx), V.shape[1]
    C = np.vstack([np.zeros((1, N)), np.cumsum(V, 0)])
    net = V.mean(1); Cn = np.concatenate([[0.0], np.cumsum(net)])
    cols = []
    full = lambda a: np.repeat(np.asarray(a)[:, None], N, 1)

    for g in groups:
        if g == "lags24":
            cols += [V[issue_idx - k] for k in range(24)]
        elif g == "history_days":
            cols += [V[issue_idx - 24 * d] for d in range(1, 7)]
            cols += [_win_mean(C, issue_idx, 72), _win_mean(C, issue_idx, 168)]
            win = np.stack([V[i - 167:i + 1] for i in issue_idx])
            cols += [win.std(1), win.min(1)]
        elif g == "neigh_knn":
            A = M["knn10"]
            cols += [V[issue_idx] @ A.T, _win_mean(C, issue_idx, 24) @ A.T, _win_mean(C, issue_idx, 168) @ A.T]
        elif g == "neigh_radius":
            m24 = _win_mean(C, issue_idx, 24)
            for r in RADII_KM:
                A = M[f"r{r}"]
                cols += [V[issue_idx] @ A.T, m24 @ A.T]
        elif g == "network":
            cols += [full(net[issue_idx]), full(_win_mean(Cn, issue_idx, 6)),
                     full(_win_mean(Cn, issue_idx, 24)), full(_win_mean(Cn, issue_idx, 168))]
        elif g == "trend":
            cols += [V[issue_idx] - V[issue_idx - 24],
                     _win_mean(C, issue_idx, 24) - _win_mean(C, issue_idx - 24, 24),
                     full(_win_mean(Cn, issue_idx, 24) - _win_mean(Cn, issue_idx - 24, 24))]
        elif g == "seasonal":
            m30 = _win_mean(C, issue_idx, 720); n30 = _win_mean(Cn, issue_idx, 720)
            cols += [np.repeat(np.asarray(clim)[None, :], n_i, 0), m30, m30 / np.maximum(n30[:, None], 1.0)]
            day = np.array([(pd.Timestamp(t).normalize() - pd.Timestamp("2024-01-01")).days for t in issue_times])
            ang = 2 * np.pi * day / 365.0
            cols += [full(np.sin(ang)), full(np.cos(ang))]
        elif g == "dow":
            dow = np.array([pd.Timestamp(t).dayofweek for t in issue_times])
            cols += [full(np.sin(2 * np.pi * dow / 7)), full(np.cos(2 * np.pi * dow / 7))]
        else:
            raise KeyError(g)
    return np.column_stack([c.reshape(-1) for c in cols])


def feature_names(groups):
    return [f for g in groups for f in GROUP_FEATURES[g]]


def feature_groups(groups):
    return np.array([g for g in groups for _ in GROUP_FEATURES[g]])
