"""Model monitoring: data drift and fairness.

* Drift: Population Stability Index (PSI) of every feature and of the model score, comparing the
  training period with a recent window. PSI above 0.2 is the usual "investigate / consider retraining" line.
* Fairness: alert rates, false-positive rates and fraud recall by customer age band and city tier.
  Neither attribute is a model input, but behaviour correlated with them can still make the model treat
  groups differently; legitimate customers in one group should not be stopped far more often than others.
"""
from __future__ import annotations

import numpy as np

from prahari.features import FEATURES


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                     # (nearly) constant feature: compare category shares
        vals = np.unique(np.concatenate([expected, actual]))
        e = np.array([(expected == v).mean() for v in vals])
        a = np.array([(actual == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(X_ref: np.ndarray, X_now: np.ndarray, s_ref: np.ndarray, s_now: np.ndarray,
                 y_now: np.ndarray | None = None) -> dict:
    feats = sorted(({"feature": f, "psi": round(psi(X_ref[:, i], X_now[:, i]), 4)} for i, f in enumerate(FEATURES)),
                   key=lambda d: -d["psi"])
    score_psi = round(psi(s_ref, s_now), 4)
    # concept drift signal: drift among the payments that were actually fraud
    status = "alert" if score_psi > 0.2 or feats[0]["psi"] > 0.25 else "watch" if feats[0]["psi"] > 0.1 else "stable"
    return {"score_psi": score_psi, "features": feats[:10], "status": status,
            "rule": "PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 (or score PSI > 0.2) investigate and consider retraining"}


def fairness_report(groups: dict[str, np.ndarray], flagged: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for name, g in groups.items():
        rows = []
        for v in sorted(set(g.tolist())):
            m = g == v
            legit, fraud = m & (y == 0), m & (y == 1)
            rows.append({"group": str(v), "payments": int(m.sum()),
                         "alert_rate": round(float(flagged[m].mean()), 5) if m.any() else 0.0,
                         "false_positive_rate": round(float(flagged[legit].mean()), 5) if legit.any() else 0.0,
                         "fraud_recall": round(float(flagged[fraud].mean()), 3) if fraud.sum() >= 5 else None})
        fps = int((flagged & (y == 0)).sum())
        fprs = [r["false_positive_rate"] for r in rows if r["payments"] >= 2000]
        ratio = (max(fprs) / min(fprs)) if fprs and min(fprs) > 0 else None
        status = ("insufficient data" if fps < 50 or ratio is None
                  else "review" if ratio > 2.0 else "ok")
        out[name] = {"groups": rows, "false_positives": fps, "max_fpr_ratio": round(ratio, 2) if ratio else None,
                     "status": status, "rule": "review if one group's false-positive rate is over 2x another's"}
    return out
