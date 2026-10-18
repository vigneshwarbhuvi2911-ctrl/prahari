"""Risk models, decision policy and explanations.

* Supervised model: gradient-boosted trees (scikit-learn HistGradientBoosting), trained on a strict
  time-based split (train on the past, tune on the next period, test on the most recent period), with
  fraud up-weighted for the 0.5% class imbalance.
* Calibration: logistic (Platt) calibration on the validation period, so the risk shown to analysts is a
  probability that always rises with the model score.
* Anomaly model: an Isolation Forest trained on normal payments only. It does not need fraud labels, so it
  can flag new attack patterns the supervised model has never seen.
* Policy: thresholds come from an alert budget (how many payments the operations team can hold or
  challenge), set on the validation period:
      HOLD     block and send to an analyst          (top ~0.3% of payments by risk)
      STEP_UP  ask the customer to confirm (OTP/call) (next ~1%, or extreme anomaly)
      ALLOW
  plus a small set of guard rules that always apply (defence in depth).
* Explanations: for every alert, each feature is reset to a typical value and the change in model score
  is measured, giving the top reasons in plain English.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

from prahari.features import DESCRIBE, FEATURES

IDX = {f: i for i, f in enumerate(FEATURES)}
# Features the anomaly model looks at: behaviour, not identity
ANOMALY_FEATURES = ["amount_log", "amount_z", "amount_vs_max", "is_night", "payer_cnt_1h", "payer_cnt_24h",
                    "new_payee", "payee_account_age_days", "device_new_for_payer", "sim_swap_hours",
                    "city_mismatch", "payee_in_distinct_24h", "payer_out_vs_inflow_2h", "payer_inflow_2h_cnt",
                    "p2m_to_non_merchant"]


@dataclass
class Policy:
    hold: float
    step_up: float
    anomaly: float
    hold_budget: float = 0.005
    step_budget: float = 0.012


@dataclass
class RiskModel:
    gbm: HistGradientBoostingClassifier
    calibrator: "PlattCalibrator"
    iforest: IsolationForest
    policy: Policy
    reference: np.ndarray                     # typical (median) normal payment, for explanations
    feature_names: list = field(default_factory=lambda: list(FEATURES))

    # ------------------------------------------------------------------------------------- scoring
    def raw(self, X: np.ndarray) -> np.ndarray:
        return self.gbm.decision_function(X)

    def prob(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(self.raw(X))

    def anomaly(self, X: np.ndarray) -> np.ndarray:
        """Higher = more unusual (sign flipped from scikit-learn's convention)."""
        return -self.iforest.score_samples(X[:, [IDX[f] for f in ANOMALY_FEATURES]])

    def decide(self, X: np.ndarray, guard: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Thresholds come from the alert budget and are applied to the raw score (same order as the probability)."""
        r, a = self.raw(X), self.anomaly(X)
        p = self.calibrator.predict(r)
        d = np.full(len(X), "ALLOW", dtype=object)
        d[(r >= self.policy.step_up) | (a >= self.policy.anomaly)] = "STEP_UP"
        d[r >= self.policy.hold] = "HOLD"
        if guard is not None:
            d[guard] = "HOLD"
        return d, p, a

    # ---------------------------------------------------------------------------------- explanation
    def explain(self, x: np.ndarray, top: int = 4) -> list[dict]:
        base = self.raw(x[None, :])[0]
        probes = np.repeat(x[None, :], len(x), axis=0)
        probes[np.arange(len(x)), np.arange(len(x))] = self.reference
        deltas = base - self.raw(probes)
        order = np.argsort(-deltas)
        out = []
        for j in order:
            if deltas[j] <= 0.05 or len(out) == top:
                break
            f = FEATURES[j]
            if not _risky_direction(f, float(x[j]), float(self.reference[j])):
                continue          # an interaction effect; a reason like "0 senders" would read as nonsense
            out.append({"feature": f, "value": round(float(x[j]), 3), "impact": round(float(deltas[j]), 3),
                        "reason": reason_text(f, float(x[j]))})
        return out


LOW_IS_RISKY = {"payee_account_age_days", "payer_account_age_days", "payee_is_merchant", "device_age_hours",
                "sim_swap_hours"}


def _risky_direction(f: str, v: float, ref: float) -> bool:
    if f == "hour":
        return True
    return v < ref if f in LOW_IS_RISKY else v > ref


def reason_text(f: str, v: float) -> str:
    base = DESCRIBE[f]
    fmt = {"sim_swap_hours": lambda: f"{base} ({v:.1f} hours)",
           "payee_account_age_days": lambda: f"{base} ({int(v)} days ago)",
           "payer_account_age_days": lambda: f"{base} ({int(v)} days old)",
           "amount_z": lambda: f"{base} ({v:.1f} standard deviations)",
           "amount_vs_max": lambda: f"{base} ({v:.1f}x the previous largest)" if v > 1 else base,
           "amount_log": lambda: f"{base} (Rs {np.expm1(v):,.0f})",
           "device_age_hours": lambda: f"{base} ({v:.1f} hours)",
           "payee_in_distinct_24h": lambda: f"{base} ({int(v)} in 24 hours)",
           "payer_out_vs_inflow_2h": lambda: f"{base} ({v:.0%} of money received in the last 2 hours)",
           "payer_inflow_2h_cnt": lambda: f"{base} ({int(v)})",
           "payer_cnt_1h": lambda: f"{base} ({int(v)} before this one)",
           "hour": lambda: f"{base} ({int(v):02d}:{int((v % 1) * 60):02d})"}
    return fmt.get(f, lambda: base)()


# ------------------------------------------------------------------------------------------- rules
def guard_rules(X: np.ndarray) -> np.ndarray:
    """Always-on rules: SIM swapped within 24h + new device + Rs 10,000 or more -> HOLD."""
    return ((X[:, IDX["sim_swap_hours"]] < 24) & (X[:, IDX["device_new_for_payer"]] == 1)
            & (np.expm1(X[:, IDX["amount_log"]]) >= 10000))


def baseline_rules(X: np.ndarray) -> np.ndarray:
    """A typical hand-written rule set, used only as the comparison baseline in the evaluation."""
    amt = np.expm1(X[:, IDX["amount_log"]])
    return ((amt >= 50000) | (X[:, IDX["payer_cnt_1h"]] >= 5)
            | ((X[:, IDX["device_new_for_payer"]] == 1) & (amt >= 10000))
            | ((X[:, IDX["is_night"]] == 1) & (amt >= 20000))
            | (X[:, IDX["remark_scam_words"]] > 0))


# ------------------------------------------------------------------------------------------ training
@dataclass
class PlattCalibrator:
    """Logistic (Platt) calibration of the raw score: strictly increasing, so a higher-risk payment never shows a
    lower probability, and the probability does not flatten at 100% the way isotonic regression does here."""
    a: float
    b: float

    @classmethod
    def fit(cls, raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        lr = LogisticRegression(C=1.0).fit(raw.reshape(-1, 1), y)
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.a * np.asarray(raw) + self.b, -40, 40)))


def train(X_tr, y_tr, X_val, y_val, seed: int = 0, hold_budget: float = 0.005, step_budget: float = 0.012) -> RiskModel:
    w = np.where(y_tr == 1, 12.0, 1.0)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, early_stopping=False, random_state=seed)
    gbm.fit(X_tr, y_tr, sample_weight=w)
    cal = PlattCalibrator.fit(gbm.decision_function(X_val), y_val)
    normal = X_tr[y_tr == 0]
    rng = np.random.default_rng(seed)
    sample = normal[rng.choice(len(normal), size=min(60000, len(normal)), replace=False)]
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=seed)
    iso.fit(sample[:, [IDX[f] for f in ANOMALY_FEATURES]])
    model = RiskModel(gbm, cal, iso, Policy(1.0, 1.0, 1e9), np.median(normal, axis=0).astype(np.float32))
    # thresholds from the alert budget on the validation period
    r_val, a_val = model.raw(X_val), model.anomaly(X_val)
    hold = float(np.quantile(r_val, 1 - hold_budget))
    step = float(np.quantile(r_val, 1 - step_budget))
    model.policy = Policy(hold=hold, step_up=min(step, hold),
                          anomaly=float(np.quantile(a_val, 1 - 0.002)), hold_budget=hold_budget,
                          step_budget=step_budget)
    return model


# ------------------------------------------------------------------------------------------- metrics
def pr_curve(y: np.ndarray, s: np.ndarray, points: int = 40) -> dict:
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    prec, rec = tp / k, tp / max(1, ys.sum())
    ap = float(np.sum(prec * ys) / max(1, ys.sum()))
    idx = np.unique(np.searchsorted(rec, np.linspace(0.02, 1.0, points)).clip(0, len(rec) - 1))
    return {"average_precision": round(ap, 4),
            "points": [{"recall": round(float(rec[i]), 4), "precision": round(float(prec[i]), 4)} for i in idx]}
