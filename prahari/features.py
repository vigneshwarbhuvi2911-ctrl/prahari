"""Online feature store.

Features are computed from state that is updated event by event, in time order, exactly as a streaming
system would see them. The SAME code builds the training set and scores live payments, so there is no
training/serving skew, and every feature is point-in-time correct: a payment is described using only what
was known *before* it happened (the payment itself is added to the state after its features are taken).

State kept per account: recent outgoing and incoming payments (deques pruned to the longest window),
running mean/variance of payment size (Welford), known payees, known devices, last SIM swap.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from prahari.sim import Account, Dataset, Txn

# Words analysts have seen in scam payment notes (maintained by the fraud team). Deliberately does not
# include the vocabulary of the later "investment scam", so the novel attack is genuinely new to the model.
SCAM_WORDS = re.compile(r"\b(kyc|refund|cashback|reward|disconnect\w*|customs|helpline|approval|otp|lottery|prize|"
                        r"penalty|blocked|verify)\b", re.I)

FEATURES = [
    "amount_log", "amount_z", "amount_vs_max", "round_amount", "hour", "is_night",
    "txn_p2p", "txn_p2m", "txn_collect",
    "payer_cnt_1h", "payer_cnt_24h", "payer_amt_24h_log", "payer_unique_payees_24h",
    "new_payee", "payee_account_age_days", "payer_account_age_days", "payee_is_merchant", "p2m_to_non_merchant",
    "device_new_for_payer", "device_age_hours", "sim_swap_hours", "city_mismatch",
    "remark_scam_words", "remark_present",
    "payee_in_cnt_1h", "payee_in_distinct_24h", "payee_in_amt_24h_log", "payee_out_in_ratio_24h",
    "payer_inflow_2h_cnt", "payer_out_vs_inflow_2h", "payer_in_distinct_24h",
]
DESCRIBE = {   # plain-English reason for each feature, used in alert explanations
    "amount_log": "large payment amount", "amount_z": "amount far above this customer's usual spending",
    "amount_vs_max": "largest payment this customer has made", "round_amount": "round-figure amount",
    "hour": "unusual time of day", "is_night": "made between midnight and 6 am",
    "txn_p2p": "person-to-person transfer", "txn_p2m": "merchant payment", "txn_collect": "approved a collect request",
    "payer_cnt_1h": "burst of payments in the last hour", "payer_cnt_24h": "many payments today",
    "payer_amt_24h_log": "high total sent today", "payer_unique_payees_24h": "paying many different people today",
"new_payee": "first payment to this payee",
    "payee_account_age_days": "payee account opened recently", "payer_account_age_days": "payer account is new",
    "payee_is_merchant": "payee is not a registered merchant",
    "p2m_to_non_merchant": "QR/merchant-style payment to an account that is not a registered merchant",
    "device_new_for_payer": "new device for this customer", "device_age_hours": "device first seen very recently",
    "sim_swap_hours": "SIM card swapped shortly before", "city_mismatch": "payment from outside the home city",
    "remark_scam_words": "payment note contains known scam words", "remark_present": "payment note present",
    "payee_in_cnt_1h": "payee receiving many payments this hour",
    "payee_in_distinct_24h": "payee receiving money from many different people",
    "payee_in_amt_24h_log": "payee received a large total today",
    "payee_out_in_ratio_24h": "payee moves money out as fast as it comes in",
    "payer_inflow_2h_cnt": "payer received several payments in the last 2 hours",
    "payer_out_vs_inflow_2h": "payer is passing on money received minutes ago",
    "payer_in_distinct_24h": "payer received money from many different people today",
}
H1, H2, H24, D7 = timedelta(hours=1), timedelta(hours=2), timedelta(hours=24), timedelta(days=7)


@dataclass
class AccState:
    out: deque = field(default_factory=deque)        # (ts, amount, payee)   last 24h
    inn: deque = field(default_factory=deque)        # (ts, amount, payer)   last 24h
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    max_amt: float = 0.0
    payees: set = field(default_factory=set)
    devices: set = field(default_factory=set)
    last_swap: datetime | None = None


def _prune(dq: deque, now: datetime, window: timedelta) -> None:
    while dq and now - dq[0][0] > window:
        dq.popleft()


class FeatureStore:
    def __init__(self, accounts: dict[str, Account], device_first_seen: dict[str, datetime],
                 home_device: dict[str, str]):
        self.accounts = accounts
        self.device_first_seen = dict(device_first_seen)
        self.state: dict[str, AccState] = defaultdict(AccState)
        for acc, dev in home_device.items():
            self.state[acc].devices.add(dev)

    def sim_swap(self, account: str, ts: datetime) -> None:
        self.state[account].last_swap = ts

    def features(self, t: Txn) -> np.ndarray:
        """Point-in-time features for a payment, from state BEFORE it."""
        p, q = self.state[t.payer], self.state[t.payee]
        for dq in (p.out, p.inn, q.out, q.inn):
            _prune(dq, t.ts, H24)
        payer, payee = self.accounts.get(t.payer), self.accounts.get(t.payee)
        la = math.log1p(t.amount)
        std = math.sqrt(p.m2 / (p.n - 1)) if p.n > 2 else 0.0
        z = (la - p.mean) / max(std, 0.35) if p.n > 2 else 0.0
        out1h = [x for x in p.out if t.ts - x[0] <= H1]
        in2h = [x for x in p.inn if t.ts - x[0] <= H2]
        q_in1h = [x for x in q.inn if t.ts - x[0] <= H1]
        q_in_amt = sum(x[1] for x in q.inn)
        q_out_amt = sum(x[1] for x in q.out)
        inflow2 = sum(x[1] for x in in2h)
        swap_h = (t.ts - p.last_swap).total_seconds() / 3600 if p.last_swap and p.last_swap <= t.ts else 999.0
        dev_first = self.device_first_seen.get(t.device, t.ts)
        hour = t.ts.hour + t.ts.minute / 60
        payee_merchant = 1.0 if payee and payee.kind == "merchant" else 0.0
        v = [
            la, z, t.amount / p.max_amt if p.max_amt else 1.0, 1.0 if t.amount >= 1000 and t.amount % 1000 == 0 else 0.0,
            hour, 1.0 if hour < 6 else 0.0,
            float(t.txn_type == "P2P"), float(t.txn_type == "P2M"), float(t.txn_type == "COLLECT"),
            len(out1h), len(p.out), math.log1p(sum(x[1] for x in p.out)), len({x[2] for x in p.out}),
            0.0 if t.payee in p.payees else 1.0,
            min((t.ts - payee.open_ts).days, 3650) if payee else 0.0,
            min((t.ts - payer.open_ts).days, 3650) if payer else 0.0,
            payee_merchant, 1.0 if t.txn_type == "P2M" and not payee_merchant else 0.0,
            0.0 if t.device in p.devices else 1.0, min((t.ts - dev_first).total_seconds() / 3600, 2000.0),
            min(swap_h, 999.0), 1.0 if payer and t.city != payer.city else 0.0,
            float(len(SCAM_WORDS.findall(t.remarks or ""))), 1.0 if t.remarks else 0.0,
            len(q_in1h), len({x[2] for x in q.inn}), math.log1p(q_in_amt),
            (q_out_amt / q_in_amt) if q_in_amt else 0.0,
            len(in2h), (t.amount / inflow2) if inflow2 else 0.0, len({x[2] for x in p.inn}),
        ]
        return np.asarray(v, dtype=np.float32)

    def update(self, t: Txn) -> None:
        p, q = self.state[t.payer], self.state[t.payee]
        p.out.append((t.ts, t.amount, t.payee))
        q.inn.append((t.ts, t.amount, t.payer))
        la = math.log1p(t.amount)
        p.n += 1
        d = la - p.mean
        p.mean += d / p.n
        p.m2 += d * (la - p.mean)
        p.max_amt = max(p.max_amt, t.amount)
        p.payees.add(t.payee)
        p.devices.add(t.device)
        self.device_first_seen.setdefault(t.device, t.ts)

    def observe(self, t: Txn) -> np.ndarray:
        x = self.features(t)
        self.update(t)
        return x


def new_store(ds: Dataset) -> FeatureStore:
    return FeatureStore(ds.accounts, {d: ts for d, ts in ds.device_first_seen.items() if ts <= ds.config.start},
                        ds.home_device)


def build_matrix(ds: Dataset, snapshot_at: datetime | None = None) -> tuple[np.ndarray, FeatureStore, FeatureStore | None]:
    """Replays the whole timeline (payments and SIM-swap events, in time order) through the store.
    Optionally returns a copy of the store as it was at `snapshot_at` (used to replay the test period live)."""
    import copy
    store = new_store(ds)
    X = np.zeros((len(ds.txns), len(FEATURES)), dtype=np.float32)
    k, swaps = 0, ds.sim_swaps
    snap = None
    for i, t in enumerate(ds.txns):
        if snapshot_at and snap is None and t.ts >= snapshot_at:
            snap = copy.deepcopy(store)
            snap.next_swap = k            # index of the first SIM-swap event not yet applied
        while k < len(swaps) and swaps[k][1] <= t.ts:
            store.sim_swap(swaps[k][0], swaps[k][1])
            k += 1
        X[i] = store.observe(t)
    return X, store, snap
