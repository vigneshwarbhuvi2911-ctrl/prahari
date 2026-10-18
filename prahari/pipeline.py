"""End-to-end pipeline and the runtime engine.

    simulate -> features (streaming) -> time split -> train + calibrate + thresholds -> score test period
             -> graph + rings -> evaluate (model, rules baseline, graph, agent, drift, fairness)

`Engine.build()` runs everything (about a minute at full scale) and `save()`/`load()` keep the result, so the
API starts instantly. The engine then serves: live replay of the test period through the online feature
store, ad-hoc scoring, the alert queue, investigations, rings, money trails and metrics.
"""
from __future__ import annotations

import copy
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from prahari import agent as A
from prahari.config import Settings
from prahari.features import FEATURES, FeatureStore, build_matrix
from prahari.graph import PaymentGraph
from prahari.llm import LLMClient, make_llm
from prahari.model import IDX, RiskModel, baseline_rules, guard_rules, pr_curve, train
from prahari.monitor import drift_report, fairness_report
from prahari.sim import Dataset, SimConfig, Txn, simulate

ARTIFACT = "prahari.joblib"
BURN_IN_DAYS = 14


def _n(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def fmt_inr(v: float) -> str:
    s = f"{int(round(v)):d}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        s = ",".join(([head] if head else []) + parts) + "," + tail
    return f"Rs {s}"


@dataclass
class Engine:
    ds: Dataset
    X: np.ndarray
    model: RiskModel
    test_start: datetime
    val_start: datetime
    snapshot: FeatureStore
    decisions: np.ndarray            # test period only
    probs: np.ndarray
    raws: np.ndarray
    anomalies: np.ndarray
    metrics: dict = field(default_factory=dict)
    rings: list = field(default_factory=list)
    knn_scale: np.ndarray | None = None
    knn_rows: np.ndarray | None = None
    knn_labels: list = field(default_factory=list)
    knn_threshold: float = 0.0
    watchlist: dict = field(default_factory=dict)
    decisions_model: np.ndarray | None = None      # before the graph escalation (for comparison)

    # ------------------------------------------------------------------------------------- building
    @classmethod
    def build(cls, cfg: SimConfig | None = None, log=print) -> "Engine":
        t0 = time.time()
        ds = simulate(cfg or SimConfig())
        log(f"simulated {len(ds.txns):,} payments, {len(ds.accounts):,} accounts in {time.time() - t0:.1f}s")
        val_start, test_start = ds.split_days()
        X, _, snap = build_matrix(ds, snapshot_at=test_start)
        log(f"features built in {time.time() - t0:.1f}s")
        y, ts = cls._labels(ds)
        # the first 14 days only warm up the feature store (customer history, known payees and devices);
        # training on them would teach the model that "everything is new"
        burn_in = ds.config.start + timedelta(days=BURN_IN_DAYS)
        tr, va, te = (ts >= burn_in) & (ts < val_start), (ts >= val_start) & (ts < test_start), ts >= test_start
        model = train(X[tr], y[tr], X[va], y[va])
        log(f"model trained in {time.time() - t0:.1f}s")
        dec, prob, anom = model.decide(X[te], guard_rules(X[te]))
        model_only = dec.copy()
        dec, watch = cls._graph_escalation(ds, test_start, dec, log)
        e = cls(ds, X, model, test_start, val_start, snap, dec, prob, model.raw(X[te]), anom)
        e.watchlist, e.decisions_model = watch, model_only
        e._knn_fit(X[tr], [t.scenario for t, keep in zip(ds.txns, tr) if keep], y[tr])
        e._post_init()
        e.rings = e.graph.rings(e.alert_txns())
        e.metrics = e.evaluate(X[tr], y[tr], log)
        log(f"pipeline finished in {time.time() - t0:.1f}s")
        return e

    @staticmethod
    def _graph_escalation(ds: Dataset, test_start: datetime, dec: np.ndarray, log=print) -> tuple[np.ndarray, dict]:
        """Daily graph job, without look-ahead: at the start of each day, rings are rebuilt from the alerts and
        payments seen so far; ring members and the accounts feeding them go on a watchlist, and a payment that the
        model sent to STEP_UP is escalated to HOLD when its payee is on the watchlist."""
        dec = dec.copy()
        test_idx = next(i for i, t in enumerate(ds.txns) if t.ts >= test_start)
        watch: dict[str, str] = {}
        day = test_start
        end = ds.txns[-1].ts
        escalated = 0
        while day <= end:
            cut = next((i for i in range(test_idx, len(ds.txns)) if ds.txns[i].ts >= day), len(ds.txns))
            prefix = ds.txns[:cut]
            g = PaymentGraph(prefix, ds.accounts)
            alerts = [t for t, d in zip(ds.txns[test_idx:cut], dec[:cut - test_idx]) if d == "HOLD"]
            members = {m for r in g.rings(alerts) for m in r["members"]}
            for acc in members | g.feeders(members, day - timedelta(days=7)):
                watch.setdefault(acc, day.isoformat())
            nxt = day + timedelta(days=1)
            for i in range(cut, len(ds.txns)):
                t = ds.txns[i]
                if t.ts >= nxt:
                    break
                k = i - test_idx
                if dec[k] == "STEP_UP" and t.payee in watch:
                    dec[k] = "HOLD"
                    escalated += 1
            day = nxt
        log(f"graph job: {len(watch)} accounts on the watchlist, {escalated} payments escalated to HOLD")
        return dec, watch

    @staticmethod
    def _labels(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([t.is_fraud for t in ds.txns], dtype=np.int8), np.array([t.ts for t in ds.txns]))

    def _knn_fit(self, X_tr, scenarios, y_tr) -> None:
        fraud = X_tr[y_tr == 1]
        self.knn_scale = X_tr.std(axis=0) + 1e-6
        self.knn_rows = fraud / self.knn_scale
        self.knn_labels = [A.SCENARIO_TO_TYPOLOGY.get(s, "unclear") for s, yy in zip(scenarios, y_tr) if yy == 1]
        nn = NearestNeighbors(n_neighbors=2).fit(self.knn_rows)
        d, _ = nn.kneighbors(self.knn_rows)
        self.knn_threshold = float(np.quantile(d[:, 1], 0.95))

    def _post_init(self, llm: LLMClient | None = None) -> None:
        """Rebuilds the in-memory indexes (also after loading from disk)."""
        ds = self.ds
        self.by_id = {t.txn_id: i for i, t in enumerate(ds.txns)}
        self.out_idx, self.in_idx = defaultdict(list), defaultdict(list)
        for i, t in enumerate(ds.txns):
            self.out_idx[t.payer].append(i)
            self.in_idx[t.payee].append(i)
        self.test_offset = next(i for i, t in enumerate(ds.txns) if t.ts >= self.test_start)
        self.graph = PaymentGraph(ds.txns, ds.accounts)
        self.knn = NearestNeighbors(n_neighbors=5).fit(self.knn_rows)
        self.llm = llm or LLMClient("mock", "mock", "mock")
        self.agent = A.build_agent(self.tools(), self.llm)
        self.cases: dict[str, dict] = {}
        self.analyst: dict[str, dict] = {}
        self.reset_live()

    # ---------------------------------------------------------------------------------- persistence
    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        keep = {k: getattr(self, k) for k in ("ds", "X", "model", "test_start", "val_start", "snapshot", "decisions",
                                              "probs", "raws", "anomalies", "metrics", "rings", "knn_scale",
                                              "knn_rows", "knn_labels", "knn_threshold", "watchlist", "decisions_model")}
        path = folder / ARTIFACT
        joblib.dump(keep, path, compress=3)
        return path

    @classmethod
    def load(cls, folder: Path, llm: LLMClient | None = None) -> "Engine":
        e = cls(**joblib.load(folder / ARTIFACT))
        e._post_init(llm)
        return e

    @classmethod
    def load_or_build(cls, settings: Settings, log=print) -> "Engine":
        if (settings.artifacts / ARTIFACT).exists():
            e = cls.load(settings.artifacts, make_llm(settings))
        else:
            e = cls.build(SimConfig.small() if settings.scale == "small" else SimConfig(), log)
            e.save(settings.artifacts)
            e._post_init(make_llm(settings))
        return e

    # ------------------------------------------------------------------------------------- helpers
    def txn(self, txn_id: str) -> Txn:
        return self.ds.txns[self.by_id[txn_id]]

    def test_txns(self) -> list[Txn]:
        return self.ds.txns[self.test_offset:]

    def alert_txns(self, include_step_up: bool = False) -> list[Txn]:
        keep = ("HOLD", "STEP_UP") if include_step_up else ("HOLD",)
        return [t for t, d in zip(self.test_txns(), self.decisions) if d in keep]

    def vpa(self, acc: str) -> str:
        a = self.ds.accounts.get(acc)
        return a.vpa if a else acc

    def row(self, txn_id: str) -> np.ndarray:
        return self.X[self.by_id[txn_id]]

    def test_pos(self, txn_id: str) -> int:
        return self.by_id[txn_id] - self.test_offset

    # ---------------------------------------------------------------------------------- live replay
    def reset_live(self) -> None:
        self.live_store = copy.deepcopy(self.snapshot)
        self.live_pos = 0
        self.live_swap = getattr(self.snapshot, "next_swap", 0)

    def live_next(self, n: int = 50) -> list[dict]:
        """Scores the next n payments of the test period as a stream: online features, model, policy."""
        test = self.test_txns()
        out = []
        for _ in range(n):
            if self.live_pos >= len(test):
                self.reset_live()
            t = test[self.live_pos]
            swaps = self.ds.sim_swaps
            while self.live_swap < len(swaps) and swaps[self.live_swap][1] <= t.ts:
                self.live_store.sim_swap(swaps[self.live_swap][0], swaps[self.live_swap][1])
                self.live_swap += 1
            t0 = time.perf_counter()
            x = self.live_store.observe(t)
            dec, p, a = self.model.decide(x[None, :], guard_rules(x[None, :]))
            if dec[0] == "STEP_UP" and self.watchlist.get(t.payee, "9999") <= t.ts.isoformat():
                dec[0] = "HOLD"                     # payee is on the graph watchlist
            ms = (time.perf_counter() - t0) * 1000
            out.append({"txn_id": t.txn_id, "ts": t.ts.isoformat(timespec="seconds"), "payer": self.vpa(t.payer),
                        "payee": self.vpa(t.payee), "amount": t.amount, "type": t.txn_type, "decision": str(dec[0]),
                        "risk": round(float(p[0]), 4), "latency_ms": round(ms, 2),
                        "is_fraud": t.is_fraud, "scenario": t.scenario})
            self.live_pos += 1
        return out

    def score_payload(self, p: dict) -> dict:
        """Scores a hypothetical payment against the current live state (the state is not changed)."""
        payer, payee = p["payer"], p["payee"]
        for acc in (payer, payee):
            if acc not in self.ds.accounts:
                raise KeyError(acc)
        ts = datetime.fromisoformat(p["ts"]) if p.get("ts") else (
            self.test_txns()[max(0, self.live_pos - 1)].ts + timedelta(minutes=1))
        t = Txn("TX-ADHOC", ts, payer, payee, float(p["amount"]), p.get("type", "P2P"),
                p.get("device") or self.ds.home_device[payer], p.get("city") or self.ds.accounts[payer].city,
                p.get("remarks", ""))
        t0 = time.perf_counter()
        x = self.live_store.features(t)
        dec, prob, anom = self.model.decide(x[None, :], guard_rules(x[None, :]))
        reasons = self.model.explain(x)
        # an ad-hoc payment is "now", so the latest watchlist applies unless a past timestamp was given
        listed = self.watchlist.get(payee, "9999") <= ts.isoformat() if p.get("ts") else payee in self.watchlist
        if dec[0] == "STEP_UP" and listed:
            dec[0] = "HOLD"                         # payee is on the graph watchlist
            reasons.insert(0, {"feature": "graph_watchlist", "value": 1.0, "impact": None,
                               "reason": "receiving account is on the mule watchlist (graph)"})
        ms = (time.perf_counter() - t0) * 1000
        return {"decision": str(dec[0]), "risk": round(float(prob[0]), 4),
                "anomaly": round(float(anom[0]), 4), "reasons": reasons,
                "latency_ms": round(ms, 2), "features": {f: round(float(v), 3) for f, v in zip(FEATURES, x)}}

    # ------------------------------------------------------------------------------------ the queue
    def alerts(self, limit: int = 100) -> list[dict]:
        out = []
        model = self.decisions_model if self.decisions_model is not None else self.decisions
        for t, d, dm, p in zip(self.test_txns(), self.decisions, model, self.probs):
            if d != "HOLD":
                continue
            out.append({"txn_id": t.txn_id, "ts": t.ts.isoformat(timespec="minutes"), "payer": self.vpa(t.payer),
                        "payee": self.vpa(t.payee), "amount": t.amount, "type": t.txn_type, "risk": round(float(p), 4),
                        "held_by": "model" if dm == "HOLD" else "graph watchlist",
                        "status": self.analyst.get(t.txn_id, {}).get("decision", "open"),
                        "investigated": t.txn_id in self.cases})
        # analyst queue: open cases first, then by money at risk (graph holds count as high risk), newest first
        return sorted(out, key=lambda a: (a["status"] != "open", -a["amount"] * max(a["risk"], 0.9 if
                      a["held_by"] != "model" else 0), a["ts"]), reverse=False)[:limit]

    def investigate(self, txn_id: str) -> dict:
        if txn_id not in self.by_id:
            raise KeyError(txn_id)
        if txn_id not in self.cases:
            self.cases[txn_id] = A.investigate(self.agent, txn_id)
        return self.cases[txn_id] | {"analyst": self.analyst.get(txn_id)}

    def record_decision(self, txn_id: str, decision: str, note: str = "") -> dict:
        if txn_id not in self.by_id:
            raise KeyError(txn_id)
        self.analyst[txn_id] = {"decision": decision, "note": note, "at": datetime.now().isoformat(timespec="seconds")}
        return self.analyst[txn_id]

    def ring_view(self) -> list[dict]:
        out = []
        for r in self.rings:
            nodes = []
            for m in r["members"]:
                prof = self.graph.profile(m)
                nodes.append({"account": m, "vpa": self.vpa(m), "score": r["scores"].get(m, 0), **prof})
            out.append(r | {"nodes": nodes})
        return out

    # ----------------------------------------------------------------------------------- agent tools
    def tools(self) -> dict:
        e = self

        def get_payment(tid):
            t = e.txn(tid)
            pa, pe = e.ds.accounts[t.payer], e.ds.accounts[t.payee]
            data = {"txn_id": t.txn_id, "time": t.ts.isoformat(timespec="minutes"), "amount": t.amount,
                    "type": t.txn_type, "payer": t.payer, "payer_vpa": pa.vpa, "payer_city": pa.city,
                    "payee": t.payee, "payee_vpa": pe.vpa, "payment_city": t.city, "device": t.device,
                    "remarks": t.remarks}
            note = f', note "{t.remarks}"' if t.remarks else ""
            return ("Payment", f"{t.txn_type} payment {t.txn_id} of {fmt_inr(t.amount)} from {t.payer} ({pa.vpa}) to "
                    f"{t.payee} ({pe.vpa}) at {t.ts:%d %b %Y %H:%M}, made from {t.city}{note}.", data)

        def model_reasons(tid):
            x = e.row(tid)
            i = e.test_pos(tid)
            risk = float(e.probs[i]) if 0 <= i < len(e.probs) else float(e.model.prob(x[None, :])[0])
            reasons = e.model.explain(x)
            txt = "; ".join(r["reason"] for r in reasons)
            return ("Model risk and reasons", f"Model risk {risk:.0%}. Main reasons: {txt}." if risk >= 0.01 else f"Model risk {risk:.1%}. Main reasons: {txt}.",
                    {"risk": round(risk, 4), "reasons": reasons,
                     "features": {f: round(float(v), 3) for f, v in zip(FEATURES, x)}})

        def customer_profile(tid):
            t = e.txn(tid)
            a = e.ds.accounts[t.payer]
            past = [e.ds.txns[i] for i in e.out_idx[t.payer] if e.ds.txns[i].ts < t.ts and
                    t.ts - e.ds.txns[i].ts <= timedelta(days=30)]
            amts = [p.amount for p in past]
            med = float(np.median(amts)) if amts else 0.0
            age = (t.ts - a.open_ts).days
            return ("Customer profile", f"Payer {t.payer} is a {a.kind} account in {a.city}, open for {age} days, with "
                    f"{len(past)} payments in the previous 30 days (median {fmt_inr(med)}, largest "
                    f"{fmt_inr(max(amts) if amts else 0)}).",
                    {"account": t.payer, "kind": a.kind, "city": a.city, "account_age_days": age,
                     "payments_30d": len(past), "median_30d": med, "max_30d": max(amts) if amts else 0})

        def device_and_sim(tid):
            t = e.txn(tid)
            prev = {e.ds.txns[i].device for i in e.out_idx[t.payer] if e.ds.txns[i].ts < t.ts}
            prev.add(e.ds.home_device[t.payer])
            first = e.ds.device_first_seen.get(t.device, t.ts)
            swaps = [s for s in e.ds.sim_swaps if s[0] == t.payer and s[1] <= t.ts and t.ts - s[1] <= timedelta(days=7)]
            hrs = round((t.ts - swaps[-1][1]).total_seconds() / 3600, 1) if swaps else None
            new = t.device not in prev
            dev_age = round((t.ts - first).total_seconds() / 3600, 1)
            seen = f"{dev_age} hours" if dev_age < 48 else f"{int(dev_age // 24)} days"
            txt = (f"Device {t.device} is {'NEW for this customer' if new else 'known for this customer'} "
                   f"(first seen {seen} before the payment). ") + \
                  (f"The SIM card was swapped {hrs} hours before the payment." if hrs is not None
                   else "No SIM swap in the previous 7 days.")
            return ("Device and SIM", txt, {"device": t.device, "device_is_new": new, "device_age_hours": dev_age,
                                            "sim_swap_hours_before": hrs})

        def recent_activity(tid):
            t = e.txn(tid)
            win = timedelta(hours=24)
            outs = [e.ds.txns[i] for i in e.out_idx[t.payer]
                    if timedelta(0) <= t.ts - e.ds.txns[i].ts <= win and e.ds.txns[i].txn_id != t.txn_id]
            ins = [e.ds.txns[i] for i in e.in_idx[t.payer] if timedelta(0) <= t.ts - e.ds.txns[i].ts <= win]
            txt = (f"In the 24 hours before this payment, {t.payer} sent {_n(len(outs), 'payment')} totalling "
                   f"{fmt_inr(sum(x.amount for x in outs))} and received {_n(len(ins), 'payment')} totalling "
                   f"{fmt_inr(sum(x.amount for x in ins))}.")
            return ("Recent activity", txt, {"sent": [{"txn_id": x.txn_id, "amount": x.amount, "to": x.payee} for x in outs[-10:]],
                                             "received": [{"txn_id": x.txn_id, "amount": x.amount, "from": x.payer}
                                                          for x in ins[-10:]]})

        def money_trail(tid):
            tr = e.graph.money_trail(tid)
            hops = [x for x in tr["edges"] if x["hop"] >= 1]
            if hops:
                last = tr["nodes"][-1]
                txt = (f"The money moved on through {_n(len(hops), 'onward transfer')} across {_n(tr['hops'], 'hop')} "
                       f"within hours, "
                       f"reaching {last['account']} ({last['vpa']}).")
            else:
                txt = "The money did not move on quickly from the receiving account."
            return ("Money trail", txt, tr)

        def ring_membership(tid):
            t = e.txn(tid)
            for r in e.rings:
                if t.payee in r["members"] or t.payer in r["members"]:
                    return ("Ring membership", f"The account is part of suspected ring {r['ring']} with {r['size']} "
                            f"accounts and {fmt_inr(r['stolen_inflow'])} of flagged inflow.",
                            {"ring": r["ring"], "size": r["size"], "members": r["members"]})
            return ("Ring membership", "The accounts are not part of any detected ring.", {"ring": None})

        def similar_cases(tid):
            x = e.row(tid)[None, :] / e.knn_scale
            d, idx = e.knn.kneighbors(x)
            near = [{"typology": e.knn_labels[j], "distance": round(float(dd), 3)} for dd, j in zip(d[0], idx[0])]
            known = near[0]["distance"] <= e.knn_threshold
            top = Counter(n["typology"] for n in near).most_common(1)[0][0]
            txt = (f"Most similar past confirmed cases: {A.TYPOLOGY_NAMES[top]}." if known
                   else "Unlike any past confirmed fraud case (possible new pattern).")
            return ("Similar past cases", txt, {"nearest": near, "known_pattern_distance": round(e.knn_threshold, 3)})

        return {f.__name__: f for f in (get_payment, model_reasons, customer_profile, device_and_sim,
                                         recent_activity, money_trail, ring_membership, similar_cases)}

    # ------------------------------------------------------------------------------------ evaluation
    def evaluate(self, X_tr, y_tr, log=print) -> dict:
        test = self.test_txns()
        Xt = self.X[self.test_offset:]
        y = np.array([t.is_fraud for t in test])
        sc = np.array([t.scenario for t in test])
        amt = np.array([t.amount for t in test])
        dec = self.decisions
        hold, flag = dec == "HOLD", dec != "ALLOW"
        days = max(1, (test[-1].ts - test[0].ts).days)
        base = baseline_rules(Xt)

        def pr(mask):
            return {"precision": round(float(y[mask].mean()), 3) if mask.any() else 0.0,
                    "recall": round(float(mask[y == 1].mean()), 3),
                    "value_recall": round(float(amt[mask & (y == 1)].sum() / amt[y == 1].sum()), 3),
                    "alerts_per_day": round(float(mask.sum() / days), 1),
                    "false_positives_per_10k": round(float((mask & (y == 0)).sum() / (y == 0).sum() * 1e4), 2)}

        per = {}
        for s in sorted(set(sc) - {"normal"}):
            m = sc == s
            per[s] = {"payments": int(m.sum()), "hold": round(float(hold[m].mean()), 3),
                      "hold_model_only": round(float((self.decisions_model[m] == "HOLD").mean()), 3)
                      if self.decisions_model is not None else None,
                      "hold_or_step_up": round(float(flag[m].mean()), 3),
                      "rules_baseline": round(float(base[m].mean()), 3),
                      "novel": s == "investment_scam"}
        # graph
        truth = {a.account_id for a in self.ds.accounts.values() if a.kind in ("mule", "cashout", "scammer")}
        active = {t.payee for t in test if t.is_fraud and t.payee in truth} | \
                 {t.payer for t in test if t.is_fraud and t.payer in truth}
        pred = {m for r in self.rings for m in r["members"]}
        purity = [Counter(self.ds.accounts[m].ring_id or self.ds.accounts[m].kind for m in r["members"]).most_common(1)[0][1]
                  / r["size"] for r in self.rings]
        novel_accounts = {t.payee for t in test if t.scenario == "investment_scam"}
        # latency through the live path (features + model + policy)
        self.reset_live()
        lat = [ev["latency_ms"] for ev in self.live_next(min(3000 if len(test) > 20000 else 300, len(test)))]
        self.reset_live()
        # agent on a sample of alerts
        fraud_alerts = [t for t in self.alert_txns() if t.is_fraud][:120]
        legit_alerts = [t for t in self.alert_txns() if not t.is_fraud][:20]
        typ_ok, typ_n, by_typ = 0, 0, defaultdict(lambda: [0, 0])
        grounded = sentences = 0
        for t in fraud_alerts + legit_alerts:
            c = A.investigate(self.agent, t.txn_id)
            sentences += len(c["narrative"])
            kept, _ = A.verify(c["narrative"], c["evidence"])
            grounded += len(kept)
            if t.is_fraud:
                want = A.SCENARIO_TO_TYPOLOGY[t.scenario]
                ok = c["typology"] == want
                typ_ok += ok
                typ_n += 1
                by_typ[t.scenario][0] += ok
                by_typ[t.scenario][1] += 1
        groups = {"age_band": np.array([self.ds.accounts[t.payer].age_band for t in test]),
                  "city_tier": np.array([f"tier {self.ds.accounts[t.payer].tier}" for t in test])}
        tr_scores = self.model.raw(X_tr[np.random.default_rng(0).choice(len(X_tr), min(50000, len(X_tr)), replace=False)])
        m = {
            "data": {"payments": len(self.ds.txns), "accounts": len(self.ds.accounts),
                     "test_payments": len(test), "test_fraud": int(y.sum()),
                     "fraud_rate": round(float(np.mean([t.is_fraud for t in self.ds.txns])), 5),
                     "periods": {"warm_up": f"{self.ds.config.start:%d %b} - {self.ds.config.start + timedelta(days=BURN_IN_DAYS - 1):%d %b %Y}",
                                 "train": f"{self.ds.config.start + timedelta(days=BURN_IN_DAYS):%d %b} - {self.val_start - timedelta(days=1):%d %b %Y}",
                                 "validation": f"{self.val_start:%d %b} - {self.test_start - timedelta(days=1):%d %b %Y}",
                                 "test": f"{self.test_start:%d %b} - {test[-1].ts:%d %b %Y}"}},
            "policy": {"hold_threshold_raw": round(self.model.policy.hold, 3),
                       "hold_budget": self.model.policy.hold_budget, "step_budget": self.model.policy.step_budget},
            "model": {"average_precision": pr_curve(y, self.raws)["average_precision"], "hold": pr(hold),
                      "hold_model_only": pr(self.decisions_model == "HOLD") if self.decisions_model is not None else None,
                      "graph_escalations": int(((dec == "HOLD") & (self.decisions_model != "HOLD")).sum())
                      if self.decisions_model is not None else 0,
                      "hold_or_step_up": pr(flag), "pr_curve": pr_curve(y, self.raws)["points"]},
            "rules_baseline": pr(base),
            "by_scenario": per,
            "graph": {"rings_found": len(self.rings), "true_rings": len(self.ds.rings),
                      "accounts_flagged": len(pred),
                      "precision": round(len(pred & truth) / max(1, len(pred)), 3),
                      "recall_active_fraud_accounts": round(len(pred & active) / max(1, len(active)), 3),
                      "ring_purity": round(float(np.mean(purity)), 3) if purity else 0.0,
                      "novel_scam_payees_linked": int(len(novel_accounts & pred))},
            "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2),
                           "p99": round(float(np.percentile(lat, 99)), 2), "samples": len(lat)},
            "agent": {"cases": len(fraud_alerts) + len(legit_alerts),
                      "typology_accuracy": round(typ_ok / max(1, typ_n), 3),
                      "by_scenario": {k: round(v[0] / v[1], 3) for k, v in by_typ.items()},
                      "grounded_sentence_rate": round(grounded / max(1, sentences), 3)},
            "drift": drift_report(X_tr, Xt, tr_scores, self.raws, y),
            "fairness": fairness_report(groups, flag, y),
        }
        log(f"model AP {m['model']['average_precision']}, HOLD precision {m['model']['hold']['precision']} "
            f"recall {m['model']['hold']['recall']}; rules recall {m['rules_baseline']['recall']}")
        return m
