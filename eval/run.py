"""Evaluation report and quality gates.

    python -m eval.run              # load saved artifacts (or build them) and write eval/REPORT.md
    python -m eval.run --rebuild    # always simulate + train from scratch first

Exits with status 1 if any quality gate fails, so CI blocks a change that makes the system worse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prahari.config import Settings
from prahari.pipeline import Engine
from prahari.sim import SimConfig

# (label, metric path, comparison, threshold). Gates apply to the full-scale build.
GATES = [
    ("Model average precision", "model.average_precision", ">=", 0.85),
    ("HOLD precision", "model.hold.precision", ">=", 0.90),
    ("HOLD recall (model + graph)", "model.hold.recall", ">=", 0.85),
    ("HOLD + step-up recall", "model.hold_or_step_up.recall", ">=", 0.90),
    ("Graph lifts HOLD recall over model only", "_graph_lift", ">=", 0.05),
    ("Model beats rules baseline on recall", "_model_vs_rules", ">=", 0.30),
    ("Novel scam held (never in training)", "by_scenario.investment_scam.hold", ">=", 0.50),
    ("Mule-ring account precision", "graph.precision", ">=", 0.90),
    ("Mule-ring account recall", "graph.recall_active_fraud_accounts", ">=", 0.85),
    ("Agent typology accuracy", "agent.typology_accuracy", ">=", 0.90),
    ("Narrative sentences grounded", "agent.grounded_sentence_rate", ">=", 1.0),
    ("Scoring latency p95 (ms)", "latency_ms.p95", "<=", 50.0),
]


def get(m: dict, path: str):
    if path == "_graph_lift":
        return round(m["model"]["hold"]["recall"] - m["model"]["hold_model_only"]["recall"], 3)
    if path == "_model_vs_rules":
        return round(m["model"]["hold_or_step_up"]["recall"] - m["rules_baseline"]["recall"], 3)
    for k in path.split("."):
        m = m[k]
    return m


def check(m: dict) -> list[tuple[str, float, str, float, bool]]:
    out = []
    for label, path, op, thr in GATES:
        v = get(m, path)
        out.append((label, v, op, thr, v >= thr if op == ">=" else v <= thr))
    return out


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def report(m: dict, gates: list) -> str:
    d, mo, rb = m["data"], m["model"], m["rules_baseline"]
    rows = [("Rules baseline (hand-written)", rb), ("Model only - HOLD", mo["hold_model_only"]),
            ("Model + graph - HOLD", mo["hold"]), ("Model + graph - HOLD or step-up", mo["hold_or_step_up"])]
    L = ["# Prahari evaluation report", "",
         "> All numbers are on **synthetic** data from the built-in simulator. They show that the system works "
         "end to end and how its parts compare; they are not a claim about real-world fraud rates.", "",
         f"- Payments simulated: **{d['payments']:,}** across {d['accounts']:,} accounts "
         f"(fraud rate {pct(d['fraud_rate'])})",
         f"- Test period: {d['periods']['test']} - {d['test_payments']:,} payments, {d['test_fraud']} fraudulent",
         f"- Time-based split: warm-up {d['periods']['warm_up']}, train {d['periods']['train']}, "
         f"validation {d['periods']['validation']}", "",
         "## Quality gates", "", "| Gate | Value | Target | Result |", "|---|---|---|---|"]
    for label, v, op, thr, ok in gates:
        L.append(f"| {label} | {v} | {op} {thr} | {'PASS' if ok else 'FAIL'} |")
    L += ["", "## Decisions on the test period", "",
          "| Approach | Precision | Recall | Value recall | Alerts / day | False positives / 10k |",
          "|---|---|---|---|---|---|"]
    for name, r in rows:
        L.append(f"| {name} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['value_recall'])} | "
                 f"{r['alerts_per_day']} | {r['false_positives_per_10k']} |")
    L += ["", f"Model average precision: **{mo['average_precision']}**. "
          f"Graph escalations (STEP_UP raised to HOLD by the mule watchlist): {mo['graph_escalations']}.", "",
          "## By fraud type", "",
          "| Fraud type | Payments | Rules | Model only HOLD | Model + graph HOLD | HOLD or step-up |",
          "|---|---|---|---|---|---|"]
    for s, r in m["by_scenario"].items():
        name = s + (" (new - not in training data)" if r["novel"] else "")
        L.append(f"| {name} | {r['payments']} | {pct(r['rules_baseline'])} | {pct(r['hold_model_only'])} | "
                 f"{pct(r['hold'])} | {pct(r['hold_or_step_up'])} |")
    g, a, lat = m["graph"], m["agent"], m["latency_ms"]
    L += ["", "## Mule-ring detection", "",
          f"- Rings found: {g['rings_found']} (simulated: {g['true_rings']}); accounts flagged: {g['accounts_flagged']}",
          f"- Account precision {pct(g['precision'])}, recall of active fraud accounts "
          f"{pct(g['recall_active_fraud_accounts'])}, ring purity {pct(g['ring_purity'])}",
          f"- Receiving accounts of the new scam linked to a ring: {g['novel_scam_payees_linked']}", "",
          "## Investigator agent", "",
          f"- Cases investigated: {a['cases']}; typology accuracy {pct(a['typology_accuracy'])}; "
          f"grounded narrative sentences {pct(a['grounded_sentence_rate'])}",
          "- By fraud type: " + ", ".join(f"{k} {pct(v)}" for k, v in a["by_scenario"].items()),
          "  (the new scam is counted correct when it is labelled `new_pattern`)", "",
          "## Latency (live path: features + model + policy, one payment at a time)", "",
          f"p50 {lat['p50']} ms, p95 {lat['p95']} ms, p99 {lat['p99']} ms over {lat['samples']} payments.", "",
          "## Monitoring", "", f"- Drift (PSI, train vs test): **{m['drift']['status']}**"]
    for grp, r in m["fairness"].items():
        L.append(f"- Fairness by {grp.replace('_', ' ')}: **{r['status']}** "
                 f"(highest / lowest false-positive rate ratio {r['max_fpr_ratio']})")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("REPORT.md")))
    args = ap.parse_args()
    s = Settings()
    if args.rebuild or not (Path(s.artifacts) / "prahari.joblib").exists():
        e = Engine.build(SimConfig())
        e.save(s.artifacts)
    else:
        e = Engine.load(s.artifacts)
    gates = check(e.metrics)
    Path(args.out).write_text(report(e.metrics, gates))
    for label, v, op, thr, ok in gates:
        print(f"{'PASS' if ok else 'FAIL'}  {label}: {v} (target {op} {thr})")
    failed = [g for g in gates if not g[4]]
    print(f"\n{len(gates) - len(failed)}/{len(gates)} gates passed; report written to {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
