"""Model policy and explanations, graph intelligence, the investigator agent and monitoring."""
import numpy as np

from prahari import agent as A
from prahari.features import FEATURES
from prahari.llm import LLMError
from prahari.model import IDX, baseline_rules, guard_rules, pr_curve
from prahari.monitor import fairness_report, psi
from tests.conftest import ScriptedLLM, first_alert


def test_policy_thresholds_and_decisions(engine):
    pol = engine.model.policy
    assert pol.hold >= pol.step_up
    d = engine.decisions
    assert set(np.unique(d)) <= {"ALLOW", "STEP_UP", "HOLD"}
    assert (d == "HOLD").mean() < 0.03 and (d != "ALLOW").mean() < 0.06
    y = np.array([t.is_fraud for t in engine.test_txns()])
    assert y[d == "HOLD"].mean() > 0.7                       # held payments are mostly fraud
    assert (d[y == 1] != "ALLOW").mean() > (baseline_rules(engine.X[engine.test_offset:])[y == 1]).mean()


def test_calibration_is_monotonic(engine):
    r = np.linspace(-15, 15, 50)
    p = engine.model.calibrator.predict(r)
    assert np.all(np.diff(p) >= -1e-9) and 0 <= p.min() and p.max() <= 1


def test_guard_rule_and_explanations(engine):
    x = engine.model.reference.copy()
    x[IDX["sim_swap_hours"]], x[IDX["device_new_for_payer"]], x[IDX["amount_log"]] = 2.0, 1.0, np.log1p(49000)
    assert guard_rules(x[None, :])[0]
    reasons = engine.model.explain(x)
    assert reasons and any("SIM card swapped" in r["reason"] for r in reasons)
    assert all(r["feature"] in FEATURES and r["impact"] > 0 for r in reasons)


def test_pr_curve():
    y = np.array([1, 0, 1, 0, 0])
    c = pr_curve(y, np.array([0.9, 0.8, 0.7, 0.2, 0.1]))
    assert abs(c["average_precision"] - (1 + 2 / 3) / 2) < 1e-3


def test_money_trail_follows_mule_hops(engine):
    t = next(t for t in engine.test_txns() if t.scenario == "ato")
    tr = engine.graph.money_trail(t.txn_id)
    assert tr["hops"] >= 2 and tr["edges"][0]["txn_id"] == t.txn_id
    kinds = {engine.ds.accounts[n["account"]].kind for n in tr["nodes"][1:]}
    assert kinds <= {"mule", "cashout"}


def test_rings_are_mostly_real_mules(engine):
    members = {m for r in engine.rings for m in r["members"]}
    assert engine.rings and members
    real = sum(engine.ds.accounts[m].kind in ("mule", "cashout", "scammer") for m in members)
    assert real / len(members) >= 0.8
    assert engine.metrics["graph"]["precision"] >= 0.8


def test_graph_watchlist_has_no_look_ahead(engine):
    """An account only escalates payments made after the day it joined the watchlist."""
    for t, d, dm in zip(engine.test_txns(), engine.decisions, engine.decisions_model):
        if d == "HOLD" and dm != "HOLD":
            assert engine.watchlist[t.payee] <= t.ts.isoformat()


def test_agent_investigates_account_takeover(engine):
    t = first_alert(engine, "ato")
    c = engine.investigate(t.txn_id)
    assert c["typology"] == "account_takeover"
    assert [e["tool"] for e in c["evidence"]] == A.PLAYBOOK
    assert [e["id"] for e in c["evidence"]] == [f"E{i}" for i in range(1, len(A.PLAYBOOK) + 1)]
    kept, removed = A.verify(c["narrative"], c["evidence"])
    assert len(kept) == len(c["narrative"]) and not removed
    r = c["report"]
    assert r["status"].startswith("draft") and r["subjects"][0]["account"] == t.payee
    assert "1930" in " ".join(r["recommended_actions"])


def test_agent_typologies_and_new_pattern(engine):
    assert engine.investigate(first_alert(engine, "mule_layering").txn_id)["typology"] == "mule_layering"
    assert engine.metrics["agent"]["typology_accuracy"] >= 0.85


def _agent_with(engine, responses):
    llm = ScriptedLLM(responses)
    return A.build_agent(engine.tools(), llm), llm


def test_llm_plan_is_restricted_to_known_tools(engine):
    agent, llm = _agent_with(engine, {"plan": {"tools": ["get_payment", "delete_database", "money_trail",
                                                         "device_and_sim", "model_reasons"]},
                                      "write": {"sentences": []}})
    c = A.investigate(agent, first_alert(engine, "ato").txn_id)
    assert [e["tool"] for e in c["evidence"]] == ["get_payment", "money_trail", "device_and_sim", "model_reasons"]
    assert c["trace"][0]["source"] == "llm"


def test_llm_narrative_is_checked_against_evidence(engine):
    t = first_alert(engine, "ato")
    amount = int(t.amount)
    good = {"text": f"The payment {t.txn_id} moved money to {t.payee}.", "evidence": ["E1"]}
    bad = {"text": f"The customer also lost Rs {amount + 12345} earlier that day.", "evidence": ["E1"]}
    agent, llm = _agent_with(engine, {"plan": LLMError("down"), "write": {"sentences": [good, bad]}})
    c = A.investigate(agent, t.txn_id)
    assert c["narrative_source"] == "llm" and [s["text"] for s in c["narrative"]] == [good["text"]]
    assert c["removed_sentences"][0]["reason"].startswith("not found in cited evidence")
    assert llm.calls[-1]["tier"] == "large"
    agent2, _ = _agent_with(engine, {"write": {"sentences": [bad]}})
    assert A.investigate(agent2, t.txn_id)["narrative_source"] == "template"


def test_psi_and_fairness():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02 and psi(a, rng.normal(1.0, 1, 5000)) > 0.25
    g = np.array(["x"] * 3000 + ["y"] * 3000)
    y = np.zeros(6000, dtype=int)
    flagged = np.zeros(6000, dtype=bool)
    flagged[:90] = True                      # 3% of group x, 0.5% of group y
    flagged[3000:3015] = True
    rep = fairness_report({"g": g}, flagged, y)["g"]
    assert rep["status"] == "review" and rep["max_fpr_ratio"] == 6.0
