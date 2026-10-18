"""Investigator agent (LangGraph).

For each alert the agent gathers evidence with tools, decides the fraud typology, writes a case narrative in
which every sentence cites its evidence, and drafts a Suspicious Transaction Report (STR) and recommended
actions for the analyst, who makes the final decision.

    plan -> act (loop: one tool per step) -> assess -> write -> report

* Planning: the LLM chooses tools from an allow-list; offline, a fixed investigation playbook is used.
* Assessment: deterministic typology rules over the evidence, backed by nearest past confirmed cases.
  If the alert looks unlike every known pattern, it is labelled "new_pattern" for the fraud strategy team.
* Writing: the LLM writes the narrative; every amount, account id and time in a sentence must appear in the
  evidence it cites, otherwise the sentence is removed (and a template narrative is used if none survive).
"""
from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from prahari.llm import PLAN_SYSTEM, WRITE_SYSTEM, LLMClient, LLMError, Usage

PLAYBOOK = ["get_payment", "model_reasons", "customer_profile", "device_and_sim", "recent_activity",
            "money_trail", "ring_membership", "similar_cases"]
TYPOLOGY_NAMES = {
    "account_takeover": "Account takeover (SIM swap / new device)",
    "collect_request_scam": "Collect-request scam (social engineering)",
    "mule_layering": "Mule account layering",
    "qr_code_swap": "QR code swap at a merchant",
    "new_pattern": "New pattern - not like known typologies",
    "unclear": "Unclear - needs analyst review",
}
SCENARIO_TO_TYPOLOGY = {"ato": "account_takeover", "collect_scam": "collect_request_scam",
                        "mule_layering": "mule_layering", "qr_swap": "qr_code_swap", "investment_scam": "new_pattern"}
ACTIONS = {
    "account_takeover": ["Keep the payment on hold and block further debits from the new device",
                         "Contact the customer on a verified channel (not the swapped SIM) and re-verify the SIM",
                         "Ask the beneficiary banks to freeze the receiving accounts",
                         "Guide the customer to report on 1930 / cybercrime.gov.in"],
    "collect_request_scam": ["Block the requesting VPA and report it to its PSP",
                             "Call the customer to confirm and warn about refund/KYC collect-request scams",
                             "Freeze onward transfers from the receiving account"],
    "mule_layering": ["Place a debit freeze on the account pending review",
                      "Trace and freeze the next accounts in the money trail",
                      "Escalate to the AML team to file an STR"],
    "qr_code_swap": ["Block the look-alike VPA and alert the genuine merchant",
                     "Notify customers who paid the look-alike VPA",
                     "Freeze onward transfers from the look-alike account"],
    "new_pattern": ["Hold the payment and escalate to the fraud strategy team",
                    "Label the case after review so the model learns the new pattern",
                    "Check other recent payments to the same payee"],
    "unclear": ["Analyst review required before any action"],
}


class InvState(TypedDict, total=False):
    txn_id: str
    plan: list
    evidence: Annotated[list, operator.add]
    typology: str
    typology_basis: str
    confidence: float
    narrative: list
    removed: list
    narrative_source: str
    report: dict
    usage: Usage
    trace: Annotated[list, operator.add]


def build_agent(tools: dict, llm: LLMClient, max_steps: int = 8):
    """`tools` maps a tool name to fn(txn_id) -> (title, summary_text, data)."""

    def plan(s: InvState) -> dict:
        t0 = time.time()
        chosen, source = list(PLAYBOOK), "playbook"
        if not llm.is_mock:
            try:
                raw = llm.complete_json(PLAN_SYSTEM.format(tools=", ".join(tools)),
                                        f"Flagged payment: {tools['get_payment'](s['txn_id'])[1]}",
                                        tier="small", step="plan", usage=s["usage"], max_tokens=200)
                picked = [t for t in raw.get("tools", []) if t in tools]
                if len(picked) >= 3:
                    chosen = list(dict.fromkeys(["get_payment"] + picked))[:max_steps]
                    source = "llm"
            except LLMError:
                pass
        return {"plan": chosen, "trace": [{"node": "plan", "source": source, "tools": chosen,
                                           "ms": round((time.time() - t0) * 1000, 1)}]}

    def act(s: InvState) -> dict:
        t0 = time.time()
        name, rest = s["plan"][0], s["plan"][1:]
        title, summary, data = tools[name](s["txn_id"])
        eid = f"E{len(s.get('evidence', [])) + 1}"
        return {"plan": rest, "evidence": [{"id": eid, "tool": name, "title": title, "summary": summary, "data": data}],
                "trace": [{"node": "act", "tool": name, "ms": round((time.time() - t0) * 1000, 1)}]}

    def assess(s: InvState) -> dict:
        typ, basis, conf = classify(s["evidence"])
        return {"typology": typ, "typology_basis": basis, "confidence": conf,
                "trace": [{"node": "assess", "typology": typ}]}

    def write(s: InvState) -> dict:
        t0 = time.time()
        template = template_narrative(s["evidence"], s["typology"])
        sentences, removed, source = template, [], "template"
        if not llm.is_mock:
            ev = "\n".join(f"[{e['id']}] {e['title']}: {e['summary']}" for e in s["evidence"])
            try:
                raw = llm.complete_json(WRITE_SYSTEM, f"Evidence:\n<<<\n{ev}\n>>>\n\nTypology assessed: "
                                        f"{TYPOLOGY_NAMES[s['typology']]}", tier="large", step="write",
                                        usage=s["usage"], max_tokens=900)
                kept, removed = verify(raw.get("sentences") or [], s["evidence"])
                if kept:
                    sentences, source = kept, "llm"
            except LLMError:
                pass
        return {"narrative": sentences, "removed": removed, "narrative_source": source,
                "trace": [{"node": "write", "source": source, "ms": round((time.time() - t0) * 1000, 1)}]}

    def report(s: InvState) -> dict:
        return {"report": str_draft(s), "trace": [{"node": "report"}]}

    g = StateGraph(InvState)
    for n, f in [("plan", plan), ("act", act), ("assess", assess), ("write", write), ("report", report)]:
        g.add_node(n, f)
    g.add_edge(START, "plan")
    g.add_edge("plan", "act")
    g.add_conditional_edges("act", lambda s: "act" if s.get("plan") else "assess", {"act": "act", "assess": "assess"})
    g.add_edge("assess", "write")
    g.add_edge("write", "report")
    g.add_edge("report", END)
    return g.compile()


# ---------------------------------------------------------------------------------------- assessment
def _ev(evidence: list, tool: str) -> dict:
    return next((e["data"] for e in evidence if e["tool"] == tool), {})


def classify(evidence: list) -> tuple[str, str, float]:
    pay, dev, feat = _ev(evidence, "get_payment"), _ev(evidence, "device_and_sim"), _ev(evidence, "model_reasons")
    f = feat.get("features", {})
    if dev.get("sim_swap_hours_before") is not None and dev["sim_swap_hours_before"] <= 48 and dev.get("device_is_new"):
        return "account_takeover", "SIM swapped and a new device used shortly before the payment", 0.9
    if pay.get("type") == "COLLECT" and (f.get("remark_scam_words", 0) > 0 or f.get("new_payee", 0) == 1):
        return "collect_request_scam", "collect request from an unknown VPA with a scam-style note", 0.85
    # pass-through: sends on most of what just came in (a victim's large payment after a small credit is not this)
    if 0.5 <= f.get("payer_out_vs_inflow_2h", 0) <= 1.2 and f.get("payer_inflow_2h_cnt", 0) >= 1:
        return "mule_layering", "the payer is passing on money received within the last 2 hours", 0.85
    if f.get("p2m_to_non_merchant", 0) == 1:
        return "qr_code_swap", "merchant-style QR payment to an account that is not a registered merchant", 0.8
    sim = _ev(evidence, "similar_cases")
    if sim.get("nearest") and sim["nearest"][0]["distance"] <= sim.get("known_pattern_distance", 0):
        votes: dict[str, int] = {}
        for c in sim["nearest"]:
            votes[c["typology"]] = votes.get(c["typology"], 0) + 1
        best = max(votes, key=votes.get)
        return best, f"closest to {votes[best]} of {len(sim['nearest'])} similar past confirmed cases", 0.6
    if sim.get("nearest"):
        return "new_pattern", "high risk but unlike any known fraud pattern", 0.5
    return "unclear", "not enough evidence", 0.3


# ------------------------------------------------------------------------------------------ narrative
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ACC = re.compile(r"\b(?:AC|TX|DV)\d{6}\b")


def _facts(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)} | set(_ACC.findall(text))


def verify(sentences: list, evidence: list) -> tuple[list, list]:
    by_id = {e["id"]: e["summary"] + " " + json.dumps(e["data"], default=str) for e in evidence}
    kept, removed = [], []
    for s in sentences:
        text = str(s.get("text", "")).strip()
        cites = [c for c in s.get("evidence") or [] if c in by_id]
        source = " ".join(by_id[c] for c in cites)
        missing = _facts(text) - _facts(source)
        if text and cites and not missing:
            kept.append({"text": text, "evidence": cites})
        elif text:
            removed.append({"text": text, "reason": "no valid evidence cited" if not cites
                            else f"not found in cited evidence: {sorted(missing)}"})
    return kept, removed


def template_narrative(evidence: list, typology: str) -> list:
    ids = {e["tool"]: e["id"] for e in evidence}
    out = []
    for tool in ("get_payment", "model_reasons", "device_and_sim", "recent_activity", "money_trail",
                 "ring_membership", "similar_cases"):
        e = next((x for x in evidence if x["tool"] == tool), None)
        if e and e["summary"]:
            out.append({"text": e["summary"], "evidence": [ids[tool]]})
    out.append({"text": f"Assessed typology: {TYPOLOGY_NAMES[typology]}.", "evidence": [ids.get("model_reasons", "E1")]})
    return out


def str_draft(s: InvState) -> dict:
    pay = _ev(s["evidence"], "get_payment")
    trail = _ev(s["evidence"], "money_trail")
    ring = _ev(s["evidence"], "ring_membership")
    subjects = [{"account": pay.get("payee"), "vpa": pay.get("payee_vpa"), "role": "receiving account"}]
    for n in trail.get("nodes", [])[2:]:
        subjects.append({"account": n["account"], "vpa": n["vpa"], "role": f"onward recipient (hop {n['hop']})"})
    txns = [{"txn_id": pay.get("txn_id"), "time": pay.get("time"), "amount": pay.get("amount")}] + \
        [{"txn_id": e["txn_id"], "time": e["ts"], "amount": e["amount"]} for e in trail.get("edges", [])[1:]]
    return {
        "title": "Suspicious Transaction Report - DRAFT for analyst approval",
        "reporting_entity": "Prahari demo PSP (fictional)",
        "victim_or_originator": {"account": pay.get("payer"), "vpa": pay.get("payer_vpa")},
        "subjects": subjects[:12],
        "transactions": txns[:20],
        "total_amount": round(sum(t["amount"] or 0 for t in txns[:20]), 2),
        "typology": TYPOLOGY_NAMES[s["typology"]],
        "grounds_for_suspicion": [x["text"] for x in s["narrative"]],
        "linked_ring": ring.get("ring"),
        "recommended_actions": ACTIONS[s["typology"]],
        "status": "draft - requires analyst review and approval before filing",
    }


def investigate(agent, txn_id: str) -> dict:
    t0 = time.time()
    s = agent.invoke({"txn_id": txn_id, "evidence": [], "trace": [], "usage": Usage()})
    return {"txn_id": txn_id, "typology": s["typology"], "typology_name": TYPOLOGY_NAMES[s["typology"]],
            "typology_basis": s["typology_basis"], "confidence": s["confidence"],
            "evidence": s["evidence"], "narrative": s["narrative"], "narrative_source": s["narrative_source"],
            "removed_sentences": s.get("removed", []), "report": s["report"], "usage": s["usage"].as_dict(),
            "trace": s["trace"], "latency_ms": int((time.time() - t0) * 1000)}
