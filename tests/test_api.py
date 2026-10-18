import json
import os

import pytest
from fastapi.testclient import TestClient

from prahari.api import create_app
from tests.conftest import first_alert


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(create_app(engine))


def test_health_summary_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    s = client.get("/api/summary").json()
    assert s["model"]["hold"]["precision"] > 0.5 and "pr_curve" not in s["model"]
    m = client.get("/api/metrics").json()
    assert {"model", "rules_baseline", "by_scenario", "graph", "agent", "drift", "fairness", "latency_ms"} <= set(m)
    assert "Prahari" in client.get("/").text


def test_live_and_stream(client):
    ev = client.get("/api/live?n=5").json()
    assert len(ev) == 5 and {"decision", "risk", "latency_ms"} <= set(ev[0])
    r = client.get("/api/stream?rate=40&limit=3")
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert len(events) == 3 and all("decision" in ev for ev in events)


def test_score_a_takeover_style_payment(client, engine):
    ex = client.get("/api/accounts/examples").json()
    payer, mule = ex[0]["account"], ex[3]["account"]
    calm = client.post("/api/score", json={"payer": payer, "payee": ex[2]["account"], "amount": 450, "type": "P2M"}).json()
    risky = client.post("/api/score", json={"payer": payer, "payee": mule, "amount": 95000, "type": "P2P",
                                            "device": "DV-NEW-99"}).json()
    assert risky["risk"] > calm["risk"] and risky["decision"] != "ALLOW"
    assert any("device" in r["reason"] for r in risky["reasons"])
    assert all(r["feature"] == "graph_watchlist" or r["impact"] > 0 for r in risky["reasons"])
    assert client.post("/api/score", json={"payer": "AC999999", "payee": mule, "amount": 10}).status_code == 404
    assert client.post("/api/score", json={"payer": "bad", "payee": mule, "amount": 10}).status_code == 422


def test_alerts_cases_decisions(client, engine):
    alerts = client.get("/api/alerts?limit=50").json()
    assert alerts and all(a["status"] == "open" for a in alerts[:5])
    tid = first_alert(engine, "collect_scam").txn_id
    c = client.get(f"/api/cases/{tid}").json()
    assert c["typology"] == "collect_request_scam" and c["report"]["title"].startswith("Suspicious Transaction")
    d = client.post(f"/api/cases/{tid}/decision", json={"decision": "confirmed_fraud", "note": "called customer"}).json()
    assert d["decision"] == "confirmed_fraud"
    assert client.get(f"/api/cases/{tid}").json()["analyst"]["note"] == "called customer"
    assert client.post(f"/api/cases/{tid}/decision", json={"decision": "maybe"}).status_code == 422
    assert client.get("/api/cases/TX000000").status_code == 404


def test_rings_and_trail(client, engine):
    rings = client.get("/api/rings").json()
    assert rings and {"nodes", "edges", "members"} <= set(rings[0])
    t = first_alert(engine, "ato")
    assert client.get(f"/api/trail/{t.txn_id}").json()["hops"] >= 1
    assert client.get("/api/trail/NOPE").status_code == 404


def test_ci_uses_the_real_langgraph():
    from tests.conftest import REAL_LANGGRAPH
    if os.environ.get("CI"):
        assert REAL_LANGGRAPH
