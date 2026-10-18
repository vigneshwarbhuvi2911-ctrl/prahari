"""HTTP API and the fraud operations dashboard."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from prahari.config import Settings
from prahari.pipeline import Engine

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("prahari")


class Payment(BaseModel):
    payer: str = Field(pattern=r"^AC\d{6}$")
    payee: str = Field(pattern=r"^AC\d{6}$")
    amount: float = Field(gt=0, le=10_000_000)
    type: Literal["P2P", "P2M", "COLLECT"] = "P2P"
    device: str | None = Field(default=None, max_length=20)
    city: str | None = Field(default=None, max_length=40)
    remarks: str = Field(default="", max_length=120)


class Decision(BaseModel):
    decision: Literal["confirmed_fraud", "false_positive", "escalated"]
    note: str = Field(default="", max_length=500)


def create_app(engine: Engine | None = None, settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Prahari", version="1.0.0",
                  description="Real-time UPI fraud intelligence: streaming risk scoring, mule-ring detection and an "
                              "investigator agent. Personal demo on synthetic data.")
    state = {"engine": engine, "error": None}
    lock = threading.Lock()

    def eng() -> Engine:
        if state["engine"] is None:
            with lock:
                if state["engine"] is None:
                    try:
                        state["engine"] = Engine.load_or_build(settings or Settings(), log.info)
                    except Exception as exc:    # surface start-up problems through /health
                        state["error"] = str(exc)
                        raise
        return state["engine"]

    def nf(fn, *a):
        try:
            return fn(*a)
        except KeyError:
            raise HTTPException(404, "not found")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        e = eng()
        return {"status": "ok", "payments": len(e.ds.txns), "llm_provider": e.llm.provider}

    @app.get("/api/summary")
    def summary():
        e = eng()
        m = e.metrics
        return {"data": m["data"], "model": {k: v for k, v in m["model"].items() if k != "pr_curve"},
                "rules_baseline": m["rules_baseline"], "graph": m["graph"], "latency_ms": m["latency_ms"],
                "agent": m["agent"], "open_alerts": sum(1 for a in e.alerts(10_000) if a["status"] == "open"),
                "llm_provider": e.llm.provider}

    @app.get("/api/live")
    def live(n: int = 25):
        with lock:
            return eng().live_next(max(1, min(n, 200)))

    @app.get("/api/stream")
    async def stream(rate: int = 40, limit: int = 0):
        """Server-sent events: the test period replayed as a live stream.

        rate = payments per second; limit = stop after this many events (0 = run until the client disconnects).
        """
        e = eng()
        rate = max(1, min(rate, 400))

        async def gen():
            sent = 0
            while not limit or sent < limit:
                with lock:
                    batch = e.live_next(max(1, rate // 10))
                for ev in batch[: (limit - sent) if limit else None]:
                    yield f"data: {json.dumps(ev)}\n\n"
                    sent += 1
                await asyncio.sleep(0.1)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/score")
    def score(p: Payment):
        with lock:
            return nf(eng().score_payload, p.model_dump())

    @app.get("/api/accounts/examples")
    def examples():
        e = eng()
        acc = e.ds.accounts
        users = [a for a in acc.values() if a.kind == "user"][:3]
        shop = next(a for a in acc.values() if a.kind == "merchant")
        mules = [a for a in acc.values() if a.kind == "mule"]
        mule = next((a for a in mules if a.account_id in e.watchlist), mules[0])
        label = "mule account on the graph watchlist" if mule.account_id in e.watchlist else "mule account"
        return [{"account": a.account_id, "vpa": a.vpa, "label": lbl} for a, lbl in
                [(users[0], "customer"), (users[1], "customer"), (shop, "shop"), (mule, label)]]

    @app.get("/api/alerts")
    def alerts(limit: int = 100):
        return eng().alerts(max(1, min(limit, 1000)))

    @app.get("/api/cases/{txn_id}")
    def case(txn_id: str):
        with lock:
            return nf(eng().investigate, txn_id)

    @app.post("/api/cases/{txn_id}/decision")
    def decide(txn_id: str, d: Decision):
        return nf(eng().record_decision, txn_id, d.decision, d.note)

    @app.get("/api/rings")
    def rings():
        return eng().ring_view()

    @app.get("/api/trail/{txn_id}")
    def trail(txn_id: str):
        return nf(eng().graph.money_trail, txn_id)

    @app.get("/api/metrics")
    def metrics():
        return eng().metrics

    return app


@lru_cache
def get_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app()
