"""Graph intelligence: follow the money and uncover mule rings.

A single-payment model sees one edge at a time. Fraud gangs are networks: stolen money lands in a first
layer of mule accounts and is passed on within minutes to a second layer and then to a cash-out account.
This module:

* builds a payment graph for a time window;
* follows the money from any payment (only hops that move most of the money on, quickly);
* scores accounts for mule behaviour (fast pass-through, many unrelated senders, account age);
* grows rings from the model's alerts through fast onward transfers, then splits them into communities
  (Louvain), so analysts get one case per gang instead of hundreds of separate alerts.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
from networkx.algorithms.community import louvain_communities

from prahari.sim import Account, Txn

FAST = timedelta(hours=3)


class PaymentGraph:
    def __init__(self, txns: list[Txn], accounts: dict[str, Account]):
        self.accounts = accounts
        self.txns = {t.txn_id: t for t in txns}
        self.out: dict[str, list[Txn]] = defaultdict(list)
        self.inn: dict[str, list[Txn]] = defaultdict(list)
        for t in txns:                          # txns arrive in time order
            self.out[t.payer].append(t)
            self.inn[t.payee].append(t)
        self._out_ts = {a: [t.ts for t in lst] for a, lst in self.out.items()}

    # --------------------------------------------------------------------------------- money trail
    def forwards(self, account: str, after: datetime, amount: float, within: timedelta = FAST) -> list[Txn]:
        """Payments out of `account` soon after it received `amount` that pass on at least 30% of it."""
        ts = self._out_ts.get(account, [])
        i = bisect_left(ts, after)
        out = []
        for t in self.out[account][i:]:
            if t.ts - after > within:
                break
            if t.amount >= 0.3 * amount and self.accounts.get(t.payee, None) and \
                    self.accounts[t.payee].kind != "merchant":
                out.append(t)
        return out

    def money_trail(self, txn_id: str, depth: int = 4) -> dict:
        """Hop-by-hop path of the money from one payment: nodes, edges and the share that moved on."""
        start = self.txns[txn_id]
        nodes = {start.payer: 0, start.payee: 1}
        edges = [_edge(start, 0)]
        frontier = [(start.payee, start.ts, start.amount, 1)]
        seen = {start.txn_id}
        while frontier:
            acc, ts, amt, d = frontier.pop(0)
            if d > depth:
                continue
            for t in self.forwards(acc, ts, amt):
                if t.txn_id in seen:
                    continue
                seen.add(t.txn_id)
                nodes.setdefault(t.payee, d + 1)
                edges.append(_edge(t, d))
                frontier.append((t.payee, t.ts, t.amount, d + 1))
        moved = sum(e["amount"] for e in edges if e["hop"] == 1)
        return {"origin": txn_id, "nodes": [self.node(a, h) for a, h in nodes.items()], "edges": edges,
                "hops": max(nodes.values()), "share_moved_on": round(moved / start.amount, 3) if start.amount else 0.0}

    def node(self, acc: str, hop: int | None = None) -> dict:
        a = self.accounts.get(acc)
        d = {"account": acc, "vpa": a.vpa if a else acc, "kind_hint": a.kind if a and a.kind in ("merchant",) else "",
             **self.profile(acc)}
        if hop is not None:
            d["hop"] = hop
        return d

    # ------------------------------------------------------------------------------- mule scoring
    def profile(self, acc: str, until: datetime | None = None) -> dict:
        inn = [t for t in self.inn.get(acc, []) if until is None or t.ts <= until]
        out = [t for t in self.out.get(acc, []) if until is None or t.ts <= until]
        in_amt = sum(t.amount for t in inn)
        fast_out = 0.0
        for t in inn:
            fast_out += sum(f.amount for f in self.forwards(acc, t.ts, t.amount))
        a = self.accounts.get(acc)
        ref = until or (max([t.ts for t in inn + out]) if inn or out else None)
        return {"in_count": len(inn), "out_count": len(out), "in_amount": round(in_amt, 2),
                "out_amount": round(sum(t.amount for t in out), 2),
                "distinct_senders": len({t.payer for t in inn}),
                "pass_through": round(min(fast_out / in_amt, 1.0), 3) if in_amt else 0.0,
                "account_age_days": (ref - a.open_ts).days if a and ref else None}

    def mule_score(self, acc: str) -> float:
        p = self.profile(acc)
        if p["in_amount"] < 5000:
            return 0.0
        s = 0.55 * p["pass_through"] + 0.25 * min(p["distinct_senders"] / 5, 1.0)
        age = p["account_age_days"]
        if age is not None and age < 90:
            s += 0.2
        return round(min(s, 1.0), 3)

    # ------------------------------------------------------------------------------- ring detection
    def rings(self, alerts: list[Txn], min_size: int = 3, threshold: float = 0.45) -> list[dict]:
        """Grow from alerted payments through fast onward transfers; keep accounts that behave like mules."""
        g = nx.Graph()
        suspects: dict[str, float] = {}

        def score(acc: str) -> float:
            if acc not in suspects:
                suspects[acc] = self.mule_score(acc)
            return suspects[acc]

        for t in alerts:
            trail = self.money_trail(t.txn_id)
            if trail["hops"] < 2 and score(t.payee) < threshold:
                continue      # money did not move on quickly and the payee does not look like a mule
            for e in trail["edges"]:
                a, b = e["payer"], e["payee"]
                score(b)
                if e["hop"] == 0:
                    g.add_node(b)                  # victim -> first account: the victim is not a member
                else:                              # fast onward transfer of stolen money: same gang
                    w = g.get_edge_data(a, b, {}).get("weight", 0) + e["amount"]
                    g.add_edge(a, b, weight=w)
                    score(a)
        out = []
        for comp in nx.connected_components(g):
            sub = g.subgraph(comp)
            parts = louvain_communities(sub, weight="weight", seed=1) if len(comp) > 12 else [comp]
            for members in parts:
                if len(members) < min_size:
                    continue
                members = sorted(members)
                mem = set(members)
                victims = {t.payer for t in alerts if t.payee in mem and t.payer not in mem}
                inflow = sum(t.amount for t in alerts if t.payee in mem and t.payer not in mem)
                out.append({"ring": f"R-{len(out) + 1:03d}", "members": members, "size": len(members),
                            "victims": len(victims), "stolen_inflow": round(inflow, 2),
                            "edges": [{"from": u, "to": v, "amount": round(d["weight"], 2)}
                                      for u, v, d in g.subgraph(members).edges(data=True)],
                            "scores": {m: suspects[m] for m in members}})
        return sorted(out, key=lambda r: -r["stolen_inflow"])


    def feeders(self, members: set[str], since: datetime) -> set[str]:
        """Accounts (not in the ring) that pass most of the money they receive into ring members within hours:
        typically the fake VPAs scammers give their victims."""
        out = set()
        for m in members:
            for t in self.inn.get(m, []):
                if t.ts < since or t.payer in members or t.payer in out:
                    continue
                a = self.accounts.get(t.payer)
                if not a or a.kind == "merchant":
                    continue
                received = [r for r in self.inn.get(t.payer, []) if timedelta(0) <= t.ts - r.ts <= FAST]
                got = sum(r.amount for r in received)
                if got and t.amount >= 0.5 * min(got, t.amount / 0.5):
                    out.add(t.payer)
        return out


def _edge(t: Txn, hop: int) -> dict:
    return {"txn_id": t.txn_id, "payer": t.payer, "payee": t.payee, "amount": t.amount,
            "ts": t.ts.isoformat(timespec="minutes"), "hop": hop}
