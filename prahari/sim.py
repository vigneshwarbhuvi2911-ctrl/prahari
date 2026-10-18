"""Synthetic UPI ecosystem with labelled fraud.

The simulator creates people, merchants, devices and SIM cards, generates ordinary payment behaviour
(daily rhythm, weekends, month-start rent, travel, phone upgrades), and injects six fraud typologies
seen in Indian digital payments:

  ato              SIM swap -> new device -> fast high-value transfers to mule accounts
  collect_scam     victim approves UPI collect requests from a fake "refund / KYC / cashback" VPA
  qr_swap          a look-alike VPA replaces a shop's QR code; customers pay the fraudster
  mule_layering    stolen money hops through 2 layers of mule accounts to a cash-out account
  investment_scam  NOVEL pattern, only in the last part of the timeline (not seen in training):
                   victims pay escalating "deposits" to fake investment VPAs over several days

Ordinary behaviour deliberately contains look-alikes of fraud signals (legitimate phone upgrades,
SIM replacements, travel, large rent payments, first-time payees) so the detection problem is not trivial.
Everything is generated from a seed and is fully reproducible. No real people, VPAs or banks.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

CITIES = [("Chennai", 1), ("Mumbai", 1), ("Bengaluru", 1), ("Delhi", 1), ("Hyderabad", 1), ("Kolkata", 1),
          ("Pune", 1), ("Coimbatore", 2), ("Madurai", 2), ("Kochi", 2), ("Jaipur", 2), ("Lucknow", 2),
          ("Indore", 2), ("Vijayawada", 2), ("Salem", 3), ("Tirunelveli", 3), ("Hubli", 3), ("Guntur", 3),
          ("Nashik", 3), ("Erode", 3)]
HANDLES = ["kaveri", "ganga", "yamuna", "godavari", "narmada"]          # fictional PSP handles
AGE_BANDS = ["18-25", "26-35", "36-50", "51-65", "65+"]
FIRST = ["Arun", "Priya", "Karthik", "Divya", "Rahul", "Anitha", "Suresh", "Meena", "Vijay", "Lakshmi", "Imran",
         "Fatima", "Joseph", "Mary", "Rohit", "Sneha", "Ganesh", "Kavya", "Manoj", "Deepa", "Arjun", "Pooja",
         "Sanjay", "Nisha", "Farhan", "Ayesha", "Vikram", "Revathi", "Naveen", "Harini"]
LAST = ["Kumar", "Raman", "Iyer", "Nair", "Reddy", "Sharma", "Gupta", "Khan", "Pillai", "Menon", "Rao", "Das",
        "Singh", "Patel", "Joseph", "Shetty", "Naidu", "Ali", "Krishnan", "Bose"]
MERCHANT_CATS = {  # category: (median amount, sigma, share)
    "grocery": (650, 0.7, 0.30), "food": (380, 0.6, 0.18), "fuel": (1400, 0.5, 0.12), "pharmacy": (700, 0.8, 0.08),
    "ecommerce": (1600, 0.9, 0.12), "utilities": (1800, 0.6, 0.07), "electronics": (9000, 0.9, 0.04),
    "travel": (4200, 0.8, 0.05), "education": (16000, 0.6, 0.04)}
SHOP_WORDS = ["Sri Annapoorna", "Murugan", "Lakshmi", "Balaji", "Sai", "Ganesh", "Royal", "City", "Green",
              "New", "Star", "Krishna", "Anand", "Selvam", "Ambika"]
P2P_REMARKS = ["", "", "", "rent", "dinner", "thanks", "share", "fees", "loan return", "gift", "trip", "groceries"]
SCAM_REMARKS = {
    "collect_scam": ["KYC update pending", "refund for cancelled order", "cashback reward credited",
                     "electricity disconnection today", "parcel customs fee", "loan approval fee"],
    "investment_scam": ["investment", "deposit for trading plan", "profit withdrawal fee", "tax to release profit",
                        "upgrade to VIP plan"],
}
SCAM_VPA = {"collect_scam": ["refund.desk", "kyc.update", "cashback.rewards", "eb.helpline", "customs.clear",
                             "loan.approval"],
            "investment_scam": ["growth.fund", "stock.tips.pro", "crypto.yield", "wealth.club"]}


@dataclass
class SimConfig:
    seed: int = 7
    n_users: int = 4000
    n_merchants: int = 300
    days: int = 75
    start: datetime = datetime(2026, 6, 1)
    novel_from_day: int = 60            # the investment scam appears only from this day
    n_rings: int = 14
    ato_victims: int = 70
    collect_victims: int = 150
    qr_swaps: int = 10
    investment_victims: int = 30
    daily_rate: float = 0.9             # average payments per user per day

    @classmethod
    def small(cls, seed: int = 11) -> "SimConfig":
        """Fast configuration for tests."""
        return cls(seed=seed, n_users=700, n_merchants=60, days=40, novel_from_day=32, n_rings=6, ato_victims=18,
                   collect_victims=30, qr_swaps=3, investment_victims=8)


@dataclass
class Account:
    account_id: str
    vpa: str
    name: str
    kind: str                 # user | merchant | mule | scammer | cashout
    city: str
    tier: int
    age_band: str
    open_ts: datetime
    category: str = ""        # merchants only
    ring_id: str = ""
    layer: int = 0            # mule layer (1, 2); cash-out = 3


@dataclass
class Txn:
    txn_id: str
    ts: datetime
    payer: str
    payee: str
    amount: float
    txn_type: str             # P2P | P2M | COLLECT
    device: str
    city: str
    remarks: str = ""
    is_fraud: int = 0
    scenario: str = "normal"
    ring_id: str = ""
    case_id: str = ""         # links all transactions of one fraud incident


@dataclass
class Dataset:
    config: SimConfig
    accounts: dict[str, Account]
    txns: list[Txn]
    sim_swaps: list[tuple[str, datetime, bool]]        # (account, time, is_fraudulent)
    device_first_seen: dict[str, datetime]
    home_device: dict[str, str]
    rings: dict[str, list[str]] = field(default_factory=dict)

    def split_days(self) -> tuple[datetime, datetime]:
        """(validation start, test start): train = first 60%, validation = next 20%, test = last 20%."""
        d = self.config.days
        return (self.config.start + timedelta(days=int(d * 0.6)), self.config.start + timedelta(days=int(d * 0.8)))


class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.r = random.Random(cfg.seed)
        self.accounts: dict[str, Account] = {}
        self.txns: list[Txn] = []
        self.sim_swaps: list[tuple[str, datetime, bool]] = []
        self.dev_first: dict[str, datetime] = {}
        self.home_dev: dict[str, str] = {}
        self.cur_dev: dict[str, list[tuple[datetime, str]]] = {}   # account -> [(from_ts, device)]
        self.rings: dict[str, list[str]] = {}
        self._n_txn = 0
        self._n_dev = 0
        self._n_case = 0

    # ------------------------------------------------------------------------------------- helpers
    def _id(self, prefix: str, n: int) -> str:
        return f"{prefix}{n:06d}"

    def _ts(self, day: int, hour: float) -> datetime:
        return self.cfg.start + timedelta(days=day, hours=hour)

    def _device(self, ts: datetime) -> str:
        self._n_dev += 1
        d = f"DV{self._n_dev:06d}"
        self.dev_first[d] = ts
        return d

    def _device_at(self, acc: str, ts: datetime) -> str:
        dev = self.home_dev[acc]
        for t, d in self.cur_dev.get(acc, []):
            if t <= ts:
                dev = d
        return dev

    def _lognormal(self, median: float, sigma: float) -> float:
        return max(1.0, round(math.exp(math.log(median) + self.r.gauss(0, sigma)), 0))

    def _name(self) -> str:
        return f"{self.r.choice(FIRST)} {self.r.choice(LAST)}"

    def _add_account(self, kind: str, open_ts: datetime, city=None, name=None, vpa=None, **kw) -> Account:
        n = len(self.accounts) + 1
        city_name, tier = city or self.r.choice(CITIES)
        name = name or self._name()
        vpa = vpa or f"{name.lower().replace(' ', '.')}{self.r.randint(1, 999)}@{self.r.choice(HANDLES)}"
        a = Account(self._id("AC", n), vpa, name, kind, city_name, tier, self.r.choice(AGE_BANDS), open_ts, **kw)
        self.accounts[a.account_id] = a
        self.home_dev[a.account_id] = self._device(open_ts)
        return a

    def _txn(self, ts, payer, payee, amount, ttype, remarks="", fraud=0, scenario="normal", ring="", case="",
             device=None, city=None) -> Txn:
        self._n_txn += 1
        t = Txn(self._id("TX", self._n_txn), ts, payer, payee, float(round(amount, 2)), ttype,
                device or self._device_at(payer, ts), city or self.accounts[payer].city, remarks, fraud, scenario,
                ring, case)
        self.txns.append(t)
        return t

    def _case(self) -> str:
        self._n_case += 1
        return f"CASE{self._n_case:05d}"

    def _hour(self) -> float:
        x = self.r.random()
        if x < 0.30:
            return self.r.uniform(8, 11.5)
        if x < 0.55:
            return self.r.uniform(12, 15.5)
        if x < 0.95:
            return self.r.uniform(17, 22.5)
        return self.r.uniform(0, 7.5) if self.r.random() < 0.6 else self.r.uniform(22.5, 23.99)

    # ------------------------------------------------------------------------------------ population
    def build_population(self) -> None:
        c = self.cfg
        weights = [6 if t == 1 else 3 if t == 2 else 1.5 for _, t in CITIES]
        self.users, self.merchants = [], []
        cats = list(MERCHANT_CATS)
        for _ in range(c.n_merchants):
            cat = self.r.choices(cats, [MERCHANT_CATS[k][2] for k in cats])[0]
            shop = f"{self.r.choice(SHOP_WORDS)} {cat.title()}"
            vpa = f"{shop.lower().replace(' ', '')}{self.r.randint(1, 99)}@{self.r.choice(HANDLES)}"
            city = self.r.choices(CITIES, weights)[0]
            opened = (c.start + timedelta(days=self.r.randint(-20, c.days - 10)) if self.r.random() < 0.05
                      else c.start - timedelta(days=self.r.randint(200, 3000)))     # some shops are new
            m = self._add_account("merchant", opened, city, shop, vpa, category=cat)
            self.merchants.append(m.account_id)
        by_city: dict[str, list[str]] = {}
        for m in self.merchants:
            by_city.setdefault(self.accounts[m].city, []).append(m)
        online = [m for m in self.merchants if self.accounts[m].category in ("ecommerce", "utilities", "travel")]
        for _ in range(c.n_users):
            city = self.r.choices(CITIES, weights)[0]
            # 4% of customers open their account during the period: legitimate "new account" look-alikes
            opened = (c.start + timedelta(days=self.r.randint(-30, c.days - 10)) if self.r.random() < 0.04
                      else c.start - timedelta(days=self.r.randint(60, 3500)))
            u = self._add_account("user", opened, city)
            self.users.append(u.account_id)
        self.profile = {}
        for u in self.users:
            a = self.accounts[u]
            local = by_city.get(a.city) or self.merchants
            self.profile[u] = {
                "rate": c.daily_rate * self.r.lognormvariate(0, 0.5),
                "shops": self.r.sample(local, min(6, len(local))) + self.r.sample(online, min(2, len(online))),
                "contacts": [],
                "p2p_median": self.r.choice([400, 800, 1500, 2500, 4000]),
                "landlord": None,
            }
        for u in self.users:
            same_city = [v for v in self.users if self.accounts[v].city == self.accounts[u].city and v != u]
            pool = same_city if len(same_city) > 10 else self.users
            self.profile[u]["contacts"] = self.r.sample(pool, min(8, len(pool)))
            if self.r.random() < 0.35:
                self.profile[u]["landlord"] = self.r.choice(pool)

    # ----------------------------------------------------------------------------- normal behaviour
    def normal_activity(self) -> None:
        c = self.cfg
        # legitimate look-alikes of fraud signals
        for u in self.r.sample(self.users, int(len(self.users) * 0.02)):           # phone upgrade
            ts = self._ts(self.r.randrange(c.days), self.r.uniform(9, 21))
            self.cur_dev.setdefault(u, []).append((ts, self._device(ts)))
        for u in self.r.sample(self.users, int(len(self.users) * 0.004)):          # SIM replacement + new phone
            ts = self._ts(self.r.randrange(c.days), self.r.uniform(10, 18))
            self.sim_swaps.append((u, ts, False))
            self.cur_dev.setdefault(u, []).append((ts + timedelta(hours=self.r.uniform(1, 30)), self._device(ts)))
        self.night_owls = set(self.r.sample(self.users, int(len(self.users) * 0.05)))   # shift workers
        travel: dict[str, tuple[int, int, str]] = {}
        for u in self.r.sample(self.users, int(len(self.users) * 0.04)):
            d0 = self.r.randrange(c.days - 6)
            travel[u] = (d0, d0 + self.r.randint(2, 6), self.r.choice(CITIES)[0])
        for day in range(c.days):
            weekday = (c.start + timedelta(days=day)).weekday()
            dom = (c.start + timedelta(days=day)).day
            boost = (1.15 if weekday >= 5 else 1.0) * (1.25 if dom <= 5 else 1.0)
            for u in self.users:
                p = self.profile[u]
                n = self._poisson(p["rate"] * boost)
                trip = travel.get(u)
                city = trip[2] if trip and trip[0] <= day <= trip[1] else None
                if self.accounts[u].open_ts > self._ts(day, 23.99):
                    continue
                for _ in range(n):
                    hour = self.r.uniform(0, 6) if u in self.night_owls and self.r.random() < 0.4 else self._hour()
                    ts = self._ts(day, hour)
                    if ts >= self.accounts[u].open_ts:
                        self._normal_txn(u, ts, city)
                if self.r.random() < 0.0015:     # occasional big legitimate transfer (gold, deposit, family)
                    v = self.r.choice(self.profile[u]["contacts"] + self.merchants)
                    if self._open(v, self._ts(day, 12)):
                        self._txn(self._ts(day, self._hour()), u, v, self.r.choice([50000, 75000, 100000, 150000, 200000]),
                                  "P2P" if self.accounts[v].kind == "user" else "P2M", self.r.choice(["", "gold", "advance", "family"]),
                                  city=city)
                if p["landlord"] and dom == 3 and self.r.random() < 0.9 and self._open(p["landlord"], self._ts(day, 9)):
                    self._txn(self._ts(day, self.r.uniform(9, 20)), u, p["landlord"],
                              self.r.choice([8000, 10000, 12000, 15000, 18000, 25000]), "P2P", "rent")

    def _spread(self, i: int, n: int, lo: int, hi: int) -> int:
        """Spreads incidents evenly over the timeline, so every period (train, validation, test) has each type."""
        return min(hi - 1, lo + int((i + self.r.random()) * (hi - lo) / max(1, n)))

    def _poisson(self, lam: float) -> int:
        l, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.r.random()
            if p <= l:
                return k
            k += 1

    def _open(self, acc: str, ts: datetime) -> bool:
        return self.accounts[acc].open_ts <= ts

    def _normal_txn(self, u: str, ts: datetime, city: str | None) -> None:
        p = self.profile[u]
        x = self.r.random()
        new_payee = self.r.random() < 0.06
        if x < 0.62:
            m = self.r.choice(self.merchants) if new_payee else self.r.choice(p["shops"])
            if not self._open(m, ts):
                return
            med, sig, _ = MERCHANT_CATS[self.accounts[m].category]
            self._txn(ts, u, m, self._lognormal(med, sig), "P2M", city=city)
        elif x < 0.95:
            v = self.r.choice(self.users) if new_payee else self.r.choice(p["contacts"])
            if not self._open(v, ts):
                return
            self._txn(ts, u, v, self._lognormal(p["p2p_median"], 0.9), "P2P", self.r.choice(P2P_REMARKS), city=city)
        else:   # legitimate collect request from a utility or online merchant
            online = [m for m in p["shops"] if self.accounts[m].category in ("ecommerce", "utilities", "travel")]
            m = self.r.choice(online or p["shops"])
            if not self._open(m, ts):
                return
            med, sig, _ = MERCHANT_CATS[self.accounts[m].category]
            self._txn(ts, u, m, self._lognormal(med, sig), "COLLECT", self.r.choice(["bill payment", "order", ""]),
                      city=city)

    # ------------------------------------------------------------------------------------ mule rings
    def build_rings(self) -> None:
        c = self.cfg
        for k in range(c.n_rings):
            rid = f"RING{k + 1:02d}"
            start_day = self._spread(k, c.n_rings, 0, c.days - 10)
            act = self._ts(start_day, 0)
            members = []
            city = self.r.choice(CITIES)
            for layer, count in ((1, self.r.randint(3, 4)), (2, 2), (3, 1)):
                for _ in range(count):
                    kind = "cashout" if layer == 3 else "mule"
                    # about half the mules are recently opened; the rest are older accounts rented or sold to
                    # the gang, with a normal-looking history
                    aged = self.r.random() < 0.5
                    opened = act - timedelta(days=self.r.randint(200, 2000) if aged else self.r.randint(3, 40))
                    a = self._add_account(kind, opened, city, ring_id=rid, layer=layer)
                    members.append(a.account_id)
                    span = max(1.0, (c.start + timedelta(days=c.days) - max(a.open_ts, c.start)).days)
                    for _ in range(self.r.randint(6, 20) if aged else self.r.randint(2, 8)):   # camouflage
                        ts = max(a.open_ts, c.start) + timedelta(days=self.r.uniform(0, span), hours=self.r.uniform(8, 21))
                        if self.r.random() < 0.6:
                            m = self.r.choice(self.merchants)
                            med, sig, _ = MERCHANT_CATS[self.accounts[m].category]
                            self._txn(ts, a.account_id, m, self._lognormal(med, sig), "P2M")
                        else:
                            self._txn(ts, a.account_id, self.r.choice(self.users), self._lognormal(1200, 0.8), "P2P",
                                      self.r.choice(P2P_REMARKS))
            self.rings[rid] = members
        self.ring_window = {}
        for rid, mem in self.rings.items():
            first = min(self.accounts[m].open_ts for m in mem)
            self.ring_window[rid] = first + timedelta(days=40)

    def _active_ring(self, ts: datetime) -> str:
        cands = [rid for rid, mem in self.rings.items()
                 if max(self.accounts[m].open_ts for m in mem) <= ts]
        return self.r.choice(cands or list(self.rings))

    def launder(self, ts: datetime, amount: float, rid: str, case: str, entry: str | None = None,
                source: str | None = None) -> None:
        """Moves stolen money through the ring, each hop within minutes:
        [source (fake VPA) ->] layer-1 mule -> layer-2 mule -> cash-out."""
        mem = self.rings[rid]
        l1 = [m for m in mem if self.accounts[m].layer == 1]
        l2 = [m for m in mem if self.accounts[m].layer == 2]
        out = [m for m in mem if self.accounts[m].layer == 3]
        if source:
            src, hops = source, [self.r.choice(l1), self.r.choice(l2), out[0]]
        else:
            src = entry or self.r.choice(l1)
            hops = [self.r.choice(l2), out[0]]
        t, amt = ts, amount
        for dst in hops:
            t = t + timedelta(minutes=self.r.uniform(4, 95))
            amt = math.floor(amt * self.r.uniform(0.86, 0.98) / 100) * 100
            if amt < 100:
                return
            parts = 2 if amt > 60000 and self.r.random() < 0.5 else 1
            for i in range(parts):
                self._txn(t + timedelta(minutes=3 * i), src, dst, amt / parts, "P2P", "", 1, "mule_layering", rid, case)
            src = dst

    # -------------------------------------------------------------------------------- fraud scenarios
    def inject_fraud(self) -> None:
        c = self.cfg
        train_days = c.novel_from_day
        # account takeover
        old_users = [u for u in self.users if self.accounts[u].open_ts < c.start]
        for i, v in enumerate(self.r.sample(old_users, c.ato_victims)):
            case = self._case()
            day = self._spread(i, c.ato_victims, 2, c.days - 2)
            swap = self._ts(day, self.r.uniform(0, 23))
            rid = self._active_ring(swap)
            self.sim_swaps.append((v, swap, True))
            t = swap + timedelta(hours=self.r.uniform(0.5, 6))
            dev = self._device(t)
            city = self.r.choice(CITIES)[0] if self.r.random() < 0.5 else self.accounts[v].city
            mules = [m for m in self.rings[rid] if self.accounts[m].layer == 1]
            for _ in range(self.r.randint(2, 4)):
                amt = self.r.choice([15000, 20000, 25000, 40000, 49000, 75000, 99000, 150000])
                mule = self.r.choice(mules)
                self._txn(t, v, mule, amt, "P2P", "", 1, "ato", rid, case, device=dev, city=city)
                self.launder(t, amt, rid, case, entry=mule)
                t += timedelta(minutes=self.r.uniform(3, 40))
        # collect-request scam through a fake helpdesk VPA
        scam_pool = []
        n_scam = max(4, c.collect_victims // 12)
        for k in range(n_scam):
            kind = self.r.choice(SCAM_VPA["collect_scam"])
            active = c.start + timedelta(days=self._spread(k, n_scam, 0, c.days - 10))
            opened = active - timedelta(days=self.r.randint(150, 900)) if self.r.random() < 0.4 else active
            a = self._add_account("scammer", opened, name="Support Desk",
                                  vpa=f"{kind}{self.r.randint(10, 99)}@{self.r.choice(HANDLES)}")
            self.scam_active = getattr(self, "scam_active", {}) | {a.account_id: active}
            scam_pool.append(a.account_id)
        for v in self.r.sample(old_users, c.collect_victims):
            s = self.r.choice(scam_pool)
            open_day = (self.scam_active[s] - c.start).days
            day = self.r.randrange(open_day + 1, min(c.days, open_day + 10))
            t, case = self._ts(day, self._hour()), self._case()
            rid = self._active_ring(t)
            remark = self.r.choice(SCAM_REMARKS["collect_scam"])
            for _ in range(self.r.choice([1, 1, 2, 3])):
                amt = self.r.choice([1999, 4999, 9999, 14999, 24999, 49999, 2500, 7500])
                self._txn(t, v, s, amt, "COLLECT", remark, 1, "collect_scam", rid, case)
                t += timedelta(minutes=self.r.uniform(2, 25))
            self.launder(t + timedelta(minutes=10), amt * 0.9, rid, case, source=s)
        # QR code swap at a shop
        established = [m for m in self.merchants if self.accounts[m].open_ts < c.start]
        for i, m in enumerate(self.r.sample(established, c.qr_swaps)):
            shop = self.accounts[m]
            fake_vpa = shop.vpa.split("@")[0].replace("0", "").rstrip("0123456789") + \
                f"{self.r.randint(1, 9)}@{self.r.choice([h for h in HANDLES if not shop.vpa.endswith(h)])}"
            day = self._spread(i, c.qr_swaps, 5, c.days - 3)
            fake = self._add_account("scammer", self._ts(day - 1, 10), (shop.city, shop.tier), shop.name, fake_vpa)
            customers = [u for u in self.users if m in self.profile[u]["shops"]] or self.r.sample(self.users, 20)
            case = self._case()
            total, t = 0.0, self._ts(day, 8)
            for u in self.r.sample(customers, min(len(customers), self.r.randint(8, 20))):
                t = t + timedelta(minutes=self.r.uniform(10, 120))
                med, sig, _ = MERCHANT_CATS[shop.category]
                amt = self._lognormal(med, sig)
                total += amt
                self._txn(t, u, fake.account_id, amt, "P2M", "", 1, "qr_swap", "", case)
            rid = self._active_ring(t)
            self.launder(t + timedelta(minutes=30), total * 0.95, rid, case, source=fake.account_id)
        # NOVEL: investment scam, only near the end of the timeline
        inv = []
        for k in range(max(2, c.investment_victims // 8)):
            kind = self.r.choice(SCAM_VPA["investment_scam"])
            a = self._add_account("scammer", c.start - timedelta(days=self.r.randint(300, 1500)), name="Wealth Advisory",
                                  vpa=f"{kind}{self.r.randint(10, 99)}@{self.r.choice(HANDLES)}")
            inv.append(a.account_id)
        for v in self.r.sample(old_users, c.investment_victims):
            s = self.r.choice(inv)
            day = self.r.randrange(train_days, c.days - 3)
            amt, case = self.r.choice([2000, 5000, 10000]), self._case()
            for i in range(self.r.randint(3, 6)):
                if day + i >= c.days:
                    break
                t = self._ts(day + i, self._hour())
                self._txn(t, v, s, amt, "P2P", self.r.choice(SCAM_REMARKS["investment_scam"]), 1, "investment_scam",
                          "", case)
                rid = self._active_ring(t)
                self.launder(t + timedelta(minutes=40), amt * 0.9, rid, case, source=s)
                amt = min(200000, amt * self.r.choice([1.5, 2, 2.5]))

    def run(self) -> Dataset:
        self.build_population()
        self.normal_activity()
        self.build_rings()
        self.inject_fraud()
        end = self.cfg.start + timedelta(days=self.cfg.days)
        self.txns = sorted((t for t in self.txns if self.cfg.start <= t.ts < end), key=lambda t: (t.ts, t.txn_id))
        return Dataset(self.cfg, self.accounts, self.txns, sorted(self.sim_swaps, key=lambda s: s[1]),
                       self.dev_first, self.home_dev, self.rings)


def simulate(cfg: SimConfig | None = None) -> Dataset:
    return Simulator(cfg or SimConfig()).run()
