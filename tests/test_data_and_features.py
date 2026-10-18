"""Simulator realism and the online feature store's point-in-time correctness."""
from collections import Counter
from datetime import timedelta

import numpy as np

from prahari.features import FEATURES, FeatureStore, new_store
from prahari.sim import SimConfig, Txn, simulate

F = {f: i for i, f in enumerate(FEATURES)}


def test_simulation_is_reproducible_and_labelled(small_ds):
    again = simulate(SimConfig.small())
    assert [(t.txn_id, t.amount) for t in again.txns[:200]] == [(t.txn_id, t.amount) for t in small_ds.txns[:200]]
    kinds = Counter(t.scenario for t in small_ds.txns)
    assert {"normal", "ato", "collect_scam", "qr_swap", "mule_layering", "investment_scam"} <= set(kinds)
    assert 0.002 < sum(t.is_fraud for t in small_ds.txns) / len(small_ds.txns) < 0.03
    assert all(t.ts == sorted(t.ts for t in small_ds.txns)[i] for i, t in enumerate(small_ds.txns[:50]))


def test_novel_attack_only_appears_late(small_ds):
    first = min(t.ts for t in small_ds.txns if t.scenario == "investment_scam")
    assert first >= small_ds.config.start + timedelta(days=small_ds.config.novel_from_day)


def test_normal_data_contains_fraud_look_alikes(small_ds):
    assert any(not fraud for _, _, fraud in small_ds.sim_swaps)              # legitimate SIM replacements
    users = [a for a in small_ds.accounts.values() if a.kind == "user"]
    assert any(a.open_ts > small_ds.config.start for a in users)             # legitimately new accounts
    big_legit = [t for t in small_ds.txns if not t.is_fraud and t.amount >= 50000]
    assert big_legit                                                         # large legitimate payments


def test_mule_rings_have_layers(small_ds):
    for rid, members in small_ds.rings.items():
        layers = Counter(small_ds.accounts[m].layer for m in members)
        assert layers[3] == 1 and layers[2] == 2 and layers[1] >= 3


def _store(ds) -> FeatureStore:
    return new_store(ds)


def test_features_are_point_in_time(small_ds):
    s = _store(small_ds)
    u, v = next(a for a in small_ds.accounts.values() if a.kind == "user"), \
        [a for a in small_ds.accounts.values() if a.kind == "user"][1]
    t1 = Txn("T1", small_ds.config.start + timedelta(days=1), u.account_id, v.account_id, 500, "P2P",
             small_ds.home_device[u.account_id], u.city)
    x1 = s.observe(t1)
    x2 = s.observe(Txn("T2", t1.ts + timedelta(minutes=5), u.account_id, v.account_id, 700, "P2P", t1.device, u.city))
    assert x1[F["new_payee"]] == 1 and x2[F["new_payee"]] == 0         # the payment itself is not "known" yet
    assert x1[F["payer_cnt_1h"]] == 0 and x2[F["payer_cnt_1h"]] == 1
    assert x2[F["device_new_for_payer"]] == 0


def test_sim_swap_and_pass_through_features(small_ds):
    s = _store(small_ds)
    users = [a for a in small_ds.accounts.values() if a.kind == "user"]
    a, b, c = users[0], users[1], users[2]
    t0 = small_ds.config.start + timedelta(days=2)
    s.sim_swap(a.account_id, t0)
    x = s.observe(Txn("A", t0 + timedelta(hours=2), a.account_id, b.account_id, 40000, "P2P", "DV-NEW", a.city))
    assert abs(x[F["sim_swap_hours"]] - 2.0) < 0.01 and x[F["device_new_for_payer"]] == 1
    y = s.observe(Txn("B", t0 + timedelta(hours=2, minutes=20), b.account_id, c.account_id, 38000, "P2P",
                      small_ds.home_device[b.account_id], b.city))
    assert y[F["payer_inflow_2h_cnt"]] == 1 and abs(y[F["payer_out_vs_inflow_2h"]] - 0.95) < 0.01


def test_live_replay_matches_training_features(engine):
    """No training/serving skew: the live path computes exactly the features the model was trained on."""
    engine.reset_live()
    events = engine.live_next(300)
    engine.reset_live()
    store = engine.live_store
    test = engine.test_txns()[:300]
    swaps, k = engine.ds.sim_swaps, engine.snapshot.next_swap
    for i, t in enumerate(test):
        while k < len(swaps) and swaps[k][1] <= t.ts:
            store.sim_swap(*swaps[k][:2])
            k += 1
        assert np.allclose(store.observe(t), engine.X[engine.test_offset + i], atol=1e-4), t.txn_id
    assert [e["txn_id"] for e in events] == [t.txn_id for t in test]
    engine.reset_live()
