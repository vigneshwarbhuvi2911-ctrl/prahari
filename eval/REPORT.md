# Prahari evaluation report

> All numbers are on **synthetic** data from the built-in simulator. They show that the system works end to end and how its parts compare; they are not a claim about real-world fraud rates.

- Payments simulated: **329,624** across 4,416 accounts (fraud rate 0.7%)
- Test period: 31 Jul - 14 Aug 2026 - 71,057 payments, 892 fraudulent
- Time-based split: warm-up 01 Jun - 14 Jun 2026, train 15 Jun - 15 Jul 2026, validation 16 Jul - 30 Jul 2026

## Quality gates

| Gate | Value | Target | Result |
|---|---|---|---|
| Model average precision | 0.9467 | >= 0.85 | PASS |
| HOLD precision | 0.999 | >= 0.9 | PASS |
| HOLD recall (model + graph) | 0.926 | >= 0.85 | PASS |
| HOLD + step-up recall | 0.963 | >= 0.9 | PASS |
| Graph lifts HOLD recall over model only | 0.225 | >= 0.05 | PASS |
| Model beats rules baseline on recall | 0.735 | >= 0.3 | PASS |
| Novel scam held (never in training) | 0.763 | >= 0.5 | PASS |
| Mule-ring account precision | 0.98 | >= 0.9 | PASS |
| Mule-ring account recall | 0.99 | >= 0.85 | PASS |
| Agent typology accuracy | 0.983 | >= 0.9 | PASS |
| Narrative sentences grounded | 1.0 | >= 1.0 | PASS |
| Scoring latency p95 (ms) | 16.97 | <= 50.0 | PASS |

## Decisions on the test period

| Approach | Precision | Recall | Value recall | Alerts / day | False positives / 10k |
|---|---|---|---|---|---|
| Rules baseline (hand-written) | 44.7% | 22.8% | 57.1% | 32.4 | 35.77 |
| Model only - HOLD | 100.0% | 70.1% | 77.5% | 44.6 | 0.0 |
| Model + graph - HOLD | 99.9% | 92.6% | 98.1% | 59.1 | 0.14 |
| Model + graph - HOLD or step-up | 54.5% | 96.3% | 98.8% | 112.5 | 102.05 |

Model average precision: **0.9467**. Graph escalations (STEP_UP raised to HOLD by the mule watchlist): 202.

## By fraud type

| Fraud type | Payments | Rules | Model only HOLD | Model + graph HOLD | HOLD or step-up |
|---|---|---|---|---|---|
| ato | 45 | 62.2% | 100.0% | 100.0% | 100.0% |
| collect_scam | 55 | 100.0% | 100.0% | 100.0% | 100.0% |
| investment_scam (new - not in training data) | 131 | 18.3% | 0.0% | 76.3% | 76.3% |
| mule_layering | 621 | 15.5% | 78.4% | 94.7% | 99.7% |
| qr_swap | 40 | 0.0% | 95.0% | 95.0% | 100.0% |

## Mule-ring detection

- Rings found: 13 (simulated: 14); accounts flagged: 100
- Account precision 98.0%, recall of active fraud accounts 99.0%, ring purity 87.9%
- Receiving accounts of the new scam linked to a ring: 3

## Investigator agent

- Cases investigated: 121; typology accuracy 98.3%; grounded narrative sentences 100.0%
- By fraud type: collect_scam 100.0%, mule_layering 97.6%, ato 100.0%, investment_scam 100.0%
  (the new scam is counted correct when it is labelled `new_pattern`)

## Latency (live path: features + model + policy, one payment at a time)

p50 10.75 ms, p95 16.97 ms, p99 20.1 ms over 3000 payments.

## Monitoring

- Drift (PSI, train vs test): **stable**
- Fairness by age band: **ok** (highest / lowest false-positive rate ratio 1.18)
- Fairness by city tier: **ok** (highest / lowest false-positive rate ratio 1.38)
