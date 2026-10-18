# Prahari design notes

This document explains how Prahari works and why it was built this way. The README has the results and run instructions.

## 1. The problem

A UPI fraud team has three jobs, and Prahari covers all three:

1. **Decide in milliseconds** whether to allow a payment, ask the customer to confirm it (step-up), or hold it.
2. **Find the network behind the fraud.** Stolen money is moved quickly through layers of mule accounts and then cashed out. Looking at one payment at a time misses this.
3. **Investigate and report.** Every held payment needs a case file. Confirmed fraud needs a Suspicious Transaction Report (STR) and fast actions, such as freezing receiving accounts and guiding the customer to the 1930 helpline.

## 2. Simulated UPI ecosystem (`sim.py`)

There is no public UPI fraud dataset with payment-level graphs and device data, so Prahari generates one. The default world has 4,000 customers, 300 merchants and 75 days of payments, about 330,000 payments in total. Customers have home cities, devices, spending habits and regular payees.

**Fraud types**

| Type | How it is simulated | Main signals |
|---|---|---|
| Account takeover | SIM swap, then a new device drains the account within hours | hours since SIM swap, new device, amount vs history |
| Collect-request scam | Unknown VPA sends a "refund / KYC" collect request | COLLECT type, new payee, scam words in the note |
| QR code swap | Look-alike VPA replaces a shop's QR | merchant-style payment to a non-merchant |
| Mule layering | Layer-1 mules forward 85-98% of each credit within hours to layer 2, then to a cash-out account | pass-through ratio, fan-in from many senders |
| **Investment scam (new)** | Appears only in the test period. Victims pay "growth fund" VPAs voluntarily from their usual device | looks normal to the model. Caught through the graph |

**Hard negatives.** Normal data deliberately contains legitimate behaviour that looks like fraud, so the model cannot learn shortcuts:

- phone upgrades and legitimate SIM replacements
- travel to other cities
- night-owl customers
- large one-off transfers (rent deposits, family support)
- new accounts and new shops

Half the mules are **aged accounts** with months of normal-looking activity. Many scam VPAs are aged too.

## 3. Streaming feature store (`features.py`)

`FeatureStore.observe(txn)` returns the 31 features for a payment using only past state, then updates that state. The state uses small deques and running (Welford) statistics, so each payment costs O(1) work.

The **same code** builds the training matrix and serves live scoring. This removes training/serving skew. A test replays the live stream and checks that it produces exactly the training matrix. The store is snapshotted at the start of the test period so the live replay can resume from there.

**Feature groups**

- **Amount:** log amount, z-score against the payer's history, ratio to the payer's largest payment, round amount.
- **Time:** hour of day, night.
- **Type:** P2P, P2M, collect.
- **Payer velocity:** payments in the last 1 hour and 24 hours, unique payees in 24 hours.
- **Relationship:** new payee, payee account age, payer account age.
- **Merchant:** payee is a merchant, P2M payment to a non-merchant.
- **Device and SIM:** device new for this payer, device age, hours since SIM swap, city mismatch.
- **Note:** scam words present.
- **Payee inflow:** payments received in the last hour, distinct senders in 24 hours, amount received, out/in ratio.
- **Pass-through:** credits received in the last 2 hours, this payment as a share of those credits.

A cumulative "payer history count" feature was removed because it grows over time and caused drift between training and test. A 14-day warm-up lets every account build history before training starts.

## 4. Models and policy (`model.py`)

**Time-based split:** warm-up (14 days), then training, validation and test, in that order.

- **Supervised model.** HistGradientBoosting with fraud up-weighted (×12) for the class imbalance.
- **Calibration.** Logistic (Platt) calibration on the validation period. Isotonic regression was tried first, but it flattened to 100% across the whole top of the score range, so a held payment and a stepped-up payment could show the same "100%" risk. Platt calibration is strictly increasing, so the displayed risk always ranks payments in the same order as the decision.
- **Anomaly model.** IsolationForest trained on normal payments only, using behaviour features. It can raise a step-up on unusual payments without needing any fraud labels.
- **Policy.** The operations team can only review so many alerts, so thresholds come from an **alert budget** set on validation: HOLD ≈ top 0.5% of payments, STEP_UP ≈ the next 1.2%. Two additions:
  - A guard rule always holds SIM swap < 24h + new device + Rs 10,000 or more, regardless of the model.
  - A payment to an account on the mule watchlist is raised from STEP_UP to HOLD.
- **Explanations.** Each feature in turn is replaced with the value of a typical legitimate payment, and the change in score is measured. Features that raise the risk become reasons. A feature whose value points the "safe" way (for example "0 senders" for a fan-in feature) is not shown as a reason, because it would read as nonsense to an analyst.
- **Rules baseline.** A hand-written rule set of the kind banks start with: Rs 50k+, 5+ payments per hour, new device + Rs 10k, night + Rs 20k, scam words. It is evaluated on the same test period.

## 5. Graph intelligence (`graph.py`, `pipeline.py`)

- **Money trail.** Prahari follows "fast forwards" from the payee of a payment: an outgoing transfer within 3 hours that carries at least 30% of the incoming amount to a non-merchant. This builds the hop-by-hop path with amounts.
- **Mule score.** Combines the share of incoming money passed on within 3 hours, the number of distinct senders, and whether the account is younger than 90 days. Accounts that received less than Rs 5,000 score 0.
- **Ring detection.** Seeds are the payees of held payments. Rings grow along fast trails, and weak single-hop links are dropped. Louvain community detection splits large components.
- **Daily graph job, with no look-ahead.** Each day, rings are rebuilt from holds up to the previous day, and their accounts go on a watchlist with the date they were added. Only later payments to those accounts can be escalated. A test checks this.

**Why the graph caught the new scam.** The investment-scam VPAs forwarded money into the same layering network that the earlier frauds used. The graph flagged them while the model, which had never seen this scam, scored them as low risk.

## 6. Investigator agent (`agent.py`)

The agent is a LangGraph state machine: `plan → act (loop, one tool per step) → assess → write → report`.

- **Tools** (8): payment, model reasons, customer profile, device and SIM, 24-hour activity, money trail, ring membership, similar past cases (nearest neighbours over feature vectors of confirmed cases in the training period).
- **Planning.** With an LLM, the plan is limited to the tool allow-list, and unknown tools are dropped. Offline, a fixed playbook is used.
- **Assessment.** Deterministic rules decide the fraud type from the evidence. If no rule fits, the nearest past cases vote. If the alert is unlike every known case, it is labelled **new_pattern** and routed to the fraud-strategy team.
- **Grounded writing.** Every sentence must cite evidence ids, and every number and account id in a sentence must appear in the evidence it cites. Failing sentences are removed and listed. If none survive, a template narrative built directly from the evidence is used.
- **Report.** The STR draft lists subjects from the money trail, transactions, total amount, grounds for suspicion and recommended actions. Its status is always "draft - requires analyst approval". Analyst decisions are recorded through the API.

## 7. Monitoring (`monitor.py`)

- **Drift.** PSI per feature and for the model score, training vs test. Status: stable < 0.1, watch 0.1-0.25, alert > 0.25 (or score PSI > 0.2).
- **Fairness.** False-positive rate of alerts (HOLD or STEP_UP) on legitimate payments, by customer age band and city tier. The highest-to-lowest ratio must stay below 2. Small groups are excluded to avoid noisy ratios.

## 8. Serving (`api.py`)

FastAPI serves the dashboard and a JSON API. `/api/stream` replays the test period as server-sent events through the online feature store. Artifacts are built once (`python -m prahari build`, baked into the Docker image) and loaded at startup. The engine holds everything in memory, which is fine for a demo.

**Production changes would be:**

- Kafka (or similar) for the event stream.
- Redis for online features.
- A graph database or stream job for continuous ring scores.
- A case-management database.
- Delayed-label retraining with champion/challenger rollout.

## 9. Testing and quality gates

There are 25 tests on a small simulated world. They cover:

- simulator realism;
- point-in-time features and live-replay parity;
- policy, calibration and explanations;
- money trails, ring precision and no look-ahead;
- agent typologies;
- the LLM path (tool allow-list, removal of ungrounded sentences);
- drift and fairness math;
- every API route.

CI runs these tests, then a full-scale build that must pass 12 quality gates (`eval/run.py`), then a Docker build that scores a risky payment.
