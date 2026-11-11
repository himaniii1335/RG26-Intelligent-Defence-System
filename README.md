# RG26 — Intelligent Defence System

Major Project-I, JUIT. A digital twin of a defence facility that scores network
threats and physical sensor anomalies **on the same zone, in the same time
window**, and escalates when both appear together.

Most published digital-twin security work keeps the cyber and the physical
pipeline separate. The contribution here is the correlation step: a port scan is
routine, an unauthorised badge read is routine, but both in the server room
within two minutes is treated as one coordinated intrusion.

## Phases

| Phase | Work | Status |
|---|---|---|
| 1 | NSL-KDD preprocessing, feature engineering, Random Forest baseline | done (`NSLKDD.ipynb`) |
| 2 | Hybrid detector, digital twin, risk correlation, API, dashboard | done (this release) |
| 3 | CICIDS2017 and TON_IoT, SHAP explanations, PostgreSQL, cloud deployment | planned |

## Detection model

Phase 1 reached ~75% test accuracy with a single Random Forest. The ceiling is
structural: `KDDTest+` contains **17 attack types absent from `KDDTrain+`**, and
a supervised classifier has no class to assign them to, so it calls them normal.

Phase 2 keeps the Random Forest and adds an Isolation Forest trained on normal
traffic only.

    RF says attack             -> threat, named family
    RF says normal + IF flags  -> threat, "Unknown / Zero-day"
    both say normal            -> normal

Measured on `KDDTest+` (`ml/artifacts/metrics.json`):

| Attack detection | RF only | Hybrid |
|---|---|---|
| Recall | 60.65% | **73.03%** |
| Precision | 96.62% | 92.80% |
| F1 | 74.52% | **81.74%** |
| False positive rate | 2.80% | 7.49% |
| U2R detection | 10.50% | **65.00%** |
| **17 unseen attack types** | 20.69% | **58.99%** |

The false positive rate roughly doubles. That is the honest price of catching
unknown attacks, and it is reported rather than hidden.

## Risk model

Each zone keeps a 120-second sliding window of scored events.

- `cyber_score` — strongest recent network threat, weighted by recency and
  attack-family severity (U2R and R2L rank above DoS: an intruder holding
  credentials is worse than a flood).
- `physical_score` — same, from rule-based sensor anomalies.
- `risk` — the mean of the two, plus a correlation bonus when **both** are
  non-zero, scaled by how critical the zone is.

Thresholds: 30 = elevated (amber), 70 = critical (red).

The physical side is rule-based on purpose. There is no labelled history of
break-ins to learn from, the rules encode written access-control policy, and a
guard has to be able to read why an alarm fired.

## Running it

```bash
pip install -r requirements.txt

# 1. train (about 2 minutes; writes ml/artifacts/)
python ml/train_hybrid.py

# 2. start the API and dashboard
uvicorn backend.main:app --port 8000
#    open http://127.0.0.1:8000

# 3. in a second terminal, drive events
python simulator/generator.py --mode demo --reset   # scripted incident
python simulator/generator.py --mode live           # continuous traffic
```

`KDDTrain+.txt` and `KDDTest+.txt` must sit in the repository root. They are
excluded from version control.

## Layout

```
ml/preprocess.py        NSL-KDD pipeline, fitted once and reused at inference
ml/train_hybrid.py      trains both models, writes metrics and figures
ml/detector.py          inference wrapper used by the backend
backend/twin.py         facility model and the physical-cyber risk score
backend/physical.py     sensor anomaly rules
backend/db.py           SQLite event and alert store
backend/main.py         FastAPI service
simulator/generator.py  event generator, replays real NSL-KDD flows
dashboard/index.html    single-page monitoring client
```

## Known limitations

- R2L detection stays low (11.98%). These attacks look like normal application
  traffic at flow level; packet payload features would be needed.
- Per-alert explanations use Random Forest importance weighted by the scaled
  feature value, an approximation of SHAP rather than SHAP itself.
- SQLite and a single process. Suitable for demonstration, not for deployment.
- The physical sensor stream is simulated; no hardware is connected.
