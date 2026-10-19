# fraudguard: credit card fraud detection, built like a product

A fraud-detection pipeline with correct metrics for imbalanced data, cost-based decision thresholds,
experiment tracking in MLflow, and (coming next) a FastAPI service, a UI and free cloud hosting.

> Status: **work in progress.** See the roadmap. Each step lands as its own PR.

## Why not just `model.fit()` and report accuracy?

Only 0.17 % of transactions in the dataset are fraud. This project therefore:

- evaluates with **PR-AUC and fraud-class precision/recall**, never accuracy or weighted F1;
- splits **by time** (train on the past, test on the future), the way a production model is used;
- picks the decision threshold on a **validation** slice from explicit costs: a missed fraud costs its
  amount, a false alarm costs a fixed review cost (both configurable);
- keeps all preprocessing **inside the sklearn Pipeline**, so serving needs no duplicated logic.

Details of what was fixed from the first notebook version are in [docs/findings.md](docs/findings.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install

make data-synthetic          # instant demo data (not for reporting results)
# or: make data              # real data from Kaggle (needs ~/.kaggle/kaggle.json)

make train                   # one model (configs/train.yaml)
make train-all               # logreg, random forest, gradient boosting
make mlflow-ui               # http://127.0.0.1:5000  compare runs
make test
```

Each run writes `models/<run>/` with `model.joblib`, `metadata.json` (threshold, metrics, config, git SHA,
library versions), `pr_curve.png` and `confusion_matrix.png`, and logs the same to MLflow.

## Project layout

```
configs/train.yaml        every knob: split, features, imbalance handling, model, costs, tracking
src/fraudguard/
  data.py                 schema validation, de-duplication, time / stratified split
  features.py             preprocessing (log-amount, hour-of-day, scaling) as a ColumnTransformer
  models.py               logreg, random forest, hist-gradient-boosting, lightgbm (+ optional SMOTE)
  metrics.py              fraud-class metrics, cost-based threshold selection
  tracking.py             MLflow wrapper (a no-op tracker is used in tests)
  train.py                CLI entry point
tests/                    metrics are checked against a brute-force reference
docs/findings.md          review of the original notebooks
notebooks/legacy/         the original exploratory notebooks, kept for history
```

## Results

## Results

Time-based hold-out: train on the earliest transactions, validate the threshold on the next slice,
test on the most recent 20% (56,745 transactions, **74 frauds**). Threshold chosen from costs on
validation data: a missed fraud costs its amount, a false alarm costs 5 (assumption).

| Model | PR-AUC | Fraud precision | Fraud recall | F1 | Caught / missed | False alarms | Cost saving* |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.734 | 0.942 | 0.662 | 0.778 | 49 / 25 | 3 | 60.8% |
| Random forest | 0.805 | 0.903 | 0.757 | 0.824 | 56 / 18 | 6 | 65.5% |
| Gradient boosting (untuned) | 0.690 | 0.739 | 0.689 | 0.713 | 51 / 23 | 18 | 44.7% |

\*Cost relative to having no model, under the assumed costs in `configs/train.yaml`.

With only 74 frauds in the test window, differences of a few transactions are within noise.
Gradient boosting runs on near-default parameters; tuning is on the roadmap.

![MLflow run comparison](docs/mlflow_runs.png)

## Roadmap

- [x] **1. Project structure**: package, config, data script, tests
- [x] **2. Correct evaluation**: time split, PR-AUC, cost-based threshold, three model families
- [ ] **3. MLflow**: model registry with a `champion` alias, promotion rule
- [ ] **4. FastAPI service**: `/predict`, `/predict/batch`, `/health`, `/model-info`
- [ ] **5. Frontend**: score a transaction, adjust the threshold, see feature contributions
- [ ] **6. Docker + docker-compose**: API, UI and MLflow
- [ ] **7. CI/CD**: GitHub Actions (lint, tests, image build, deploy)
- [ ] **8. Monitoring**: drift report and prediction logging
- [ ] **9. Hosting**: free-tier deployment and a live demo link

## Data and credits

Dataset: *Credit Card Fraud Detection*, Machine Learning Group, ULB, distributed on Kaggle
(<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>). Features `V1`-`V28` are PCA components.
The cost values in the config are illustrative assumptions, not real bank figures.
