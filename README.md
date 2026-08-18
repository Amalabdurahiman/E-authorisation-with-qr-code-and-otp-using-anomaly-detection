# E-Authorisation Using QR Code and OTP with Anomaly Detection

MSc Advanced Computer Science final project — University of Hertfordshire, 2024.
Supervisor: Vitoria Wilkinson. Awarded Merit.

A multi-factor authentication system combining QR code and OTP verification with machine
learning anomaly detection, built to flag suspicious authentication attempts that pass
credential checks but deviate from normal behaviour.

---

## Problem

Credential-based authentication fails when credentials are stolen: the login is valid, so
rule-based checks pass. Adding a behavioural layer means an attempt can be flagged as
anomalous even when the password and OTP are correct.

The research question was which unsupervised anomaly detection approach performs best on
authentication-style data, and how far class imbalance limits that performance.

## Approach

| Stage | What was done |
|---|---|
| Authentication | Flask application with QR enrolment, TOTP verification (`pyotp`), SQLite user store |
| Detection | Isolation Forest and One-Class SVM, trained on network intrusion data as a proxy for authentication behaviour |
| Imbalance handling | SMOTE oversampling plus synthetic anomaly generation at varying ratios |
| Evaluation | Stratified k-fold cross-validation; precision, recall, F1 and ROC-AUC |

## Datasets

- **KDDCup99** — established benchmark, but dated traffic patterns
- **KDDTest+** (NSL-KDD) — deduplicated variant addressing known KDDCup99 flaws
- **UNSW-NB15** — modern traffic and attack types; gave the most realistic evaluation

## Why not accuracy

Anomalies are a small minority of records in all three datasets. A model predicting
"normal" for everything scores high accuracy while detecting nothing. Evaluation therefore
used precision, recall and F1, with recall treated as the primary metric — a missed
intrusion costs more than a false alarm that triggers a second authentication factor.

## Findings

- **Isolation Forest** scaled well across high-dimensional data and was computationally cheap.
- **One-Class SVM** detected non-linear boundaries the Isolation Forest missed, at higher cost.
- **Class imbalance was the binding constraint** on both models, not model choice — the
  clearest single finding of the project.
- **SMOTE and synthetic anomaly generation raised recall without a corresponding fall in
  precision**, confirming that the limitation was data representation rather than algorithm.
- **A combined model** taking both detectors' predictions improved the precision–recall
  balance over either alone.
- **Dataset choice materially changed results.** UNSW-NB15 produced the most credible
  evaluation; conclusions drawn from KDDCup99 alone would have been optimistic.

## Limitations

- Network intrusion datasets are a proxy for authentication behaviour, not a direct
  substitute. Real authentication telemetry would test this properly.
- Synthetic anomalies inherit the distribution of the real ones. They mitigate imbalance
  but cannot introduce genuinely novel attack patterns.
- Detection thresholds were tuned against these datasets and would need recalibration
  against live traffic.
- The Flask application is a research prototype demonstrating integration, not a
  production authentication service.

## Repository structure

```
app.py                                  Flask application: QR enrolment, OTP, session handling
Isolation_forest_webpage.py             Detection integrated into the authentication flow
Isolation-test.py                       Isolation Forest baseline
One-Class svm Test.py                   One-Class SVM baseline
Combined Model Test.py                  Combined detector
Combined with KDDTrain+ dataset.py      Combined model, KDDTrain+
Combined Model with adjusted kddtest+.py  Combined model, tuned on KDDTest+
Combined model with KDDTest++.py        Combined model, KDDTest+ variant
Combined with SMOTE Test.py             SMOTE-balanced training
Combined with synthetic data.py         Synthetic anomaly generation
Combined with Synthetic 0.2.py          Synthetic anomalies at 0.2 ratio
Cross-Validation Test for 2 models.py   Stratified k-fold evaluation
UNSW Dataset with isolation.py          Isolation Forest, UNSW-NB15
UNSW Dataset with oneclass SVM.py       One-Class SVM, UNSW-NB15
*.html                                  Application templates
```

## Running it

```bash
pip install -r requirements.txt
python app.py                     # authentication prototype at http://127.0.0.1:5000
python "Cross-Validation Test for 2 models.py"   # model evaluation
```

KDDCup99 and KDDTest+ are not included in this repository and must be downloaded
separately. `UNSW_NB15_testing-set.csv` is included.

## Stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, Flask, pyotp, qrcode, SQLite.

## Further work

Validation against real authentication telemetry; online learning so thresholds adapt as
behaviour drifts; and calibration of the false-positive rate against the friction cost of
an additional verification step.
