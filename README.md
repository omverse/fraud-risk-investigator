# Fraud & Risk Investigator

A fraud detection and risk investigation system that scores transactions using machine learning, retrieves policy and past-case context from a vector database, and generates grounded, human-readable alert explanations.

---

## 1. Problem
Financial fraud detection operates under severe class imbalance: only ~3.5% of transactions are fraudulent. Traditional evaluation metrics like raw classification accuracy are deceptive because a naive model that predicts "never fraud" achieves ~96.5% accuracy while catching zero fraudulent transactions. In production risk systems, the cost of a False Negative (a missed fraudulent transaction resulting in direct chargebacks and financial loss) is typically orders of magnitude higher than a False Positive (a temporary transaction verification or customer friction). The objective is to build an honest, robust ML system evaluated on precision-recall metrics and cost-sensitive thresholds rather than arbitrary default cutoffs.

---

## 2. Data
The system is built on the IEEE-CIS Fraud Detection dataset:
- **`train_transaction.csv`**: 590,540 transactions across 394 features.
- **`train_identity.csv`**: 144,233 identity records across 41 features.
- **Joined Dataset**: Left-joined on `TransactionID` $\rightarrow$ 590,540 rows $\times$ 434 columns.
- **Input Features**: 431 features used for modeling after dropping `isFraud` (target label), `TransactionID` (arbitrary identifier), and raw `TransactionDT` (absolute timestamp that cannot generalize forward in tree-based splits).
- **Categorical Handling**: 31 categorical columns (14 from transaction, 17 from identity). Categorical levels are strictly learned from the training set only; unseen categories in validation and test are encoded as missing values to prevent data leakage.
- **Masked Columns**: Column meanings are masked per Vesta Corporation's schema. Only category-level descriptions are documented: `D` are time deltas, `M` are match flags, `C` are counting metrics, `dist` are geographic/network distances, and `V` are Vesta-engineered features. No unverified domain interpretations are asserted.

---

## 3. Time-Based Split
Transactions are ordered chronologically by `TransactionDT` (time elapsed since a reference point) and split by row positions using a stable sort to mirror real-world production deployment:
- **Train (Oldest 70%)**: 413,378 rows | Fraud rate: 3.517% | `TransactionDT`: 86,400 to 10,437,996 (~121 days)
- **Validation (Middle 15%)**: 88,581 rows | Fraud rate: 3.434% | `TransactionDT`: 10,438,003 to 13,151,840 (~31 days)
- **Test (Newest 15%)**: 88,581 rows | Fraud rate: 3.480% | `TransactionDT`: 13,151,880 to 15,811,131 (~31 days)

**Why no random shuffling**: A fraud model in production must predict future transactions using only past historical data. Random shuffling allows trees to memorize future behavioral patterns and fraud rings, yielding unrealistically optimistic scores that collapse when deployed live.

---

## 4. Baseline Results (Day 1)
Baseline model: `XGBClassifier(tree_method="hist", enable_categorical=True, eval_metric="aucpr", early_stopping_rounds=50)` trained on Train and stopped at best iteration on Validation.

| Metric | Baseline Score | Context |
| :--- | :--- | :--- |
| **Dummy PR-AUC** | **0.0348** | Baseline to beat (equals test fraud prevalence of 3.48%) |
| **Validation PR-AUC** | **0.5630** | Peak validation score (best iteration = 599) |
| **Test PR-AUC** | **0.4686** | 13.5x over dummy baseline; gap reflects validation stopping selection & temporal drift |
| **Test ROC-AUC** | **0.8626** | Strong global ranking capability |
| **Precision (at 0.5 cutoff)** | **75.4%** | 949 true frauds out of 1,258 flags |
| **Recall (at 0.5 cutoff)** | **30.8%** | Only 949 out of 3,083 actual frauds caught (**69.2% missed**) |

### Confusion Matrix (Test Set @ Threshold 0.5)
- **True Negatives (TN)**: 85,189 | **False Positives (FP)**: 309
- **False Negatives (FN)**: 2,134 | **True Positives (TP)**: 949

![Day 1 Precision-Recall Curve](results/day1_pr_curve.png)

---

## 5. Temporal Drift & Adversarial Validation (Day 2)
Adversarial classifiers (distinguishing sample time periods without fraud labels) were trained to quantify distribution shifts over time:
- **Train vs. Validation ROC-AUC**: **0.8702** (the periods are clearly distinguishable)
- **Validation vs. Test ROC-AUC**: **0.7284** (closer in time, but still distinct)
- **Train vs. Test ROC-AUC**: **0.8886** (largest shift across the full 6-month timeline)

### Drift Evidence
1. **`M1` Match Feature**: Missingness plunged from **69.6%** in Train down to **25.7%** in Test. Crucially, the fraud rate among missing `M1` records surged from **2.8%** to **9.9%**, while `M1 == True` remained constant (~2.0%).
2. **`id_31` Browser Field**: Chronological progression of browser version numbers (Chrome 62/63 $\rightarrow$ 64 $\rightarrow$ 65 $\rightarrow$ 66).
3. **Identity Coverage**: Identity table join match rate dropped from ~42% in early periods to 17–20% in later periods.

---

## 6. Feature Engineering & Stationarity
To reduce vulnerability to drift while maintaining stationary signal:
- **Relative Time Features**:
  - `hour_of_day = (TransactionDT // 3600) % 24`
  - `day_of_week = (TransactionDT // 86400) % 7`
- **Browser & OS Categorical Aggregation**: Grouped high-cardinality version strings into broad families (`chrome`, `safari`, `firefox`, `edge`, `windows`, `ios`, `mac`, `android`, etc.).
- **Backward-Looking Card Velocity & Ratios (`card1`)**:
  - Running transaction counts over rolling time horizons: `card1_trans_count_1h`, `card1_trans_count_24h`, `card1_trans_count_7d`.
  - Velocity: `card1_sec_since_prev` (seconds elapsed since prior transaction on this card; first transaction is strictly `NaN`).
  - Spending deviation: `card1_amt_ratio_to_prev_avg` (ratio of current amount to card's historical cumulative average; first transaction is strictly `NaN`).
- **Normalized Epoch Deltas**:
  - `D1_norm = D1 - (TransactionDT / 86400.0)`: Subtracting elapsed days anchors the initial card registration point to a fixed timeline epoch, transforming a shifting delta into a stationary client fingerprint.

---

## 7. Noise Study & Controlled Feature Ablation
### Training Shuffle Noise Study
Re-running the baseline across 5 training row-order permutations revealed intrinsic XGBoost `hist` variance:
- **Baseline Validation PR-AUC**: Mean = **0.5600**, $\sigma(n-1) = 0.0071$, Range = **0.0200** (0.5496 to 0.5696).
- **Methodological Rule**: Any observed validation gain $< 0.015 - 0.020$ is indistinguishable from single-run training noise.

### 12-Run Controlled Ablation (Fixed 350 Trees, No Early Stopping)
To isolate feature signal from early-stopping selection bias, models were trained for exactly 350 trees across 3 random seeds ($n-1$ sample standard deviation):

| Feature Configuration | Val PR-AUC Mean | Val PR-AUC $\sigma(n-1)$ | Val PR-AUC Range | $\Delta$ vs Baseline Mean (0.5600) | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Day 1 Raw Baseline (Fixed 350 Trees)** | 0.5624 | 0.0103 | 0.0195 | +0.0024 | Working reference |
| **Core Rolling Windows** | 0.5655 | 0.0039 | 0.0073 | +0.0055 | Within noise band; kept for stationarity |
| **Core + `D1_norm`** | **0.5809** | **0.0006** | **0.0012** | **+0.0209** | **Proven signal exceeding noise ($\sim 3\sigma$)** |
| **Core + `D1_norm` + `D2_norm`** | 0.5807 | 0.0096 | 0.0180 | +0.0207 | Zero mean gain; 16x higher variance |
| **Core + Six D-Columns Normalized** | 0.5669 | 0.0034 | 0.0064 | +0.0069 | Earlier 0.6068 was early-stopping artifact |

**Locked Feature Set**: **Core Rolling Windows + `D1_norm`**.

---

## 8. Class Imbalance & Model Comparison
### Handling Class Imbalance (3.5% Fraud Prevalence)
Tested across 3 runs each (350 fixed trees, locked features):
- **Unweighted Training**: Mean Val PR-AUC = **0.5809** ($\sigma = 0.0006$) $\rightarrow$ **Selected**.
- **Mild Weighting (`scale_pos_weight = 5.0`)**: Mean Val PR-AUC = 0.5810 ($\sigma = 0.0018$, no measurable benefit).
- **Full Inverse Weighting (`scale_pos_weight = 27.4`)**: Mean Val PR-AUC = 0.5403 (-0.0406 vs unweighted).
- **Random Undersampling 1:4**: Mean Val PR-AUC = 0.5664 (-0.0145 vs unweighted).
- **Random Undersampling 1:1**: Mean Val PR-AUC = 0.5282 (-0.0527 vs unweighted).

> **Client Overlap Note**: 98.9% of validation transactions come from `card1` identifiers seen in training. Unseen cards represent only 1.1% (49 frauds), meaning validation metrics predominantly measure returning clients; novel card performance is statistically unmeasurable on validation.

### Model Architecture Comparison (3 Runs Each)
- **XGBoost (Locked, Unweighted)**: Val PR-AUC = **0.5809** ($\sigma = 0.0006$) | Training Time: **22.2s**
- **Random Forest (150 trees, max depth 14, sqrt features)**: Val PR-AUC = **0.4787** ($\sigma = 0.0017$) | Training Time: **31.2s**
- **Isolation Forest (Unsupervised Anomaly Scoring)**: Val PR-AUC = **0.1842** | Fit Time: **1.6s**

---

## 9. Isotonic Calibration & Cost-Based Threshold Optimization
### Isotonic Recalibration (Validation 50/50 Split)
Because raw tree outputs severely under-predicted probability in lower deciles, an isotonic calibrator was fit on the first half of validation (44,290 rows) and evaluated on the second half (44,291 rows):
- **Half 2 Actual Fraud Rate**: **3.272%**
- **Raw Mean Predicted Probability**: **2.233%** (Brier score: `0.02265`)
- **Calibrated Mean Predicted Probability**: **3.215%** (Brier score: `0.02245`)
- **Calibration Status**: Improved but imperfect. While the aggregate mean matches within $-0.00057$, deciles 1 through 9 remain slightly under-predicted while the top decile is slightly over-predicted (23.6% calibrated vs 21.9% actual).

### Alert-Rate Optimization ($k = 1 \dots 300$ per 1,000 Transactions)
Total validation fraud loss under "Flag Nothing" = **\$496,908** (3,042 frauds). Optimization evaluated across review costs of \$5, \$10, and \$25:

| Review Cost | Optimal Alert Rate ($k^*$) | Total Alerts | Precision | Recall | Total Financial Cost | Net Savings vs Flag Nothing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **\$10 (Headline)** | **97 / 1k** | **8,592** | **25.51%** | **72.06%** | **\$198,569** | **+\$298,339** (60.0% savings) |
| **\$5** | **152 / 1k** | 13,464 | 17.83% | 78.93% | \$145,359 | +\$351,548 (70.7% savings) |
| **\$25** | **44 / 1k** | 3,898 | 46.07% | 59.04% | \$285,229 | +\$211,679 (42.6% savings) |

> **Operational Insight**: The cost curve is flat around the optimum: alert rates between 85 and 110 per 1,000 achieve virtually identical total costs. At the default Day 1 cutoff of 0.5 (15.9 alerts/1k), total cost was **\$342,467** (missing 62.5% of frauds). Operating at 97 / 1k captures an additional **\$143,898 in net savings**.

![Day 2 Cost Curves](results/day2_cost_curves.png)

### Amount-Aware Rule ($\mathbb{E}[\text{Loss}] = p_{\text{cal}} \times \text{TransactionAmt} \ge c_{\text{rev}}$)
Evaluated on Validation Half 2:
- At **\$5 review cost**: Amount-aware rule cost **\$67,060** vs **\$82,775** for the best tuned global threshold (**+\$15,715 additional savings** with 33.6% fewer alerts).
- At **\$10 review cost**: Amount-aware rule cost **\$93,930** vs **\$111,029** for the best tuned global threshold (**+\$17,099 additional savings** with 22.1% fewer alerts).
- At **\$25 review cost**: Amount-aware rule cost **\$134,375** vs **\$148,599** for the best tuned global threshold (**+\$14,223 additional savings**).

---

## 10. Final Test Set Evaluation (Scored Once)
The test set (88,581 rows, 3,083 frauds, \$469,608.50 total fraud loss) was scored strictly once using locked parameters and alert rates decided on validation:

### Model Performance Summary
| Model | Test PR-AUC | Test ROC-AUC | Precision @ 0.5 | Recall @ 0.5 | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Day 1 Baseline** | 0.4681 | 0.8656 | 75.44% | 30.78% | Stopped on validation |
| **Model A (Train-Only, 350 Trees)** | **0.4756** | **0.8569** | **72.77%** | **34.67%** | +0.0074 PR-AUC gain over baseline |
| **Model B (Retrained Train+Val, 350 Trees)** | **0.5482** | **0.8954** | — | — | Supplementary; +0.0726 PR-AUC gain |

> **Evaluation Honesty**: As expected, Model A test PR-AUC (0.4756) is lower than validation PR-AUC (0.5809). Because all modeling decisions were selected on validation, validation scores reflect selection optimism, whereas the test set measures true forward generalization across temporal drift. Retraining on the most recent data (Model B) recovers this gap to 0.5482.

### Performance at Prescribed Alert Rates
Evaluating Model A, the Day 1 Baseline, and Model B at identical fixed alert rates on the test set:

| Review Cost | Policy / Alert Rate | Precision | Recall | Total Cost | Context / Savings |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **\$10 (Headline)** | **Flag Nothing** | 0.0% | 0.0% | \$469,609 | Baseline loss (all 3,083 frauds missed) |
| | **Flag Everything** | 3.48% | 100.0% | \$885,810 | Excessive review overhead |
| | **Threshold 0.5 (Model A)** | 72.77% | 34.67% | \$351,361 | 16.6 alerts/1k; 2,014 frauds missed |
| | **Day 1 Baseline @ 97/1k** | 22.75% | 63.41% | \$507,280 | 1,955 frauds caught |
| | **Model A @ 97 / 1k (8,592 alerts)** | **23.14%** | **64.48%** | **\$515,779** | 1,988 frauds caught (+33 over baseline) |
| | **Model B (Retrained) @ 97 / 1k** | **25.15%** | **70.09%** | **\$512,751** | 2,161 frauds caught |
| **\$5** | **Model A @ 152 / 1k (13,464 alerts)** | 16.47% | 71.91% | \$471,637 | 2,217 frauds caught |
| | **Model B (Retrained) @ 152 / 1k** | 17.70% | 77.29% | \$459,806 | 2,383 frauds caught |
| **\$25** | **Model A @ 44 / 1k (3,898 alerts)** | 40.71% | 51.48% | \$547,931 | 1,587 frauds caught |
| | **Model B (Retrained) @ 44 / 1k** | 45.54% | 57.57% | \$545,742 | 1,775 frauds caught |

### The Power of the Amount-Aware Rule on Test
Under a fixed alert rate, ranking purely by probability flags many small-dollar frauds ($<\$20$) while missing high-dollar frauds. Using calibrated probabilities with the expected loss rule ($\mathbb{E}[\text{Loss}] = p_{\text{cal}} \times \text{amount} \ge c_{\text{rev}}$) produces decisive savings:

| Review Cost | Operating Rule | Alerts / 1k | Precision | Recall | Total Financial Cost | Net Savings vs Fixed Rate | Net Savings vs Flag Nothing |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **\$10 (Headline)** | **Fixed Alert Rate (97/1k)** | 97.0 / 1k | 23.14% | 64.48% | \$515,779 | — | -\$46,170 |
| | **Amount-Aware Rule** | **98.9 / 1k** | **19.47%** | **55.34%** | **\$183,126** | **+\$332,653** | **+\$286,483 (61.0% saved)** |
| **\$5** | **Fixed Alert Rate (152/1k)**| 152.0 / 1k | 16.47% | 71.91% | \$471,637 | — | -\$2,028 |
| | **Amount-Aware Rule** | **156.7 / 1k** | **14.41%** | **64.87%** | **\$137,608** | **+\$334,029** | **+\$332,000 (70.7% saved)** |
| **\$25** | **Fixed Alert Rate (44/1k)** | 44.0 / 1k | 40.71% | 51.48% | \$547,931 | — | -\$78,322 |
| | **Amount-Aware Rule** | **46.9 / 1k** | **26.77%** | **36.10%** | **\$265,321** | **+\$282,610** | **+\$204,287 (43.5% saved)** |

**Key Takeaway**: A \$5,000 transaction with a 15% fraud probability represents an expected loss of \$750 (overwhelmingly justifying a \$10 review), whereas a \$10 transaction with an 80% fraud probability represents only \$8 expected loss. Incorporating calibrated probabilities and transaction amounts into alerting decisions saves over **\$330,000** compared to any probability-only ranking policy.
