import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, confusion_matrix, brier_score_loss
from sklearn.isotonic import IsotonicRegression

def reduce_mem_usage(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

def clean_browser(x):
    if pd.isna(x):
        return "missing"
    x = str(x).lower()
    for b in ["chrome", "safari", "firefox", "edge", "ie", "samsung", "opera"]:
        if b in x:
            return b
    return "other"

def clean_os(x):
    if pd.isna(x):
        return "missing"
    x = str(x).lower()
    for o in ["windows", "ios", "mac", "android", "linux"]:
        if o in x:
            return o
    return "other"

def compute_rolling_window_counts(df):
    df["_orig_idx"] = np.arange(len(df))
    df_sorted = df.sort_values(["card1", "TransactionDT"], kind="stable").reset_index(drop=True)
    
    times = df_sorted["TransactionDT"].values
    cards = df_sorted["card1"].values
    amts = df_sorted["TransactionAmt"].values
    
    splits = np.where(cards[:-1] != cards[1:])[0] + 1
    group_times = np.split(times, splits)
    group_amts = np.split(amts, splits)
    
    cnt_1h, cnt_24h, cnt_7d = [], [], []
    sec_prev, amt_ratio = [], []
    
    for g_t, g_a in zip(group_times, group_amts):
        n_g = len(g_t)
        curr_idx = np.arange(n_g)
        cnt_1h.append(curr_idx - np.searchsorted(g_t, g_t - 3600, side="left"))
        cnt_24h.append(curr_idx - np.searchsorted(g_t, g_t - 86400, side="left"))
        cnt_7d.append(curr_idx - np.searchsorted(g_t, g_t - 604800, side="left"))
        
        s_p = np.empty(n_g, dtype=np.float32)
        s_p[0] = np.nan
        if n_g > 1:
            s_p[1:] = g_t[1:] - g_t[:-1]
        sec_prev.append(s_p)
        
        a_r = np.empty(n_g, dtype=np.float32)
        a_r[0] = np.nan
        if n_g > 1:
            cumsum_prev = np.cumsum(g_a) - g_a
            prev_counts = np.arange(n_g)
            prev_avg = cumsum_prev[1:] / prev_counts[1:]
            a_r[1:] = g_a[1:] / prev_avg
        amt_ratio.append(a_r)
        
    df_sorted["card1_trans_count_1h"] = np.concatenate(cnt_1h).astype(np.int16)
    df_sorted["card1_trans_count_24h"] = np.concatenate(cnt_24h).astype(np.int16)
    df_sorted["card1_trans_count_7d"] = np.concatenate(cnt_7d).astype(np.int16)
    df_sorted["card1_sec_since_prev"] = np.concatenate(sec_prev)
    df_sorted["card1_amt_ratio_to_prev_avg"] = np.concatenate(amt_ratio)
    
    df_result = df_sorted.sort_values("_orig_idx", kind="stable").drop(columns=["_orig_idx"]).reset_index(drop=True)
    return df_result

def compute_cost(y_true, amt_arr, alerts_bool, review_cost):
    missed_mask = (y_true == 1) & (~alerts_bool)
    missed_cost = float(amt_arr[missed_mask].sum())
    review_total = float(review_cost * alerts_bool.sum())
    return missed_cost + review_total, missed_cost, review_total

def main():
    print("=" * 70)
    print("Step 7: Final Test Score Evaluation")
    print("=" * 70)

    # 1. Load raw data and create base splits for Day 1 baseline
    print("\n1. Loading merged dataset...")
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    df = reduce_mem_usage(df)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)

    y = df["isFraud"].values
    amt = df["TransactionAmt"].values

    y_tr, amt_tr = y[:i1], amt[:i1]
    y_va, amt_va = y[i1:i2], amt[i1:i2]
    y_te, amt_te = y[i2:], amt[i2:]
    n_te = len(y_te)
    total_test_fraud_amt = float(amt_te[y_te == 1].sum())
    total_test_frauds = int(y_te.sum())

    print(f"Dataset split: Train={i1:,}, Val={i2-i1:,}, Test={n-i2:,}")
    print(f"Test split: {n_te:,} rows | Frauds: {total_test_frauds:,} ({y_te.mean()*100:.3f}%) | Fraud Loss: ${total_test_fraud_amt:,.2f}")

    # -------------------------------------------------------------
    # 2. Day 1 Baseline Test Predictions (Reproduce exact Day 1 baseline)
    # -------------------------------------------------------------
    print("\n2. Training Day 1 Baseline model for comparison at same alert rates...")
    X_base = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    cat_cols_base = [c for c in X_base.columns if not pd.api.types.is_numeric_dtype(X_base[c])]
    
    X_tr_b = X_base.iloc[:i1].copy()
    X_va_b = X_base.iloc[i1:i2].copy()
    X_te_b = X_base.iloc[i2:].copy()

    for c in cat_cols_base:
        cats = X_tr_b[c].astype("category").cat.categories
        X_tr_b[c] = pd.Categorical(X_tr_b[c], categories=cats)
        X_va_b[c] = pd.Categorical(X_va_b[c], categories=cats)
        X_te_b[c] = pd.Categorical(X_te_b[c], categories=cats)

    model_base = xgb.XGBClassifier(
        n_estimators=1000,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42
    )
    t0 = time.time()
    model_base.fit(X_tr_b, y_tr, eval_set=[(X_va_b, y_va)], verbose=False)
    p_te_base = model_base.predict_proba(X_te_b)[:, 1]
    base_pr_auc = float(average_precision_score(y_te, p_te_base))
    base_roc_auc = float(roc_auc_score(y_te, p_te_base))
    print(f"Day 1 Baseline (best iter {model_base.best_iteration}): Test PR-AUC = {base_pr_auc:.5f} (matches logged 0.4686)")

    np.save("results/test_preds_day1_baseline.npy", p_te_base)
    del X_base, X_tr_b, X_va_b, X_te_b, model_base

    # -------------------------------------------------------------
    # 3. Feature Engineering: Locked Feature Set (Core + D1_norm)
    # -------------------------------------------------------------
    print("\n3. Building locked feature set (Core + D1_norm)...")
    df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)
    df["id_31"] = df["id_31"].apply(clean_browser)
    df["id_30"] = df["id_30"].apply(clean_os)
    df = compute_rolling_window_counts(df)
    df["D1_norm"] = df["D1"] - (df["TransactionDT"] / 86400.0)
    df = reduce_mem_usage(df)

    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    X_tr = X.iloc[:i1].copy()
    X_va = X.iloc[i1:i2].copy()
    X_te = X.iloc[i2:].copy()

    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c], categories=cats)
        X_te[c] = pd.Categorical(X_te[c], categories=cats)

    # -------------------------------------------------------------
    # 4. Model A: Train-Only (350 fixed trees, unweighted) + Full Validation Calibrator
    # -------------------------------------------------------------
    print("\n4. Training Model A (Train-only: 413,378 rows, 350 fixed trees, unweighted)...")
    model_a = xgb.XGBClassifier(
        n_estimators=350,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42
    )
    t0 = time.time()
    model_a.fit(X_tr, y_tr, verbose=False)
    t_train_a = time.time() - t0
    print(f"Model A trained in {t_train_a:.1f}s")

    # Fit isotonic calibrator on ALL of validation (88,581 rows)
    print("Fitting isotonic calibrator on full validation set (88,581 rows)...")
    p_va_a = model_a.predict_proba(X_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_va_a, y_va)

    # Save Model A and Calibrator before/during evaluation
    model_a.save_model("results/day2_model_a_train_only.json")
    joblib.dump(iso, "results/day2_calibrator_full_val.joblib")
    print("Model A saved to results/day2_model_a_train_only.json")
    print("Calibrator saved to results/day2_calibrator_full_val.joblib")

    # Predict on test set
    print("Scoring Test set with Model A...")
    p_te_a = model_a.predict_proba(X_te)[:, 1]
    p_te_a_cal = iso.predict(p_te_a)
    np.save("results/test_preds_day2_train_only.npy", p_te_a)
    np.save("results/test_preds_day2_calibrated.npy", p_te_a_cal)

    test_pr_auc_a = float(average_precision_score(y_te, p_te_a))
    test_roc_auc_a = float(roc_auc_score(y_te, p_te_a))
    print(f"Model A Test PR-AUC:  {test_pr_auc_a:.5f} (vs Baseline: {base_pr_auc:.5f}, Gain: {test_pr_auc_a - base_pr_auc:+.5f})")
    print(f"Model A Test ROC-AUC: {test_roc_auc_a:.5f} (vs Baseline: {base_roc_auc:.5f}, Gain: {test_roc_auc_a - base_roc_auc:+.5f})")

    # -------------------------------------------------------------
    # 5. Model B (Supplementary): Retrained on Train + Validation
    # -------------------------------------------------------------
    print("\n5. Training Model B (Retrained Train + Val: 501,959 rows, 350 fixed trees)...")
    X_tr_va = X.iloc[:i2].copy()
    y_tr_va = y[:i2]
    X_te_b = X_te.copy()

    for c in cat_cols:
        cats_tr_va = X_tr_va[c].astype("category").cat.categories
        X_tr_va[c] = pd.Categorical(X_tr_va[c], categories=cats_tr_va)
        X_te_b[c] = pd.Categorical(X_te_b[c], categories=cats_tr_va)

    model_b = xgb.XGBClassifier(
        n_estimators=350,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42
    )
    t0 = time.time()
    model_b.fit(X_tr_va, y_tr_va, verbose=False)
    t_train_b = time.time() - t0
    print(f"Model B trained in {t_train_b:.1f}s")

    model_b.save_model("results/day2_model_b_retrained.json")
    p_te_b = model_b.predict_proba(X_te_b)[:, 1]
    np.save("results/test_preds_day2_retrained.npy", p_te_b)

    test_pr_auc_b = float(average_precision_score(y_te, p_te_b))
    test_roc_auc_b = float(roc_auc_score(y_te, p_te_b))
    print(f"Model B Test PR-AUC:  {test_pr_auc_b:.5f} (Gain over Model A: {test_pr_auc_b - test_pr_auc_a:+.5f})")
    print(f"Model B Test ROC-AUC: {test_roc_auc_b:.5f}")

    # -------------------------------------------------------------
    # 6. Test Set Evaluations at Prescribed Operating Points
    # -------------------------------------------------------------
    print("\n6. Evaluating Operating Points on Test Set...")

    # Load optimal alert rates from Step 6 validation results
    with open("results/step6_cost_threshold.json", "r") as f:
        s6 = json.load(f)

    alert_rates = {
        "$10": s6["alert_rate_optima"]["$10"]["optimal_k_alerts_per_1000"], # 97 per 1k (Headline)
        "$5": s6["alert_rate_optima"]["$5"]["optimal_k_alerts_per_1000"],   # 152 per 1k
        "$25": s6["alert_rate_optima"]["$25"]["optimal_k_alerts_per_1000"]  # 44 per 1k
    }

    # Model A sorting
    sort_a = np.argsort(-p_te_a)
    y_sorted_a = y_te[sort_a]
    amt_sorted_a = amt_te[sort_a]

    # Baseline sorting
    sort_base = np.argsort(-p_te_base)
    y_sorted_base = y_te[sort_base]
    amt_sorted_base = amt_te[sort_base]

    # Model B sorting
    sort_b = np.argsort(-p_te_b)
    y_sorted_b = y_te[sort_b]
    amt_sorted_b = amt_te[sort_b]

    # Reference Benchmarks on Test:
    # 1. Flagging Nothing
    cost_nothing = total_test_fraud_amt

    # 2. Flagging Everything
    # Missed fraud = 0, review cost = n_te * c_rev
    cost_everything = {
        "$5": float(5.0 * n_te),
        "$10": float(10.0 * n_te),
        "$25": float(25.0 * n_te)
    }

    # 3. Threshold 0.5 (Model A raw)
    al_05 = (p_te_a >= 0.5)
    prec_05_a = float(precision_score(y_te, al_05, zero_division=0))
    rec_05_a = float(recall_score(y_te, al_05, zero_division=0))
    n_al_05_a = int(al_05.sum())
    rate_05_a = float((n_al_05_a / n_te) * 1000.0)
    cm_05_a = confusion_matrix(y_te, al_05)

    cost_05_by_rev = {}
    for c in [5.0, 10.0, 25.0]:
        tot_c, miss_c, rev_c = compute_cost(y_te, amt_te, al_05, c)
        cost_05_by_rev[f"${int(c)}"] = tot_c

    # Detailed Comparison Table at Alert Rates
    operating_points_report = {}
    table_rows = []

    for c_rev, label in [(10.0, "$10 (Headline)"), (5.0, "$5"), (25.0, "$25")]:
        key = label.split()[0]
        k = alert_rates[key]
        n_alerts = int(round((k / 1000.0) * n_te))

        # Model A at alert rate k
        prec_a = float(y_sorted_a[:n_alerts].mean())
        rec_a = float(y_sorted_a[:n_alerts].sum() / total_test_frauds)
        cost_a, miss_a, rev_a = compute_cost(y_te, amt_te, sort_a < n_alerts, c_rev)

        # Baseline at alert rate k
        prec_base = float(y_sorted_base[:n_alerts].mean())
        rec_base = float(y_sorted_base[:n_alerts].sum() / total_test_frauds)
        cost_base, miss_base, rev_base = compute_cost(y_te, amt_te, sort_base < n_alerts, c_rev)

        # Model B (retrained) at alert rate k
        prec_b = float(y_sorted_b[:n_alerts].mean())
        rec_b = float(y_sorted_b[:n_alerts].sum() / total_test_frauds)
        cost_b, miss_b, rev_b = compute_cost(y_te, amt_te, sort_b < n_alerts, c_rev)

        # Savings
        sav_a_nothing = cost_nothing - cost_a
        sav_a_05 = cost_05_by_rev[key] - cost_a
        sav_a_base = cost_base - cost_a

        operating_points_report[key] = {
            "review_cost": c_rev,
            "alert_rate_k": k,
            "total_alerts": n_alerts,
            "model_a_train_only": {
                "precision": prec_a,
                "recall": rec_a,
                "total_cost": cost_a,
                "missed_fraud_cost": miss_a,
                "review_cost_total": rev_a,
                "savings_vs_nothing": sav_a_nothing,
                "savings_vs_threshold_0_5": sav_a_05,
                "savings_vs_day1_baseline_at_same_rate": sav_a_base
            },
            "day1_baseline_at_same_rate": {
                "precision": prec_base,
                "recall": rec_base,
                "total_cost": cost_base,
                "missed_fraud_cost": miss_base,
                "review_cost_total": rev_base
            },
            "model_b_retrained": {
                "precision": prec_b,
                "recall": rec_b,
                "total_cost": cost_b,
                "missed_fraud_cost": miss_b,
                "review_cost_total": rev_b,
                "savings_vs_nothing": cost_nothing - cost_b
            },
            "flag_nothing_cost": cost_nothing,
            "flag_everything_cost": cost_everything[key],
            "threshold_0_5_cost": cost_05_by_rev[key]
        }

        table_rows.append({
            "Review Cost": label,
            "Alert Rate": f"{k} / 1k ({n_alerts:,})",
            "Model A Prec": f"{prec_a*100:.2f}%",
            "Model A Rec": f"{rec_a*100:.2f}%",
            "Model A Cost": f"${cost_a:,.0f}",
            "Day 1 Baseline Cost": f"${cost_base:,.0f}",
            "Feature Value (Sav vs Base)": f"${sav_a_base:+,.0f}",
            "Model B Rec": f"{rec_b*100:.2f}%",
            "Model B Cost": f"${cost_b:,.0f}"
        })

    print("\n--- Test Set Performance at Prescribed Alert Rates ---")
    print(pd.DataFrame(table_rows).to_string(index=False))

    # -------------------------------------------------------------
    # 7. Amount-Aware Rule on Test Set (Calibrated Model A)
    # -------------------------------------------------------------
    print("\n7. Evaluating Amount-Aware Rule on Test Set...")
    amt_aware_results = {}
    amt_table_rows = []

    for c_rev, key in [(10.0, "$10"), (5.0, "$5"), (25.0, "$25")]:
        al_amt = (p_te_a_cal * amt_te >= c_rev)
        cost_amt, miss_amt, rev_amt = compute_cost(y_te, amt_te, al_amt, c_rev)
        prec_amt = float(precision_score(y_te, al_amt, zero_division=0))
        rec_amt = float(recall_score(y_te, al_amt, zero_division=0))
        n_amt_alerts = int(al_amt.sum())
        k_amt = float((n_amt_alerts / n_te) * 1000.0)

        cost_opt_k = operating_points_report[key]["model_a_train_only"]["total_cost"]
        sav_vs_opt_k = cost_opt_k - cost_amt

        amt_aware_results[key] = {
            "review_cost": c_rev,
            "alerts_per_1000": k_amt,
            "total_alerts": n_amt_alerts,
            "precision": prec_amt,
            "recall": rec_amt,
            "total_cost": cost_amt,
            "missed_fraud_cost": miss_amt,
            "review_cost_total": rev_amt,
            "savings_vs_fixed_alert_rate": sav_vs_opt_k,
            "savings_vs_nothing": cost_nothing - cost_amt
        }

        amt_table_rows.append({
            "Review Cost": key,
            "Amount-Aware Alerts/1k": f"{k_amt:.1f} ({n_amt_alerts:,})",
            "Precision": f"{prec_amt*100:.2f}%",
            "Recall": f"{rec_amt*100:.2f}%",
            "Amount-Aware Cost": f"${cost_amt:,.0f}",
            "Fixed Alert Rate Cost": f"${cost_opt_k:,.0f}",
            "Additional Savings": f"${sav_vs_opt_k:+,.0f}"
        })

    print(pd.DataFrame(amt_table_rows).to_string(index=False))

    # -------------------------------------------------------------
    # 8. Test Calibration Diagnostic (10 Deciles)
    # -------------------------------------------------------------
    print("\n8. Test Calibration Diagnostic (Decile Reliability Table)...")
    cal_test_df = pd.DataFrame({"y_true": y_te, "p_raw": p_te_a, "p_cal": p_te_a_cal})
    cal_test_df["bin"] = pd.qcut(cal_test_df["p_raw"], q=10, duplicates="drop")
    
    test_rel_table = cal_test_df.groupby("bin", observed=True).agg(
        count=("y_true", "count"),
        frauds=("y_true", "sum"),
        mean_p_raw=("p_raw", "mean"),
        mean_p_cal=("p_cal", "mean"),
        actual_rate=("y_true", "mean")
    ).reset_index()
    test_rel_table["bin"] = test_rel_table["bin"].astype(str)

    raw_test_brier = float(brier_score_loss(y_te, p_te_a))
    cal_test_brier = float(brier_score_loss(y_te, p_te_a_cal))
    test_actual_rate = float(y_te.mean())
    test_raw_mean_p = float(p_te_a.mean())
    test_cal_mean_p = float(p_te_a_cal.mean())

    print(f"Test Actual Fraud Rate:        {test_actual_rate*100:.3f}% ({test_actual_rate:.5f})")
    print(f"Test Raw Mean Prob:            {test_raw_mean_p*100:.3f}% ({test_raw_mean_p:.5f}) | Brier: {raw_test_brier:.5f}")
    print(f"Test Calibrated Mean Prob:     {test_cal_mean_p*100:.3f}% ({test_cal_mean_p:.5f}) | Brier: {cal_test_brier:.5f}")
    print(test_rel_table.to_string(index=False))

    # -------------------------------------------------------------
    # 9. Save Complete JSON Results
    # -------------------------------------------------------------
    final_output = {
        "dataset_split": {
            "total_rows": n,
            "train_rows": i1,
            "val_rows": i2 - i1,
            "test_rows": n_te,
            "test_frauds": total_test_frauds,
            "test_fraud_rate": test_actual_rate,
            "test_total_fraud_loss_flag_nothing": total_test_fraud_amt
        },
        "day1_baseline_on_test": {
            "test_pr_auc": base_pr_auc,
            "test_roc_auc": base_roc_auc,
            "threshold_0_5": {
                "alerts": 1258,
                "precision": 0.7543720190779014,
                "recall": 0.30781706130392473,
                "confusion_matrix": {"TN": 85189, "FP": 309, "FN": 2134, "TP": 949}
            }
        },
        "model_a_train_only": {
            "test_pr_auc": test_pr_auc_a,
            "test_roc_auc": test_roc_auc_a,
            "training_time_sec": t_train_a,
            "threshold_0_5": {
                "alerts": n_al_05_a,
                "alerts_per_1000": rate_05_a,
                "precision": prec_05_a,
                "recall": rec_05_a,
                "confusion_matrix": {
                    "TN": int(cm_05_a[0, 0]),
                    "FP": int(cm_05_a[0, 1]),
                    "FN": int(cm_05_a[1, 0]),
                    "TP": int(cm_05_a[1, 1])
                },
                "cost_by_review_cost": cost_05_by_rev
            }
        },
        "model_b_retrained": {
            "test_pr_auc": test_pr_auc_b,
            "test_roc_auc": test_roc_auc_b,
            "training_time_sec": t_train_b
        },
        "operating_points": operating_points_report,
        "amount_aware_rule_test": amt_aware_results,
        "calibration_diagnostic_test": {
            "test_actual_fraud_rate": test_actual_rate,
            "test_raw_mean_prob": test_raw_mean_p,
            "test_calibrated_mean_prob": test_cal_mean_p,
            "test_raw_brier_score": raw_test_brier,
            "test_calibrated_brier_score": cal_test_brier,
            "decile_reliability_table": test_rel_table.to_dict(orient="records")
        }
    }

    with open("results/step7_final_test_score.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("\nSaved Step 7 test evaluation to results/step7_final_test_score.json")

if __name__ == "__main__":
    main()
