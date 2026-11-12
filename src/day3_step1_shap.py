import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
        
    new_cols = pd.DataFrame({
        "card1_trans_count_1h": np.concatenate(cnt_1h).astype(np.int16),
        "card1_trans_count_24h": np.concatenate(cnt_24h).astype(np.int16),
        "card1_trans_count_7d": np.concatenate(cnt_7d).astype(np.int16),
        "card1_sec_since_prev": np.concatenate(sec_prev),
        "card1_amt_ratio_to_prev_avg": np.concatenate(amt_ratio)
    }, index=df_sorted.index)
    df_sorted = pd.concat([df_sorted, new_cols], axis=1)
    df_result = df_sorted.sort_values("_orig_idx", kind="stable").drop(columns=["_orig_idx"]).reset_index(drop=True)
    return df_result

PLAIN_MEANING_FEATURES = {
    "TransactionAmt": "Transaction amount (USD)",
    "ProductCD": "Product code",
    "card4": "Card network",
    "card6": "Card type (credit/debit)",
    "P_emaildomain": "Purchaser email domain",
    "R_emaildomain": "Recipient email domain",
    "DeviceType": "Device type (desktop/mobile)",
    "DeviceInfo": "Device information",
    "id_30": "OS family",
    "id_31": "Browser family",
    "id_33": "Screen resolution",
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
    "card1_trans_count_1h": "Card transaction count (past 1h)",
    "card1_trans_count_24h": "Card transaction count (past 24h)",
    "card1_trans_count_7d": "Card transaction count (past 7d)",
    "card1_sec_since_prev": "Seconds since previous card transaction",
    "card1_amt_ratio_to_prev_avg": "Amount ratio vs card historical average"
}

def get_feature_description(col_name):
    if col_name in PLAIN_MEANING_FEATURES:
        return False, PLAIN_MEANING_FEATURES[col_name]
    return True, "an anonymised feature"

def compute_single_feature_adv_auc(X_tr_col, X_va_col):
    """Compute Train (0) vs Validation (1) single-feature adversarial ROC-AUC."""
    s_tr = X_tr_col.reset_index(drop=True)
    s_va = X_va_col.reset_index(drop=True)
    df_1d = pd.DataFrame({"feat": pd.concat([s_tr, s_va], ignore_index=True)})
    y_adv = np.concatenate([np.zeros(len(s_tr), dtype=int), np.ones(len(s_va), dtype=int)])
    if not pd.api.types.is_numeric_dtype(df_1d["feat"]):
        df_1d["feat"] = df_1d["feat"].astype("category")
    X_a_tr, X_a_va, y_a_tr, y_a_va = train_test_split(
        df_1d, y_adv, test_size=0.30, random_state=42, stratify=y_adv
    )
    clf = xgb.XGBClassifier(
        n_estimators=25,
        max_depth=3,
        learning_rate=0.2,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_a_tr, y_a_tr, verbose=False)
    p_a_va = clf.predict_proba(X_a_va)[:, 1]
    return float(roc_auc_score(y_a_va, p_a_va))

def format_val(val):
    if pd.isna(val):
        return "MISSING"
    if isinstance(val, (float, np.floating)):
        return round(float(val), 4)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return str(val)

def main():
    print("=" * 70)
    print("Day 3 Step 1: SHAP Explainability on Locked Model A (Validation Sample)")
    print("=" * 70)

    # 1. Load data & engineer locked features
    print("\n1. Loading data and building locked features (Core + D1_norm)...")
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    df = reduce_mem_usage(df)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)

    df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)
    df["id_31"] = df["id_31"].apply(clean_browser)
    df["id_30"] = df["id_30"].apply(clean_os)
    df = compute_rolling_window_counts(df)
    df["D1_norm"] = df["D1"] - (df["TransactionDT"] / 86400.0)
    df = reduce_mem_usage(df)

    y_va = df["isFraud"].iloc[i1:i2].values
    amt_va = df["TransactionAmt"].iloc[i1:i2].values
    txid_va = df["TransactionID"].iloc[i1:i2].values

    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    del df
    X_tr = X.iloc[:i1].copy()
    X_va = X.iloc[i1:i2].copy().reset_index(drop=True)
    del X

    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c], categories=cats)

    # 2. Load saved Model A and Calibrator
    print("\n2. Loading saved Model A and full-validation isotonic calibrator...")
    model_a = xgb.XGBClassifier()
    model_a.load_model("results/day2_model_a_train_only.json")
    iso = joblib.load("results/day2_calibrator_full_val.joblib")

    feat_names = model_a.get_booster().feature_names
    X_tr = X_tr[feat_names]
    X_va = X_va[feat_names]

    p_va = model_a.predict_proba(X_va)[:, 1]
    p_va_cal = iso.predict(p_va)

    # Top 97 alerts per 1,000 on full validation
    n_va = len(X_va)
    n_alerts_97 = int(round(0.097 * n_va))
    sort_idx_va = np.argsort(-p_va, kind="stable")
    alert_mask_va = np.zeros(n_va, dtype=bool)
    alert_mask_va[sort_idx_va[:n_alerts_97]] = True

    # 3. Sample 5,000 validation rows (fixed seed=42)
    print("\n3. Sampling 5,000 validation rows (seed=42)...")
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(n_va, size=5000, replace=False))

    X_sample = X_va.iloc[sample_idx].copy()
    y_sample = y_va[sample_idx]
    amt_sample = amt_va[sample_idx]
    txid_sample = txid_va[sample_idx]
    p_sample = p_va[sample_idx]
    p_cal_sample = p_va_cal[sample_idx]
    alert_sample = alert_mask_va[sample_idx]

    tp_mask = (y_sample == 1) & alert_sample
    fp_mask = (y_sample == 0) & alert_sample
    fn_mask = (y_sample == 1) & (~alert_sample)
    tn_mask = (y_sample == 0) & (~alert_sample)

    print(f"Sample stats (5,000 rows): Alerts in sample (top 97/1k) = {alert_sample.sum()} ({alert_sample.mean()*100:.2f}%)")
    print(f"  TP (True Alerts):  {tp_mask.sum()}")
    print(f"  FP (False Alerts): {fp_mask.sum()}")
    print(f"  FN (Missed Fraud): {fn_mask.sum()}")
    print(f"  TN (Clean Clear):  {tn_mask.sum()}")

    # 4. Run SHAP TreeExplainer on the 5,000 validation rows
    print("\n4. Running shap.TreeExplainer on 5,000 validation rows...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model_a)
    shap_values = explainer.shap_values(X_sample)
    shap_time = time.time() - t0
    base_value = float(explainer.expected_value)
    print(f"SHAP TreeExplainer completed in {shap_time:.2f}s | Shape: {shap_values.shape} | Base log-odds: {base_value:.5f}")

    # Verify additivity against raw margins
    margins = model_a.predict(X_sample, output_margin=True)
    max_add_err = float(np.max(np.abs((base_value + shap_values.sum(axis=1)) - margins)))
    print(f"SHAP Additivity check max error: {max_add_err:.6f}")

    np.save("results/day3_shap_values_val5000.npy", shap_values)
    np.save("results/day3_shap_sample_indices.npy", sample_idx)

    # 5. Rank Top 20 SHAP Features and compare with Adversarial Validation Drift
    print("\n5. Computing Top 20 features by mean(|SHAP|) and single-feature drift AUCs...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top20_idx = np.argsort(-mean_abs_shap)[:20]

    # Day 2 Adversarial Validation top 10 by gain + D columns
    adv_gain_top10 = {"M1", "dist1", "id_13", "id_31", "V4", "D15", "D11", "M7", "C12", "M9"}
    d_cols_growing = {"D1", "D2", "D4", "D6", "D10", "D11", "D14", "D15"}

    top20_rows = []
    for rank, f_idx in enumerate(top20_idx, start=1):
        fname = feat_names[f_idx]
        m_shap = float(mean_abs_shap[f_idx])
        is_masked, desc = get_feature_description(fname)
        adv_auc = compute_single_feature_adv_auc(X_tr[fname], X_va[fname])
        in_adv_top10 = fname in adv_gain_top10
        notes = []
        if in_adv_top10:
            notes.append("Top-10 Adv Gain")
        if fname in d_cols_growing:
            notes.append("Growing D-col")
        if fname == "D1_norm":
            notes.append("Epoch-normalized D1")
        if adv_auc >= 0.60:
            notes.append("High drift AUC")
        elif adv_auc >= 0.55:
            notes.append("Moderate drift AUC")

        top20_rows.append({
            "rank": rank,
            "feature": fname,
            "mean_abs_shap": m_shap,
            "is_masked": is_masked,
            "description": desc,
            "single_feat_adv_auc_tr_va": round(adv_auc, 4),
            "in_day2_adv_top10": in_adv_top10,
            "drift_notes": "; ".join(notes) if notes else "Low drift"
        })

    top20_df = pd.DataFrame(top20_rows)
    print("\n--- Top 20 SHAP Features (5,000 Validation Rows) vs Temporal Drift ---")
    print(top20_df.to_string(index=False))

    # 6. Select 15 representative alerts (5 TP, 5 FP, 5 FN) & extract Top 5 contributions each
    print("\n6. Selecting 5 True Alerts (TP), 5 False Alerts (FP), 5 Missed Frauds (FN)...")
    def pick_5_indices(bool_mask):
        idxs = np.where(bool_mask)[0]
        # Pick 5 evenly spaced rows sorted by raw predicted probability so we cover a spread
        order = np.argsort(-p_sample[idxs])
        sorted_idxs = idxs[order]
        if len(sorted_idxs) <= 5:
            return sorted_idxs
        pick_pos = np.linspace(0, len(sorted_idxs) - 1, 5, dtype=int)
        return sorted_idxs[pick_pos]

    tp_sample_pos = pick_5_indices(tp_mask)
    fp_sample_pos = pick_5_indices(fp_mask)
    fn_sample_pos = pick_5_indices(fn_mask)

    selected_cases = []
    for category, pos_list in [("true_alert_TP", tp_sample_pos), ("false_alert_FP", fp_sample_pos), ("missed_fraud_FN", fn_sample_pos)]:
        print(f"\n--- {category} ---")
        for pos in pos_list:
            sv_row = shap_values[pos]
            top5_f_idx = np.argsort(-np.abs(sv_row))[:5]
            top5_contribs = []
            for f_i in top5_f_idx:
                fname = feat_names[f_i]
                is_masked, desc = get_feature_description(fname)
                raw_val = format_val(X_sample.iloc[pos, f_i])
                top5_contribs.append({
                    "feature": fname,
                    "readable_name": desc,
                    "is_masked": is_masked,
                    "value": raw_val,
                    "shap_contribution": round(float(sv_row[f_i]), 5)
                })
            case_info = {
                "category": category,
                "val_index": int(sample_idx[pos]),
                "sample_pos": int(pos),
                "transaction_id": int(txid_sample[pos]),
                "actual_fraud": int(y_sample[pos]),
                "in_top_97_per_1k": bool(alert_sample[pos]),
                "amount_usd": round(float(amt_sample[pos]), 2),
                "raw_prob": round(float(p_sample[pos]), 5),
                "calibrated_prob": round(float(p_cal_sample[pos]), 5),
                "expected_loss_usd": round(float(p_cal_sample[pos] * amt_sample[pos]), 2),
                "top_5_shap_contributions": top5_contribs
            }
            selected_cases.append(case_info)
            contrib_str = ", ".join([
                f"{c['feature']} ({c['readable_name']}={c['value']}): {c['shap_contribution']:+.3f}"
                for c in top5_contribs
            ])
            print(f"  TxID {case_info['transaction_id']} | Amt=${case_info['amount_usd']:.2f} | CalP={case_info['calibrated_prob']*100:.1f}% | ExpLoss=${case_info['expected_loss_usd']:.2f} | Top5: {contrib_str}")

    # 7. Check drift summary across Top 20 SHAP
    high_or_mod_drift = [r for r in top20_rows if r["single_feat_adv_auc_tr_va"] >= 0.55 or r["in_day2_adv_top10"]]
    print(f"\nSummary: {len(high_or_mod_drift)} of top 20 SHAP features show moderate/high drift (Adv AUC >= 0.55 or Day 2 Top-10).")

    # Save JSONs
    step1_out = {
        "metadata": {
            "model": "day2_model_a_train_only.json",
            "sample_size": 5000,
            "random_seed": 42,
            "shap_time_sec": round(shap_time, 3),
            "base_value_log_odds": base_value,
            "additivity_max_error": max_add_err,
            "sample_confusion_at_97_per_1k": {
                "alerts_in_sample": int(alert_sample.sum()),
                "TP": int(tp_mask.sum()),
                "FP": int(fp_mask.sum()),
                "FN": int(fn_mask.sum()),
                "TN": int(tn_mask.sum())
            }
        },
        "top_20_shap_features": top20_rows,
        "selected_15_cases": selected_cases
    }
    with open("results/day3_step1_shap.json", "w") as f:
        json.dump(step1_out, f, indent=2)
    print("\nSaved Day 3 Step 1 SHAP results to results/day3_step1_shap.json")

if __name__ == "__main__":
    main()
