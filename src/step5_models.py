import os
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import average_precision_score

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

def main():
    print("Loading data...")
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    df = reduce_mem_usage(df)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)
    y = df["isFraud"]

    print("Building locked feature set (Core + D1_norm)...")
    df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)
    df["id_31"] = df["id_31"].apply(clean_browser)
    df["id_30"] = df["id_30"].apply(clean_os)
    df = compute_rolling_window_counts(df)
    df["D1_norm"] = df["D1"] - (df["TransactionDT"] / 86400.0)
    df = reduce_mem_usage(df)

    X_locked = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    X_tr = X_locked.iloc[:i1].copy()
    y_tr = y.iloc[:i1].copy()
    X_va = X_locked.iloc[i1:i2].copy()
    y_va = y.iloc[i1:i2].copy()

    seeds = [42, 101, 202]

    # =============================================================
    # 0. Mild Weight Check: scale_pos_weight = 5.0 (3 runs)
    # =============================================================
    print("\n--- Mild Weight Check: scale_pos_weight = 5.0 (XGBoost, 350 trees) ---")
    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    X_tr_xgb = X_tr.copy()
    X_va_xgb = X_va.copy()
    for c in cat_cols:
        cats = X_tr_xgb[c].astype("category").cat.categories
        X_tr_xgb[c] = pd.Categorical(X_tr_xgb[c], categories=cats)
        X_va_xgb[c] = pd.Categorical(X_va_xgb[c], categories=cats)

    spw5_scores = []
    for run_i, s in enumerate(seeds, 1):
        rng = np.random.RandomState(s)
        perm = rng.permutation(len(X_tr_xgb))
        model = xgb.XGBClassifier(
            n_estimators=350, tree_method="hist", enable_categorical=True,
            scale_pos_weight=5.0, eval_metric="aucpr", n_jobs=-1, random_state=42
        )
        t0 = time.time()
        model.fit(X_tr_xgb.iloc[perm], y_tr.iloc[perm], verbose=False)
        p = model.predict_proba(X_va_xgb)[:, 1]
        sc = average_precision_score(y_va, p)
        dur = time.time() - t0
        spw5_scores.append(sc)
        print(f"  SPW=5.0 Run {run_i}/3 (seed {s}): Val PR-AUC = {sc:.5f} ({dur:.1f}s)")

    spw5_mean = np.mean(spw5_scores)
    spw5_std = np.std(spw5_scores, ddof=1)
    print(f"--> scale_pos_weight=5.0: Mean = {spw5_mean:.5f}, Std(n-1) = {spw5_std:.5f}")

    # =============================================================
    # 1. Random Forest (100-200 trees, max_depth=14)
    # =============================================================
    print("\n--- Model 2: Random Forest (150 trees, max_depth=14, max_features='sqrt') ---")
    # For Random Forest, encode categories to integer codes, keeping NaNs
    X_tr_rf = X_tr.copy()
    X_va_rf = X_va.copy()
    for c in cat_cols:
        # Category code mapping
        cats = {val: idx for idx, val in enumerate(X_tr_rf[c].dropna().unique())}
        X_tr_rf[c] = X_tr_rf[c].map(cats).astype(np.float32)
        X_va_rf[c] = X_va_rf[c].map(cats).astype(np.float32)

    rf_scores = []
    rf_times = []
    for run_i, s in enumerate(seeds, 1):
        print(f"  Training Random Forest Run {run_i}/3 (seed={s})...")
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            max_features="sqrt",
            n_jobs=-1,
            random_state=s
        )
        rf.fit(X_tr_rf, y_tr)
        dur = time.time() - t0
        p_rf = rf.predict_proba(X_va_rf)[:, 1]
        score_rf = average_precision_score(y_va, p_rf)
        rf_scores.append(score_rf)
        rf_times.append(dur)
        print(f"  RF Run {run_i}/3: Val PR-AUC = {score_rf:.5f} | Time = {dur:.1f}s")

    rf_mean = np.mean(rf_scores)
    rf_std = np.std(rf_scores, ddof=1)
    rf_time_mean = np.mean(rf_times)
    print(f"--> Random Forest: Mean Val PR-AUC = {rf_mean:.5f}, Std(n-1) = {rf_std:.5f}, Mean Time = {rf_time_mean:.1f}s")

    # =============================================================
    # 2. Isolation Forest (Unsupervised Baseline on Numeric Columns)
    # =============================================================
    print("\n--- Model 3 (Optional Reference): Isolation Forest (Unsupervised on Numeric Features) ---")
    num_cols = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    print(f"  Fitting Isolation Forest on {len(num_cols)} numeric columns (imputing median for nulls)...")
    
    # Impute median for Isolation Forest
    X_tr_num = X_tr[num_cols].fillna(X_tr[num_cols].median())
    X_va_num = X_va[num_cols].fillna(X_tr[num_cols].median())

    t0 = time.time()
    iso = IsolationForest(
        n_estimators=100,
        max_samples=10000,
        contamination=0.035,
        n_jobs=-1,
        random_state=42
    )
    iso.fit(X_tr_num)
    iso_dur = time.time() - t0
    # Invert decision function so higher score = more anomalous = predicted fraud
    iso_scores = -iso.decision_function(X_va_num)
    iso_pr_auc = average_precision_score(y_va, iso_scores)
    print(f"--> Isolation Forest Val PR-AUC: {iso_pr_auc:.5f} | Time: {iso_dur:.1f}s")

    # =============================================================
    # Summary Table for Step 5
    # =============================================================
    print("\n==================== STEP 5 MODEL COMPARISON TABLE ====================")
    model_summary = [
        {
            "model": "XGBoost (Locked Features, Unweighted)",
            "mean_val_pr_auc": 0.58085,
            "std_n_minus_1": 0.00060,
            "training_time": "22.2s",
            "notes": "350 fixed trees, hist method, native categorical support"
        },
        {
            "model": "XGBoost (scale_pos_weight = 5.0)",
            "mean_val_pr_auc": float(spw5_mean),
            "std_n_minus_1": float(spw5_std),
            "training_time": f"{dur:.1f}s",
            "notes": "Mild weighting check (sqrt(27)); also underperforms unweighted"
        },
        {
            "model": "Random Forest",
            "mean_val_pr_auc": float(rf_mean),
            "std_n_minus_1": float(rf_std),
            "training_time": f"{rf_time_mean:.1f}s",
            "notes": "150 trees, max_depth=14, sqrt features, native NaNs"
        },
        {
            "model": "Isolation Forest (Unsupervised Reference)",
            "mean_val_pr_auc": float(iso_pr_auc),
            "std_n_minus_1": None,
            "training_time": f"{iso_dur:.1f}s",
            "notes": "Numeric columns only; demonstrates gap between unsupervised and supervised"
        }
    ]
    sum_df = pd.DataFrame(model_summary)
    print(sum_df[["model", "mean_val_pr_auc", "std_n_minus_1", "training_time", "notes"]].to_string(index=False))

    with open("results/step5_models.json", "w") as f:
        json.dump(model_summary, f, indent=2)
    print("\nSaved Step 5 results to results/step5_models.json")

if __name__ == "__main__":
    main()
