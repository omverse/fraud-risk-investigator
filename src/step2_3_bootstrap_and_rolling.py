import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
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
    """
    Computes rolling window counts strictly backward-looking per card1.
    Windows: 1 hour (3600s), 24 hours (86400s), 7 days (604800s).
    Also computes seconds since previous transaction and amount ratio to previous avg.
    """
    print("\n--- Computing Rolling-Window Card Features (1h, 24h, 7d) ---")
    t0 = time.time()
    # Ensure dataframe has a stable original index to re-align
    df["_orig_idx"] = np.arange(len(df))
    # Sort by card1 and TransactionDT
    df_sorted = df.sort_values(["card1", "TransactionDT"], kind="stable").reset_index(drop=True)
    
    times = df_sorted["TransactionDT"].values
    cards = df_sorted["card1"].values
    amts = df_sorted["TransactionAmt"].values
    
    # Identify card boundaries
    splits = np.where(cards[:-1] != cards[1:])[0] + 1
    group_times = np.split(times, splits)
    group_amts = np.split(amts, splits)
    
    cnt_1h, cnt_24h, cnt_7d = [], [], []
    sec_prev, amt_ratio = [], []
    
    for g_t, g_a in zip(group_times, group_amts):
        n_g = len(g_t)
        # 1. Rolling counts
        curr_idx = np.arange(n_g)
        cnt_1h.append(curr_idx - np.searchsorted(g_t, g_t - 3600, side="left"))
        cnt_24h.append(curr_idx - np.searchsorted(g_t, g_t - 86400, side="left"))
        cnt_7d.append(curr_idx - np.searchsorted(g_t, g_t - 604800, side="left"))
        
        # 2. Sec since previous
        s_p = np.empty(n_g, dtype=np.float32)
        s_p[0] = np.nan
        if n_g > 1:
            s_p[1:] = g_t[1:] - g_t[:-1]
        sec_prev.append(s_p)
        
        # 3. Amount ratio to previous average
        a_r = np.empty(n_g, dtype=np.float32)
        a_r[0] = np.nan
        if n_g > 1:
            cumsum_prev = np.cumsum(g_a) - g_a
            prev_counts = np.arange(n_g)
            # For index 0, prev_counts is 0 (handled by nan above)
            prev_avg = cumsum_prev[1:] / prev_counts[1:]
            a_r[1:] = g_a[1:] / prev_avg
        amt_ratio.append(a_r)
        
    df_sorted["card1_trans_count_1h"] = np.concatenate(cnt_1h).astype(np.int16)
    df_sorted["card1_trans_count_24h"] = np.concatenate(cnt_24h).astype(np.int16)
    df_sorted["card1_trans_count_7d"] = np.concatenate(cnt_7d).astype(np.int16)
    df_sorted["card1_sec_since_prev"] = np.concatenate(sec_prev)
    df_sorted["card1_amt_ratio_to_prev_avg"] = np.concatenate(amt_ratio)
    
    # Restore original TransactionDT stable sort
    df_result = df_sorted.sort_values("_orig_idx", kind="stable").drop(columns=["_orig_idx"]).reset_index(drop=True)
    print(f"Rolling-window card features computed in {time.time() - t0:.2f}s")
    
    return df_result

def run_paired_bootstrap(y_true, p_base, p_new, n_iterations=200, seed=42):
    print(f"\n--- Running Paired Bootstrap ({n_iterations} iterations) ---")
    np.random.seed(seed)
    n = len(y_true)
    diffs = []
    y_arr = np.array(y_true)
    
    for i in range(n_iterations):
        indices = np.random.randint(0, n, size=n)
        y_b = y_arr[indices]
        # Only compute if at least one positive present
        if np.sum(y_b) == 0:
            continue
        pr_base = average_precision_score(y_b, p_base[indices])
        pr_new = average_precision_score(y_b, p_new[indices])
        diffs.append(pr_new - pr_base)
        
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    mean_diff = np.mean(diffs)
    pct_positive = np.mean(diffs > 0) * 100
    
    print(f"Mean PR-AUC Difference: {mean_diff:+.5f}")
    print(f"95% Bootstrap Confidence Interval: [{ci_lower:+.5f}, {ci_upper:+.5f}]")
    print(f"Percentage of iterations where new > baseline: {pct_positive:.1f}%")
    return mean_diff, ci_lower, ci_upper, pct_positive

def train_and_eval_xgb(X_tr, y_tr, X_va, y_va, name="Model"):
    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c], categories=cats)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_va = model.predict_proba(X_va)[:, 1]
    score = average_precision_score(y_va, p_va)
    duration = time.time() - t0
    print(f"[{name}] Val PR-AUC: {score:.5f} | Best Iteration: {model.best_iteration} | Train Time: {duration:.2f}s")
    return p_va, score, duration, model

def main():
    print("Loading raw merged data...")
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    df = reduce_mem_usage(df)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)

    # =============================================================
    # 1. Baseline Model Validation Predictions
    # =============================================================
    print("\n================ 1. Generating Baseline Val Predictions ================")
    y = df["isFraud"]
    X_base = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    
    p_va_base, score_base, _, _ = train_and_eval_xgb(
        X_base.iloc[:i1].copy(), y.iloc[:i1],
        X_base.iloc[i1:i2].copy(), y.iloc[i1:i2],
        name="Baseline (Day 1)"
    )
    os.makedirs("results", exist_ok=True)
    np.save("results/val_preds_baseline.npy", p_va_base)

    # =============================================================
    # 2. Features v1 (unbounded count) Model Validation Predictions
    # =============================================================
    print("\n================ 2. Generating Features v1 Val Predictions ================")
    df_v1 = df.copy()
    df_v1["hour_of_day"] = ((df_v1["TransactionDT"] // 3600) % 24).astype(np.int8)
    df_v1["day_of_week"] = ((df_v1["TransactionDT"] // 86400) % 7).astype(np.int8)
    df_v1["id_31"] = df_v1["id_31"].apply(clean_browser)
    df_v1["id_30"] = df_v1["id_30"].apply(clean_os)

    # v1 unbounded card features
    df_v1["card1_prev_trans_count"] = df_v1.groupby("card1").cumcount()
    prev_dt = df_v1.groupby("card1")["TransactionDT"].shift(1)
    df_v1["card1_sec_since_prev"] = df_v1["TransactionDT"] - prev_dt
    prev_amt_cumsum = df_v1.groupby("card1")["TransactionAmt"].cumsum() - df_v1["TransactionAmt"]
    prev_amt_avg = prev_amt_cumsum / df_v1["card1_prev_trans_count"]
    df_v1["card1_amt_ratio_to_prev_avg"] = df_v1["TransactionAmt"] / prev_amt_avg
    first_mask = df_v1["card1_prev_trans_count"] == 0
    df_v1.loc[first_mask, ["card1_prev_trans_count", "card1_sec_since_prev", "card1_amt_ratio_to_prev_avg"]] = np.nan

    X_v1 = df_v1.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    p_va_v1, score_v1, _, _ = train_and_eval_xgb(
        X_v1.iloc[:i1].copy(), y.iloc[:i1],
        X_v1.iloc[i1:i2].copy(), y.iloc[i1:i2],
        name="Features v1 (Unbounded Count)"
    )
    np.save("results/val_preds_features_v1.npy", p_va_v1)
    del df_v1, X_v1

    # =============================================================
    # 3. Paired Bootstrap on Validation Predictions
    # =============================================================
    y_va = y.iloc[i1:i2].values
    mean_diff, ci_l, ci_u, pct_pos = run_paired_bootstrap(
        y_va, p_va_base, p_va_v1, n_iterations=200, seed=42
    )

    # =============================================================
    # 4. Step 3: Rolling Window Card Features (1h, 24h, 7d)
    # =============================================================
    print("\n================ 4. Training Model with Rolling Window Counts ================")
    df_rolling = df.copy()
    df_rolling["hour_of_day"] = ((df_rolling["TransactionDT"] // 3600) % 24).astype(np.int8)
    df_rolling["day_of_week"] = ((df_rolling["TransactionDT"] // 86400) % 7).astype(np.int8)
    df_rolling["id_31"] = df_rolling["id_31"].apply(clean_browser)
    df_rolling["id_30"] = df_rolling["id_30"].apply(clean_os)
    df_rolling = compute_rolling_window_counts(df_rolling)

    X_rolling = df_rolling.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    p_va_rolling, score_rolling, dur_rolling, _ = train_and_eval_xgb(
        X_rolling.iloc[:i1].copy(), y.iloc[:i1],
        X_rolling.iloc[i1:i2].copy(), y.iloc[i1:i2],
        name="Features v2 (Rolling Windows 1h/24h/7d)"
    )
    np.save("results/val_preds_features_rolling.npy", p_va_rolling)

    # =============================================================
    # 5. D-column Normalization (D - TransactionDT/86400)
    # =============================================================
    print("\n================ 5. Testing Growing D-Columns Normalization ================")
    growing_d_cols = ["D1", "D2", "D4", "D10", "D11", "D15"]
    df_dnorm = df_rolling.copy()
    day_series = df_dnorm["TransactionDT"] / 86400.0
    for d in growing_d_cols:
        df_dnorm[f"{d}_norm"] = df_dnorm[d] - day_series

    X_dnorm = df_dnorm.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    p_va_dnorm, score_dnorm, dur_dnorm, _ = train_and_eval_xgb(
        X_dnorm.iloc[:i1].copy(), y.iloc[:i1],
        X_dnorm.iloc[i1:i2].copy(), y.iloc[i1:i2],
        name="Features v2 + Normalized D-cols"
    )
    np.save("results/val_preds_features_dnorm.npy", p_va_dnorm)

    # Compare rolling with baseline via paired bootstrap
    print("\n--- Paired Bootstrap: Rolling Windows vs Baseline ---")
    run_paired_bootstrap(y_va, p_va_base, p_va_rolling, n_iterations=200, seed=42)

    # Summary table
    print("\n==================== SUMMARY TABLE ====================")
    print(f"1. Baseline Validation PR-AUC:             {score_base:.5f}")
    print(f"2. Features v1 (Unbounded count):          {score_v1:.5f} (Diff: {score_v1 - score_base:+.5f})")
    print(f"   v1 95% Bootstrap CI on Diff:            [{ci_l:+.5f}, {ci_u:+.5f}] ({pct_pos:.1f}% positive)")
    print(f"3. Features v2 (Rolling Windows 1h/24h/7d): {score_rolling:.5f} (Diff: {score_rolling - score_base:+.5f})")
    print(f"4. Features v2 + Normalized D-cols:        {score_dnorm:.5f} (Diff: {score_dnorm - score_rolling:+.5f} vs rolling)")

if __name__ == "__main__":
    main()
