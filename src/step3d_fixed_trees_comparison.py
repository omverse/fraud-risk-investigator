import os
import time
import json
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

def train_fixed_trees(X_tr, y_tr, X_va, y_va, n_trees=350, random_state=42):
    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c], categories=cats)

    model = xgb.XGBClassifier(
        n_estimators=n_trees,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=random_state
    )
    t0 = time.time()
    # No early stopping: train strictly to n_trees
    model.fit(X_tr, y_tr, verbose=False)
    p_va = model.predict_proba(X_va)[:, 1]
    score = average_precision_score(y_va, p_va)
    dur = time.time() - t0
    return float(score), dur

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

    print("Building base features...")
    df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)
    df["id_31"] = df["id_31"].apply(clean_browser)
    df["id_30"] = df["id_30"].apply(clean_os)
    df = compute_rolling_window_counts(df)
    df = reduce_mem_usage(df)

    day_series = df["TransactionDT"] / 86400.0

    # 4 Configurations:
    # 1. core
    # 2. core + D1
    # 3. core + D1 + D2
    # 4. core + six D (D1, D2, D4, D10, D11, D15)
    
    base_X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()

    configs = {
        "1. Core Rolling": [],
        "2. Core + D1": ["D1"],
        "3. Core + D1 + D2": ["D1", "D2"],
        "4. Core + Six D": ["D1", "D2", "D4", "D10", "D11", "D15"]
    }

    seeds = [42, 101, 202]
    all_results = []

    print("\n================ Beginning 12 Fixed-Tree Runs (n_trees=350, No Early Stopping) ================")
    
    for cfg_name, d_cols in configs.items():
        print(f"\n--- Configuration: {cfg_name} ---")
        X_cfg = base_X.copy()
        for d in d_cols:
            X_cfg[f"{d}_norm"] = df[d] - day_series

        X_tr_raw = X_cfg.iloc[:i1].copy()
        y_tr_raw = y.iloc[:i1].copy()
        X_va = X_cfg.iloc[i1:i2].copy()
        y_va = y.iloc[i1:i2].copy()

        cfg_scores = []
        for run_idx, s in enumerate(seeds, 1):
            # Shuffle training rows with seed s
            rng = np.random.RandomState(s)
            perm = rng.permutation(len(X_tr_raw))
            X_tr_shuf = X_tr_raw.iloc[perm].copy()
            y_tr_shuf = y_tr_raw.iloc[perm].copy()

            score, dur = train_fixed_trees(X_tr_shuf, y_tr_shuf, X_va.copy(), y_va.copy(), n_trees=350, random_state=42)
            cfg_scores.append(score)
            print(f"  Run {run_idx}/3 (seed {s}): Val PR-AUC = {score:.5f} ({dur:.1f}s)")
            all_results.append({
                "config": cfg_name,
                "run": run_idx,
                "seed": s,
                "val_pr_auc": score,
                "train_time_s": dur
            })

    # Summary table
    res_df = pd.DataFrame(all_results)
    summary = res_df.groupby("config")["val_pr_auc"].agg(["mean", "std", "min", "max"]).reset_index()
    summary["gain_vs_baseline_mean"] = summary["mean"] - 0.5600
    summary["spread"] = summary["max"] - summary["min"]

    print("\n==================== 12-RUN SUMMARY TABLE ====================")
    print(summary.to_string(index=False))

    # Save to JSON
    with open("results/step3d_fixed_trees_comparison.json", "w") as f:
        json.dump({
            "individual_runs": all_results,
            "summary": summary.to_dict(orient="records")
        }, f, indent=2)
    print("\nSaved to results/step3d_fixed_trees_comparison.json")

if __name__ == "__main__":
    main()
