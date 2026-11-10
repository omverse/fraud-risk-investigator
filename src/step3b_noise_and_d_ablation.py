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

def train_and_eval(X_tr, y_tr, X_va, y_va, random_state=42, verbose=False):
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
        random_state=random_state
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=verbose)
    p_va = model.predict_proba(X_va)[:, 1]
    score = average_precision_score(y_va, p_va)
    dur = time.time() - t0
    return p_va, score, model.best_iteration, dur

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

    # =============================================================
    # Step 3b: Noise Study - 5 Baseline Runs with Shuffled Training Rows
    # =============================================================
    print("\n================ Step 3b: Noise Study (5 Baseline Runs) ================")
    y = df["isFraud"]
    X_base = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()

    X_base_tr = X_base.iloc[:i1].copy()
    y_base_tr = y.iloc[:i1].copy()
    X_base_va = X_base.iloc[i1:i2].copy()
    y_base_va = y.iloc[i1:i2].copy()

    seeds = [42, 101, 202, 303, 404]
    noise_results = []

    for run_i, s in enumerate(seeds, 1):
        print(f"\nRunning Baseline Run {run_i}/5 (seed={s})...")
        # Shuffle training row positions only (keep validation set intact)
        rng = np.random.RandomState(s)
        perm = rng.permutation(len(X_base_tr))
        X_tr_shuf = X_base_tr.iloc[perm].copy()
        y_tr_shuf = y_base_tr.iloc[perm].copy()

        p_va, score, best_iter, dur = train_and_eval(
            X_tr_shuf, y_tr_shuf, X_base_va.copy(), y_base_va.copy(), random_state=42
        )
        print(f"  Run {run_i}: Val PR-AUC = {score:.5f} | Best Iteration = {best_iter} | Time = {dur:.1f}s")
        noise_results.append({
            "run": run_i, "seed": s, "val_pr_auc": score, "best_iteration": best_iter, "time_s": dur
        })

    noise_df = pd.DataFrame(noise_results)
    mean_val = noise_df["val_pr_auc"].mean()
    std_val = noise_df["val_pr_auc"].std()
    min_val = noise_df["val_pr_auc"].min()
    max_val = noise_df["val_pr_auc"].max()
    spread_val = max_val - min_val

    mean_iter = noise_df["best_iteration"].mean()
    min_iter = noise_df["best_iteration"].min()
    max_iter = noise_df["best_iteration"].max()

    print("\n--- Step 3b Noise Study Summary ---")
    print(f"Validation PR-AUC across 5 runs:")
    print(f"  Mean:   {mean_val:.5f}")
    print(f"  Std:    {std_val:.5f}")
    print(f"  Min:    {min_val:.5f}")
    print(f"  Max:    {max_val:.5f}")
    print(f"  Spread: {spread_val:.5f} ({spread_val*100:.2f}%)")
    print(f"Best Iteration: Mean = {mean_iter:.1f}, Min = {min_iter}, Max = {max_iter}")

    # =============================================================
    # Step 3c: Per-Column D-Ablation
    # =============================================================
    print("\n================ Step 3c: Per-Column D-Ablation ================")
    # Build core rolling features first
    df_core = df.copy()
    df_core["hour_of_day"] = ((df_core["TransactionDT"] // 3600) % 24).astype(np.int8)
    df_core["day_of_week"] = ((df_core["TransactionDT"] // 86400) % 7).astype(np.int8)
    df_core["id_31"] = df_core["id_31"].apply(clean_browser)
    df_core["id_30"] = df_core["id_30"].apply(clean_os)
    df_core = compute_rolling_window_counts(df_core)
    df_core = reduce_mem_usage(df_core)

    day_series = df_core["TransactionDT"] / 86400.0

    # Base rolling features without any D normalization
    X_core = df_core.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    p_core, score_core, iter_core, _ = train_and_eval(
        X_core.iloc[:i1].copy(), y.iloc[:i1].copy(),
        X_core.iloc[i1:i2].copy(), y.iloc[i1:i2].copy()
    )
    print(f"\n[Core Rolling Model (No D-norm)] Val PR-AUC: {score_core:.5f} | Best Iter: {iter_core}")

    # Test columns: D1, D2, D4, D6, D10, D11, D14, D15
    candidate_d_cols = ["D1", "D2", "D4", "D6", "D10", "D11", "D14", "D15"]
    ablation_results = []

    for d in candidate_d_cols:
        print(f"\nTesting single normalization: {d}_norm = {d} - day...")
        X_single = X_core.copy()
        X_single[f"{d}_norm"] = df_core[d] - day_series
        
        p_d, score_d, iter_d, dur_d = train_and_eval(
            X_single.iloc[:i1].copy(), y.iloc[:i1].copy(),
            X_single.iloc[i1:i2].copy(), y.iloc[i1:i2].copy()
        )
        gain_vs_core = score_d - score_core
        gain_vs_base_mean = score_d - mean_val
        is_above_noise = bool(gain_vs_core > spread_val)
        print(f"  {d}_norm Val PR-AUC: {score_d:.5f} | Gain vs Core: {gain_vs_core:+.5f} | Above noise spread ({spread_val:.4f})? {is_above_noise}")
        ablation_results.append({
            "normalized_col": d,
            "val_pr_auc": float(score_d),
            "gain_vs_core": float(gain_vs_core),
            "gain_vs_base_mean": float(gain_vs_base_mean),
            "best_iter": int(iter_d),
            "time_s": float(dur_d),
            "above_noise_spread": bool(is_above_noise)
        })

    ablation_df = pd.DataFrame(ablation_results).sort_values("gain_vs_core", ascending=False)
    print("\n--- Single-Column D-Ablation Table ---")
    print(ablation_df.to_string(index=False))

    # Test combined winners (columns with positive gain above noise)
    top_cols = ablation_df[ablation_df["gain_vs_core"] > 0]["normalized_col"].tolist()
    print(f"\nTesting combined normalization with columns: {top_cols}...")
    X_comb = X_core.copy()
    for d in top_cols:
        X_comb[f"{d}_norm"] = df_core[d] - day_series

    p_comb, score_comb, iter_comb, dur_comb = train_and_eval(
        X_comb.iloc[:i1].copy(), y.iloc[:i1].copy(),
        X_comb.iloc[i1:i2].copy(), y.iloc[i1:i2].copy()
    )
    print(f"[Combined Winners ({len(top_cols)} cols)] Val PR-AUC: {score_comb:.5f} | Best Iter: {iter_comb} | Time: {dur_comb:.1f}s")
    print(f"Gain vs Core: {score_comb - score_core:+.5f} | Gain vs Baseline Mean: {score_comb - mean_val:+.5f}")

    # Paired Bootstrap: Combined Winners vs Core and vs Baseline
    print("\n--- Paired Bootstrap (200 iterations): Combined Winners vs Baseline ---")
    y_va_arr = y.iloc[i1:i2].values
    np.random.seed(42)
    diffs_vs_base = []
    diffs_vs_core = []
    for _ in range(200):
        b_idx = np.random.randint(0, len(y_va_arr), size=len(y_va_arr))
        if np.sum(y_va_arr[b_idx]) == 0:
            continue
        pr_b = average_precision_score(y_va_arr[b_idx], p_va[b_idx])
        pr_c = average_precision_score(y_va_arr[b_idx], p_core[b_idx])
        pr_w = average_precision_score(y_va_arr[b_idx], p_comb[b_idx])
        diffs_vs_base.append(pr_w - pr_b)
        diffs_vs_core.append(pr_w - pr_c)

    ci_base = np.percentile(diffs_vs_base, [2.5, 97.5])
    ci_core = np.percentile(diffs_vs_core, [2.5, 97.5])
    print(f"Combined vs Baseline 95% CI: [{ci_base[0]:+.5f}, {ci_base[1]:+.5f}]")
    print(f"Combined vs Core Rolling 95% CI: [{ci_core[0]:+.5f}, {ci_core[1]:+.5f}]")

    # Save summary
    summary_data = {
        "baseline_5_runs": {
            "mean_val_pr_auc": float(mean_val),
            "std_val_pr_auc": float(std_val),
            "min_val_pr_auc": float(min_val),
            "max_val_pr_auc": float(max_val),
            "spread_yardstick": float(spread_val),
            "runs": noise_results
        },
        "d_ablation": ablation_results,
        "combined_d_norm": {
            "columns": top_cols,
            "val_pr_auc": float(score_comb),
            "gain_vs_baseline_mean": float(score_comb - mean_val),
            "gain_vs_core": float(score_comb - score_core),
            "ci_vs_baseline": [float(ci_base[0]), float(ci_base[1])],
            "ci_vs_core": [float(ci_core[0]), float(ci_core[1])]
        }
    }
    import json
    with open("results/step3b_noise_and_d_ablation.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print("\nResults saved to results/step3b_noise_and_d_ablation.json")

if __name__ == "__main__":
    main()
