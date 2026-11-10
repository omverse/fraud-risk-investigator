import os
import json
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

def train_fixed_trees(X_tr, y_tr, X_va, y_va, n_trees=350, scale_pos_weight=1.0, random_state=42):
    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c], categories=cats)

    model = xgb.XGBClassifier(
        n_estimators=n_trees,
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=random_state
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, verbose=False)
    p_va = model.predict_proba(X_va)[:, 1]
    score = average_precision_score(y_va, p_va)
    dur = time.time() - t0
    return float(score), p_va, dur

def main():
    print("Loading raw data...")
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    df = reduce_mem_usage(df)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)
    y = df["isFraud"]

    # =============================================================
    # 0. Raw Day 1 Baseline at 350 Fixed Trees (3 runs)
    # =============================================================
    print("\n================ 0. Day 1 Baseline at 350 Fixed Trees (3 runs) ================")
    X_base = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    X_base_tr = X_base.iloc[:i1].copy()
    y_base_tr = y.iloc[:i1].copy()
    X_base_va = X_base.iloc[i1:i2].copy()
    y_base_va = y.iloc[i1:i2].copy()

    seeds = [42, 101, 202]
    base_scores = []
    for run_i, s in enumerate(seeds, 1):
        rng = np.random.RandomState(s)
        perm = rng.permutation(len(X_base_tr))
        score, _, dur = train_fixed_trees(
            X_base_tr.iloc[perm].copy(), y_base_tr.iloc[perm].copy(),
            X_base_va.copy(), y_base_va.copy(),
            n_trees=350, scale_pos_weight=1.0, random_state=42
        )
        base_scores.append(score)
        print(f"  Day 1 Base Run {run_i}/3 (seed {s}): Val PR-AUC = {score:.5f} ({dur:.1f}s)")
    base_mean = np.mean(base_scores)
    base_std = np.std(base_scores)
    print(f"--> Day 1 Baseline (350 fixed trees): Mean = {base_mean:.5f}, Std = {base_std:.5f}, Range = [{min(base_scores):.5f}, {max(base_scores):.5f}]")
    del X_base, X_base_tr, y_base_tr, X_base_va, y_base_va

    # =============================================================
    # Build Locked Feature Set: Core + D1_norm
    # =============================================================
    print("\nBuilding locked feature set (Core + D1_norm)...")
    df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)
    df["id_31"] = df["id_31"].apply(clean_browser)
    df["id_30"] = df["id_30"].apply(clean_os)
    df = compute_rolling_window_counts(df)
    df["D1_norm"] = df["D1"] - (df["TransactionDT"] / 86400.0)
    df = reduce_mem_usage(df)

    X_locked = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    X_tr_full = X_locked.iloc[:i1].copy()
    y_tr_full = y.iloc[:i1].copy()
    X_va = X_locked.iloc[i1:i2].copy()
    y_va = y.iloc[i1:i2].copy()

    # Calculate scale_pos_weight
    n_neg = (y_tr_full == 0).sum()
    n_pos = (y_tr_full == 1).sum()
    spw = n_neg / n_pos
    print(f"\nTrain set class distribution: {n_neg:,} negative, {n_pos:,} positive (scale_pos_weight = {spw:.3f})")

    # =============================================================
    # Step 4 Experiments: 3 runs each
    # 1. Unweighted (Baseline for locked features)
    # 2. scale_pos_weight = spw (~27.43)
    # 3. Random Undersampling (1:1)
    # 4. Random Undersampling (1:4)
    # =============================================================
    results = {}

    # Setup 1: Unweighted
    print("\n--- Setup 1: No Weighting (Unweighted) ---")
    s1_scores = []
    for run_i, s in enumerate(seeds, 1):
        rng = np.random.RandomState(s)
        perm = rng.permutation(len(X_tr_full))
        score, p_va, dur = train_fixed_trees(
            X_tr_full.iloc[perm].copy(), y_tr_full.iloc[perm].copy(),
            X_va.copy(), y_va.copy(),
            n_trees=350, scale_pos_weight=1.0, random_state=42
        )
        s1_scores.append(score)
        print(f"  Run {run_i}/3 (seed {s}): Val PR-AUC = {score:.5f} ({dur:.1f}s)")
    results["No Weighting"] = s1_scores

    # Setup 2: scale_pos_weight (~27.43)
    print(f"\n--- Setup 2: scale_pos_weight = {spw:.3f} ---")
    s2_scores = []
    for run_i, s in enumerate(seeds, 1):
        rng = np.random.RandomState(s)
        perm = rng.permutation(len(X_tr_full))
        score, p_va_spw, dur = train_fixed_trees(
            X_tr_full.iloc[perm].copy(), y_tr_full.iloc[perm].copy(),
            X_va.copy(), y_va.copy(),
            n_trees=350, scale_pos_weight=spw, random_state=42
        )
        s2_scores.append(score)
        print(f"  Run {run_i}/3 (seed {s}): Val PR-AUC = {score:.5f} ({dur:.1f}s)")
    results["scale_pos_weight (27.4)"] = s2_scores

    # Setup 3: Random Undersampling 1:1
    print("\n--- Setup 3: Random Undersampling (1:1 ratio) ---")
    s3_scores = []
    pos_idx = np.where(y_tr_full == 1)[0]
    neg_idx = np.where(y_tr_full == 0)[0]
    for run_i, s in enumerate(seeds, 1):
        rng = np.random.RandomState(s)
        sampled_neg = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        sub_idx = np.concatenate([pos_idx, sampled_neg])
        rng.shuffle(sub_idx)
        
        score, _, dur = train_fixed_trees(
            X_tr_full.iloc[sub_idx].copy(), y_tr_full.iloc[sub_idx].copy(),
            X_va.copy(), y_va.copy(),
            n_trees=350, scale_pos_weight=1.0, random_state=42
        )
        s3_scores.append(score)
        print(f"  Run {run_i}/3 (seed {s}): Val PR-AUC = {score:.5f} ({dur:.1f}s, train rows: {len(sub_idx):,})")
    results["Undersampling (1:1)"] = s3_scores

    # Setup 4: Random Undersampling 1:4 (retaining more majority data)
    print("\n--- Setup 4: Random Undersampling (1:4 ratio) ---")
    s4_scores = []
    for run_i, s in enumerate(seeds, 1):
        rng = np.random.RandomState(s)
        sampled_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx)*4), replace=False)
        sub_idx = np.concatenate([pos_idx, sampled_neg])
        rng.shuffle(sub_idx)
        
        score, _, dur = train_fixed_trees(
            X_tr_full.iloc[sub_idx].copy(), y_tr_full.iloc[sub_idx].copy(),
            X_va.copy(), y_va.copy(),
            n_trees=350, scale_pos_weight=1.0, random_state=42
        )
        s4_scores.append(score)
        print(f"  Run {run_i}/3 (seed {s}): Val PR-AUC = {score:.5f} ({dur:.1f}s, train rows: {len(sub_idx):,})")
    results["Undersampling (1:4)"] = s4_scores

    # =============================================================
    # Client Overlap Check (Validation evaluation on seen vs unseen card1)
    # =============================================================
    print("\n================ Client Overlap Check (Seen vs Unseen card1) ================")
    train_cards = set(df["card1"].iloc[:i1].unique())
    val_cards = df["card1"].iloc[i1:i2]
    seen_mask = val_cards.isin(train_cards).values
    unseen_mask = ~seen_mask
    print(f"Validation rows: {len(val_cards):,}")
    print(f"  Seen cards in train:   {seen_mask.sum():,} ({seen_mask.mean()*100:.1f}%) | Fraud rate: {y_va[seen_mask].mean()*100:.2f}%")
    print(f"  Unseen cards in train: {unseen_mask.sum():,} ({unseen_mask.mean()*100:.1f}%) | Fraud rate: {y_va[unseen_mask].mean()*100:.2f}%")

    pr_seen = average_precision_score(y_va[seen_mask], p_va[seen_mask])
    pr_unseen = average_precision_score(y_va[unseen_mask], p_va[unseen_mask])
    print(f"Validation PR-AUC on SEEN cards:   {pr_seen:.5f}")
    print(f"Validation PR-AUC on UNSEEN cards: {pr_unseen:.5f}")

    # =============================================================
    # Summary Table
    # =============================================================
    print("\n==================== STEP 4 SUMMARY TABLE ====================")
    rows = []
    # Add Day 1 baseline
    rows.append({
        "setup": "Day 1 Baseline (Raw, Fixed 350)",
        "mean": base_mean,
        "std": base_std,
        "min": min(base_scores),
        "max": max(base_scores),
        "spread": max(base_scores) - min(base_scores)
    })
    for k, v in results.items():
        rows.append({
            "setup": f"Locked Features ({k})",
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(min(v)),
            "max": float(max(v)),
            "spread": float(max(v) - min(v))
        })
    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False))

    # Save to JSON
    with open("results/step4_imbalance.json", "w") as f:
        json.dump({
            "results": rows,
            "individual_runs": results,
            "client_overlap": {
                "seen_count": int(seen_mask.sum()),
                "unseen_count": int(unseen_mask.sum()),
                "pr_auc_seen": float(pr_seen),
                "pr_auc_unseen": float(pr_unseen)
            }
        }, f, indent=2)
    print("\nSaved Step 4 results to results/step4_imbalance.json")

if __name__ == "__main__":
    main()
