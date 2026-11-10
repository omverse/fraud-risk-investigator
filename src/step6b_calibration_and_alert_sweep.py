import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score, brier_score_loss
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

def compute_total_cost(y_true, amt_arr, alerts_bool, review_cost):
    missed_frauds_mask = (y_true == 1) & (~alerts_bool)
    missed_cost = float(amt_arr[missed_frauds_mask].sum())
    review_total = float(review_cost * alerts_bool.sum())
    return missed_cost + review_total, missed_cost, review_total

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

    print("Building locked feature set (Core + D1_norm)...")
    df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)
    df["id_31"] = df["id_31"].apply(clean_browser)
    df["id_30"] = df["id_30"].apply(clean_os)
    df = compute_rolling_window_counts(df)
    df["D1_norm"] = df["D1"] - (df["TransactionDT"] / 86400.0)
    df = reduce_mem_usage(df)

    y = df["isFraud"]
    val_amt = df["TransactionAmt"].iloc[i1:i2].values
    val_y = y.iloc[i1:i2].values
    n_val = len(val_y)

    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    X_tr = X.iloc[:i1].copy()
    y_tr = y.iloc[:i1].copy()
    X_va = X.iloc[i1:i2].copy()
    y_va = y.iloc[i1:i2].copy()

    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c], categories=cats)

    print("\nTraining locked XGBoost model on train split (350 fixed trees)...")
    model = xgb.XGBClassifier(
        n_estimators=350,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, verbose=False)
    p_va = model.predict_proba(X_va)[:, 1]
    val_pr_auc = average_precision_score(val_y, p_va)
    print(f"Model trained in {time.time() - t0:.1f}s | Val PR-AUC = {val_pr_auc:.5f}")

    # =============================================================
    # 1. Isotonic Recalibration on Validation Split
    # =============================================================
    print("\n================ 1. Isotonic Recalibration (Split Validation 50/50) ================")
    # Split validation chronologically into two halves
    half_idx = n_val // 2
    p_va_h1, p_va_h2 = p_va[:half_idx], p_va[half_idx:]
    y_va_h1, y_va_h2 = val_y[:half_idx], val_y[half_idx:]
    amt_h1, amt_h2 = val_amt[:half_idx], val_amt[half_idx:]
    n_h2 = len(y_va_h2)

    print(f"Validation Half 1 (Calibration Fit):  {len(y_va_h1):,} rows | Fraud rate: {y_va_h1.mean()*100:.3f}%")
    print(f"Validation Half 2 (Calibration Test): {len(y_va_h2):,} rows | Fraud rate: {y_va_h2.mean()*100:.3f}%")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_va_h1, y_va_h1)

    p_cal_h2 = iso.predict(p_va_h2)

    actual_rate_h2 = float(y_va_h2.mean())
    raw_mean_p_h2 = float(p_va_h2.mean())
    cal_mean_p_h2 = float(p_cal_h2.mean())
    raw_brier = float(brier_score_loss(y_va_h2, p_va_h2))
    cal_brier = float(brier_score_loss(y_va_h2, p_cal_h2))

    print(f"\nHalf 2 Actual Fraud Rate:        {actual_rate_h2:.5f} ({actual_rate_h2*100:.3f}%)")
    print(f"Half 2 Raw Mean Prob:            {raw_mean_p_h2:.5f} ({raw_mean_p_h2*100:.3f}%) | Brier: {raw_brier:.5f}")
    print(f"Half 2 Calibrated Mean Prob:     {cal_mean_p_h2:.5f} ({cal_mean_p_h2*100:.3f}%) | Brier: {cal_brier:.5f}")
    print(f"Calibrated Difference to Actual: {cal_mean_p_h2 - actual_rate_h2:+.5f}")

    # Decile Reliability Table on Half 2
    cal_df = pd.DataFrame({"y_true": y_va_h2, "p_raw": p_va_h2, "p_cal": p_cal_h2})
    cal_df["bin"] = pd.qcut(cal_df["p_raw"], q=10, duplicates="drop")
    rel_table = cal_df.groupby("bin", observed=True).agg(
        count=("y_true", "count"),
        mean_p_raw=("p_raw", "mean"),
        mean_p_cal=("p_cal", "mean"),
        actual_rate=("y_true", "mean")
    ).reset_index()
    rel_table["bin"] = rel_table["bin"].astype(str)
    print("\n--- Reliability Table on Validation Half 2 (10 Deciles) ---")
    print(rel_table.to_string(index=False))

    # =============================================================
    # 2. Alert-Rate Optimization (k from 1 to 300 per 1,000 txns)
    # =============================================================
    print("\n================ 2. Alert-Rate Sweep on Validation Set (k = 1 to 300 per 1k) ================")
    # Sort entire validation set by predicted probability descending
    sort_order = np.argsort(-p_va)
    y_sorted = val_y[sort_order]
    amt_sorted = val_amt[sort_order]
    total_val_fraud_amt = float(val_amt[val_y == 1].sum())

    review_costs = [5.0, 10.0, 25.0]
    k_range = np.arange(1, 301)
    
    alert_sweep_curves = {c: [] for c in review_costs}
    optima_by_cost = {}

    for c_rev in review_costs:
        best_k = None
        min_cost = float("inf")
        best_prec = 0
        best_rec = 0
        best_n_alerts = 0

        for k in k_range:
            n_alerts = int(round((k / 1000.0) * n_val))
            alerts_mask = np.zeros(n_val, dtype=bool)
            alerts_mask[:n_alerts] = True
            
            # Since data is sorted, alerts are first n_alerts elements
            missed_fraud_amt = float(amt_sorted[n_alerts:][y_sorted[n_alerts:] == 1].sum())
            review_total = float(c_rev * n_alerts)
            tot_cost = missed_fraud_amt + review_total
            alert_sweep_curves[c_rev].append(tot_cost)

            if tot_cost < min_cost:
                min_cost = tot_cost
                best_k = k
                best_n_alerts = n_alerts
                best_prec = float(y_sorted[:n_alerts].mean())
                best_rec = float(y_sorted[:n_alerts].sum() / val_y.sum())

        optima_by_cost[f"${int(c_rev)}"] = {
            "review_cost": c_rev,
            "optimal_k_alerts_per_1000": int(best_k),
            "total_alerts": int(best_n_alerts),
            "precision": float(best_prec),
            "recall": float(best_rec),
            "optimal_cost": float(min_cost),
            "cost_flag_nothing": total_val_fraud_amt,
            "savings_vs_nothing": float(total_val_fraud_amt - min_cost)
        }

    print("\n--- Alert-Rate Optima Summary (k* from 1 to 300) ---")
    opt_rows = []
    for k, v in optima_by_cost.items():
        opt_rows.append({
            "Review Cost": k,
            "Optimal Alert Rate (k*)": f"{v['optimal_k_alerts_per_1000']} / 1k",
            "Total Alerts": f"{v['total_alerts']:,}",
            "Precision": f"{v['precision']*100:.2f}%",
            "Recall": f"{v['recall']*100:.2f}%",
            "Total Cost": f"${v['optimal_cost']:,.0f}",
            "Savings vs. Nothing": f"${v['savings_vs_nothing']:,.0f}"
        })
    print(pd.DataFrame(opt_rows).to_string(index=False))

    # Fixed Capacities: 10, 20, 50 alerts per 1,000
    print("\n--- Fixed Analyst Capacities (10, 20, 50 alerts per 1,000) ---")
    fixed_caps = [10, 20, 50]
    cap_rows = []
    cap_data = {}
    for cap in fixed_caps:
        n_alerts = int(round((cap / 1000.0) * n_val))
        prec = float(y_sorted[:n_alerts].mean())
        rec = float(y_sorted[:n_alerts].sum() / val_y.sum())
        missed_amt = float(amt_sorted[n_alerts:][y_sorted[n_alerts:] == 1].sum())
        
        row_dict = {
            "capacity_k": cap,
            "alerts_count": n_alerts,
            "precision": prec,
            "recall": rec
        }
        for c_rev in review_costs:
            tot_c = missed_amt + (c_rev * n_alerts)
            row_dict[f"cost_${int(c_rev)}"] = tot_c

        cap_data[f"{cap}_per_1k"] = row_dict
        cap_rows.append({
            "Capacity": f"{cap} / 1k",
            "Alerts": f"{n_alerts:,}",
            "Precision": f"{prec*100:.1f}%",
            "Recall": f"{rec*100:.1f}%",
            "Cost @ $5": f"${row_dict['cost_$5']:,.0f}",
            "Cost @ $10": f"${row_dict['cost_$10']:,.0f}",
            "Cost @ $25": f"${row_dict['cost_$25']:,.0f}"
        })
    print(pd.DataFrame(cap_rows).to_string(index=False))

    # Day 1 threshold 0.5 stats from JSON/computation
    alerts_05 = (p_va >= 0.5)
    prec_05 = float(precision_score(val_y, alerts_05, zero_division=0))
    rec_05 = float(recall_score(val_y, alerts_05, zero_division=0))
    count_05 = int(alerts_05.sum())
    rate_05 = float((count_05 / n_val) * 1000.0)
    print(f"\nDay 1 Threshold 0.5 on Validation: {count_05:,} alerts ({rate_05:.1f} / 1k) | Precision = {prec_05*100:.2f}% | Recall = {rec_05*100:.2f}%")

    # =============================================================
    # 3. Global vs. Amount-Aware Rule on Validation Half 2
    # =============================================================
    print("\n================ 3. Global vs. Amount-Aware Rule on Validation Half 2 ================")
    tot_fraud_h2 = float(amt_h2[y_va_h2 == 1].sum())
    print(f"Half 2 Total Fraud Loss (Flag Nothing): ${tot_fraud_h2:,.2f}")

    # Benchmark: Cutoff 0.5 on Half 2
    alerts_05_h2 = (p_cal_h2 >= 0.5)
    
    amt_rule_comparisons = []
    amt_rule_data = {}

    for c_rev in review_costs:
        # Best global threshold on Half 2
        best_t_h2 = None
        min_cost_glob_h2 = float("inf")
        best_prec_glob = 0
        best_rec_glob = 0
        best_k_glob = 0

        for t in np.linspace(0.005, 0.50, 100):
            al = (p_cal_h2 >= t)
            cost, _, _ = compute_total_cost(y_va_h2, amt_h2, al, c_rev)
            if cost < min_cost_glob_h2:
                min_cost_glob_h2 = cost
                best_t_h2 = t
                best_k_glob = (al.sum() / n_h2) * 1000.0
                best_prec_glob = precision_score(y_va_h2, al, zero_division=0)
                best_rec_glob = recall_score(y_va_h2, al, zero_division=0)

        # Amount-aware rule: p_cal * amt >= c_rev
        al_amt = (p_cal_h2 * amt_h2 >= c_rev)
        cost_amt, _, _ = compute_total_cost(y_va_h2, amt_h2, al_amt, c_rev)
        k_amt = (al_amt.sum() / n_h2) * 1000.0
        prec_amt = precision_score(y_va_h2, al_amt, zero_division=0)
        rec_amt = recall_score(y_va_h2, al_amt, zero_division=0)

        cost_05_h2, _, _ = compute_total_cost(y_va_h2, amt_h2, alerts_05_h2, c_rev)
        savings_over_glob = min_cost_glob_h2 - cost_amt

        amt_rule_data[f"${int(c_rev)}"] = {
            "review_cost": c_rev,
            "best_global_threshold": float(best_t_h2),
            "best_global_cost": float(min_cost_glob_h2),
            "best_global_alerts_per_1000": float(best_k_glob),
            "amount_aware_cost": float(cost_amt),
            "amount_aware_alerts_per_1000": float(k_amt),
            "amount_aware_precision": float(prec_amt),
            "amount_aware_recall": float(rec_amt),
            "savings_vs_best_global": float(savings_over_glob),
            "cost_at_0_5": float(cost_05_h2)
        }

        amt_rule_comparisons.append({
            "Review Cost": f"${int(c_rev)}",
            "Best Global T": f"{best_t_h2:.3f}",
            "Global Alerts/1k": f"{best_k_glob:.1f}",
            "Best Global Cost": f"${min_cost_glob_h2:,.0f}",
            "Amount-Aware Alerts/1k": f"{k_amt:.1f}",
            "Amount-Aware Prec": f"{prec_amt*100:.1f}%",
            "Amount-Aware Rec": f"{rec_amt*100:.1f}%",
            "Amount-Aware Cost": f"${cost_amt:,.0f}",
            "Additional Savings": f"${savings_over_glob:,.0f}"
        })
    print(pd.DataFrame(amt_rule_comparisons).to_string(index=False))

    # =============================================================
    # 4. Generate Corrected Plot: Cost vs Alert Rate (1 to 300 per 1k)
    # =============================================================
    print("\n================ 4. Generating Corrected Plot ================")
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for (c_rev, costs), color in zip(alert_sweep_curves.items(), colors):
        k_opt = optima_by_cost[f"${int(c_rev)}"]["optimal_k_alerts_per_1000"]
        c_opt = optima_by_cost[f"${int(c_rev)}"]["optimal_cost"]
        plt.plot(k_range, costs, label=f"Review Cost = \\${int(c_rev)} (Opt: {k_opt}/1k, Cost: \\${c_opt:,.0f})", color=color, lw=2.2)
        plt.scatter([k_opt], [c_opt], color=color, s=85, zorder=5)

    # Benchmark lines
    plt.axhline(y=total_val_fraud_amt, color="crimson", linestyle="--", alpha=0.8, label=f"Flag Nothing (\\${total_val_fraud_amt:,.0f})")
    plt.axvline(x=rate_05, color="dimgray", linestyle=":", alpha=0.8, label=f"Day 1 (0.5 Cutoff: {rate_05:.1f}/1k)")

    # Capacity vertical lines
    for cap in fixed_caps:
        plt.axvline(x=cap, color="lightgray", linestyle="-.", alpha=0.6)

    plt.xlabel("Alert Rate (Alerts per 1,000 Transactions)", fontsize=12)
    plt.ylabel("Total Financial Cost ($ USD)", fontsize=12)
    plt.title("Step 6: Total Cost vs. Alert Rate on Validation Set", fontsize=14, pad=12)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    plt.xlim(1, 300)
    plt.ylim(100000, 550000)
    plt.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    plt.grid(True, alpha=0.3)

    plot_path = "results/day2_cost_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Corrected plot saved to: {plot_path}")

    # =============================================================
    # 5. Save Complete JSON Results
    # =============================================================
    save_data = {
        "calibration": {
            "validation_half2": {
                "actual_fraud_rate": actual_rate_h2,
                "raw_mean_prob": raw_mean_p_h2,
                "calibrated_mean_prob": cal_mean_p_h2,
                "raw_brier_score": raw_brier,
                "calibrated_brier_score": cal_brier,
                "deciles": rel_table.to_dict(orient="records")
            }
        },
        "alert_rate_optima": optima_by_cost,
        "fixed_capacities": cap_data,
        "day1_cutoff_0_5_validation": {
            "alerts_count": count_05,
            "alerts_per_1000": rate_05,
            "precision": prec_05,
            "recall": rec_05
        },
        "amount_aware_rule_half2": amt_rule_data
    }
    with open("results/step6_cost_threshold.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("Saved complete results to results/step6_cost_threshold.json")

if __name__ == "__main__":
    main()
