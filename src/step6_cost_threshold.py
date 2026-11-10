import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score, brier_score_loss
from sklearn.calibration import calibration_curve

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
    """
    Total Cost = (sum of amounts of missed frauds) + (review_cost * number of alerts)
    """
    missed_frauds_mask = (y_true == 1) & (~alerts_bool)
    missed_cost = amt_arr[missed_frauds_mask].sum()
    review_total = review_cost * alerts_bool.sum()
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

    print("Building locked features (Core + D1_norm)...")
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

    print("\nTraining final locked XGBoost model on training set (350 fixed trees)...")
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
    # 1. Calibration Analysis
    # =============================================================
    print("\n================ 1. Calibration Analysis ================")
    actual_fraud_rate = float(val_y.mean())
    mean_pred_prob = float(p_va.mean())
    brier = float(brier_score_loss(val_y, p_va))

    print(f"Actual Validation Fraud Rate: {actual_fraud_rate:.5f} ({actual_fraud_rate*100:.3f}%)")
    print(f"Mean Predicted Probability:   {mean_pred_prob:.5f} ({mean_pred_prob*100:.3f}%)")
    print(f"Difference (Mean P - Actual): {mean_pred_prob - actual_fraud_rate:+.5f}")
    print(f"Brier Score Loss:             {brier:.5f}")

    # 10 Bins Calibration Table (quantile-based and uniform)
    print("\nReliability Bins (10 Quantile Deciles):")
    calib_df = pd.DataFrame({"y_true": val_y, "p_pred": p_va})
    calib_df["bin"] = pd.qcut(calib_df["p_pred"], q=10, duplicates="drop")
    bin_stats = calib_df.groupby("bin", observed=True).agg(
        count=("y_true", "count"),
        mean_pred=("p_pred", "mean"),
        actual_rate=("y_true", "mean")
    ).reset_index()
    bin_stats["bin"] = bin_stats["bin"].astype(str)
    print(bin_stats.to_string(index=False))

    # =============================================================
    # 2. Benchmark Costs on Validation Set
    # =============================================================
    total_fraud_loss = float(val_amt[val_y == 1].sum())
    total_txns = n_val
    print("\n================ 2. Benchmark Scenarios ================")
    print(f"Total Transactions in Validation: {total_txns:,}")
    print(f"Total Fraud Count in Validation:  {val_y.sum():,}")
    print(f"Total Fraud Amount in Validation: ${total_fraud_loss:,.2f}")
    print(f"Scenario 1: Flagging Nothing -> Total Cost = ${total_fraud_loss:,.2f} (all fraud lost, $0 review)")

    # Day 1 threshold 0.5 alerts
    alerts_05 = (p_va >= 0.5)
    prec_05 = precision_score(val_y, alerts_05, zero_division=0)
    rec_05 = recall_score(val_y, alerts_05, zero_division=0)
    alerts_05_count = int(alerts_05.sum())

    # =============================================================
    # 3. Cost-based Threshold Sweep (0.01 to 0.99)
    # =============================================================
    print("\n================ 3. Cost-Based Threshold Sweep ================")
    review_costs = [5.0, 10.0, 25.0]
    thresholds = np.linspace(0.01, 0.99, 99)
    
    cost_curves = {c: [] for c in review_costs}
    cost_summary = {}

    for c_rev in review_costs:
        cost_flag_everything = c_rev * total_txns
        cost_05, missed_05, rev_05 = compute_total_cost(val_y, val_amt, alerts_05, c_rev)
        
        best_t = None
        min_cost = float("inf")
        best_alerts_k = 0
        best_prec = 0
        best_rec = 0

        for t in thresholds:
            alerts = (p_va >= t)
            tot_cost, _, _ = compute_total_cost(val_y, val_amt, alerts, c_rev)
            cost_curves[c_rev].append(tot_cost)

            if tot_cost < min_cost:
                min_cost = tot_cost
                best_t = t
                best_alerts_k = (alerts.sum() / total_txns) * 1000.0
                best_prec = precision_score(val_y, alerts, zero_division=0)
                best_rec = recall_score(val_y, alerts, zero_division=0)

        # Amount-Aware Rule: alert if p_va * amount >= c_rev
        alerts_amt_aware = (p_va * val_amt >= c_rev)
        cost_amt_aware, missed_amt, rev_amt = compute_total_cost(val_y, val_amt, alerts_amt_aware, c_rev)
        alerts_k_amt_aware = (alerts_amt_aware.sum() / total_txns) * 1000.0
        prec_amt_aware = precision_score(val_y, alerts_amt_aware, zero_division=0)
        rec_amt_aware = recall_score(val_y, alerts_amt_aware, zero_division=0)

        savings_vs_nothing = total_fraud_loss - min_cost
        savings_vs_05 = cost_05 - min_cost

        cost_summary[f"${int(c_rev)}"] = {
            "review_cost": c_rev,
            "cost_flag_nothing": total_fraud_loss,
            "cost_flag_everything": cost_flag_everything,
            "day1_threshold_0_5": {
                "total_cost": float(cost_05),
                "alerts_per_1000": float((alerts_05_count / total_txns) * 1000.0),
                "precision": float(prec_05),
                "recall": float(rec_05)
            },
            "best_global_threshold": {
                "threshold": float(round(best_t, 2)),
                "total_cost": float(min_cost),
                "alerts_per_1000": float(best_alerts_k),
                "precision": float(best_prec),
                "recall": float(best_rec),
                "savings_vs_nothing": float(savings_vs_nothing),
                "savings_vs_05": float(savings_vs_05)
            },
            "amount_aware_rule": {
                "rule": f"p * amount >= ${int(c_rev)}",
                "total_cost": float(cost_amt_aware),
                "alerts_per_1000": float(alerts_k_amt_aware),
                "precision": float(prec_amt_aware),
                "recall": float(rec_amt_aware),
                "savings_vs_best_global": float(min_cost - cost_amt_aware)
            }
        }

    # Print Cost Table
    print("\n--- Cost Optimization Summary Table ---")
    rows = []
    for k, v in cost_summary.items():
        g = v["best_global_threshold"]
        d = v["day1_threshold_0_5"]
        a = v["amount_aware_rule"]
        rows.append({
            "Review Cost": k,
            "Optimal Threshold": g["threshold"],
            "Alerts / 1,000": f"{g['alerts_per_1000']:.1f}",
            "Precision": f"{g['precision']*100:.1f}%",
            "Recall": f"{g['recall']*100:.1f}%",
            "Optimal Cost": f"${g['total_cost']:,.0f}",
            "Cost @ 0.5 Cutoff": f"${d['total_cost']:,.0f}",
            "Flag Nothing Cost": f"${v['cost_flag_nothing']:,.0f}",
            "Savings vs Nothing": f"${g['savings_vs_nothing']:,.0f}",
            "Savings vs 0.5": f"${g['savings_vs_05']:,.0f}",
            "Amount-Aware Cost": f"${a['total_cost']:,.0f}"
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # =============================================================
    # 4. Plot Cost Curves
    # =============================================================
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(9, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (c_rev, costs), color in zip(cost_curves.items(), colors):
        best_t = cost_summary[f"${int(c_rev)}"]["best_global_threshold"]["threshold"]
        min_c = cost_summary[f"${int(c_rev)}"]["best_global_threshold"]["total_cost"]
        plt.plot(thresholds, costs, label=f"Review Cost = ${int(c_rev)} (Best T = {best_t:.2f}, Min = ${min_c:,.0f})", color=color, lw=2)
        plt.scatter([best_t], [min_c], color=color, s=80, zorder=5)

    plt.axhline(y=total_fraud_loss, color="black", linestyle="--", alpha=0.7, label=f"Flag Nothing (${total_fraud_loss:,.0f})")
    plt.axvline(x=0.5, color="gray", linestyle=":", alpha=0.7, label="Day 1 Cutoff (0.5)")
    plt.xlabel("Alert Threshold", fontsize=12)
    plt.ylabel("Total Financial Cost ($ USD)", fontsize=12)
    plt.title("Step 6: Total Cost vs. Alert Threshold on Validation Set", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plot_path = "results/day2_cost_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nCost curves plot saved to: {plot_path}")

    # Save to JSON
    with open("results/step6_cost_threshold.json", "w") as f:
        json.dump({
            "calibration": {
                "actual_fraud_rate": actual_fraud_rate,
                "mean_pred_prob": mean_pred_prob,
                "brier_score": brier,
                "decile_bins": bin_stats.to_dict(orient="records")
            },
            "costs": cost_summary
        }, f, indent=2)
    print("Saved Step 6 results to results/step6_cost_threshold.json")

if __name__ == "__main__":
    main()
