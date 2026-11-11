import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, confusion_matrix, brier_score_loss

def compute_cost(y_true, amt_arr, alerts_bool, review_cost):
    missed_mask = (y_true == 1) & (~alerts_bool)
    missed_cost = float(amt_arr[missed_mask].sum())
    review_total = float(review_cost * alerts_bool.sum())
    return missed_cost + review_total, missed_cost, review_total

def paired_bootstrap_ci(vec_diff, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(vec_diff)
    obs = float(vec_diff.sum())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = vec_diff[idx].sum()
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return {
        "observed_diff": obs,
        "boot_mean": float(boots.mean()),
        "ci_95_lower": float(ci_lo),
        "ci_95_upper": float(ci_hi),
        "prop_positive": float(np.mean(boots > 0)),
        "n_boot": n_boot,
        "seed": seed
    }

def main():
    print("=" * 70)
    print("Recomputing Step 7 Test Costs directly from Saved Predictions")
    print("=" * 70)

    # 1. Load ground truth
    print("\n1. Loading test set ground truth from raw data...")
    trans = pd.read_csv("data/train_transaction.csv", usecols=["TransactionID", "TransactionDT", "isFraud", "TransactionAmt"])
    trans = trans.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    n = len(trans)
    i2 = int(0.85 * n)
    test_df = trans.iloc[i2:].copy().reset_index(drop=True)
    del trans

    y_te = test_df["isFraud"].values
    amt_te = test_df["TransactionAmt"].values
    n_te = len(y_te)
    total_test_frauds = int(y_te.sum())
    total_test_fraud_amt = float(amt_te[y_te == 1].sum())

    print(f"Test Set: {n_te:,} rows | Frauds: {total_test_frauds:,} ({y_te.mean()*100:.3f}%) | Fraud Loss: ${total_test_fraud_amt:,.2f}")

    # 2. Load saved prediction arrays
    print("\n2. Loading saved prediction arrays from results/...")
    p_base = np.load("results/test_preds_day1_baseline.npy")
    p_a = np.load("results/test_preds_day2_train_only.npy")
    p_b = np.load("results/test_preds_day2_retrained.npy")
    p_cal = np.load("results/test_preds_day2_calibrated.npy")

    assert len(p_base) == n_te, f"Baseline length mismatch: {len(p_base)} vs {n_te}"
    assert len(p_a) == n_te, f"Model A length mismatch: {len(p_a)} vs {n_te}"
    assert len(p_b) == n_te, f"Model B length mismatch: {len(p_b)} vs {n_te}"
    assert len(p_cal) == n_te, f"Calibrated length mismatch: {len(p_cal)} vs {n_te}"

    # Overall PR-AUC and ROC-AUC
    base_pr_auc = float(average_precision_score(y_te, p_base))
    base_roc_auc = float(roc_auc_score(y_te, p_base))

    pr_auc_a = float(average_precision_score(y_te, p_a))
    roc_auc_a = float(roc_auc_score(y_te, p_a))

    pr_auc_b = float(average_precision_score(y_te, p_b))
    roc_auc_b = float(roc_auc_score(y_te, p_b))

    print(f"Baseline (Stable Sort):   Test PR-AUC = {base_pr_auc:.5f} | ROC-AUC = {base_roc_auc:.5f}")
    print(f"Model A (Train-Only):     Test PR-AUC = {pr_auc_a:.5f} | ROC-AUC = {roc_auc_a:.5f} (Gain: {pr_auc_a - base_pr_auc:+.5f})")
    print(f"Model B (Retrained T+V):  Test PR-AUC = {pr_auc_b:.5f} | ROC-AUC = {roc_auc_b:.5f} (Gain over Model A: {pr_auc_b - pr_auc_a:+.5f})")

    # Helper to get boolean alert mask in original row order
    def get_top_k_mask(p, k):
        n_alerts = int(round((k / 1000.0) * n_te))
        sort_idx = np.argsort(-p, kind="stable")
        mask = np.zeros(n_te, dtype=bool)
        mask[sort_idx[:n_alerts]] = True
        return mask

    # 3. Strict Policy Subset Consistency Checks (smaller policy strictly inside larger policy)
    print("\n3. Running Policy Subset & Recovered Dollar Consistency Checks...")
    al_05_a = (p_a >= 0.5)
    mask_a_44 = get_top_k_mask(p_a, 44)
    mask_a_97 = get_top_k_mask(p_a, 97)
    mask_a_152 = get_top_k_mask(p_a, 152)

    # Check Model A strict subset hierarchy: (p >= 0.5) <= (44/1k) <= (97/1k) <= (152/1k)
    assert np.all(al_05_a <= mask_a_44), "Model A: threshold 0.5 is NOT a subset of 44/1k!"
    assert np.all(mask_a_44 <= mask_a_97), "Model A: 44/1k is NOT a subset of 97/1k!"
    assert np.all(mask_a_97 <= mask_a_152), "Model A: 97/1k is NOT a subset of 152/1k!"

    rec_a_05 = float(amt_te[(y_te == 1) & al_05_a].sum())
    rec_a_44 = float(amt_te[(y_te == 1) & mask_a_44].sum())
    rec_a_97 = float(amt_te[(y_te == 1) & mask_a_97].sum())
    rec_a_152 = float(amt_te[(y_te == 1) & mask_a_152].sum())
    assert rec_a_05 <= rec_a_44 <= rec_a_97 <= rec_a_152, "Model A: recovered dollars failed nested subset check!"
    print(f"  [PASS] Model A subset & recovery check: ${rec_a_05:,.0f} (0.5) <= ${rec_a_44:,.0f} (44/1k) <= ${rec_a_97:,.0f} (97/1k) <= ${rec_a_152:,.0f} (152/1k)")

    for name, p in [("Baseline", p_base), ("Model B", p_b)]:
        m_44 = get_top_k_mask(p, 44)
        m_97 = get_top_k_mask(p, 97)
        m_152 = get_top_k_mask(p, 152)
        assert np.all(m_44 <= m_97) and np.all(m_97 <= m_152), f"{name}: failed subset inclusion!"
        r_44 = float(amt_te[(y_te == 1) & m_44].sum())
        r_97 = float(amt_te[(y_te == 1) & m_97].sum())
        r_152 = float(amt_te[(y_te == 1) & m_152].sum())
        assert r_44 <= r_97 <= r_152, f"{name}: failed recovered dollars monotonicity!"
        print(f"  [PASS] {name} subset & recovery check: ${r_44:,.0f} (44/1k) <= ${r_97:,.0f} (97/1k) <= ${r_152:,.0f} (152/1k)")

    # 4. Threshold 0.5 Benchmarks
    print("\n4. Threshold 0.5 Evaluation (all numbers from single saved prediction arrays)...")
    al_05_base = (p_base >= 0.5)
    prec_05_base = float(precision_score(y_te, al_05_base, zero_division=0))
    rec_05_base = float(recall_score(y_te, al_05_base, zero_division=0))
    n_al_05_base = int(al_05_base.sum())
    cm_base = confusion_matrix(y_te, al_05_base)
    cost_05_base_10, miss_05_base, _ = compute_cost(y_te, amt_te, al_05_base, 10.0)

    prec_05_a = float(precision_score(y_te, al_05_a, zero_division=0))
    rec_05_a = float(recall_score(y_te, al_05_a, zero_division=0))
    n_al_05_a = int(al_05_a.sum())
    rate_05_a = float((n_al_05_a / n_te) * 1000.0)
    cm_a = confusion_matrix(y_te, al_05_a)
    cost_05_a = {}
    for c in [5.0, 10.0, 25.0]:
        tot, miss, rev = compute_cost(y_te, amt_te, al_05_a, c)
        cost_05_a[f"${int(c)}"] = tot

    print(f"Baseline @ 0.5: Alerts={n_al_05_base:,} ({n_al_05_base/n_te*1000:.1f}/1k) | Prec={prec_05_base*100:.2f}% | Rec={rec_05_base*100:.2f}% (TP={cm_base[1,1]}, FP={cm_base[0,1]}, FN={cm_base[1,0]}) | Cost@$10=${cost_05_base_10:,.0f}")
    print(f"Model A  @ 0.5: Alerts={n_al_05_a:,} ({rate_05_a:.1f}/1k) | Prec={prec_05_a*100:.2f}% | Rec={rec_05_a*100:.2f}% (TP={cm_a[1,1]}, FP={cm_a[0,1]}, FN={cm_a[1,0]}) | Cost@$10=${cost_05_a['$10']:,.0f}")

    # 5. Corrected Fixed Alert Rate Costs
    print("\n5. Corrected Alert Rate Evaluation (97, 152, 44 alerts per 1,000 txns)...")
    alert_rates = [
        (10.0, 97, "$10 (Headline)"),
        (5.0, 152, "$5"),
        (25.0, 44, "$25")
    ]

    operating_points_data = {}
    report_rows = []
    masks_by_k = {"Model A": {}, "Baseline": {}, "Model B": {}}

    for c_rev, k, label in alert_rates:
        key = label.split()[0]
        n_alerts = int(round((k / 1000.0) * n_te))

        mask_a = get_top_k_mask(p_a, k)
        mask_base = get_top_k_mask(p_base, k)
        mask_b = get_top_k_mask(p_b, k)

        masks_by_k["Model A"][k] = mask_a
        masks_by_k["Baseline"][k] = mask_base
        masks_by_k["Model B"][k] = mask_b

        # Model A
        cost_a, miss_a, rev_a = compute_cost(y_te, amt_te, mask_a, c_rev)
        caught_amt_a = total_test_fraud_amt - miss_a
        prec_a = float(y_te[mask_a].mean())
        rec_a = float(y_te[mask_a].sum() / total_test_frauds)

        # Baseline
        cost_base, miss_base, rev_base = compute_cost(y_te, amt_te, mask_base, c_rev)
        caught_amt_base = total_test_fraud_amt - miss_base
        prec_base = float(y_te[mask_base].mean())
        rec_base = float(y_te[mask_base].sum() / total_test_frauds)

        # Model B
        cost_b, miss_b, rev_b = compute_cost(y_te, amt_te, mask_b, c_rev)
        caught_amt_b = total_test_fraud_amt - miss_b
        prec_b = float(y_te[mask_b].mean())
        rec_b = float(y_te[mask_b].sum() / total_test_frauds)

        sav_a_nothing = total_test_fraud_amt - cost_a
        sav_a_05 = cost_05_a[key] - cost_a
        sav_a_base = cost_base - cost_a

        operating_points_data[key] = {
            "review_cost": c_rev,
            "alert_rate_k": k,
            "total_alerts": n_alerts,
            "model_a_train_only": {
                "precision": prec_a,
                "recall": rec_a,
                "recovered_fraud_cost": caught_amt_a,
                "recovered_fraud_share": float(caught_amt_a / total_test_fraud_amt),
                "missed_fraud_cost": miss_a,
                "review_cost_total": rev_a,
                "total_cost": cost_a,
                "savings_vs_nothing": sav_a_nothing,
                "savings_vs_threshold_0_5": sav_a_05,
                "savings_vs_day1_baseline_at_same_rate": sav_a_base
            },
            "day1_baseline_at_same_rate": {
                "precision": prec_base,
                "recall": rec_base,
                "recovered_fraud_cost": caught_amt_base,
                "recovered_fraud_share": float(caught_amt_base / total_test_fraud_amt),
                "missed_fraud_cost": miss_base,
                "review_cost_total": rev_base,
                "total_cost": cost_base,
                "savings_vs_nothing": total_test_fraud_amt - cost_base
            },
            "model_b_retrained": {
                "precision": prec_b,
                "recall": rec_b,
                "recovered_fraud_cost": caught_amt_b,
                "recovered_fraud_share": float(caught_amt_b / total_test_fraud_amt),
                "missed_fraud_cost": miss_b,
                "review_cost_total": rev_b,
                "total_cost": cost_b,
                "savings_vs_nothing": total_test_fraud_amt - cost_b
            },
            "flag_nothing_cost": total_test_fraud_amt,
            "flag_everything_cost": float(c_rev * n_te),
            "threshold_0_5_cost": cost_05_a[key]
        }

        report_rows.append({
            "Review Cost": label,
            "Alert Rate": f"{k} / 1k ({n_alerts:,})",
            "Model A Prec": f"{prec_a*100:.2f}%",
            "Model A Rec": f"{rec_a*100:.2f}%",
            "Model A Recovered": f"${caught_amt_a:,.0f} ({caught_amt_a/total_test_fraud_amt*100:.1f}%)",
            "Model A Cost": f"${cost_a:,.0f}",
            "Baseline Cost": f"${cost_base:,.0f}",
            "Savings vs Base": f"${sav_a_base:+,.0f}",
            "Model B Rec": f"{rec_b*100:.2f}%",
            "Model B Cost": f"${cost_b:,.0f}"
        })

    print(pd.DataFrame(report_rows).to_string(index=False))

    # 6. Amount-Aware Rule on Test Set
    print("\n6. Evaluating Amount-Aware Rule with Calibrated Probabilities...")
    amt_aware_results = {}
    amt_report_rows = []
    masks_amt = {}

    for c_rev, k, label in alert_rates:
        key = label.split()[0]
        al_amt = (p_cal * amt_te >= c_rev)
        masks_amt[key] = al_amt
        n_amt_alerts = int(al_amt.sum())
        k_amt = float((n_amt_alerts / n_te) * 1000.0)
        
        prec_amt = float(precision_score(y_te, al_amt, zero_division=0))
        rec_amt = float(recall_score(y_te, al_amt, zero_division=0))
        cost_amt, miss_amt, rev_amt = compute_cost(y_te, amt_te, al_amt, c_rev)
        caught_amt = total_test_fraud_amt - miss_amt

        cost_opt_k = operating_points_data[key]["model_a_train_only"]["total_cost"]
        sav_vs_opt_k = cost_opt_k - cost_amt

        amt_aware_results[key] = {
            "review_cost": c_rev,
            "alerts_per_1000": k_amt,
            "total_alerts": n_amt_alerts,
            "precision": prec_amt,
            "recall": rec_amt,
            "recovered_fraud_cost": caught_amt,
            "recovered_fraud_share": float(caught_amt / total_test_fraud_amt),
            "missed_fraud_cost": miss_amt,
            "review_cost_total": rev_amt,
            "total_cost": cost_amt,
            "savings_vs_fixed_alert_rate": sav_vs_opt_k,
            "savings_vs_nothing": total_test_fraud_amt - cost_amt
        }

        amt_report_rows.append({
            "Review Cost": label,
            "Alerts/1k": f"{k_amt:.1f} ({n_amt_alerts:,})",
            "Precision": f"{prec_amt*100:.2f}%",
            "Recall": f"{rec_amt*100:.2f}%",
            "Recovered Fraud": f"${caught_amt:,.0f} ({caught_amt/total_test_fraud_amt*100:.1f}%)",
            "Amount-Aware Cost": f"${cost_amt:,.0f}",
            "Fixed Rate Cost": f"${cost_opt_k:,.0f}",
            "Savings vs Fixed": f"${sav_vs_opt_k:+,.0f}"
        })

    print(pd.DataFrame(amt_report_rows).to_string(index=False))

    # 7. Paired Bootstrap (1,000 resamples over test rows)
    print("\n7. Paired Bootstrap Over Test Rows (1,000 resamples, 95% CIs)...")
    def row_cost(m, c_rev):
        return y_te * amt_te * (~m) + c_rev * m
    def row_recov(m):
        return y_te * amt_te * m

    boot_results = {
        "model_a_vs_baseline_97_per_1k_recovered_dollars": paired_bootstrap_ci(
            row_recov(masks_by_k["Model A"][97]) - row_recov(masks_by_k["Baseline"][97])
        ),
        "model_a_vs_baseline_97_per_1k_cost_savings_at_10": paired_bootstrap_ci(
            row_cost(masks_by_k["Baseline"][97], 10.0) - row_cost(masks_by_k["Model A"][97], 10.0)
        ),
        "model_b_vs_model_a_97_per_1k_recovered_dollars": paired_bootstrap_ci(
            row_recov(masks_by_k["Model B"][97]) - row_recov(masks_by_k["Model A"][97])
        ),
        "model_b_vs_model_a_97_per_1k_cost_savings_at_10": paired_bootstrap_ci(
            row_cost(masks_by_k["Model A"][97], 10.0) - row_cost(masks_by_k["Model B"][97], 10.0)
        ),
        "amount_aware_vs_fixed_97_per_1k_savings_at_10": paired_bootstrap_ci(
            row_cost(masks_by_k["Model A"][97], 10.0) - row_cost(masks_amt["$10"], 10.0)
        ),
        "amount_aware_vs_fixed_152_per_1k_savings_at_5": paired_bootstrap_ci(
            row_cost(masks_by_k["Model A"][152], 5.0) - row_cost(masks_amt["$5"], 5.0)
        ),
        "amount_aware_vs_fixed_44_per_1k_savings_at_25": paired_bootstrap_ci(
            row_cost(masks_by_k["Model A"][44], 25.0) - row_cost(masks_amt["$25"], 25.0)
        )
    }

    for k_boot, v_boot in boot_results.items():
        print(f"  {k_boot}:")
        print(f"    Observed = ${v_boot['observed_diff']:+,.0f} | 95% CI = [${v_boot['ci_95_lower']:+,.0f}, ${v_boot['ci_95_upper']:+,.0f}] | P(>0) = {v_boot['prop_positive']*100:.1f}%")

    # 8. Test Calibration Diagnostic
    print("\n8. Test Calibration Diagnostic...")
    cal_test_df = pd.DataFrame({"y_true": y_te, "p_raw": p_a, "p_cal": p_cal})
    cal_test_df["bin"] = pd.qcut(cal_test_df["p_raw"], q=10, duplicates="drop")
    test_rel_table = cal_test_df.groupby("bin", observed=True).agg(
        count=("y_true", "count"),
        frauds=("y_true", "sum"),
        mean_p_raw=("p_raw", "mean"),
        mean_p_cal=("p_cal", "mean"),
        actual_rate=("y_true", "mean")
    ).reset_index()
    test_rel_table["bin"] = test_rel_table["bin"].astype(str)

    raw_test_brier = float(brier_score_loss(y_te, p_a))
    cal_test_brier = float(brier_score_loss(y_te, p_cal))

    # 9. Save updated results JSONs
    final_output = {
        "dataset_split": {
            "total_rows": n,
            "train_rows": i2 - (int(0.15 * n)),
            "val_rows": int(0.15 * n),
            "test_rows": n_te,
            "test_frauds": total_test_frauds,
            "test_fraud_rate": float(y_te.mean()),
            "test_total_fraud_loss_flag_nothing": total_test_fraud_amt
        },
        "day1_baseline_on_test": {
            "description": "Baseline model trained with stable-sort pipeline on Day 1 features, evaluated on single saved test array (test_preds_day1_baseline.npy)",
            "test_pr_auc": base_pr_auc,
            "test_roc_auc": base_roc_auc,
            "threshold_0_5": {
                "alerts": n_al_05_base,
                "precision": prec_05_base,
                "recall": rec_05_base,
                "confusion_matrix": {
                    "TN": int(cm_base[0, 0]),
                    "FP": int(cm_base[0, 1]),
                    "FN": int(cm_base[1, 0]),
                    "TP": int(cm_base[1, 1])
                },
                "total_cost_at_10": cost_05_base_10
            }
        },
        "model_a_train_only": {
            "test_pr_auc": pr_auc_a,
            "test_roc_auc": roc_auc_a,
            "training_time_sec": 16.92,
            "threshold_0_5": {
                "alerts": n_al_05_a,
                "alerts_per_1000": rate_05_a,
                "precision": prec_05_a,
                "recall": rec_05_a,
                "confusion_matrix": {
                    "TN": int(cm_a[0, 0]),
                    "FP": int(cm_a[0, 1]),
                    "FN": int(cm_a[1, 0]),
                    "TP": int(cm_a[1, 1])
                },
                "cost_by_review_cost": cost_05_a
            }
        },
        "model_b_retrained": {
            "test_pr_auc": pr_auc_b,
            "test_roc_auc": roc_auc_b,
            "training_time_sec": 27.28
        },
        "operating_points": operating_points_data,
        "amount_aware_rule_test": amt_aware_results,
        "paired_bootstrap_1000": boot_results,
        "calibration_diagnostic_test": {
            "test_actual_fraud_rate": float(y_te.mean()),
            "test_raw_mean_prob": float(p_a.mean()),
            "test_calibrated_mean_prob": float(p_cal.mean()),
            "test_raw_brier_score": raw_test_brier,
            "test_calibrated_brier_score": cal_test_brier,
            "decile_reliability_table": test_rel_table.to_dict(orient="records")
        }
    }

    with open("results/step7_final_test_score.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("\nUpdated results/step7_final_test_score.json")

if __name__ == "__main__":
    main()
