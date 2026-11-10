import os
import json
import numpy as np
import pandas as pd

def main():
    print("Compiling complete Day 2 results into results/day2_results.json...")

    # Load individual step json files
    with open("results/day1_baseline.json", "r") as f:
        d1 = json.load(f)
    with open("results/step3b_noise_and_d_ablation.json", "r") as f:
        s3b = json.load(f)
    with open("results/step3d_fixed_trees_comparison.json", "r") as f:
        s3d = json.load(f)
    with open("results/step4_imbalance.json", "r") as f:
        s4 = json.load(f)
    with open("results/step5_models.json", "r") as f:
        s5 = json.load(f)
    with open("results/step6_cost_threshold.json", "r") as f:
        s6 = json.load(f)
    
    # Load step 7 if available
    s7 = {}
    if os.path.exists("results/step7_final_test_score.json"):
        with open("results/step7_final_test_score.json", "r") as f:
            s7 = json.load(f)

    # Dataset splits and category breakdown
    trans_cats = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"] + [f"M{i}" for i in range(1, 10)]
    id_cats = [f"id_{i:02d}" for i in range(12, 39)] # 12 to 38
    # Actual identity categories from day1
    id_cats_actual = [c for c in d1.get("categorical_columns", []) if c.startswith("id_")]

    dataset_summary = {
        "total_rows": 590540,
        "total_columns": 434,
        "features_used": 431,
        "splits": {
            "train": {
                "rows": 413378,
                "share": 0.70,
                "fraud_count": 14538,
                "fraud_rate": 0.03516878014795175,
                "time_range": [86400, 10437996],
                "duration_days": 121
            },
            "validation": {
                "rows": 88581,
                "share": 0.15,
                "fraud_count": 3042,
                "fraud_rate": 0.034341450209412856,
                "time_range": [10438003, 13151840],
                "duration_days": 31
            },
            "test": {
                "rows": 88581,
                "share": 0.15,
                "fraud_count": 3083,
                "fraud_rate": 0.03480430339463316,
                "time_range": [13151880, 15811131],
                "duration_days": 31
            }
        },
        "categorical_columns_breakdown": {
            "total_count": len(d1.get("categorical_columns", [])),
            "transaction_count": len([c for c in d1.get("categorical_columns", []) if not c.startswith("id_")]),
            "identity_count": len([c for c in d1.get("categorical_columns", []) if c.startswith("id_")]),
            "transaction_columns": [c for c in d1.get("categorical_columns", []) if not c.startswith("id_")],
            "identity_columns": [c for c in d1.get("categorical_columns", []) if c.startswith("id_")]
        }
    }

    day2_compiled = {
        "metadata": {
            "project": "Fraud & Risk Investigator",
            "phase": "Day 2: Better features, imbalance, models and the alert threshold",
            "date": "2026-11-10",
            "locked_configuration": {
                "model": "XGBoost (350 fixed trees, unweighted, tree_method='hist')",
                "feature_set": "Core Rolling Windows + D1_norm",
                "calibration": "Isotonic Regression",
                "operating_point_headline": "97 alerts per 1,000 transactions ($10 review cost)"
            }
        },
        "dataset": dataset_summary,
        "step1_adversarial_validation": {
            "train_vs_val_roc_auc": 0.87020,
            "val_vs_test_roc_auc": 0.72835,
            "train_vs_test_roc_auc": 0.88864,
            "drift_diagnostics": {
                "M1_missing_rate_shift": "Train 69.6% missing -> Test 25.7% missing (fraud rate among missing rose 2.8% -> 9.9%)",
                "id_31_browser_shift": "Chronological progression Chrome 62/63 -> 64 -> 65 -> 66",
                "identity_coverage_drop": "Identity table match dropped from ~42% to 17-20% in later periods"
            }
        },
        "step2_and_3_feature_engineering": {
            "added_features": [
                "hour_of_day: (TransactionDT // 3600) % 24",
                "day_of_week: (TransactionDT // 86400) % 7",
                "id_31: broad browser family (chrome, safari, firefox, edge, ie, samsung, opera, other, missing)",
                "id_30: broad OS family (windows, ios, mac, android, linux, other, missing)",
                "card1_trans_count_1h: transactions on same card in past 1 hour",
                "card1_trans_count_24h: transactions on same card in past 24 hours",
                "card1_trans_count_7d: transactions on same card in past 7 days",
                "card1_sec_since_prev: seconds since previous transaction on card (NaN for first)",
                "card1_amt_ratio_to_prev_avg: transaction amount divided by historical card mean (NaN for first)",
                "D1_norm: D1 - (TransactionDT / 86400.0) anchoring registration day to static epoch"
            ]
        },
        "step3b_noise_study": s3b.get("noise_study_baseline", {}),
        "step3d_feature_ablation_fixed_trees": s3d,
        "step4_imbalance_comparison": s4,
        "step5_model_comparison": s5,
        "step6_validation_cost_and_calibration": s6,
        "step7_final_test_evaluation": s7
    }

    with open("results/day2_results.json", "w") as f:
        json.dump(day2_compiled, f, indent=2)
    print("Successfully compiled complete results to results/day2_results.json")

if __name__ == "__main__":
    main()
