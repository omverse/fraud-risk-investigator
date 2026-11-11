import os
import json
import numpy as np
import pandas as pd

def main():
    print("Compiling complete Day 2 results into results/day2_results.json...")

    # Dynamically compute dataset shape, splits, and category counts from raw data
    print("Loading raw CSVs to compute dataset splits and category counts directly...")
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")

    trans_cols_set = set(train_trans.columns) - {"TransactionID", "isFraud", "TransactionDT"}
    id_cols_set = set(train_id.columns) - {"TransactionID"}

    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)

    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    trans_cat_cols = [c for c in cat_cols if c in trans_cols_set]
    id_cat_cols = [c for c in cat_cols if c in id_cols_set]

    def get_split_stats(start, end):
        sub = df.iloc[start:end]
        dt_min = int(sub["TransactionDT"].min())
        dt_max = int(sub["TransactionDT"].max())
        return {
            "rows": int(len(sub)),
            "share": round(len(sub) / n, 4),
            "fraud_count": int(sub["isFraud"].sum()),
            "fraud_rate": float(sub["isFraud"].mean()),
            "time_range": [dt_min, dt_max],
            "duration_days": round((dt_max - dt_min) / 86400.0, 1)
        }

    dataset_summary = {
        "total_rows": int(n),
        "total_columns": int(df.shape[1]),
        "features_used": int(X.shape[1]),
        "splits": {
            "train": get_split_stats(0, i1),
            "validation": get_split_stats(i1, i2),
            "test": get_split_stats(i2, n)
        },
        "categorical_columns_breakdown": {
            "total_count": int(len(cat_cols)),
            "transaction_count": int(len(trans_cat_cols)),
            "identity_count": int(len(id_cat_cols)),
            "transaction_columns": trans_cat_cols,
            "identity_columns": id_cat_cols
        }
    }
    print(f"Computed Splits: Train={dataset_summary['splits']['train']['rows']:,}, Val={dataset_summary['splits']['validation']['rows']:,}, Test={dataset_summary['splits']['test']['rows']:,}")
    print(f"Computed Categories: {len(cat_cols)} total ({len(trans_cat_cols)} transaction, {len(id_cat_cols)} identity)")

    del df, X

    # Load individual step json files
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
    with open("results/step7_final_test_score.json", "r") as f:
        s7 = json.load(f)

    day2_compiled = {
        "metadata": {
            "project": "Fraud & Risk Investigator",
            "phase": "Day 2: Better features, imbalance, models and the alert threshold",
            "date": "2026-11-11",
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
