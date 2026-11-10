import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve
)

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

def main():
    print("Loading data...")
    t0 = time.time()
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id
    print(f"Data merged: {df.shape} in {time.time() - t0:.2f}s")

    print("Downcasting memory...")
    df = reduce_mem_usage(df)

    # Sort strictly by TransactionDT with a stable sort
    print("Sorting by TransactionDT (stable sort)...")
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)

    # Capture time ranges and splits before dropping TransactionDT
    dt_series = df["TransactionDT"]
    splits_info = {
        "train": {
            "rows": int(i1),
            "share": round(i1 / n, 2),
            "fraud_rate": float(df["isFraud"].iloc[:i1].mean()),
            "min_transaction_dt": int(dt_series.iloc[:i1].min()),
            "max_transaction_dt": int(dt_series.iloc[:i1].max())
        },
        "validation": {
            "rows": int(i2 - i1),
            "share": round((i2 - i1) / n, 2),
            "fraud_rate": float(df["isFraud"].iloc[i1:i2].mean()),
            "min_transaction_dt": int(dt_series.iloc[i1:i2].min()),
            "max_transaction_dt": int(dt_series.iloc[i1:i2].max())
        },
        "test": {
            "rows": int(n - i2),
            "share": round((n - i2) / n, 2),
            "fraud_rate": float(df["isFraud"].iloc[i2:].mean()),
            "min_transaction_dt": int(dt_series.iloc[i2:].min()),
            "max_transaction_dt": int(dt_series.iloc[i2:].max())
        }
    }

    y = df["isFraud"]
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    trans_known_cats = {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", 
                        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"}
    trans_cat_cols = [c for c in cat_cols if c in trans_known_cats]
    id_cat_cols = [c for c in cat_cols if c not in trans_known_cats]
    print(f"Number of categorical columns detected: {len(cat_cols)} ({len(trans_cat_cols)} transaction, {len(id_cat_cols)} identity)")

    print("Splitting into Train (70%), Validation (15%), Test (15%)...")
    X_tr, X_va, X_te = (X.iloc[:i1].copy(), X.iloc[i1:i2].copy(), X.iloc[i2:].copy())
    y_tr, y_va, y_te = y.iloc[:i1], y.iloc[i1:i2], y.iloc[i2:]

    print("Encoding categorical variables based on Train set only...")
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        for part in (X_tr, X_va, X_te):
            part[c] = pd.Categorical(part[c], categories=cats)

    # Compute dummy baseline PR-AUC
    # A dummy model that predicts the constant positive rate or zeros
    dummy_pred = np.full(len(y_te), y_te.mean())
    dummy_pr = average_precision_score(y_te, dummy_pred)
    print(f"\nDummy baseline PR-AUC: {dummy_pr:.5f} (matches test fraud rate: {y_te.mean():.5f})")

    # XGBoost Baseline
    print("\nTraining XGBoost baseline with hist and early stopping...")
    t_train = time.time()
    model = xgb.XGBClassifier(
        n_estimators=1000,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=50
    )
    print(f"Training completed in {time.time() - t_train:.2f}s")
    print(f"Best iteration: {model.best_iteration}")

    # Predictions
    print("\nPredicting on Validation set...")
    p_va = model.predict_proba(X_va)[:, 1]
    val_pr_auc = average_precision_score(y_va, p_va)
    val_roc_auc = roc_auc_score(y_va, p_va)

    print("Predicting on Test set (scored once)...")
    p_te = model.predict_proba(X_te)[:, 1]
    test_pr_auc = average_precision_score(y_te, p_te)
    test_roc_auc = roc_auc_score(y_te, p_te)

    # Metrics at default 0.5 threshold on test set
    y_te_pred_05 = (p_te >= 0.5).astype(int)
    prec_05 = precision_score(y_te, y_te_pred_05, zero_division=0)
    rec_05 = recall_score(y_te, y_te_pred_05, zero_division=0)
    cm_05 = confusion_matrix(y_te, y_te_pred_05)

    print("\n================ Results Summary ================")
    print(f"Dummy PR-AUC:                  {dummy_pr:.5f}")
    print(f"Validation PR-AUC:             {val_pr_auc:.5f}")
    print(f"Test PR-AUC:                   {test_pr_auc:.5f}")
    print(f"Validation-Test PR-AUC Gap:    {val_pr_auc - test_pr_auc:+.5f}")
    print(f"Test ROC-AUC:                  {test_roc_auc:.5f}")
    print(f"Precision at threshold 0.5:    {prec_05:.5f}")
    print(f"Recall at threshold 0.5:       {rec_05:.5f}")
    print(f"Best iteration:                {model.best_iteration}")
    print("\nConfusion Matrix at threshold 0.5:")
    print(cm_05)
    print(f"TN: {cm_05[0, 0]:,}, FP: {cm_05[0, 1]:,}")
    print(f"FN: {cm_05[1, 0]:,}, TP: {cm_05[1, 1]:,}")

    # Plot Precision-Recall curve
    os.makedirs("results", exist_ok=True)
    precision_curve, recall_curve, _ = precision_recall_curve(y_te, p_te)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label=f"XGBoost Baseline (PR-AUC = {test_pr_auc:.4f})", color="royalblue", lw=2)
    plt.axhline(y=dummy_pr, color="crimson", linestyle="--", label=f"Dummy Baseline (PR-AUC = {dummy_pr:.4f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve (Test Set - Day 1 Baseline)", fontsize=14)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, alpha=0.3)
    pr_curve_path = "results/day1_pr_curve.png"
    plt.savefig(pr_curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPrecision-Recall curve saved to: {pr_curve_path}")

    # Save metrics JSON
    import sys, platform, sklearn
    metrics = {
        "dataset": {
            "total_rows": int(n),
            "total_features": int(X.shape[1]),
            "categorical_features_count": len(cat_cols),
            "transaction_cat_features_count": len(trans_cat_cols),
            "identity_cat_features_count": len(id_cat_cols),
            "dropped_columns": ["isFraud", "TransactionID", "TransactionDT"]
        },
        "splits": splits_info,
        "model_config": {
            "model_name": "XGBClassifier",
            "random_seed": 42,
            "parameters": {
                "n_estimators": 1000,
                "tree_method": "hist",
                "enable_categorical": True,
                "eval_metric": "aucpr",
                "early_stopping_rounds": 50,
                "n_jobs": -1,
                "random_state": 42
            }
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "libraries": {
                "xgboost": xgb.__version__,
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "matplotlib": plt.matplotlib.__version__
            }
        },
        "metrics": {
            "dummy_pr_auc": float(dummy_pr),
            "validation_pr_auc": float(val_pr_auc),
            "test_pr_auc": float(test_pr_auc),
            "validation_test_pr_auc_gap": float(val_pr_auc - test_pr_auc),
            "test_roc_auc": float(test_roc_auc),
            "best_iteration": int(model.best_iteration),
            "threshold_0_5": {
                "precision": float(prec_05),
                "recall": float(rec_05),
                "confusion_matrix": {
                    "true_negatives": int(cm_05[0, 0]),
                    "false_positives": int(cm_05[0, 1]),
                    "false_negatives": int(cm_05[1, 0]),
                    "true_positives": int(cm_05[1, 1])
                }
            }
        },
        "categorical_columns": cat_cols
    }
    with open("results/day1_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to results/day1_baseline.json")

if __name__ == "__main__":
    main()
