import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def reduce_mem_usage(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.int8(127):
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

def run_adversarial(df, i_start, i_end1, i_end2, period1_name, period2_name):
    print(f"\n--- Running Adversarial Validation: {period1_name} (0) vs {period2_name} (1) ---")
    df_p1 = df.iloc[i_start:i_end1].copy()
    df_p2 = df.iloc[i_end1:i_end2].copy()

    df_p1["target_adv"] = 0
    df_p2["target_adv"] = 1

    df_adv = pd.concat([df_p1, df_p2], axis=0).reset_index(drop=True)
    del df_p1, df_p2

    y = df_adv["target_adv"]
    # Strictly exclude isFraud, TransactionID, TransactionDT
    X = df_adv.drop(columns=["isFraud", "TransactionID", "TransactionDT", "target_adv"])

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_cols:
        X[c] = X[c].astype("category")

    # Split into train/validation for adversarial model (stratified 75% train / 25% test)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42
    )

    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"Training finished in {time.time() - t0:.2f}s, best iteration: {clf.best_iteration}")

    p_val = clf.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, p_val)
    print(f"Adversarial ROC-AUC ({period1_name} vs {period2_name}): {roc:.5f}")

    # Top features by gain
    gain_importance = clf.get_booster().get_score(importance_type='gain')
    feat_imp = pd.Series(gain_importance).sort_values(ascending=False)
    print(f"\nTop 10 features separating {period1_name} and {period2_name} (by gain):")
    print(feat_imp.head(10))

    # Weight/frequency importance as well
    weight_importance = clf.get_booster().get_score(importance_type='weight')
    feat_weight = pd.Series(weight_importance).sort_values(ascending=False)
    print(f"\nTop 10 features separating {period1_name} and {period2_name} (by split count):")
    print(feat_weight.head(10))

    return roc, feat_imp.head(10), feat_weight.head(10)

def main():
    print("Loading data...")
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    print("Downcasting...")
    df = reduce_mem_usage(df)

    print("Sorting by TransactionDT with stable sort...")
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1 = int(0.70 * n)
    i2 = int(0.85 * n)

    # 1. Train vs Validation
    roc_tr_va, top10_gain, top10_weight = run_adversarial(df, 0, i1, i2, "Train", "Validation")

    # 2. Validation vs Test (Optional diagnostic mentioned in doc)
    roc_va_te, top10_va_te_gain, _ = run_adversarial(df, i1, i2, n, "Validation", "Test")

if __name__ == "__main__":
    main()
