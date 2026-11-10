import time
import re
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
    if "chrome" in x:
        return "chrome"
    elif "safari" in x:
        return "safari"
    elif "firefox" in x:
        return "firefox"
    elif "edge" in x:
        return "edge"
    elif "ie" in x or "internet explorer" in x:
        return "ie"
    elif "samsung" in x:
        return "samsung"
    elif "opera" in x:
        return "opera"
    else:
        return "other"

def clean_os(x):
    if pd.isna(x):
        return "missing"
    x = str(x).lower()
    if "windows" in x:
        return "windows"
    elif "ios" in x:
        return "ios"
    elif "mac" in x or "os x" in x:
        return "mac"
    elif "android" in x:
        return "android"
    elif "linux" in x:
        return "linux"
    else:
        return "other"

def compute_card_features(df):
    """
    Computes backward-looking card features using earlier rows only.
    Data MUST be sorted by TransactionDT (stable sort).
    card1 is used as proxy identifier.
    """
    print("\n--- Computing Backward-Looking Card Features ---")
    t0 = time.time()
    
    # Group by card1
    # 1. Number of earlier transactions on that card (cumcount)
    df["card1_prev_trans_count"] = df.groupby("card1").cumcount()
    # Card's first transaction gets NaN / missing
    
    # 2. Seconds since previous transaction
    df["card1_prev_dt"] = df.groupby("card1")["TransactionDT"].shift(1)
    df["card1_sec_since_prev"] = df["TransactionDT"] - df["card1_prev_dt"]
    df.drop(columns=["card1_prev_dt"], inplace=True)
    
    # 3. Amount divided by average of earlier amounts
    # Cumulative sum of previous amounts divided by number of previous transactions
    prev_amt_cumsum = df.groupby("card1")["TransactionAmt"].cumsum() - df["TransactionAmt"]
    prev_amt_avg = prev_amt_cumsum / df["card1_prev_trans_count"]
    df["card1_amt_ratio_to_prev_avg"] = df["TransactionAmt"] / prev_amt_avg
    
    # A card's first transaction should get missing value
    first_trans_mask = df["card1_prev_trans_count"] == 0
    df.loc[first_trans_mask, "card1_prev_trans_count"] = np.nan
    df.loc[first_trans_mask, "card1_sec_since_prev"] = np.nan
    df.loc[first_trans_mask, "card1_amt_ratio_to_prev_avg"] = np.nan
    
    print(f"Card features computed in {time.time() - t0:.2f}s")
    
    # Verification check by hand
    sample_card = df.loc[df["card1"].duplicated(keep=False), "card1"].iloc[0]
    sample_rows = df[df["card1"] == sample_card][
        ["TransactionID", "TransactionDT", "card1", "TransactionAmt",
         "card1_prev_trans_count", "card1_sec_since_prev", "card1_amt_ratio_to_prev_avg"]
    ].head(3)
    print("\nVerification by hand (sample card):")
    print(sample_rows)
    assert pd.isna(sample_rows.iloc[0]["card1_prev_trans_count"]), "First transaction count should be NaN!"
    assert pd.isna(sample_rows.iloc[0]["card1_sec_since_prev"]), "First transaction sec_since_prev should be NaN!"
    assert pd.isna(sample_rows.iloc[0]["card1_amt_ratio_to_prev_avg"]), "First transaction amt_ratio should be NaN!"
    print("Verification passed: First transaction gets NaN!")
    
    return df

def main():
    print("Loading data...")
    t0 = time.time()
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id
    print(f"Loaded merged data in {time.time() - t0:.2f}s: {df.shape}")

    print("Sorting by TransactionDT with stable sort...")
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    # -------------------------------------------------------------
    # Step 2: Time Features
    # -------------------------------------------------------------
    print("\n--- Adding Step 2 Time Features ---")
    df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)

    # Step 2b: Browser & OS grouping to reduce drift
    print("Grouping id_31 and id_30 into browser and OS families...")
    df["id_31"] = df["id_31"].apply(clean_browser)
    df["id_30"] = df["id_30"].apply(clean_os)

    # -------------------------------------------------------------
    # Step 3: Backward-looking Card Features
    # -------------------------------------------------------------
    df = compute_card_features(df)

    print("\nDowncasting memory...")
    df = reduce_mem_usage(df)

    # -------------------------------------------------------------
    # Prepare Train / Val splits
    # -------------------------------------------------------------
    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)

    y = df["isFraud"]
    # Keep raw TransactionDT and TransactionID out of features
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")

    X_tr = X.iloc[:i1].copy()
    X_va = X.iloc[i1:i2].copy()
    y_tr = y.iloc[:i1]
    y_va = y.iloc[i1:i2]

    # Clean up memory before training
    del df, X, y
    import gc
    gc.collect()

    # Encode categories from train only
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c], categories=cats)

    print(f"\nTraining set shape: {X_tr.shape}, Validation set shape: {X_va.shape}")
    print("\nTraining XGBoost with new time + card features...")
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
    train_duration = time.time() - t_train
    print(f"Training completed in {train_duration:.2f}s")
    print(f"Best iteration: {model.best_iteration}")

    # Evaluate validation PR-AUC
    p_va = model.predict_proba(X_va)[:, 1]
    val_pr_auc = average_precision_score(y_va, p_va)

    print("\n================ Results Comparison ================")
    print(f"Day 1 Baseline Validation PR-AUC:      0.56300")
    print(f"Day 2 (+ Time & Card) Validation PR-AUC: {val_pr_auc:.5f}")
    print(f"Validation PR-AUC Change:               {val_pr_auc - 0.56300:+.5f}")
    print(f"Training Time:                          {train_duration:.2f}s")

if __name__ == "__main__":
    main()
