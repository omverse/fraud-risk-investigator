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

    # -------------------------------------------------------------
    # 1. Train vs Test Adversarial Validation
    # -------------------------------------------------------------
    print("\n================ 1. Train vs Test Adversarial Validation ================")
    df_train = df.iloc[:i1].copy()
    df_test = df.iloc[i2:].copy()

    df_train["target_adv"] = 0
    df_test["target_adv"] = 1

    df_adv = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    del df_train, df_test

    y_adv = df_adv["target_adv"]
    X_adv = df_adv.drop(columns=["isFraud", "TransactionID", "TransactionDT", "target_adv"])

    cat_cols = [c for c in X_adv.columns if not pd.api.types.is_numeric_dtype(X_adv[c])]
    for c in cat_cols:
        X_adv[c] = X_adv[c].astype("category")

    X_train_adv, X_val_adv, y_train_adv, y_val_adv = train_test_split(
        X_adv, y_adv, test_size=0.25, random_state=42, stratify=y_adv
    )

    clf_adv = xgb.XGBClassifier(
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
    clf_adv.fit(X_train_adv, y_train_adv, eval_set=[(X_val_adv, y_val_adv)], verbose=50)
    p_val_adv = clf_adv.predict_proba(X_val_adv)[:, 1]
    roc_train_test = roc_auc_score(y_val_adv, p_val_adv)
    print(f"\nTrain vs Test Adversarial ROC-AUC: {roc_train_test:.5f} (trained in {time.time() - t0:.2f}s, best_iteration: {clf_adv.best_iteration})")

    top10_gain = pd.Series(clf_adv.get_booster().get_score(importance_type='gain')).sort_values(ascending=False).head(10)
    print("\nTop 10 features separating Train and Test (by gain):")
    print(top10_gain)

    del df_adv, X_adv, y_adv, X_train_adv, X_val_adv, y_train_adv, y_val_adv, clf_adv

    # -------------------------------------------------------------
    # 2. Monthly value shares for M1 and id_31 across 30-day periods
    # -------------------------------------------------------------
    print("\n================ 2. Monthly Value Shares for M1 and id_31 ================")
    # 30-day period definition
    df["period_30d"] = (df["TransactionDT"] // (86400 * 30)).astype(int)
    print(f"30-day period counts:\n{df['period_30d'].value_counts().sort_index()}")

    # M1 distribution per 30-day period
    print("\nM1 Distribution by 30-day period (percentage of total rows in period):")
    m1_crosstab = pd.crosstab(
        df["period_30d"], 
        df["M1"].fillna("MISSING"), 
        normalize="index"
    ) * 100
    print(m1_crosstab.round(2))

    # id_31 top browsers over time
    print("\nid_31 Distribution: Top values across 30-day periods (% of non-null):")
    # Find top 8 most frequent id_31 values overall
    top_browsers = df["id_31"].value_counts().head(8).index.tolist()
    df["id_31_top"] = df["id_31"].apply(lambda x: x if x in top_browsers else ("MISSING" if pd.isna(x) else "OTHER"))
    id31_crosstab = pd.crosstab(
        df["period_30d"], 
        df["id_31_top"], 
        normalize="index"
    ) * 100
    print(id31_crosstab.round(2))

    # Also show id_31 non-null percentage per period
    id31_avail = df.groupby("period_30d")["id_31"].apply(lambda s: s.notna().mean() * 100)
    print("\nid_31 availability (% with identity row) by 30-day period:")
    print(id31_avail.round(2))

    # -------------------------------------------------------------
    # 3. Check which D columns grow with time
    # -------------------------------------------------------------
    print("\n================ 3. Analysis of D Columns Growing With Time ================")
    d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in df.columns]
    df["day"] = df["TransactionDT"] / 86400.0

    d_stats = []
    for d in d_cols:
        valid_mask = df[d].notna()
        if valid_mask.sum() > 100:
            corr = df.loc[valid_mask, d].corr(df.loc[valid_mask, "day"])
            mean_p0 = df.loc[df["period_30d"] == 0, d].mean()
            mean_p5 = df.loc[df["period_30d"] == 5, d].mean()
            median_p0 = df.loc[df["period_30d"] == 0, d].median()
            median_p5 = df.loc[df["period_30d"] == 5, d].median()
            d_stats.append({
                "column": d,
                "corr_with_time": corr,
                "mean_month0": mean_p0,
                "mean_month5": mean_p5,
                "mean_diff": mean_p5 - mean_p0,
                "median_month0": median_p0,
                "median_month5": median_p5,
                "median_diff": median_p5 - median_p0
            })

    d_df = pd.DataFrame(d_stats).sort_values("corr_with_time", ascending=False)
    print(d_df.to_string(index=False))

if __name__ == "__main__":
    main()
