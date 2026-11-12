import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import json
import time
import numpy as np
import pandas as pd
from src.explain_transaction import (
    FraudTransactionExplainer,
    build_training_reference_stats,
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
    df = df.copy()
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

    new_cols = pd.DataFrame({
        "card1_trans_count_1h": np.concatenate(cnt_1h).astype(np.int16),
        "card1_trans_count_24h": np.concatenate(cnt_24h).astype(np.int16),
        "card1_trans_count_7d": np.concatenate(cnt_7d).astype(np.int16),
        "card1_sec_since_prev": np.concatenate(sec_prev),
        "card1_amt_ratio_to_prev_avg": np.concatenate(amt_ratio),
    }, index=df_sorted.index)
    df_sorted = pd.concat([df_sorted, new_cols], axis=1)
    df_result = df_sorted.sort_values("_orig_idx", kind="stable").drop(columns=["_orig_idx"]).reset_index(drop=True)
    return df_result


def row_to_json_safe_dict(series_or_row):
    out = {}
    for k, v in series_or_row.items():
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = str(v)
    return out


def main():
    print("=" * 72)
    print("Day 3 Step 3: Building Single-Transaction Explainer & 15 Test Fixtures")
    print("=" * 72)

    # 1. Load data identically to Step 1
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    df = reduce_mem_usage(df)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)

    df = df.copy()
    df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)
    df["id_31"] = df["id_31"].apply(clean_browser)
    df["id_30"] = df["id_30"].apply(clean_os)
    df = compute_rolling_window_counts(df)
    df["D1_norm"] = df["D1"] - (df["TransactionDT"] / 86400.0)
    df = reduce_mem_usage(df)

    y_tr = df["isFraud"].iloc[:i1].values
    y_va = df["isFraud"].iloc[i1:i2].values

    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    del df
    X_tr = X.iloc[:i1].copy().reset_index(drop=True)
    X_va_raw = X.iloc[i1:i2].copy().reset_index(drop=True)
    del X

    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)

    with open("results/day3_step1_shap.json") as f:
        step1_data = json.load(f)
    selected_cases = step1_data["selected_15_cases"]
    shap_5000 = np.load("results/day3_shap_values_val5000.npy")

    # Save training reference stats (lightweight artifact for Day 6 app & Step 3)
    d1_missing_shift = {
        "train_missing_share": round(float(X_tr["D1"].isna().mean()), 5),
        "train_missing_fraud_rate": round(float(y_tr[X_tr["D1"].isna().values].mean()), 5),
        "val_missing_share": round(float(X_va_raw["D1"].isna().mean()), 5),
        "val_missing_fraud_rate": round(float(y_va[X_va_raw["D1"].isna().values].mean()), 5),
    }

    import xgboost as xgb
    tmp_m = xgb.XGBClassifier()
    tmp_m.load_model("results/day2_model_a_train_only.json")
    feat_names = list(tmp_m.get_booster().feature_names)
    del tmp_m

    ref_stats = build_training_reference_stats(X_tr, feat_names, d1_regime_stats=d1_missing_shift)
    with open("results/day3_training_reference_stats.json", "w") as f:
        json.dump(ref_stats, f, indent=2)
    print("Saved training reference stats to results/day3_training_reference_stats.json")

    # Instantiate reusable single-transaction explainer from artifacts
    explainer = FraudTransactionExplainer.from_artifacts()

    # Build self-contained 15 test fixtures (unclamped raw features + ground truth labels)
    fixtures = []
    explanations = []
    os.makedirs("results/day3_explanations", exist_ok=True)

    max_prob_err = 0.0
    max_group_err = 0.0
    t0 = time.time()

    for idx_case, case in enumerate(selected_cases, start=1):
        val_idx = case["val_index"]
        sample_pos = case["sample_pos"]
        tx_id = case["transaction_id"]

        # Unclamped raw feature row for fixture serialization & unseen category inspection
        raw_row = X_va_raw.iloc[val_idx]
        raw_dict = row_to_json_safe_dict(raw_row[feat_names])
        raw_cats = {c: (None if pd.isna(raw_row[c]) else str(raw_row[c])) for c in cat_cols}

        fixture_record = {
            "fixture_id": idx_case,
            "selection_rule": "Uniform rank-quantile slice inside outcome bucket at top 97/1k operating point (seed=42 5,000-row validation sample)",
            "category": case["category"],
            "val_index": val_idx,
            "transaction_id": tx_id,
            "actual_fraud": case["actual_fraud"],
            "in_top_97_per_1k": case["in_top_97_per_1k"],
            "features": raw_dict,
        }
        fixtures.append(fixture_record)

        # Call single-transaction function cleanly using raw_dict directly (live single-row SHAP)
        exp_out = explainer.explain_transaction(
            raw_dict,
            transaction_id=tx_id,
            raw_unclamped_cats=raw_cats,
            extra_context={
                "fixture_id": idx_case,
                "validation_outcome_bucket": case["category"],
                "actual_fraud": case["actual_fraud"],
                "in_top_97_per_1k": case["in_top_97_per_1k"],
                "selection_rule": fixture_record["selection_rule"],
            },
        )

        # Verify against Step 1 predictions & precomputed 5,000-row SHAP
        p_err = abs(exp_out["transaction_summary"]["raw_prob"] - case["raw_prob"])
        max_prob_err = max(max_prob_err, p_err)
        max_group_err = max(max_group_err, exp_out["groups_additivity_check"]["abs_error"])

        assert p_err < 1e-4, f"Prob mismatch on Tx {tx_id}: {p_err}"
        assert exp_out["groups_additivity_check"]["passed"], f"Group additivity failed on Tx {tx_id}"

        # Save individual alert JSON
        indiv_path = f"results/day3_explanations/tx_{tx_id}.json"
        with open(indiv_path, "w") as f:
            json.dump(exp_out, f, indent=2)
        explanations.append(exp_out)

        caution_codes = [c["code"] for c in exp_out["data_cautions"]]
        print(
            f"  [{idx_case:2d}/15] Tx {tx_id} ({case['category']:15s}) | "
            f"CalP={exp_out['transaction_summary']['calibrated_prob_display']:>9s} | "
            f"GrpAddErr={exp_out['groups_additivity_check']['abs_error']:.2e} | "
            f"Cautions={caution_codes}"
        )

    dt_all = time.time() - t0
    with open("results/day3_step3_fixtures_15.json", "w") as f:
        json.dump(fixtures, f, indent=2)
    with open("results/day3_step3_explanations_15.json", "w") as f:
        json.dump(explanations, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Step 3 Complete: 15 single-transaction explanations generated in {dt_all:.2f}s ({dt_all/15*1000:.1f} ms/alert)")
    print(f"Max raw_prob diff vs Step 1: {max_prob_err:.6f} | Max group additivity error: {max_group_err:.2e}")
    print("Saved:")
    print("  - results/day3_training_reference_stats.json")
    print("  - results/day3_step3_fixtures_15.json")
    print("  - results/day3_step3_explanations_15.json")
    print("  - results/day3_explanations/tx_<transaction_id>.json (15 files)")


if __name__ == "__main__":
    main()
