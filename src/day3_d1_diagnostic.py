import os
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


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


def summarize_series(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"count_valid": 0}
    return {
        "count_valid": int(len(vals)),
        "min": round(float(np.min(vals)), 4),
        "p01": round(float(np.percentile(vals, 1)), 4),
        "p05": round(float(np.percentile(vals, 5)), 4),
        "median": round(float(np.median(vals)), 4),
        "mean": round(float(np.mean(vals)), 4),
        "p95": round(float(np.percentile(vals, 95)), 4),
        "p99": round(float(np.percentile(vals, 99)), 4),
        "max": round(float(np.max(vals)), 4),
    }


def eval_slice(y_true, p_pred, label):
    n = int(len(y_true))
    pos = int(y_true.sum())
    prev = float(pos / n) if n > 0 else 0.0
    if pos == 0 or pos == n:
        return {"label": label, "n": n, "frauds": pos, "fraud_rate": round(prev, 5), "pr_auc": None, "roc_auc": None}
    pr = float(average_precision_score(y_true, p_pred))
    roc = float(roc_auc_score(y_true, p_pred))
    return {
        "label": label,
        "n": n,
        "frauds": pos,
        "fraud_rate": round(prev, 5),
        "pr_auc": round(pr, 5),
        "pr_lift_vs_prevalence": round(pr / prev, 2) if prev > 0 else None,
        "roc_auc": round(roc, 5),
        "mean_raw_prob": round(float(np.mean(p_pred)), 5),
    }


def main():
    print("=" * 72)
    print("Day 3 Diagnostic: D1 = 0 vs D1 > 0 and D1_norm Out-of-Range Behavior")
    print("=" * 72)

    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    df = reduce_mem_usage(df)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)

    df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)
    df["id_31"] = df["id_31"].apply(clean_browser)
    df["id_30"] = df["id_30"].apply(clean_os)
    df = compute_rolling_window_counts(df)
    df["D1_norm"] = df["D1"] - (df["TransactionDT"] / 86400.0)
    df = reduce_mem_usage(df)

    epoch_day_all = (df["TransactionDT"].values.astype(float) / 86400.0)
    d1_all = df["D1"].values.astype(float)
    d1_norm_all = df["D1_norm"].values.astype(float)
    y_all = df["isFraud"].values.astype(int)

    epoch_tr, epoch_va = epoch_day_all[:i1], epoch_day_all[i1:i2]
    d1_tr, d1_va = d1_all[:i1], d1_all[i1:i2]
    d1n_tr, d1n_va = d1_norm_all[:i1], d1_norm_all[i1:i2]
    y_tr, y_va = y_all[:i1], y_all[i1:i2]

    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    del df
    X_tr = X.iloc[:i1].copy()
    X_va = X.iloc[i1:i2].copy().reset_index(drop=True)
    del X

    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c], categories=cats)

    # Load locked Model A (do not retrain/alter locked model)
    model_a = xgb.XGBClassifier()
    model_a.load_model("results/day2_model_a_train_only.json")
    feat_names = model_a.get_booster().feature_names
    X_tr = X_tr[feat_names]
    X_va = X_va[feat_names]
    p_va_a = model_a.predict_proba(X_va)[:, 1]

    # Regime masks
    def get_regimes(d1_arr):
        is_nan = np.isnan(d1_arr)
        is_zero = (~is_nan) & (d1_arr == 0)
        is_pos = (~is_nan) & (d1_arr > 0)
        is_neg = (~is_nan) & (d1_arr < 0)
        return {"D1 == 0": is_zero, "D1 > 0": is_pos, "D1 < 0": is_neg, "D1 is NaN": is_nan}

    reg_tr = get_regimes(d1_tr)
    reg_va = get_regimes(d1_va)

    print("\n1. Share of rows & fraud rates by D1 regime:")
    regime_summary = {"train": {}, "validation": {}}
    for split_name, reg_dict, y_split in [("train", reg_tr, y_tr), ("validation", reg_va, y_va)]:
        n_split = len(y_split)
        print(f"  {split_name.upper()} (N={n_split:,}):")
        for r_name, mask in reg_dict.items():
            cnt = int(mask.sum())
            share = cnt / n_split
            frauds = int(y_split[mask].sum()) if cnt > 0 else 0
            frate = (frauds / cnt) if cnt > 0 else 0.0
            regime_summary[split_name][r_name] = {
                "count": cnt,
                "share": round(share, 5),
                "frauds": frauds,
                "fraud_rate": round(frate, 5),
            }
            if cnt > 0:
                print(f"    {r_name:10s}: {cnt:7,d} ({share*100:6.2f}%) | frauds={frauds:5,d} ({frate*100:5.2f}%)")

    # 2. Epoch day and D1_norm ranges
    tr_valid_min = float(np.nanmin(d1n_tr))
    tr_valid_max = float(np.nanmax(d1n_tr))
    tr_d1_0_min = float(np.nanmin(d1n_tr[reg_tr["D1 == 0"]]))
    tr_d1_0_max = float(np.nanmax(d1n_tr[reg_tr["D1 == 0"]]))

    print("\n2. Epoch day & D1_norm ranges against Train range:")
    print(f"  Train epoch day range: [{epoch_tr.min():.2f}, {epoch_tr.max():.2f}]")
    print(f"  Val   epoch day range: [{epoch_va.min():.2f}, {epoch_va.max():.2f}]")
    print(f"  Train D1_norm overall min..max: [{tr_valid_min:.4f}, {tr_valid_max:.4f}]")
    print(f"  Train D1 == 0 D1_norm min..max: [{tr_d1_0_min:.4f}, {tr_d1_0_max:.4f}]")

    range_summary = {
        "train_epoch_day_range": [round(float(epoch_tr.min()), 4), round(float(epoch_tr.max()), 4)],
        "val_epoch_day_range": [round(float(epoch_va.min()), 4), round(float(epoch_va.max()), 4)],
        "train_d1_norm_overall": summarize_series(d1n_tr),
        "train_d1_norm_D1_eq_0": summarize_series(d1n_tr[reg_tr["D1 == 0"]]),
        "train_d1_norm_D1_gt_0": summarize_series(d1n_tr[reg_tr["D1 > 0"]]),
        "val_d1_norm_overall": summarize_series(d1n_va),
        "val_d1_norm_D1_eq_0": summarize_series(d1n_va[reg_va["D1 == 0"]]),
        "val_d1_norm_D1_gt_0": summarize_series(d1n_va[reg_va["D1 > 0"]]),
    }

    # Check out-of-range rates in validation
    val_d1_0_vals = d1n_va[reg_va["D1 == 0"]]
    val_d1_pos_vals = d1n_va[reg_va["D1 > 0"]]
    val_all_valid = d1n_va[~np.isnan(d1n_va)]

    oor = {
        "train_global_min_d1_norm": round(tr_valid_min, 4),
        "train_global_max_d1_norm": round(tr_valid_max, 4),
        "val_all_below_train_min_count": int(np.sum(val_all_valid < tr_valid_min)),
        "val_all_below_train_min_share": round(float(np.mean(val_all_valid < tr_valid_min)), 5),
        "val_D1_eq_0_below_train_min_count": int(np.sum(val_d1_0_vals < tr_valid_min)),
        "val_D1_eq_0_below_train_min_share": round(float(np.mean(val_d1_0_vals < tr_valid_min)), 5),
        "val_D1_gt_0_below_train_min_count": int(np.sum(val_d1_pos_vals < tr_valid_min)),
        "val_D1_gt_0_below_train_min_share": round(float(np.mean(val_d1_pos_vals < tr_valid_min)), 5),
        "val_D1_gt_0_above_train_max_count": int(np.sum(val_d1_pos_vals > tr_valid_max)),
        "val_D1_gt_0_above_train_max_share": round(float(np.mean(val_d1_pos_vals > tr_valid_max)), 5),
        "corr_d1_norm_vs_epoch_tr_D1_eq_0": round(float(np.corrcoef(d1n_tr[reg_tr["D1 == 0"]], epoch_tr[reg_tr["D1 == 0"]])[0, 1]), 5),
        "corr_d1_norm_vs_epoch_va_D1_eq_0": round(float(np.corrcoef(d1n_va[reg_va["D1 == 0"]], epoch_va[reg_va["D1 == 0"]])[0, 1]), 5),
        "corr_d1_norm_vs_epoch_tr_D1_gt_0": round(float(np.corrcoef(d1n_tr[reg_tr["D1 > 0"]], epoch_tr[reg_tr["D1 > 0"]])[0, 1]), 5),
        "corr_d1_norm_vs_epoch_va_D1_gt_0": round(float(np.corrcoef(d1n_va[reg_va["D1 > 0"]], epoch_va[reg_va["D1 > 0"]])[0, 1]), 5),
    }

    print(f"  Val D1 == 0 D1_norm min..max: [{range_summary['val_d1_norm_D1_eq_0']['min']}, {range_summary['val_d1_norm_D1_eq_0']['max']}]")
    print(f"  Val D1 > 0  D1_norm min..max: [{range_summary['val_d1_norm_D1_gt_0']['min']}, {range_summary['val_d1_norm_D1_gt_0']['max']}]")
    print(f"  Val D1 == 0 strictly below train min ({tr_valid_min:.4f}): {oor['val_D1_eq_0_below_train_min_count']:,} / {len(val_d1_0_vals):,} ({oor['val_D1_eq_0_below_train_min_share']*100:.2f}%)")
    print(f"  Val D1 > 0  strictly below train min ({tr_valid_min:.4f}): {oor['val_D1_gt_0_below_train_min_count']:,} / {len(val_d1_pos_vals):,} ({oor['val_D1_gt_0_below_train_min_share']*100:.2f}%)")
    print(f"  Corr(D1_norm, epoch_day): Tr(D1=0)={oor['corr_d1_norm_vs_epoch_tr_D1_eq_0']}, Tr(D1>0)={oor['corr_d1_norm_vs_epoch_tr_D1_gt_0']}, Va(D1>0)={oor['corr_d1_norm_vs_epoch_va_D1_gt_0']}")

    # 3. PR-AUC for Locked Model A on D1 == 0 vs D1 > 0
    print("\n3. Locked Model A Validation PR-AUC by D1 regime:")
    model_a_slices = {
        "overall_val": eval_slice(y_va, p_va_a, "Overall Validation"),
        "D1_eq_0": eval_slice(y_va[reg_va["D1 == 0"]], p_va_a[reg_va["D1 == 0"]], "Val D1 == 0"),
        "D1_gt_0": eval_slice(y_va[reg_va["D1 > 0"]], p_va_a[reg_va["D1 > 0"]], "Val D1 > 0"),
        "D1_is_nan": eval_slice(y_va[reg_va["D1 is NaN"]], p_va_a[reg_va["D1 is NaN"]], "Val D1 is NaN"),
        "D1_norm_below_train_min": eval_slice(
            y_va[~np.isnan(d1n_va) & (d1n_va < tr_valid_min)],
            p_va_a[~np.isnan(d1n_va) & (d1n_va < tr_valid_min)],
            "Val D1_norm < Train min",
        ),
        "D1_norm_inside_train_range": eval_slice(
            y_va[~np.isnan(d1n_va) & (d1n_va >= tr_valid_min)],
            p_va_a[~np.isnan(d1n_va) & (d1n_va >= tr_valid_min)],
            "Val D1_norm >= Train min",
        ),
    }
    for k, s in model_a_slices.items():
        if s["n"] > 0:
            print(f"  {s['label']:30s} | N={s['n']:6,d} | Frauds={s['frauds']:4,d} ({s['fraud_rate']*100:5.2f}%) | PR-AUC={s['pr_auc']} (lift={s.get('pr_lift_vs_prevalence')}x) | ROC-AUC={s['roc_auc']}")

    # 4. Controlled Diagnostic Reference: Core vs Core + D1_norm over 3 shuffled runs + single unshuffled ref
    print("\n4a. Single-run unshuffled diagnostic (Core vs Locked Model A, train->val only):")
    t0 = time.time()
    core_cols = [c for c in feat_names if c != "D1_norm"]
    model_core = xgb.XGBClassifier(
        n_estimators=350,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )
    model_core.fit(X_tr[core_cols], y_tr, verbose=False)
    p_va_core = model_core.predict_proba(X_va[core_cols])[:, 1]
    print(f"  Unshuffled Core-only fit + pred in {time.time() - t0:.1f}s")

    core_slices = {
        "overall_val": eval_slice(y_va, p_va_core, "Overall Validation (Core-only)"),
        "D1_eq_0": eval_slice(y_va[reg_va["D1 == 0"]], p_va_core[reg_va["D1 == 0"]], "Val D1 == 0 (Core-only)"),
        "D1_gt_0": eval_slice(y_va[reg_va["D1 > 0"]], p_va_core[reg_va["D1 > 0"]], "Val D1 > 0 (Core-only)"),
        "D1_is_nan": eval_slice(y_va[reg_va["D1 is NaN"]], p_va_core[reg_va["D1 is NaN"]], "Val D1 is NaN (Core-only)"),
    }
    d1_norm_val_gain = {}
    for k in ["overall_val", "D1_eq_0", "D1_gt_0", "D1_is_nan"]:
        a_s = model_a_slices[k]
        c_s = core_slices[k]
        if a_s["pr_auc"] is not None and c_s["pr_auc"] is not None:
            diff_pr = round(a_s["pr_auc"] - c_s["pr_auc"], 5)
            diff_roc = round(a_s["roc_auc"] - c_s["roc_auc"], 5)
            d1_norm_val_gain[k] = {
                "core_only_pr_auc": c_s["pr_auc"],
                "model_a_pr_auc": a_s["pr_auc"],
                "delta_pr_auc": diff_pr,
                "core_only_roc_auc": c_s["roc_auc"],
                "model_a_roc_auc": a_s["roc_auc"],
                "delta_roc_auc": diff_roc,
            }
            print(f"  {k:12s}: Core={c_s['pr_auc']:.5f} -> Model A={a_s['pr_auc']:.5f} (Delta PR-AUC={diff_pr:+.5f}, Delta ROC={diff_roc:+.5f})")

    print("\n4b. 3-Shuffled-Run Slice Ablation (seeds [42, 101, 202], 350 trees, Core vs Core+D1_norm):")
    shuffle_seeds = [42, 101, 202]
    eval_masks = {
        "overall_val": np.ones(len(y_va), dtype=bool),
        "D1_eq_0": reg_va["D1 == 0"],
        "D1_gt_0": reg_va["D1 > 0"],
        "D1_norm_below_train_min": (~np.isnan(d1n_va)) & (d1n_va < tr_valid_min),
        "D1_norm_inside_train_range": (~np.isnan(d1n_va)) & (d1n_va >= tr_valid_min),
        "D1_is_nan": reg_va["D1 is NaN"],
    }
    multi_runs = {"Core": {k: [] for k in eval_masks}, "Core+D1_norm": {k: [] for k in eval_masks}}
    multi_runs_roc = {"Core": {k: [] for k in eval_masks}, "Core+D1_norm": {k: [] for k in eval_masks}}

    for seed in shuffle_seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].copy()
        y_tr_shuf = y_tr[perm]

        for cfg_name, cols in [("Core", core_cols), ("Core+D1_norm", feat_names)]:
            t_cfg = time.time()
            m_cfg = xgb.XGBClassifier(
                n_estimators=350,
                tree_method="hist",
                enable_categorical=True,
                eval_metric="aucpr",
                n_jobs=-1,
                random_state=42,
            )
            m_cfg.fit(X_tr_shuf[cols], y_tr_shuf, verbose=False)
            p_cfg = m_cfg.predict_proba(X_va[cols])[:, 1]
            for sl_key, sl_mask in eval_masks.items():
                pr_val = float(average_precision_score(y_va[sl_mask], p_cfg[sl_mask]))
                roc_val = float(roc_auc_score(y_va[sl_mask], p_cfg[sl_mask]))
                multi_runs[cfg_name][sl_key].append(pr_val)
                multi_runs_roc[cfg_name][sl_key].append(roc_val)
            print(f"  seed={seed} {cfg_name:12s} ({time.time()-t_cfg:.1f}s) -> Overall={multi_runs[cfg_name]['overall_val'][-1]:.5f} | D1=0:{multi_runs[cfg_name]['D1_eq_0'][-1]:.5f} | D1>0:{multi_runs[cfg_name]['D1_gt_0'][-1]:.5f}")

    three_run_slice_summary = {}
    print("\n  Summary across 3 shuffled runs (mean +/- n-1 std):")
    for sl_key in eval_masks:
        c_prs = np.array(multi_runs["Core"][sl_key])
        d_prs = np.array(multi_runs["Core+D1_norm"][sl_key])
        diffs = d_prs - c_prs
        c_rocs = np.array(multi_runs_roc["Core"][sl_key])
        d_rocs = np.array(multi_runs_roc["Core+D1_norm"][sl_key])
        roc_diffs = d_rocs - c_rocs
        three_run_slice_summary[sl_key] = {
            "core_pr_auc_runs": [round(float(x), 5) for x in c_prs],
            "core_pr_auc_mean": round(float(c_prs.mean()), 5),
            "core_pr_auc_std_n1": round(float(c_prs.std(ddof=1)), 5),
            "core_d1norm_pr_auc_runs": [round(float(x), 5) for x in d_prs],
            "core_d1norm_pr_auc_mean": round(float(d_prs.mean()), 5),
            "core_d1norm_pr_auc_std_n1": round(float(d_prs.std(ddof=1)), 5),
            "paired_delta_pr_auc_runs": [round(float(x), 5) for x in diffs],
            "delta_pr_auc_mean": round(float(diffs.mean()), 5),
            "delta_pr_auc_std_n1": round(float(diffs.std(ddof=1)), 5),
            "core_roc_auc_mean": round(float(c_rocs.mean()), 5),
            "core_d1norm_roc_auc_mean": round(float(d_rocs.mean()), 5),
            "delta_roc_auc_mean": round(float(roc_diffs.mean()), 5),
        }
        print(
            f"  {sl_key:26s} | Core={c_prs.mean():.5f}±{c_prs.std(ddof=1):.5f} | "
            f"Core+D1_norm={d_prs.mean():.5f}±{d_prs.std(ddof=1):.5f} | "
            f"Delta PR={diffs.mean():+.5f}±{diffs.std(ddof=1):.5f} | Delta ROC={roc_diffs.mean():+.5f}"
        )

    # 5. SHAP D1_norm breakdown on saved 5,000-row validation sample
    print("\n5. Saved 5,000-row SHAP attribution on D1_norm by D1 regime:")
    shap_vals = np.load("results/day3_shap_values_val5000.npy")
    sample_idx = np.load("results/day3_shap_sample_indices.npy")
    d1_norm_col_idx = feat_names.index("D1_norm")
    shap_d1n = shap_vals[:, d1_norm_col_idx]
    d1_s = d1_va[sample_idx]
    d1n_s = d1n_va[sample_idx]

    s_zero = (~np.isnan(d1_s)) & (d1_s == 0)
    s_pos = (~np.isnan(d1_s)) & (d1_s > 0)
    s_nan = np.isnan(d1_s)
    s_oor = (~np.isnan(d1n_s)) & (d1n_s < tr_valid_min)
    s_in = (~np.isnan(d1n_s)) & (d1n_s >= tr_valid_min)

    shap_d1n_breakdown = {}
    for label, m in [
        ("sample_all", np.ones(len(sample_idx), dtype=bool)),
        ("sample_D1_eq_0", s_zero),
        ("sample_D1_gt_0", s_pos),
        ("sample_D1_is_nan", s_nan),
        ("sample_D1_norm_below_train_min", s_oor),
        ("sample_D1_norm_inside_train_range", s_in),
    ]:
        if m.sum() > 0:
            shap_d1n_breakdown[label] = {
                "n": int(m.sum()),
                "mean_abs_shap_d1_norm": round(float(np.abs(shap_d1n[m]).mean()), 5),
                "mean_signed_shap_d1_norm": round(float(shap_d1n[m].mean()), 5),
                "pct_positive_shap": round(float(np.mean(shap_d1n[m] > 0)), 4),
            }
            print(f"  {label:32s}: N={int(m.sum()):4d} | mean|SHAP|={shap_d1n_breakdown[label]['mean_abs_shap_d1_norm']:.5f} | mean_signed={shap_d1n_breakdown[label]['mean_signed_shap_d1_norm']:+.5f}")

    # Check variance/saturation of SHAP(D1_norm) inside below-train-min region
    if s_oor.sum() > 0:
        oor_std = float(np.std(shap_d1n[s_oor]))
        oor_uniq = int(len(np.unique(np.round(shap_d1n[s_oor], 5))))
        shap_d1n_breakdown["sample_D1_norm_below_train_min"]["std_signed_shap"] = round(oor_std, 5)
        shap_d1n_breakdown["sample_D1_norm_below_train_min"]["unique_shap_values_5dp"] = oor_uniq
        print(f"  Out-of-range SHAP(D1_norm) spread: std={oor_std:.5f}, unique_values={oor_uniq} across N={int(s_oor.sum())}")

    out = {
        "metadata": {
            "purpose": "Diagnosis only on train vs validation (locked Model A unchanged, test set untouched)",
            "train_rows": int(len(y_tr)),
            "val_rows": int(len(y_va)),
        },
        "d1_regime_shares_and_fraud_rates": regime_summary,
        "d1_norm_ranges_vs_train": range_summary,
        "out_of_range_and_clock_correlations": oor,
        "model_a_validation_slices": model_a_slices,
        "core_only_vs_model_a_unshuffled_single_run": d1_norm_val_gain,
        "core_vs_core_d1norm_3_shuffled_runs_by_slice": three_run_slice_summary,
        "shap_val5000_d1_norm_breakdown": shap_d1n_breakdown,
    }
    with open("results/day3_d1_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved diagnostic to results/day3_d1_diagnostic.json")


if __name__ == "__main__":
    main()
