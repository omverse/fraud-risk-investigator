import os
import re
import copy
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats
from scipy.stats import spearmanr
from lime.discretize import BaseDiscretizer
from lime.lime_tabular import LimeTabularExplainer

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


def _discretize_array_with_missing(x, qts, missing_bin):
    """Map finite values via np.searchsorted(qts, x) and non-finite/NaN to missing_bin."""
    if np.ndim(x) == 0:
        if not np.isfinite(x):
            return missing_bin
        return int(np.searchsorted(qts, x))
    x_arr = np.asarray(x, dtype=np.float64)
    res = np.empty(len(x_arr), dtype=int)
    mask_miss = ~np.isfinite(x_arr)
    res[mask_miss] = missing_bin
    if np.any(~mask_miss):
        res[~mask_miss] = np.searchsorted(qts, x_arr[~mask_miss])
    return res


class NaNAwareQuartileDiscretizer(BaseDiscretizer):
    """Quartile discretizer preserving NaN as an explicit MISSING bin for XGBoost."""

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        self.to_discretize = [x for x in range(data.shape[1]) if x not in categorical_features]
        self.data_stats = None
        self.names = {}
        self.lambdas = {}
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.missing_bin = {}
        self.random_state = np.random.RandomState(random_state) if isinstance(random_state, (int, np.integer)) else (
            random_state if random_state is not None else np.random.RandomState(42)
        )

        for feature in self.to_discretize:
            col = data[:, feature]
            valid = col[np.isfinite(col)]
            name = feature_names[feature]

            if len(valid) == 0:
                qts = np.array([], dtype=np.float64)
                m_bin = 0
                self.missing_bin[feature] = m_bin
                self.names[feature] = [f"{name} is MISSING"]
                self.lambdas[feature] = lambda x, q=qts, mb=m_bin: _discretize_array_with_missing(x, q, mb)
                self.means[feature] = []
                self.stds[feature] = []
                self.mins[feature] = []
                self.maxs[feature] = []
                continue

            qts = np.unique(np.percentile(valid, [25, 50, 75]))
            n_qts = len(qts)
            obs_bins = n_qts + 1
            m_bin = obs_bins
            self.missing_bin[feature] = m_bin

            b_min = float(np.min(valid))
            b_max = float(np.max(valid))

            bin_names = [f"{name} <= {qts[0]:.2f}"]
            for i in range(n_qts - 1):
                bin_names.append(f"{qts[i]:.2f} < {name} <= {qts[i + 1]:.2f}")
            bin_names.append(f"{name} > {qts[n_qts - 1]:.2f}")
            bin_names.append(f"{name} is MISSING")
            self.names[feature] = bin_names

            self.lambdas[feature] = lambda x, q=qts, mb=m_bin: _discretize_array_with_missing(x, q, mb)
            disc = self.lambdas[feature](col)

            self.mins[feature] = [b_min] + qts.tolist()
            self.maxs[feature] = qts.tolist() + [b_max]
            self.means[feature] = []
            self.stds[feature] = []
            for b in range(obs_bins):
                sel = col[(disc == b) & np.isfinite(col)]
                if len(sel) == 0:
                    mid = 0.5 * (self.mins[feature][b] + self.maxs[feature][b])
                    self.means[feature].append(float(mid))
                    self.stds[feature].append(1e-11)
                else:
                    self.means[feature].append(float(np.mean(sel)))
                    self.stds[feature].append(float(np.std(sel)) + 1e-11)

    def bins(self, data, labels):
        return []

    def get_undiscretize_values(self, feature, values):
        vals = np.asarray(values, dtype=int)
        orig_shape = vals.shape
        vals_flat = vals.ravel()
        ret = np.empty(len(vals_flat), dtype=np.float64)

        m_bin = self.missing_bin[feature]
        is_miss = vals_flat == m_bin
        ret[is_miss] = np.nan

        obs_mask = ~is_miss
        if np.any(obs_mask):
            if len(self.means[feature]) == 0:
                ret[obs_mask] = np.nan
            else:
                obs_idx = np.clip(vals_flat[obs_mask], 0, len(self.means[feature]) - 1)
                mins = np.asarray(self.mins[feature])[obs_idx]
                maxs = np.asarray(self.maxs[feature])[obs_idx]
                means = np.asarray(self.means[feature])[obs_idx]
                stds = np.asarray(self.stds[feature])[obs_idx]

                # Clip lower/upper equal boundaries cleanly instead of LIME's default minz fallback
                obs_out = mins.copy()
                need_sample = (maxs > mins) & (stds > 0)
                if np.any(need_sample):
                    minz = (mins[need_sample] - means[need_sample]) / stds[need_sample]
                    maxz = (maxs[need_sample] - means[need_sample]) / stds[need_sample]
                    sampled = scipy.stats.truncnorm.rvs(
                        minz,
                        maxz,
                        loc=means[need_sample],
                        scale=stds[need_sample],
                        random_state=self.random_state,
                    )
                    obs_out[need_sample] = np.clip(sampled, mins[need_sample], maxs[need_sample])
                ret[obs_mask] = obs_out

        return ret.reshape(orig_shape)


def parse_lime_rule_to_base_feature(rule_str, feature_names_set):
    """Map LIME discretized/categorical rule string back to its base feature name."""
    # Check interval pattern: "<num> < <feat> <= <num>"
    m_interval = re.match(r"^[+-]?\d+(?:\.\d+)?\s*<\s*(.+?)\s*<=\s*[+-]?\d+(?:\.\d+)?$", rule_str)
    if m_interval and m_interval.group(1) in feature_names_set:
        return m_interval.group(1)

    # Check "<feat> <= <num>" or "<feat> > <num>"
    m_bound = re.match(r"^(.+?)\s*(?:<=|>)\s*[+-]?\d+(?:\.\d+)?$", rule_str)
    if m_bound and m_bound.group(1) in feature_names_set:
        return m_bound.group(1)

    # Check "<feat> is MISSING"
    if rule_str.endswith(" is MISSING"):
        cand = rule_str[: -len(" is MISSING")]
        if cand in feature_names_set:
            return cand

    # Check categorical "<feat>=<value>"
    if "=" in rule_str:
        prefix = rule_str.split("=", 1)[0]
        if prefix in feature_names_set:
            return prefix

    raise ValueError(f"Could not map LIME rule string '{rule_str}' back to any base feature.")


def format_val(val):
    if pd.isna(val):
        return "MISSING"
    if isinstance(val, (float, np.floating)):
        return round(float(val), 4)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return str(val)


def main():
    print("=" * 72)
    print("Day 3 Step 2: LIME Comparison Across 3 Seeds on Same 15 Alerts")
    print("=" * 72)

    # 1. Load data & engineer locked features identically to Step 1
    print("\n1. Loading data & reconstructing locked features (Core + D1_norm)...")
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id

    df = reduce_mem_usage(df)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(0.70 * n), int(0.85 * n)

    df = df.copy()
    extra_cols = pd.DataFrame({
        "hour_of_day": ((df["TransactionDT"] // 3600) % 24).astype(np.int8),
        "day_of_week": ((df["TransactionDT"] // 86400) % 7).astype(np.int8),
        "id_31": df["id_31"].apply(clean_browser),
        "id_30": df["id_30"].apply(clean_os),
    })
    df["hour_of_day"] = extra_cols["hour_of_day"]
    df["day_of_week"] = extra_cols["day_of_week"]
    df["id_31"] = extra_cols["id_31"]
    df["id_30"] = extra_cols["id_30"]
    df = compute_rolling_window_counts(df)
    df["D1_norm"] = (df["D1"] - (df["TransactionDT"] / 86400.0)).astype(np.float32)
    df = reduce_mem_usage(df)

    y_tr = df["isFraud"].iloc[:i1].values
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    del df
    X_tr = X.iloc[:i1].copy().reset_index(drop=True)
    X_va = X.iloc[i1:i2].copy().reset_index(drop=True)
    del X

    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    cat_categories = {}
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        cat_categories[c] = cats
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c], categories=cats)

    # 2. Load locked Model A & Step 1 SHAP artifacts
    print("\n2. Loading Model A and Day 3 Step 1 SHAP outputs...")
    model_a = xgb.XGBClassifier()
    model_a.load_model("results/day2_model_a_train_only.json")
    feat_names = model_a.get_booster().feature_names
    feat_set = set(feat_names)
    X_tr = X_tr[feat_names]
    X_va = X_va[feat_names]

    with open("results/day3_step1_shap.json") as f:
        step1_data = json.load(f)
    selected_cases = step1_data["selected_15_cases"]
    shap_values_5000 = np.load("results/day3_shap_values_val5000.npy")
    shap_time_total = float(step1_data["metadata"]["shap_time_sec"])

    # 3. Sample 10,000 training rows for LIME background & build encoder/decoder
    print("\n3. Sampling 10,000 training rows for LIME background...")
    rng_bg = np.random.default_rng(42)
    bg_idx = np.sort(rng_bg.choice(len(X_tr), size=10000, replace=False))
    X_bg = X_tr.iloc[bg_idx].copy().reset_index(drop=True)
    bg_fraud_rate = float(y_tr[bg_idx].mean())
    print(f"  10,000 training background rows sampled | Fraud rate in background = {bg_fraud_rate*100:.3f}%")

    cat_col_set = set(cat_cols)
    cat_indices = [i for i, c in enumerate(feat_names) if c in cat_col_set]
    missing_code = {c: len(cat_categories[c]) for c in cat_cols}
    categorical_names = {
        i: list(cat_categories[feat_names[i]]) + ["MISSING"]
        for i in cat_indices
    }

    def df_to_lime_matrix(df_in):
        mat = np.empty((len(df_in), len(feat_names)), dtype=np.float64)
        for j, c in enumerate(feat_names):
            if c in cat_col_set:
                codes = df_in[c].cat.codes.values.astype(np.int32)
                codes = np.where(codes < 0, missing_code[c], codes)
                mat[:, j] = codes
            else:
                mat[:, j] = df_in[c].values.astype(np.float64)
        return mat

    def lime_matrix_to_df(arr):
        cols = {}
        for j, c in enumerate(feat_names):
            if c in cat_col_set:
                codes = np.rint(arr[:, j]).astype(np.int32)
                codes = np.where(codes >= missing_code[c], -1, codes)
                cols[c] = pd.Categorical.from_codes(codes, categories=cat_categories[c])
            else:
                cols[c] = arr[:, j].astype(np.float32)
        return pd.DataFrame(cols)

    def predict_fn_lime(arr):
        df_pred = lime_matrix_to_df(arr)
        return model_a.predict_proba(df_pred)

    t_setup_0 = time.time()
    M_bg = df_to_lime_matrix(X_bg)
    val_indices_15 = [c["val_index"] for c in selected_cases]
    X_15 = X_va.iloc[val_indices_15].copy().reset_index(drop=True)
    M_15 = df_to_lime_matrix(X_15)

    # Verify round-trip prediction exactness for all 15 alerts
    p_check = predict_fn_lime(M_15)[:, 1]
    max_pred_diff = max(abs(float(p_check[k]) - selected_cases[k]["raw_prob"]) for k in range(15))
    print(f"  Round-trip predict_fn max diff vs Step 1 raw_prob = {max_pred_diff:.6f}")
    assert max_pred_diff < 1e-4, f"Round-trip prediction mismatch: {max_pred_diff}"

    base_disc = NaNAwareQuartileDiscretizer(
        M_bg,
        categorical_features=cat_indices,
        feature_names=feat_names,
        random_state=42,
    )
    # Build clean surrogate matrix for StandardScaler (avoid empty-slice warning if any column is all-NaN)
    all_nan_cols = np.all(np.isnan(M_bg), axis=0)
    M_bg_for_init = M_bg.copy()
    if np.any(all_nan_cols):
        M_bg_for_init[:, all_nan_cols] = 0.0
        print(f"  Note: {int(all_nan_cols.sum())} columns were 100% NaN in 10k sample.")

    t_setup = time.time() - t_setup_0
    print(f"  Background matrix & NaN-aware discretizer built in {t_setup:.2f}s")

    # 4. Run LIME across 3 seeds on the 15 alerts
    seeds = [42, 43, 44]
    num_samples = 5000
    num_features = 10

    explainers = {}
    t_init_0 = time.time()
    for s in seeds:
        disc_s = copy.deepcopy(base_disc)
        disc_s.random_state = np.random.RandomState(s)
        explainers[s] = LimeTabularExplainer(
            training_data=M_bg_for_init,
            mode="classification",
            feature_names=feat_names,
            categorical_features=cat_indices,
            categorical_names=categorical_names,
            class_names=["legit", "fraud"],
            discretizer=disc_s,
            feature_selection="auto",
            random_state=s,
        )
    t_explainer_init = (time.time() - t_init_0) / len(seeds)
    print(f"  LimeTabularExplainer avg init time per seed: {t_explainer_init:.2f}s")

    print(f"\n4. Running LIME on 15 alerts x {len(seeds)} seeds ({15*len(seeds)} explanations, num_samples={num_samples}, num_features={num_features})...")

    all_timings = []
    per_case_records = []
    seed_summaries = {s: {"top5_overlap_counts": [], "top10_hits_of_shap5": [], "top5_sign_matches_all5": [],
                          "top5_sign_matches_overlap": [], "local_r2": [], "spearman_signed_on_top10": []} for s in seeds}

    for k, case in enumerate(selected_cases):
        sample_pos = case["sample_pos"]
        sv_full = shap_values_5000[sample_pos]
        abs_sv = np.abs(sv_full)
        shap_rank_order = np.argsort(-abs_sv)
        shap_rank_map = {feat_names[f_i]: r + 1 for r, f_i in enumerate(shap_rank_order)}
        shap_val_map = {feat_names[f_i]: float(sv_full[f_i]) for f_i in range(len(feat_names))}
        shap_top5_list = [c["feature"] for c in case["top_5_shap_contributions"]]
        shap_top5_set = set(shap_top5_list)

        row_vec = M_15[k]
        runs_by_seed = {}

        for s in seeds:
            exp_t0 = time.time()
            exp = explainers[s].explain_instance(
                data_row=row_vec,
                predict_fn=predict_fn_lime,
                labels=(1,),
                num_features=num_features,
                num_samples=num_samples,
            )
            dt = time.time() - exp_t0
            all_timings.append(dt)

            local_pairs = exp.local_exp[1]
            rule_pairs = exp.as_list(label=1)
            assert len(local_pairs) == len(rule_pairs)

            top10_items = []
            for r_rank, ((col_idx, w), (rule_str, w_list)) in enumerate(zip(local_pairs, rule_pairs), start=1):
                base_feat = feat_names[col_idx]
                parsed_base = parse_lime_rule_to_base_feature(rule_str, feat_set)
                assert parsed_base == base_feat, f"Mismatch {parsed_base} != {base_feat} for '{rule_str}'"
                assert abs(w - w_list) < 1e-9

                shap_v = shap_val_map[base_feat]
                shap_r = shap_rank_map[base_feat]
                same_sign = (w > 0 and shap_v > 0) or (w < 0 and shap_v < 0)
                top10_items.append({
                    "lime_rank": r_rank,
                    "feature": base_feat,
                    "lime_rule": rule_str,
                    "raw_value": format_val(X_15.iloc[k, col_idx]),
                    "lime_weight_prob": round(float(w), 6),
                    "shap_rank": int(shap_r),
                    "shap_value_log_odds": round(float(shap_v), 5),
                    "in_shap_top5": base_feat in shap_top5_set,
                    "same_sign_as_shap": bool(same_sign),
                })

            top5_items = top10_items[:5]
            lime_top5_set = {x["feature"] for x in top5_items}
            lime_top10_set = {x["feature"] for x in top10_items}
            overlap_5 = lime_top5_set & shap_top5_set
            overlap_10_vs_5 = lime_top10_set & shap_top5_set

            sign_match_all5 = sum(1 for x in top5_items if x["same_sign_as_shap"])
            overlap_items = [x for x in top5_items if x["in_shap_top5"]]
            sign_match_overlap = sum(1 for x in overlap_items if x["same_sign_as_shap"])

            # Spearman correlation across LIME's top 10 features between LIME prob weight and SHAP log-odds
            lime_w_10 = np.array([x["lime_weight_prob"] for x in top10_items])
            shap_v_10 = np.array([x["shap_value_log_odds"] for x in top10_items])
            sp_corr = spearmanr(lime_w_10, shap_v_10).statistic
            sp_corr = 0.0 if np.isnan(sp_corr) else float(sp_corr)

            local_r2 = float(exp.score) if hasattr(exp, "score") else None
            local_pred_prob = float(exp.local_pred[0]) if hasattr(exp, "local_pred") else None

            seed_summaries[s]["top5_overlap_counts"].append(len(overlap_5))
            seed_summaries[s]["top10_hits_of_shap5"].append(len(overlap_10_vs_5))
            seed_summaries[s]["top5_sign_matches_all5"].append(sign_match_all5)
            if len(overlap_items) > 0:
                seed_summaries[s]["top5_sign_matches_overlap"].append(sign_match_overlap / len(overlap_items))
            seed_summaries[s]["local_r2"].append(local_r2)
            seed_summaries[s]["spearman_signed_on_top10"].append(sp_corr)

            runs_by_seed[str(s)] = {
                "seed": s,
                "runtime_sec": round(dt, 4),
                "local_r2": round(local_r2, 5) if local_r2 is not None else None,
                "local_pred_prob": round(local_pred_prob, 5) if local_pred_prob is not None else None,
                "top5_overlap_with_shap_count": len(overlap_5),
                "top5_overlap_features": sorted(overlap_5),
                "top5_jaccard_with_shap": round(len(overlap_5) / len(lime_top5_set | shap_top5_set), 4),
                "shap_top5_in_lime_top10_count": len(overlap_10_vs_5),
                "sign_agreement_lime_top5_vs_full_shap": f"{sign_match_all5}/5",
                "sign_agreement_on_overlap_only": f"{sign_match_overlap}/{len(overlap_items)}" if overlap_items else "N/A (0 overlap)",
                "spearman_signed_top10_vs_shap": round(sp_corr, 4),
                "top_5_lime": top5_items,
                "ranks_6_to_10_lime": top10_items[5:],
            }

        # Between-seed stability metrics for this alert
        s_sets = [({x["feature"] for x in runs_by_seed[str(s)]["top_5_lime"]}) for s in seeds]
        s_ordered = [([x["feature"] for x in runs_by_seed[str(s)]["top_5_lime"]]) for s in seeds]
        pair_overlaps = [
            len(s_sets[0] & s_sets[1]),
            len(s_sets[0] & s_sets[2]),
            len(s_sets[1] & s_sets[2]),
        ]
        pair_jaccards = [
            len(s_sets[0] & s_sets[1]) / len(s_sets[0] | s_sets[1]),
            len(s_sets[0] & s_sets[2]) / len(s_sets[0] | s_sets[2]),
            len(s_sets[1] & s_sets[2]) / len(s_sets[1] | s_sets[2]),
        ]
        top1_matches = sum([
            s_ordered[0][0] == s_ordered[1][0],
            s_ordered[0][0] == s_ordered[2][0],
            s_ordered[1][0] == s_ordered[2][0],
        ])
        core_3way = s_sets[0] & s_sets[1] & s_sets[2]
        union_3way = s_sets[0] | s_sets[1] | s_sets[2]

        # Check any sign flips across seeds for any feature appearing in >=2 seeds' top 10
        feat_signs_by_seed = {}
        feat_weights_by_seed = {}
        for s in seeds:
            for item in runs_by_seed[str(s)]["top_5_lime"] + runs_by_seed[str(s)]["ranks_6_to_10_lime"]:
                f = item["feature"]
                feat_signs_by_seed.setdefault(f, []).append(1 if item["lime_weight_prob"] >= 0 else -1)
                feat_weights_by_seed.setdefault(f, []).append(item["lime_weight_prob"])
        sign_flips_top10 = [f for f, sgns in feat_signs_by_seed.items() if len(set(sgns)) > 1]

        per_case_records.append({
            "case_index_in_15": k + 1,
            "category": case["category"],
            "transaction_id": case["transaction_id"],
            "amount_usd": case["amount_usd"],
            "raw_prob": case["raw_prob"],
            "calibrated_prob": case["calibrated_prob"],
            "shap_top5_features": shap_top5_list,
            "seed_stability": {
                "mean_pairwise_top5_overlap_out_of_5": round(float(np.mean(pair_overlaps)), 4),
                "mean_pairwise_top5_jaccard": round(float(np.mean(pair_jaccards)), 4),
                "core_3way_shared_top5_count": len(core_3way),
                "core_3way_shared_top5_features": sorted(core_3way),
                "union_unique_features_across_3_top5s": len(union_3way),
                "top1_feature_pairwise_match_rate": round(top1_matches / 3.0, 4),
                "features_flipping_sign_across_seeds_top10": sign_flips_top10,
            },
            "runs_by_seed": runs_by_seed,
        })

        print(
            f"  [{k+1:2d}/15] {case['category']:15s} Tx {case['transaction_id']} | "
            f"p={case['raw_prob']:.4f} | "
            f"SHAP∩LIME5 mean={np.mean([runs_by_seed[str(s)]['top5_overlap_with_shap_count'] for s in seeds]):.2f}/5 | "
            f"SeedOverlap={np.mean(pair_overlaps):.2f}/5 (Union={len(union_3way)}) | "
            f"R2={np.mean([runs_by_seed[str(s)]['local_r2'] for s in seeds]):.3f}"
        )

    # 5. Aggregate Metrics Summary
    all_shap_overlaps = [runs["top5_overlap_with_shap_count"] for c in per_case_records for runs in c["runs_by_seed"].values()]
    all_shap_jaccards = [runs["top5_jaccard_with_shap"] for c in per_case_records for runs in c["runs_by_seed"].values()]
    all_shap10_hits = [runs["shap_top5_in_lime_top10_count"] for c in per_case_records for runs in c["runs_by_seed"].values()]
    all_r2 = [runs["local_r2"] for c in per_case_records for runs in c["runs_by_seed"].values()]
    all_spearman = [runs["spearman_signed_top10_vs_shap"] for c in per_case_records for runs in c["runs_by_seed"].values()]
    all_sign_all5 = [float(np.mean(seed_summaries[s]["top5_sign_matches_all5"])) / 5.0 for s in seeds]
    all_sign_ov = [v for s in seeds for v in seed_summaries[s]["top5_sign_matches_overlap"]]

    seed_pair_overlaps = [c["seed_stability"]["mean_pairwise_top5_overlap_out_of_5"] for c in per_case_records]
    seed_pair_jaccards = [c["seed_stability"]["mean_pairwise_top5_jaccard"] for c in per_case_records]
    seed_core_3way = [c["seed_stability"]["core_3way_shared_top5_count"] for c in per_case_records]
    seed_union_counts = [c["seed_stability"]["union_unique_features_across_3_top5s"] for c in per_case_records]
    seed_top1_rates = [c["seed_stability"]["top1_feature_pairwise_match_rate"] for c in per_case_records]
    total_sign_flips = sum(len(c["seed_stability"]["features_flipping_sign_across_seeds_top10"]) for c in per_case_records)

    # Breakdown by category (TP, FP, FN)
    by_cat = {}
    for cat in ["true_alert_TP", "false_alert_FP", "missed_fraud_FN"]:
        sub = [c for c in per_case_records if c["category"] == cat]
        sub_ov = [runs["top5_overlap_with_shap_count"] for c in sub for runs in c["runs_by_seed"].values()]
        sub_seed_ov = [c["seed_stability"]["mean_pairwise_top5_overlap_out_of_5"] for c in sub]
        sub_r2 = [runs["local_r2"] for c in sub for runs in c["runs_by_seed"].values()]
        by_cat[cat] = {
            "mean_top5_overlap_with_shap_out_of_5": round(float(np.mean(sub_ov)), 4),
            "mean_seed_pairwise_overlap_out_of_5": round(float(np.mean(sub_seed_ov)), 4),
            "mean_local_r2": round(float(np.mean(sub_r2)), 4),
        }

    aggregate = {
        "timing": {
            "background_sample_rows": 10000,
            "background_fraud_rate": round(bg_fraud_rate, 5),
            "discretizer_and_matrix_setup_sec": round(t_setup, 3),
            "explainer_init_avg_sec_per_seed": round(t_explainer_init, 3),
            "total_45_explanations_sec": round(float(np.sum(all_timings)), 3),
            "mean_sec_per_explanation": round(float(np.mean(all_timings)), 4),
            "median_sec_per_explanation": round(float(np.median(all_timings)), 4),
            "min_sec_per_explanation": round(float(np.min(all_timings)), 4),
            "max_sec_per_explanation": round(float(np.max(all_timings)), 4),
            "shap_5000_rows_total_sec": shap_time_total,
            "shap_amortized_sec_per_row": round(shap_time_total / 5000.0, 6),
            "lime_vs_shap_per_row_speed_ratio": round(float(np.mean(all_timings)) / (shap_time_total / 5000.0), 1),
        },
        "shap_vs_lime_alignment": {
            "mean_top5_overlap_out_of_5": round(float(np.mean(all_shap_overlaps)), 4),
            "mean_top5_overlap_pct": round(float(np.mean(all_shap_overlaps)) / 5.0 * 100.0, 2),
            "min_max_top5_overlap_across_45_runs": [int(np.min(all_shap_overlaps)), int(np.max(all_shap_overlaps))],
            "mean_top5_jaccard": round(float(np.mean(all_shap_jaccards)), 4),
            "mean_shap_top5_inside_lime_top10_out_of_5": round(float(np.mean(all_shap10_hits)), 4),
            "sign_agreement_rate_all_lime_top5_vs_shap": round(float(np.mean(all_sign_all5)), 4),
            "sign_agreement_rate_on_top5_overlap_only": round(float(np.mean(all_sign_ov)), 4) if all_sign_ov else None,
            "n_runs_with_at_least_1_overlap": len(all_sign_ov),
            "mean_spearman_signed_lime_top10_vs_shap": round(float(np.mean(all_spearman)), 4),
            "mean_local_ridge_r2": round(float(np.mean(all_r2)), 4),
            "min_max_local_ridge_r2": [round(float(np.min(all_r2)), 4), round(float(np.max(all_r2)), 4)],
        },
        "lime_seed_to_seed_stability": {
            "seeds": seeds,
            "mean_pairwise_top5_overlap_out_of_5": round(float(np.mean(seed_pair_overlaps)), 4),
            "mean_pairwise_top5_jaccard": round(float(np.mean(seed_pair_jaccards)), 4),
            "mean_3way_shared_top5_out_of_5": round(float(np.mean(seed_core_3way)), 4),
            "mean_unique_features_in_union_of_3_top5s": round(float(np.mean(seed_union_counts)), 4),
            "mean_top1_pairwise_match_rate": round(float(np.mean(seed_top1_rates)), 4),
            "total_feature_sign_flips_in_top10_across_15_cases": int(total_sign_flips),
        },
        "by_alert_category": by_cat,
    }

    print("\n" + "=" * 72)
    print("SUMMARY METRICS: Day 3 Step 2 LIME vs SHAP")
    print("=" * 72)
    print(json.dumps(aggregate, indent=2))

    out = {
        "metadata": {
            "model": "day2_model_a_train_only.json",
            "background_rows": 10000,
            "alerts_explained": 15,
            "seeds": seeds,
            "num_samples_per_explanation": num_samples,
            "num_features_requested": num_features,
            "discretizer": "NaNAwareQuartileDiscretizer (quantile bins + explicit MISSING bin)",
            "round_trip_max_pred_diff": round(max_pred_diff, 7),
        },
        "aggregate_summary": aggregate,
        "cases": per_case_records,
    }
    with open("results/day3_step2_lime.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved complete Step 2 LIME results to results/day3_step2_lime.json")


if __name__ == "__main__":
    main()
