import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import re
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

PLAIN_MEANING_FEATURES = {
    "TransactionAmt": "Transaction amount (USD)",
    "ProductCD": "Product code",
    "card4": "Card network",
    "card6": "Card type (credit/debit)",
    "P_emaildomain": "Purchaser email domain",
    "R_emaildomain": "Recipient email domain",
    "DeviceType": "Device type (desktop/mobile)",
    "DeviceInfo": "Device information",
    "id_30": "OS family",
    "id_31": "Browser family",
    "id_33": "Screen resolution",
    "hour_of_day": "Hour of day (dataset relative clock; offset mod 24h)",
    "day_of_week": "Day of week (dataset relative clock; offset mod 7d)",
    "card1_trans_count_1h": "Card transaction count (past 1h)",
    "card1_trans_count_24h": "Card transaction count (past 24h)",
    "card1_trans_count_7d": "Card transaction count (past 7d)",
    "card1_sec_since_prev": "Seconds since previous card transaction",
    "card1_amt_ratio_to_prev_avg": "Amount ratio vs card historical average",
}

GROUP_DEFINITIONS = [
    ("amount", "Amount (TransactionAmt)"),
    ("card_history", "Card-history features (rolling velocity & prior-avg ratio)"),
    ("C_columns", "C columns (anonymised counting features)"),
    ("D_columns", "D columns (anonymised time-delta features)"),
    ("M_flags", "M flags (anonymised match flags)"),
    ("V_columns", "V columns (Vesta-engineered anonymised features)"),
    ("card_and_address", "Card and address fields (anonymised card/address IDs & plain card metadata)"),
    ("device_and_identity", "Device and identity fields (DeviceType/Info & id_01–id_38)"),
    ("transaction_context_and_distance", "Transaction context & distance (ProductCD, email domains, relative clock, anonymised dist1/dist2)"),
]

DISCLAIMER_TEXT = "This explanation describes what drove the model's score, not why the fraud happened."


def classify_feature_group(col_name):
    """Map every model feature into a mutually exclusive documented group."""
    if col_name == "TransactionAmt":
        return "amount"
    if col_name.startswith("card1_"):
        return "card_history"
    if re.match(r"^C\d+$", col_name):
        return "C_columns"
    if re.match(r"^D\d+(_norm)?$", col_name):
        return "D_columns"
    if re.match(r"^M\d+$", col_name):
        return "M_flags"
    if re.match(r"^V\d+$", col_name):
        return "V_columns"
    if re.match(r"^(card\d+|addr\d+)$", col_name):
        return "card_and_address"
    if col_name in ("DeviceType", "DeviceInfo") or re.match(r"^id_\d+$", col_name):
        return "device_and_identity"
    if col_name in ("ProductCD", "dist1", "dist2", "P_emaildomain", "R_emaildomain", "hour_of_day", "day_of_week"):
        return "transaction_context_and_distance"
    raise ValueError(f"Unmapped feature '{col_name}' cannot be assigned to any group.")


def describe_feature(col_name):
    if col_name in PLAIN_MEANING_FEATURES:
        return False, PLAIN_MEANING_FEATURES[col_name]
    return True, "an anonymised feature"


def describe_masked_category(col_name):
    """Coarse category label allowed for anonymised columns in downstream summaries."""
    if col_name not in PLAIN_MEANING_FEATURES:
        if re.match(r"^C\d+$", col_name):
            return "a counting-type feature"
        if re.match(r"^D\d+(_norm)?$", col_name):
            return "a time-delta feature"
        if re.match(r"^M\d+$", col_name):
            return "a match flag"
        if re.match(r"^V\d+$", col_name):
            return "a Vesta-engineered anonymised feature"
        if re.match(r"^(card\d+|addr\d+)$", col_name):
            return "an anonymised card or address identifier"
        if re.match(r"^id_\d+$", col_name):
            return "an anonymised identity feature"
        if col_name in ("dist1", "dist2"):
            return "an anonymised distance feature"
        return "an anonymised feature"
    return PLAIN_MEANING_FEATURES[col_name]


def _format_val(val):
    if val is None or pd.isna(val):
        return "MISSING"
    if isinstance(val, (float, np.floating)):
        return round(float(val), 4)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return str(val)


def build_training_reference_stats(X_tr, feat_names, d1_regime_stats=None):
    """Extract numeric min/max/quantiles and categorical vocabularies strictly from Training data."""
    numeric_ranges = {}
    categorical_levels = {}
    for c in feat_names:
        s = X_tr[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(s):
            cats = [str(x) for x in s.cat.categories] if isinstance(s.dtype, pd.CategoricalDtype) else sorted(s.dropna().astype(str).unique().tolist())
            categorical_levels[c] = cats
        else:
            vals = s.values.astype(np.float64)
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                numeric_ranges[c] = {
                    "min": round(float(np.min(finite)), 4),
                    "p01": round(float(np.percentile(finite, 1)), 4),
                    "p99": round(float(np.percentile(finite, 99)), 4),
                    "max": round(float(np.max(finite)), 4),
                    "train_missing_rate": round(float(np.mean(~np.isfinite(vals))), 5),
                }
            else:
                numeric_ranges[c] = {"min": None, "p01": None, "p99": None, "max": None, "train_missing_rate": 1.0}

    return {
        "metadata": {"source": "X_tr (oldest 70% training split, 413,378 rows)"},
        "feature_count": len(feat_names),
        "numeric_ranges": numeric_ranges,
        "categorical_levels": categorical_levels,
        "d1_missing_shift": d1_regime_stats or {
            "train_missing_share": 0.00066,
            "train_missing_fraud_rate": 0.00366,
            "val_missing_share": 0.01078,
            "val_missing_fraud_rate": 0.04084,
        },
    }


class FraudTransactionExplainer:
    """Single-transaction explainer combining locked Model A, isotonic calibrator, TreeSHAP, and drift cautions."""

    def __init__(self, model, calibrator, ref_stats):
        self.model = model
        self.calibrator = calibrator
        self.ref_stats = ref_stats
        self.feat_names = list(model.get_booster().feature_names)
        self.tree_explainer = shap.TreeExplainer(model)
        # TreeExplainer updates expected_value from XGBoost pred_contribs on first shap_values() call
        self.base_value = None

        # Precompute exhaustive group -> feature index map
        self.group_indices = {g_key: [] for g_key, _ in GROUP_DEFINITIONS}
        for idx, col in enumerate(self.feat_names):
            g_key = classify_feature_group(col)
            self.group_indices[g_key].append(idx)
        assert sum(len(v) for v in self.group_indices.values()) == len(self.feat_names)

        self.numeric_ranges = ref_stats.get("numeric_ranges", {})
        self.categorical_levels = ref_stats.get("categorical_levels", {})
        self.d1_missing_shift = ref_stats.get("d1_missing_shift", {})

    @classmethod
    def from_artifacts(
        cls,
        model_path="results/day2_model_a_train_only.json",
        calibrator_path="results/day2_calibrator_full_val.joblib",
        ref_stats_path="results/day3_training_reference_stats.json",
    ):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        calibrator = joblib.load(calibrator_path)
        with open(ref_stats_path) as f:
            ref_stats = json.load(f)
        return cls(model=model, calibrator=calibrator, ref_stats=ref_stats)

    def _prepare_single_row_df(self, tx):
        """Normalize a 1-row DataFrame, Series, or dict to match Model A's exact feature schema and dtypes."""
        if isinstance(tx, pd.DataFrame):
            if len(tx) != 1:
                raise ValueError(f"Expected exactly 1 transaction row, got {len(tx)}")
            row_df = tx.copy().reset_index(drop=True)
        elif isinstance(tx, pd.Series):
            row_df = tx.to_frame().T.reset_index(drop=True)
        elif isinstance(tx, dict):
            row_df = pd.DataFrame([tx])
        else:
            raise TypeError(f"Unsupported transaction type: {type(tx)}")

        cols = {}
        for col in self.feat_names:
            val = row_df[col].iloc[0] if col in row_df.columns else np.nan
            if col in self.categorical_levels:
                cats = pd.Index(self.categorical_levels[col])
                cols[col] = pd.Categorical([val], categories=cats)
            else:
                cols[col] = pd.Series([val], dtype="float32")
        return pd.DataFrame(cols)[self.feat_names]

    def explain_transaction(
        self,
        tx,
        transaction_id=None,
        raw_unclamped_cats=None,
        shap_row=None,
        extra_context=None,
        additivity_tol=1e-4,
    ):
        """Produce a grounded, group-additive explanation JSON for one transaction."""
        row_df = self._prepare_single_row_df(tx)

        # Model predictions
        p_raw = float(self.model.predict_proba(row_df)[:, 1][0])
        raw_log_odds = float(self.model.predict(row_df, output_margin=True)[0])
        p_cal = float(self.calibrator.predict([p_raw])[0])

        amt = float(row_df["TransactionAmt"].iloc[0]) if pd.notna(row_df["TransactionAmt"].iloc[0]) else 0.0
        exp_loss = float(p_cal * amt)
        cal_display = "above 99%" if p_cal >= 0.99 else f"{p_cal * 100:.2f}%"

        # TreeSHAP attribution (live single-row or verified precomputed vector)
        if shap_row is None:
            sv = np.asarray(self.tree_explainer.shap_values(row_df)[0], dtype=np.float64)
            base_val = float(np.asarray(self.tree_explainer.expected_value).ravel()[0])
            self.base_value = base_val
        else:
            sv = np.asarray(shap_row, dtype=np.float64)
            assert len(sv) == len(self.feat_names)
            if self.base_value is None:
                _ = self.tree_explainer.shap_values(row_df)
                self.base_value = float(np.asarray(self.tree_explainer.expected_value).ravel()[0])
            base_val = self.base_value

        # Top 5 individual SHAP contributions (sorted by |SHAP|)
        sorted_idx = np.argsort(-np.abs(sv))
        top5_list = []
        for rank_i, f_idx in enumerate(sorted_idx[:5], start=1):
            fname = self.feat_names[f_idx]
            is_masked, display_name = describe_feature(fname)
            val = row_df.iloc[0, f_idx]
            shap_v = float(sv[f_idx])
            top5_list.append({
                "rank": rank_i,
                "display_name": display_name,
                "feature_code": fname,
                "is_masked": is_masked,
                "masked_category": describe_masked_category(fname) if is_masked else None,
                "value": _format_val(val),
                "direction": "increases_fraud_score" if shap_v >= 0 else "decreases_fraud_score",
                "shap_contribution_log_odds": round(shap_v, 5),
            })

        # Grouped SHAP contributions + additivity verification
        grouped_list = []
        total_group_shap = 0.0
        for g_key, g_desc in GROUP_DEFINITIONS:
            idxs = self.group_indices[g_key]
            g_sum = float(np.sum(sv[idxs]))
            total_group_shap += g_sum
            grouped_list.append({
                "group": g_key,
                "description": g_desc,
                "feature_count": len(idxs),
                "total_shap_log_odds": round(g_sum, 5),
                "direction": "increases_fraud_score" if g_sum >= 0 else "decreases_fraud_score",
            })
        grouped_list_sorted = sorted(grouped_list, key=lambda x: abs(x["total_shap_log_odds"]), reverse=True)

        reconstructed_log_odds = base_val + total_group_shap
        additivity_err = abs(reconstructed_log_odds - raw_log_odds)
        if additivity_err > additivity_tol:
            raise AssertionError(
                f"Grouped SHAP additivity check failed: err={additivity_err:.6e} > tol={additivity_tol}"
            )

        # Data cautions (D1_norm below training min, D1 missing, and any top feature outside training range/vocab)
        cautions = []
        d1_val = row_df["D1"].iloc[0]
        d1_norm_val = row_df["D1_norm"].iloc[0]
        d1_norm_stats = self.numeric_ranges.get("D1_norm", {})
        tr_d1n_min = d1_norm_stats.get("min")
        tr_d1n_max = d1_norm_stats.get("max")

        if pd.isna(d1_val):
            cautions.append({
                "code": "D1_MISSING",
                "severity": "warning",
                "feature_code": "D1",
                "display_name": "an anonymised feature",
                "observed_value": "MISSING",
                "message": (
                    "Anonymised time-delta feature D1 is missing. Missingness and fraud prevalence shifted sharply "
                    f"between training ({self.d1_missing_shift.get('train_missing_share', 0.00066)*100:.2f}% rows, "
                    f"{self.d1_missing_shift.get('train_missing_fraud_rate', 0.00366)*100:.2f}% fraud) "
                    f"and validation ({self.d1_missing_shift.get('val_missing_share', 0.01078)*100:.2f}% rows, "
                    f"{self.d1_missing_shift.get('val_missing_fraud_rate', 0.04084)*100:.2f}% fraud)."
                ),
            })

        d1n_rank = int(np.where(sorted_idx == self.feat_names.index("D1_norm"))[0][0]) + 1
        if d1n_rank <= 5 and pd.notna(d1_norm_val) and tr_d1n_min is not None and float(d1_norm_val) < float(tr_d1n_min):
            d1_zero_note = (
                " Because D1 == 0 on this row, D1_norm equals minus the transaction epoch day and cannot stay inside historical bounds."
                if (pd.notna(d1_val) and float(d1_val) == 0.0)
                else ""
            )
            cautions.append({
                "code": "D1_NORM_BELOW_TRAIN_MIN",
                "severity": "warning",
                "shap_rank": d1n_rank,
                "feature_code": "D1_norm",
                "display_name": "an anonymised feature",
                "masked_category": "a time-delta feature",
                "observed_value": round(float(d1_norm_val), 4),
                "train_range": [tr_d1n_min, tr_d1n_max],
                "message": (
                    f"Top-{d1n_rank} driver D1_norm (a time-delta feature, value {float(d1_norm_val):.2f}) sits below the training minimum "
                    f"({tr_d1n_min:.2f}). Tree splits route all below-range rows down the leftmost branch regardless of magnitude."
                    + d1_zero_note
                ),
            })
        elif d1n_rank <= 5 and pd.notna(d1_norm_val) and tr_d1n_max is not None and float(d1_norm_val) > float(tr_d1n_max):
            cautions.append({
                "code": "D1_NORM_ABOVE_TRAIN_MAX",
                "severity": "info",
                "shap_rank": d1n_rank,
                "feature_code": "D1_norm",
                "display_name": "an anonymised feature",
                "masked_category": "a time-delta feature",
                "observed_value": round(float(d1_norm_val), 4),
                "train_range": [tr_d1n_min, tr_d1n_max],
                "message": f"Top-{d1n_rank} driver D1_norm (a time-delta feature, value {float(d1_norm_val):.2f}) exceeds training maximum ({tr_d1n_max:.2f}).",
            })

        # Check top-10 SHAP drivers for out-of-training-range numerics or unseen categories
        seen_caution_feats = {"D1", "D1_norm"}
        for rank_i, f_idx in enumerate(sorted_idx[:10], start=1):
            fname = self.feat_names[f_idx]
            if fname in seen_caution_feats:
                continue
            is_masked, display_name = describe_feature(fname)
            val = row_df.iloc[0, f_idx]

            if fname in self.numeric_ranges and pd.notna(val):
                fval = float(val)
                nr = self.numeric_ranges[fname]
                tmin, tmax = nr.get("min"), nr.get("max")
                if tmin is not None and (fval < tmin or fval > tmax):
                    seen_caution_feats.add(fname)
                    bound_dir = "below training minimum" if fval < tmin else "above training maximum"
                    cautions.append({
                        "code": "TOP_FEATURE_OUT_OF_TRAIN_RANGE",
                        "severity": "warning" if rank_i <= 5 else "info",
                        "shap_rank": rank_i,
                        "feature_code": fname,
                        "display_name": display_name,
                        "observed_value": round(fval, 4),
                        "train_range": [tmin, tmax],
                        "message": (
                            f"Top-{rank_i} driver ({display_name}) value {fval:.4g} is {bound_dir} "
                            f"(training range: [{tmin}, {tmax}])."
                        ),
                    })
            elif fname in self.categorical_levels and raw_unclamped_cats and fname in raw_unclamped_cats:
                raw_cat = raw_unclamped_cats[fname]
                if raw_cat is not None and not pd.isna(raw_cat) and str(raw_cat) not in set(self.categorical_levels[fname]):
                    seen_caution_feats.add(fname)
                    cautions.append({
                        "code": "UNSEEN_CATEGORICAL_VALUE",
                        "severity": "info",
                        "shap_rank": rank_i,
                        "feature_code": fname,
                        "display_name": display_name,
                        "observed_value": str(raw_cat),
                        "message": (
                            f"Top-{rank_i} driver ({display_name}) had category '{raw_cat}' unseen in training; "
                            "encoded as MISSING per pipeline leakage rules."
                        ),
                    })

        # Hour-of-day dataset clock formatting
        hr_val = int(row_df["hour_of_day"].iloc[0]) if pd.notna(row_df["hour_of_day"].iloc[0]) else None
        dow_val = int(row_df["day_of_week"].iloc[0]) if pd.notna(row_df["day_of_week"].iloc[0]) else None

        output = {
            "transaction_id": int(transaction_id) if transaction_id is not None else None,
            "transaction_summary": {
                "amount_usd": round(amt, 2),
                "raw_prob": round(p_raw, 5),
                "raw_log_odds": round(raw_log_odds, 5),
                "calibrated_prob": round(p_cal, 5),
                "calibrated_prob_display": cal_display,
                "calibration_note": (
                    "Isotonic calibration hits >= 0.99 in top bin; displayed as 'above 99%' rather than certainty."
                    if p_cal >= 0.99
                    else None
                ),
                "expected_loss_usd": round(exp_loss, 2),
                "amount_aware_review_flags": {
                    "at_5_usd_review_cost": exp_loss >= 5.0,
                    "at_10_usd_review_cost": exp_loss >= 10.0,
                    "at_25_usd_review_cost": exp_loss >= 25.0,
                },
                "hour_of_day": {
                    "value": hr_val,
                    "day_of_week": dow_val,
                    "clock_label": "Dataset's own clock (TransactionDT elapsed seconds mod 86400 / 3600; not verified local wall-clock time)",
                },
            },
            "top_5_individual_contributions": top5_list,
            "grouped_contributions": grouped_list_sorted,
            "groups_additivity_check": {
                "base_value_log_odds": round(base_val, 6),
                "sum_of_group_shap": round(total_group_shap, 6),
                "reconstructed_log_odds": round(reconstructed_log_odds, 6),
                "model_raw_log_odds": round(raw_log_odds, 6),
                "abs_error": float(f"{additivity_err:.3e}"),
                "tolerance": additivity_tol,
                "passed": bool(additivity_err <= additivity_tol),
            },
            "data_cautions": cautions,
            "disclaimer": DISCLAIMER_TEXT,
        }
        if extra_context:
            output["fixture_metadata"] = extra_context
        return output
