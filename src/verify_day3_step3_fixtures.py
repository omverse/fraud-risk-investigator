import json
from src.explain_transaction import FraudTransactionExplainer, DISCLAIMER_TEXT


def main():
    with open("results/day3_step3_fixtures_15.json") as f:
        fixtures = json.load(f)
    with open("results/day3_step3_explanations_15.json") as f:
        saved_explanations = json.load(f)

    assert len(fixtures) == 15
    assert len(saved_explanations) == 15

    explainer = FraudTransactionExplainer.from_artifacts()

    for fix, saved in zip(fixtures, saved_explanations):
        tx_id = fix["transaction_id"]
        # Re-run single-transaction explainer cold from fixture feature dict alone
        live = explainer.explain_transaction(fix["features"], transaction_id=tx_id)

        # 1. Exact prediction match & calibrated probability cap rule
        s_sum = saved["transaction_summary"]
        l_sum = live["transaction_summary"]
        assert abs(s_sum["raw_prob"] - l_sum["raw_prob"]) < 1e-6
        assert "100" not in l_sum["calibrated_prob_display"]
        if l_sum["calibrated_prob"] >= 0.99:
            assert l_sum["calibrated_prob_display"] == "above 99%"

        # 2. Expected loss matches cal_prob * amount
        exp_calc = round(l_sum["calibrated_prob"] * l_sum["amount_usd"], 2)
        assert abs(exp_calc - l_sum["expected_loss_usd"]) <= 0.02

        # 3. Hour of day explicitly labelled as data's own clock
        assert "Dataset's own clock" in l_sum["hour_of_day"]["clock_label"]

        # 4. Masked columns strictly named 'an anonymised feature'
        l_sum_top5 = live["top_5_individual_contributions"]
        for item in l_sum_top5:
            if item["is_masked"]:
                assert item["display_name"] == "an anonymised feature"
        assert len(l_sum_top5) == 5

        # 5. Groups additivity check within 1e-4 tolerance
        chk = live["groups_additivity_check"]
        assert chk["passed"] is True
        assert chk["abs_error"] <= 1e-4
        assert len(live["grouped_contributions"]) == 9

        # 6. Disclaimer exact match
        assert live["disclaimer"] == DISCLAIMER_TEXT

        # 7. Individual per-alert file matches
        with open(f"results/day3_explanations/tx_{tx_id}.json") as f_ind:
            indiv = json.load(f_ind)
        assert indiv["transaction_id"] == tx_id
        assert indiv["groups_additivity_check"]["passed"] is True

    # 8. Edge-case check: synthetic missing D1 triggers D1_MISSING caution + additivity holds
    mod_missing_d1 = dict(fixtures[0]["features"])
    mod_missing_d1["D1"] = None
    mod_missing_d1["D1_norm"] = None
    res_d1m = explainer.explain_transaction(mod_missing_d1, transaction_id=-1)
    assert any(c["code"] == "D1_MISSING" for c in res_d1m["data_cautions"])
    assert res_d1m["groups_additivity_check"]["passed"] is True

    # 9. Edge-case check: out-of-range top feature triggers TOP_FEATURE_OUT_OF_TRAIN_RANGE
    mod_oor_amt = dict(fixtures[3]["features"])
    mod_oor_amt["TransactionAmt"] = 999999.0
    res_oor = explainer.explain_transaction(mod_oor_amt, transaction_id=-2)
    assert any(c["code"] == "TOP_FEATURE_OUT_OF_TRAIN_RANGE" for c in res_oor["data_cautions"])
    assert res_oor["groups_additivity_check"]["passed"] is True

    print("[PASS] All 15 Step 3 fixtures + 2 synthetic edge-case caution checks succeeded.")


if __name__ == "__main__":
    main()
