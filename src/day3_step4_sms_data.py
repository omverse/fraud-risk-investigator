"""Day 3 Step 4: Prepare UCI SMS Spam Collection + Generic Modern Scam/Normal Test Set.

Responsibilities:
1. Load `ucirvine/sms_spam` (UCI SMS Spam Collection v.1, early-2010s English SMS).
2. Inspect and remove duplicate messages BEFORE splitting so identical texts cannot
   appear in multiple splits (also check for any label conflicts across duplicates).
3. Save a deterministic stratified 80/10/10 train/val/test split (seed=42) to data/sms_spam/.
4. Create a standalone 80-message generic modern test set (40 scam-style, 40 normal)
   with no real phone numbers, account numbers, or live links — held out strictly
   for single-use end-of-track evaluation in Step 7 (never used for training or tuning).
5. Record the locked experimental rules and summary statistics for Steps 5-7.
"""

import json
import os
import re
from sklearn.model_selection import train_test_split
import pandas as pd


SPLIT_SEED = 42

LOCKED_EXPERIMENT_RULES = {
    "dataset_hf_id": "ucirvine/sms_spam",
    "dataset_name": "UCI SMS Spam Collection v.1",
    "era_and_language": "Compiled in the early 2010s (Almeida et al., ACM DocEng 2011); English SMS messages only.",
    "hf_hub_license_field": "unknown",
    "uci_repository_license": "CC BY 4.0 (UCI Machine Learning Repository, Dataset ID 228)",
    "deduplication_policy": "Drop whitespace-collapsed lowercase duplicates (keeping first stripped text) before splitting; assert zero exact or normalized text overlap across train/val/test.",
    "split_ratios": {"train": 0.80, "val": 0.10, "test": 0.10},
    "split_seed": SPLIT_SEED,
    "metrics": ["pr_auc_spam", "f1_spam", "precision_spam", "recall_spam"],
    "threshold_selection_rule": "Maximize validation F1 on the spam class using the validation split only.",
    "models": [
        "TF-IDF + Logistic Regression (Step 5 baseline)",
        "Hugging Face Zero-Shot NLI Classifier (Step 5 baseline)",
        "Fine-tuned DistilBERT (Step 6: 3 seeds, 2-3 epochs, report mean +/- sample std with ddof=1)",
    ],
    "test_hygiene_rule": "No hyperparameter tuning or threshold adjustment after viewing UCI test set or modern test set numbers.",
    "modern_test_set_rule": "Generic modern scam/normal messages (no real numbers or links), evaluated once at the end, never used for training.",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def build_modern_test_messages():
    """Return 80 generic modern messages (40 scam-style, 40 normal) with zero real PII/links."""
    scam_messages = [
        "[Bank Alert] Unusual sign-in detected on your checking account. Verify your identity within 2 hours at [LINK] or access will be suspended.",
        "USPS Delivery Notice: Your parcel cannot be delivered due to an incomplete street address. Confirm details and pay $1.99 redelivery at [LINK].",
        "ExpressToll Final Notice: You have an unpaid toll balance of $6.40. Settle today at [LINK] to avoid a $50 late penalty and registration hold.",
        "Security Alert: A wire transfer of $2,450 was queued from your savings profile. If you did not authorize this, cancel via [LINK] immediately.",
        "CoinExchange Notice: A login from a new device in an unrecognized region requested a withdrawal. Lock your wallet now at [LINK].",
        "Hi! I work with a digital commerce recruiting team. Earn $150-$320/day rating app listings flexibly from home. Reply YES to get onboarded.",
        "Your streaming membership payment failed today. Update your billing card at [LINK] within 24 hours so your account is not terminated.",
        "IRS Tax Refund Update: An unclaimed credit of $842.00 is ready for direct deposit. Submit your routing info at [LINK].",
        "Apple Pay Notice: Your digital wallet has been temporarily restricted for suspicious activity. Re-activate at [LINK].",
        "Hey Mom, it's me! I dropped my phone in water and I'm texting from a friend's number. Can you send $280 via instant pay right now? I'll pay you back tonight.",
        "Billing Support: Your annual cloud security license ($399.00) auto-renewed today. Call our cancellation desk at [PHONE] within 12 hours for a full refund.",
        "Courier Alert: Customs clearance is paused on shipment #0000 due to missing duty tax of $3.50. Complete payment at [LINK].",
        "Your mobile carrier upgraded your loyalty points! You have 14,200 points expiring tonight—redeem for a free tablet at [LINK].",
        "Fraud Dept Notice: Reply with the 6-digit verification code we just sent to block an unauthorized card charge at Metro Electronics.",
        "PayService: You received an incoming payment of $650.00 pending verification. Accept funds at [LINK] before the hold expires.",
        "DMV Reminder: Our records show an unresolved traffic citation. Pay the $35 administrative fee at [LINK] by Friday to prevent license suspension.",
        "URGENT HR Request: Hi, this is the CEO in a client meeting and can't talk. Can you pick up five $100 digital gift cards and send the codes ASAP?",
        "Your debit card ending in 0000 was flagged for three overseas attempts. Confirm whether these were yours at [LINK] right away.",
        "Crypto Airdrop Confirmed: Connect your wallet at [LINK] to claim your verified 1,500 token allocation before midnight.",
        "Pharmacy Notice: Your prescription order needs urgent card re-verification. Enter payment info at [LINK] so shipping isn't delayed.",
        "Electric Utility Alert: Scheduled disconnection notice for tomorrow at 9 AM due to a past-due invoice. Pay immediately at [LINK].",
        "Someone uploaded your personal email address to a leaked data list. Click [LINK] to run a complimentary identity wipe now.",
        "Quick favor—our accounting portal is down and the vendor needs a $1,200 urgent wire today. Can you push it through and confirm?",
        "Notice from Social Benefits Office: Your monthly stipend was put on hold for identity reconfirmation. Complete form at [LINK].",
        "Parcel locker full: Reschedule doorstep drop-off for package [ID] for free only today at [LINK].",
        "Congratulations! Your mobile number was picked in this week's supermarket customer appreciation giveaway for a $500 voucher at [LINK].",
        "WebMail Admin: Your mailbox storage is 99% full and incoming messages will bounce. Upgrade storage credentials at [LINK].",
        "Hi, sorry to bother you—did I send money to your account by mistake? Please return the $200 via instant transfer before the app locks it.",
        "Airline Rebooking Desk: Your flight was cancelled. Reclaim your $420 voucher or rebook seat selection immediately at [LINK].",
        "Loan Pre-Approval: You qualify for up to $15,000 same-day cash regardless of credit score! Apply with zero paperwork at [LINK].",
        "Bank Fraud Specialist calling soon: Please keep the 2FA security code handy and read it to the agent to verify you own the device.",
        "Your package from Global Express is held at local customs depot. Confirm recipient DOB and zip code at [LINK] to release.",
        "Security Warning: Unusual API key access detected on your payment merchant dashboard. Reset root password via [LINK].",
        "Guaranteed daily returns of 4% trading AI-managed forex pools. Limited slots this week—message [CONTACT] for VIP invite.",
        "Student Loan Relief Program: Forgiveness applications close tonight! Submit your SSN last-4 at [LINK] to lock zero interest.",
        "Your tax return was rejected due to a mismatch. Upload corrected banking details at [LINK] within 48 hours.",
        "Attention Cardholder: A $912 purchase at Luxury Watches Ltd is pending. If unauthorized, enter your full card PIN at [LINK] to cancel.",
        "Remote Data Entry Offer: Earn $45/hr setting up merchant profiles on your laptop, no experience needed. Register at [LINK].",
        "Medical Insurance Rebate: A $275 overpayment refund is pending for your household. Claim deposit via [LINK].",
        "Your cloud drive access expires in 3 hours due to failed card billing. Click [LINK] now to avoid permanent photo deletion.",
    ]

    normal_messages = [
        "Your login verification code is 482910. It expires in 10 minutes. Our team will never call or text asking you for this code.",
        "Order #A1042 from Downtown Roasters is out for delivery today between 2 PM and 4 PM. Track it directly inside the official store app.",
        "Card Alert: A charge of $24.50 at Corner Market on card ending in 0000 was approved. If you made this purchase, no action is needed.",
        "Running about 10 minutes late to the 3 PM sync because bridge traffic is backed up. Please go ahead and start without me.",
        "Reminder: Your dental cleaning with Dr. Patel is scheduled for tomorrow at 9:30 AM. Reply YES to confirm or call the office front desk.",
        "Gate update: Flight AA100 now boards at Gate B14 at 6:15 PM. Please check the airline app for the latest seat and boarding pass details.",
        "Hey, did the electric utility bill auto-pay from our joint account this morning, or should I submit it tonight?",
        "Customer note: I am traveling out of town this week and bought two train tickets using my debit card at the station kiosk.",
        "Your library hold 'Designing Data-Intensive Applications' is ready for pickup at the main branch desk until Friday.",
        "Receipt from City Pharmacy: Copay of $15.00 paid on card ending in 0000. Questions? Contact us through the patient portal.",
        "Thanks for joining today's sprint review! I pushed the updated API schema notes to the shared team wiki.",
        "Your monthly statement for savings account ending in 0000 is now ready to view when you sign in to online banking.",
        "Soccer practice is moved to Field 3 tonight because of wet turf on Field 1. Bring an extra water bottle.",
        "Customer explanation: This $480 hotel charge is for my work conference stay in Chicago from Tuesday to Thursday.",
        "Rideshare update: Your driver Sam has arrived in a silver sedan near the north pickup curb.",
        "Friendly reminder from your building manager: Water will be shut off briefly Thursday 10 AM to 11 AM for pipe maintenance.",
        "Two-step sign-in alert: A new browser signed in to your workspace account from macOS. If this was you, you can ignore this message.",
        "Hey, are we still meeting at 12:30 for lunch near the office, or do you want to push it to 1:00?",
        "Your recurring transfer of $250.00 to Emergency Savings completed successfully on schedule.",
        "Customer context: Sending $150 to my sister for our shared parents' anniversary dinner gift—she paid the restaurant deposit.",
        "Clinic update: Lab results from yesterday's blood panel have been posted to your secure health chart.",
        "Auto-pay confirmation: Your internet bill of $65.00 was paid on time today. Next scheduled payment is next month.",
        "Can you review the pull request when you have a free 15 minutes? I split the calibration script into two smaller helpers.",
        "Your grocery pickup order is ready in spot #4. Check in using the store app when you arrive in the parking lot.",
        "Hotel reservation confirmed: Check-in Sunday 3 PM, confirmation code stored in your travel itinerary folder.",
        "Cardholder note: I placed three small coffee and bakery orders back-to-back while buying snacks for the whole office team.",
        "Transit alert: Red Line trains are running 8 minutes behind schedule due to signal testing downtown.",
        "Your warranty registration for the new laptop stand was saved. Keep your receipt for your records.",
        "Happy birthday! Let us know when you're free this weekend so everyone can grab dinner together.",
        "Low balance courtesy alert: Checking account ending in 0000 dropped below your $200 reminder threshold.",
        "Customer note: Purchasing a replacement monitor online from a new electronics vendor after my desk screen cracked.",
        "Prescription ready: Your refill is ready for pickup at the Main Street counter during regular store hours.",
        "Team calendar update: Tomorrow's standup moved from 9:30 AM to 10:00 AM to accommodate the deployment window.",
        "Your direct deposit payroll credit posted this morning. Log in to your payroll portal to download your pay stub.",
        "Thanks for calling customer care earlier. Your travel notice for May 12 to May 19 has been added to your card profile.",
        "Parking session for zone 402 expires in 15 minutes. Extend your meter time inside the city parking app if needed.",
        "Order delayed slightly: One book from your shipment will arrive Thursday instead of Wednesday at no extra shipping cost.",
        "Customer note: Buying two concert tickets on resale for next month—using the same home Wi-Fi and device as usual.",
        "Your monthly mobile plan invoice is $45.00 and will auto-pay on the 20th. Manage settings anytime in the carrier app.",
        "Quick question—do you have the PDF from yesterday's design workshop handy so I can link it in the project doc?",
    ]

    assert len(scam_messages) == 40
    assert len(normal_messages) == 40

    rows = []
    for idx, msg in enumerate(normal_messages, start=1):
        rows.append({"id": f"modern_ham_{idx:02d}", "sms": msg, "label": 0, "label_name": "ham"})
    for idx, msg in enumerate(scam_messages, start=1):
        rows.append({"id": f"modern_spam_{idx:02d}", "sms": msg, "label": 1, "label_name": "spam"})

    # Sanity check: ensure no duplicate texts within modern set
    texts = [r["sms"].strip().lower() for r in rows]
    assert len(set(texts)) == len(rows), "Duplicate detected in modern test messages!"
    return rows


def main():
    from datasets import load_dataset

    print("=" * 72)
    print("Day 3 Step 4: UCI SMS Spam Collection Prep & Modern Test Set")
    print("=" * 72)

    ds = load_dataset("ucirvine/sms_spam", split="train")
    raw_df = ds.to_pandas()
    assert list(raw_df.columns) == ["sms", "label"]

    raw_n = len(raw_df)
    raw_spam = int((raw_df["label"] == 1).sum())
    raw_ham = int((raw_df["label"] == 0).sum())
    raw_spam_rate = raw_spam / raw_n

    # Inspect exact, stripped, and whitespace/case-normalized duplicate counts + label conflicts
    raw_df["sms_stripped"] = raw_df["sms"].astype(str).str.strip()
    raw_df["sms_norm"] = raw_df["sms_stripped"].map(_normalize_whitespace).str.lower()

    exact_dups = int(raw_df["sms"].duplicated(keep="first").sum())
    stripped_dups = int(raw_df["sms_stripped"].duplicated(keep="first").sum())
    norm_dups = int(raw_df["sms_norm"].duplicated(keep="first").sum())

    # Check if any duplicate text carries conflicting ham/spam labels
    exact_label_nunique = raw_df.groupby("sms")["label"].nunique()
    stripped_label_nunique = raw_df.groupby("sms_stripped")["label"].nunique()
    norm_label_nunique = raw_df.groupby("sms_norm")["label"].nunique()

    exact_conflicts = int((exact_label_nunique > 1).sum())
    stripped_conflicts = int((stripped_label_nunique > 1).sum())
    norm_conflicts = int((norm_label_nunique > 1).sum())

    # Deduplicate BEFORE splitting on whitespace/case-normalized text (preserving first raw stripped message)
    dedup_df = (
        raw_df.drop_duplicates(subset=["sms_norm"], keep="first")
        .copy()
        .reset_index(drop=True)
    )
    # Keep readable stripped text of the first occurrence while asserting zero normalized overlap across splits
    dedup_df["sms"] = dedup_df["sms_stripped"]
    dedup_df = dedup_df[["sms", "sms_norm", "label"]].copy()

    dedup_n = len(dedup_df)
    dedup_spam = int((dedup_df["label"] == 1).sum())
    dedup_ham = int((dedup_df["label"] == 0).sum())
    dedup_spam_rate = dedup_spam / dedup_n

    # Stratified 80 / 10 / 10 split with fixed seed=42
    train_df, temp_df = train_test_split(
        dedup_df,
        test_size=0.20,
        random_state=SPLIT_SEED,
        stratify=dedup_df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SPLIT_SEED,
        stratify=temp_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Zero-overlap checks on BOTH exact stripped text and normalized lowercase+whitespace text
    tr_s, va_s, te_s = set(train_df["sms"]), set(val_df["sms"]), set(test_df["sms"])
    assert len(tr_s & va_s) == 0 and len(tr_s & te_s) == 0 and len(va_s & te_s) == 0
    tr_n, va_n, te_n = set(train_df["sms_norm"]), set(val_df["sms_norm"]), set(test_df["sms_norm"])
    assert len(tr_n & va_n) == 0 and len(tr_n & te_n) == 0 and len(va_n & te_n) == 0

    train_out = train_df[["sms", "label"]].copy()
    val_out = val_df[["sms", "label"]].copy()
    test_out = test_df[["sms", "label"]].copy()

    os.makedirs("data/sms_spam", exist_ok=True)
    train_out.to_csv("data/sms_spam/train.csv", index=False)
    val_out.to_csv("data/sms_spam/val.csv", index=False)
    test_out.to_csv("data/sms_spam/test.csv", index=False)

    # Build and save generic modern held-out test set (never used for training or tuning)
    modern_rows = build_modern_test_messages()
    modern_df = pd.DataFrame(modern_rows)
    mod_texts_lower = set(modern_df["sms"].str.strip().str.lower())
    uci_texts_lower = set(dedup_df["sms"].str.strip().str.lower())
    assert len(mod_texts_lower & uci_texts_lower) == 0, "Modern set accidentally overlapped UCI!"

    with open("results/day3_sms_modern_test.json", "w") as f:
        json.dump(modern_rows, f, indent=2)

    def _split_stats(d):
        n_s = len(d)
        sp = int((d["label"] == 1).sum())
        hm = int((d["label"] == 0).sum())
        return {"rows": n_s, "ham": hm, "spam": sp, "spam_rate": round(sp / n_s, 5)}

    summary = {
        "locked_rules": LOCKED_EXPERIMENT_RULES,
        "raw_dataset": {
            "rows": raw_n,
            "ham": raw_ham,
            "spam": raw_spam,
            "spam_rate": round(raw_spam_rate, 5),
        },
        "duplicate_audit": {
            "exact_duplicates_removed": exact_dups,
            "stripped_duplicates_removed": stripped_dups,
            "normalized_case_ws_duplicates": norm_dups,
            "exact_label_conflicts": exact_conflicts,
            "stripped_label_conflicts": stripped_conflicts,
            "normalized_label_conflicts": norm_conflicts,
            "dedup_method_applied": "drop_duplicates(subset=['sms_norm'], keep='first') preserving original sms_stripped text",
        },
        "deduplicated_dataset": {
            "rows": dedup_n,
            "ham": dedup_ham,
            "spam": dedup_spam,
            "spam_rate": round(dedup_spam_rate, 5),
        },
        "splits": {
            "train_80": _split_stats(train_df),
            "val_10": _split_stats(val_df),
            "test_10": _split_stats(test_df),
        },
        "modern_test_set": {
            "path": "results/day3_sms_modern_test.json",
            "rows": len(modern_rows),
            "ham": int((modern_df["label"] == 0).sum()),
            "spam": int((modern_df["label"] == 1).sum()),
            "spam_rate": round(float((modern_df["label"] == 1).mean()), 5),
            "usage_constraint": "Held out strictly for single-use final evaluation at Step 7; never used for training or threshold selection.",
        },
    }

    with open("results/day3_step4_sms_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Raw dataset       : {raw_n} rows (ham={raw_ham}, spam={raw_spam}, spam_rate={raw_spam_rate*100:.2f}%)")
    print(f"Duplicates found  : exact={exact_dups}, stripped={stripped_dups}, case+ws_norm={norm_dups}")
    print(f"Label conflicts   : exact={exact_conflicts}, stripped={stripped_conflicts}, case+ws_norm={norm_conflicts}")
    print(f"After dedup       : {dedup_n} rows (ham={dedup_ham}, spam={dedup_spam}, spam_rate={dedup_spam_rate*100:.2f}%)")
    print(f"Splits (seed={SPLIT_SEED}): train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"  Train spam rate : {summary['splits']['train_80']['spam_rate']*100:.2f}% ({summary['splits']['train_80']['spam']}/{len(train_df)})")
    print(f"  Val spam rate   : {summary['splits']['val_10']['spam_rate']*100:.2f}% ({summary['splits']['val_10']['spam']}/{len(val_df)})")
    print(f"  Test spam rate  : {summary['splits']['test_10']['spam_rate']*100:.2f}% ({summary['splits']['test_10']['spam']}/{len(test_df)})")
    print(f"Modern held-out   : {len(modern_rows)} rows (40 ham, 40 spam) saved to results/day3_sms_modern_test.json")


if __name__ == "__main__":
    main()
