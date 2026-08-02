import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import DataStore
from safety import analyze as safety_analyze

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
store = DataStore(DATASET_DIR)
samples = list(csv.DictReader(open(os.path.join(DATASET_DIR, "sample_messages.csv"), newline="", encoding="utf-8")))

for mid in ["sample_msg_052", "sample_msg_053"]:
    gt = next(r for r in samples if r["message_id"] == mid)
    print("=" * 60)
    print(f"=== {mid} ===")
    ct = gt["conversation_type"]
    bid = gt["business_id"]
    sid = gt["sender_user_id"]
    txt = gt["message_text"]
    mt = gt["media_type"]
    fc = gt["forwarded_count"]
    print(f"  conversation_type: {ct}")
    print(f"  business_id: {bid}")
    print(f"  sender_user_id: {sid}")
    print(f"  message_text: {txt}")
    print(f"  media_type: {mt}")
    print(f"  forwarded_count: {fc}")
    print(f"  GT action: {gt['action']}")
    print(f"  GT message_type: {gt['message_type']}")

    business = store.business_info(bid) if bid else {}
    membership = store.group_membership(gt.get("group_id", ""), gt["user_id"]) if gt.get("group_id") else {}
    sender_hist = store.sender_history_to_user(gt["user_id"], sid)
    sender_first_contact = ct == "personal" and len(sender_hist) == 0

    signals = safety_analyze(gt, "", business, membership, sender_first_contact)

    print()
    print("  --- Risk signals ---")
    for k in ["has_url", "asks_otp_or_pin", "urgency_language", "asks_payment",
              "highly_forwarded", "forwarded_count", "business_domain_mismatch",
              "business_unverified", "business_high_reports", "business_uses_new_domain",
              "sender_first_contact", "group_muted_by_user"]:
        print(f"  {k:30s} {signals[k]}")

    print()
    print(f"  risk_score: {signals['risk_score']}")
    print(f"  hard_mute_scam: {signals['hard_mute_scam']}")
    print()
    print("  --- Rule breakdown ---")
    o, u, p, m, h, r = signals["asks_otp_or_pin"], signals["has_url"], signals["asks_payment"], signals["business_domain_mismatch"], signals["business_unverified"], signals["urgency_language"]
    nd = signals["business_uses_new_domain"]
    hr = signals["business_high_reports"]
    hf = signals["highly_forwarded"]
    print(f"  OTP+URL+unverified: {o} and ({u} or {p}) and {h} => {o and (u or p) and h} (+3)")
    print(f"  OTP+URL+mismatch:   {o} and ({u} or {p}) and {m} => {o and (u or p) and m} (+3)")
    print(f"  biz_mismatch:       {m} => {m} (+2)")
    print(f"  urgency+pay+url/unv:{r} and {p} and ({u} or {h}) => {r and p and (u or h)} (+2)")
    print(f"  new_domain+payment: {nd} and {p} => {nd and p} (+2)")
    print(f"  high_reports:       {hr} => {hr} (+1)")
    print(f"  forwarded+url/urg:  {hf} and ({u} or {r}) => {hf and (u or r)} (+1)")
    print()
    print(f"  Total risk_score: {signals['risk_score']} (threshold >= 4)")
    print(f"  hard_mute_scam: {signals['hard_mute_scam']}")
    print()
