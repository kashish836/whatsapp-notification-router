"""
safety.py
---------
Deterministic, explainable risk signals computed from structured metadata
(not vibes). These feed into the LLM prompt as hard facts, AND serve as a
rule-based fallback/override so a network failure or a model hallucination
can never accidentally `notify` a user about an obvious scam.

Design principle from the problem statement:
  "clear scam or safety risk should be muted regardless of the user's
   usual engagement" -> these checks are user-preference-independent.
"""

from __future__ import annotations
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+|[a-z0-9-]+\.(?:in|com|net|xyz|link|info)\b", re.I)
OTP_RE = re.compile(r"\botp\b|\bpin\b|\bcvv\b|verification code|login code|\bdigit code\b|\b\d+\s*digit\b", re.I)
URGENCY_RE = re.compile(
    r"act now|immediately|expire|suspend|blocked|last chance|limited time|"
    r"reactivate|confirm your|update your kyc|failed.{0,15}deliver|reattempt fee|"
    r"click here|claim now|winner|lottery|prize", re.I)
PAYMENT_ASK_RE = re.compile(r"pay\b|payment|fee|refund|transfer|deposit|wallet", re.I)


def domain_mismatch(business: dict) -> bool:
    official = (business.get("official_domain") or "").strip().lower()
    used = (business.get("domain_used_by_sender") or "").strip().lower()
    if not official or not used:
        return False
    return official != used


def analyze(message_row: dict, media_text: str, business: dict, group_membership: dict,
            sender_first_contact: bool) -> dict:
    """Returns a dict of boolean/scalar risk signals plus a hard_mute verdict."""
    text = (message_row.get("message_text") or "") + " " + (media_text or "")
    forwarded_count = int(message_row.get("forwarded_count") or 0)

    has_url = bool(URL_RE.search(text))
    asks_otp = bool(OTP_RE.search(text))
    urgency_language = bool(URGENCY_RE.search(text))
    asks_payment = bool(PAYMENT_ASK_RE.search(text))
    highly_forwarded = forwarded_count >= 5

    biz_mismatch = domain_mismatch(business) if business else False
    biz_unverified = bool(business) and str(business.get("verified", "0")) not in ("1", "1.0", "True", "true")
    biz_high_reports = bool(business) and float(business.get("user_reports_30d") or 0) >= 5
    biz_new_domain = bool(business) and float(business.get("domain_used_by_sender_age_days") or 9999) < 30

    signals = {
        "has_url": has_url,
        "asks_otp_or_pin": asks_otp,
        "urgency_language": urgency_language,
        "asks_payment": asks_payment,
        "highly_forwarded": highly_forwarded,
        "forwarded_count": forwarded_count,
        "business_domain_mismatch": biz_mismatch,
        "business_unverified": biz_unverified,
        "business_high_reports": biz_high_reports,
        "business_uses_new_domain": biz_new_domain,
        "sender_first_contact": sender_first_contact,
        "group_muted_by_user": str(group_membership.get("group_muted_by_user", "0")) in ("1", "1.0", "True", "true"),
    }

    # Hard-mute triggers: high-confidence, user-preference-independent scam patterns.
    # Kept conservative (multiple corroborating signals) to avoid false positives
    # on legitimate transactional/business messages.
    risk_score = 0
    if asks_otp and (has_url or asks_payment) and biz_unverified:
        risk_score += 3
    if asks_otp and (has_url or asks_payment) and biz_mismatch:
        risk_score += 3
    if asks_otp and (urgency_language or asks_payment) and not (bool(business) and not biz_unverified and not biz_mismatch):
        risk_score += 3
    if biz_mismatch:
        risk_score += 2
    if urgency_language and asks_payment and (has_url or biz_unverified):
        risk_score += 2
    if biz_new_domain and asks_payment:
        risk_score += 2
    if biz_high_reports:
        risk_score += 1
    if highly_forwarded and (has_url or urgency_language):
        risk_score += 1

    signals["risk_score"] = risk_score
    signals["hard_mute_scam"] = risk_score >= 4 or (asks_otp and (urgency_language or asks_payment) and not (bool(business) and not biz_unverified and not biz_mismatch))
    return signals
