"""
router.py  (Groq edition)
--------------------------
The decision engine. For one message:

  1. Pulls user / group / business / relationship context (data_loader).
  2. Resolves media (media.py) into text if image/voice.
  3. Computes deterministic risk signals (safety.py).
  4. Retrieves candidate evidence messages (retrieval.py).
  5. Calls Groq with all of the above as structured context, forcing a
     structured decision via JSON mode — no fragile free-text parsing.
  6. Applies a safety override: if safety.py flags a hard-mute scam pattern,
     the final action is forced to `mute` / `scam` regardless of what the
     model said — guarantees "risk overrides personalization" even if the
     LLM call fails, is rate-limited, or disagrees.

Returns a dict matching the exact output.csv schema.
"""

from __future__ import annotations
import json
import time
from typing import Literal

from pydantic import BaseModel

from data_loader import DataStore
from retrieval import find_evidence
from safety import analyze as safety_analyze
from media import get_media_text

ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_TYPES = {"personal", "urgent", "event", "payment", "business_update",
                  "promotion", "greeting", "forward", "spam", "scam", "unknown"}


class RouteDecision(BaseModel):
    action: Literal["notify", "digest", "mute"]
    message_type: Literal["personal", "urgent", "event", "payment", "business_update",
                           "promotion", "greeting", "forward", "spam", "scam", "unknown"]
    reason: str
    confidence: float
    evidence_message_ids: list[str]


SYSTEM_PROMPT = """You are the decision engine of a WhatsApp Message Notification Router.

For every message, classify it into:
- action: "notify" (interrupt now), "digest" (safe but low priority, show later), "mute" (repetitive, unwanted, low-value, suspicious, scam-like, or unsafe)
- message_type: personal, urgent, event, payment, business_update, promotion, greeting, forward, spam, scam, unknown

Core principles:
1. PERSONALIZE to this specific user: their engagement history with this sender/group/business,
   their quiet hours, how often they open/reply/dismiss/report similar messages.
2. A muted group can still contain something the user should be notified about (e.g. a direct
   @mention, an admin emergency alert, a personal ask directed at them).
3. Repetition matters: if message_events show this user has ignored/dismissed near-identical
   messages before, lean toward digest or mute even if the content itself looks legitimate.
4. Risk overrides personalization: if `risk_signals` indicate a likely scam/phishing pattern
   (OTP/PIN requests, mismatched sender domain, urgency + payment ask, unverified sender with
   new domain, high report counts), route to mute/scam or mute/spam REGARDLESS of how engaged
   this user usually is with this business or sender.
5. Use the retrieved `evidence_candidates` as your evidence_message_ids ONLY when they genuinely
   support the reasoning (same sender repeating a pattern, or a clear precedent for how this user
   treats this type of message). Do not cite irrelevant history just to fill the field. Only use
   IDs that appear in evidence_candidates — never invent an id.
6. Confidence should reflect how much the evidence actually supports the call — vague context
   should get a mid confidence (0.5-0.7), strong direct precedent or an unambiguous scam pattern
   should get high confidence (0.85+).
7. Keep `reason` to one concrete, specific sentence — name the actual signal, not a generic label.

message_type disambiguation:
- "urgent" = time-sensitive personal or group message requiring immediate action (e.g. a direct request, emergency, deadline, someone needs help NOW). Think: would the user be annoyed if they saw this 2 hours late?
- "event" = scheduled or informational announcement about a future event (e.g. meeting, festival, ceremony, school circular, community gathering). Think: is this informing about something happening at a specific time/place?

Business message priority order (when a message could fit multiple types):
1. If it advertises a discount, sale, coupon, special offer, OR showcases/lists a specific product or item for sale (e.g. photos of items, "available now", pickup/delivery details for a purchase) — even if no discount is mentioned and even if it also mentions a date or time window (e.g. "pickup this weekend") — classify as "promotion". Selling or showcasing a product ALWAYS wins over a date/logistics mention.
2. Only if there is NO discount, offer, or product listing, and it announces a specific date/time-bound happening (event with no item for sale — meeting, festival, ceremony, RSVP, appointment) → "event"
3. Otherwise, if it is purely informational (order status, policy change, account/service notice) with no offer, no product listing, and no specific date/time-bound happening → "business_update"

- "personal" = a direct message from a known contact that doesn't fit other specific categories (casual chat, question, sharing something).
- "greeting" = generic hi/hello/festival wishes with no actionable content.
- "forward" = a message forwarded from elsewhere (check forwarded_count), not originally from the sender.

When media_extracted_text is empty (voice note with no transcription, image with no OCR):
- Do NOT default to "unknown" just because you lack the text. The message still has context.
- Primary signal: look at evidence_message_ids, especially messages from the same sender. What type of content does this sender typically send to this user? If evidence shows they send personal check-ins → personal. If they send event announcements → event. If they send promotions → promotion. Follow the sender's established pattern.
- Secondary signals: group context (admin announcements in active groups are often urgent), user engagement history (if they usually reply to this sender quickly, a new voice note is likely important), conversation type (a direct voice note from a known contact is more likely personal/urgent than a broadcast in a large group).
- Only use "unknown" when evidence is also sparse (few or no evidence candidates found) or contradictory (evidence points to multiple unrelated types with no clear dominant pattern).

Repetition and dismissal evidence for muting:
- When evidence_candidates show the user has previously dismissed, ignored, or muted messages from this sender or about this topic, that is strong evidence for "digest" or "mute" even if the current message looks benign on its surface.
- If 2+ evidence messages show the user dismissed similar content, prefer "mute" over "digest".
- If the evidence shows the user never opens messages from this sender/group, lean toward "mute".
- Cite the specific dismissed/ignored message_ids in evidence_message_ids to justify the decision.

Default-to-mute rules (override digest unless the user actively engages with this sender):
- Forwarded chain messages (forwarded_count >= 1 and message looks like a chain/forward): default to "mute"/"forward" unless evidence shows the user typically opens or replies to forwarded content from this sender.
- Generic greetings (Good morning, Happy Diwali, etc. with no personal/actionable content): default to "mute"/"greeting" unless the user has a history of replying to or engaging with messages from this sender.
- Repetitive promotional content from senders with no prior engagement: default to "mute"/"promotion" if the user has never opened or replied to messages from this business/sender, even if the current message looks slightly different from past ones.

Respond ONLY with the structured JSON decision, nothing else."""


def build_prompt_context(store: DataStore, row: dict, media_text: str,
                          evidence: list[dict], risk_signals: dict) -> dict:
    user = store.user_profile(row["user_id"])
    ctx = {
        "message": {
            "message_id": row["message_id"],
            "conversation_type": row["conversation_type"],
            "created_at": row["created_at"],
            "message_text": row.get("message_text", ""),
            "media_type": row.get("media_type", ""),
            "media_extracted_text": media_text,
            "forwarded_count": row.get("forwarded_count", "0"),
        },
        "user_profile": {
            "do_not_disturb_window": user.get("do_not_disturb_window", ""),
            "messages_opened_30d": user.get("messages_opened_30d", ""),
            "messages_replied_30d": user.get("messages_replied_30d", ""),
            "notifications_dismissed_30d": user.get("notifications_dismissed_30d", ""),
            "messages_reported_30d": user.get("messages_reported_30d", ""),
        },
        "risk_signals": risk_signals,
        "evidence_candidates": evidence,
    }

    if row["conversation_type"] == "group" and row.get("group_id"):
        group = store.group_info(row["group_id"])
        membership = store.group_membership(row["group_id"], row["user_id"])
        sender_membership = store.group_membership(row["group_id"], row.get("sender_user_id", ""))
        ctx["group_context"] = {
            "group_name": group.get("group_name", ""),
            "group_type": group.get("group_type", ""),
            "member_count": group.get("member_count", ""),
            "admin_count": group.get("admin_count", ""),
            "messages_30d": group.get("messages_30d", ""),
            "user_role_in_group": membership.get("role", ""),
            "user_group_muted": membership.get("group_muted_by_user", ""),
            "user_dismissed_30d_in_group": membership.get("notifications_dismissed_30d", ""),
            "sender_role_in_group": sender_membership.get("role", ""),
            "is_direct_mention": (f"@{row['user_id']}" in (row.get("message_text") or "")),
        }

    if row["conversation_type"] == "business" and row.get("business_id"):
        biz = store.business_info(row["business_id"])
        rel = store.user_business_relationship(row["user_id"], row["business_id"])
        ctx["business_context"] = {
            "display_name": biz.get("display_name", ""),
            "category": biz.get("category", ""),
            "verified": biz.get("verified", ""),
            "official_domain": biz.get("official_domain", ""),
            "domain_used_by_sender": biz.get("domain_used_by_sender", ""),
            "account_age_days": biz.get("account_age_days", ""),
            "user_reports_30d": biz.get("user_reports_30d", ""),
            "user_relationship": rel.get("why_user_knows_account", "no prior relationship"),
            "allows_promotions": rel.get("allows_promotions", ""),
            "promotions_opted_out_at": rel.get("promotions_opted_out_at", ""),
            "activity_count_180d": rel.get("activity_count_180d", ""),
            "user_messages_opened_30d_this_biz": rel.get("messages_opened_30d", ""),
            "user_messages_dismissed_30d_this_biz": rel.get("messages_dismissed_30d", ""),
        }

    if row["conversation_type"] == "personal":
        sender_hist = store.sender_history_to_user(row["user_id"], row.get("sender_user_id", ""))
        ctx["personal_context"] = {
            "sender_user_id": row.get("sender_user_id", ""),
            "prior_messages_from_sender": len(sender_hist),
        }

    return ctx


def route_message(client, model: str, store: DataStore, row: dict) -> dict:
    media_path = store.media_path(row.get("media_type", ""), row.get("media_id", ""))
    media_text = get_media_text(client, model, row.get("media_type", ""), media_path)

    business = store.business_info(row.get("business_id", "")) if row.get("business_id") else {}
    membership = store.group_membership(row.get("group_id", ""), row["user_id"]) if row.get("group_id") else {}
    sender_hist = store.sender_history_to_user(row["user_id"], row.get("sender_user_id", ""))
    sender_first_contact = row["conversation_type"] == "personal" and len(sender_hist) == 0

    risk_signals = safety_analyze(row, media_text, business, membership, sender_first_contact)
    evidence = find_evidence(store, row, media_text=media_text, top_k=3)
    context = build_prompt_context(store, row, media_text, evidence, risk_signals)

    decision = _call_llm(client, model, context)

    # --- deterministic safety override ---
    if risk_signals.get("hard_mute_scam"):
        decision["action"] = "mute"
        if decision.get("message_type") not in ("scam", "spam"):
            decision["message_type"] = "scam"
        decision["confidence"] = max(decision.get("confidence", 0.5), 0.9)
        signal_note = _explain_risk(risk_signals)
        decision["reason"] = f"Muted by safety rule: {signal_note}."

    return _finalize(row, decision, evidence)


def _explain_risk(signals: dict) -> str:
    parts = []
    if signals.get("asks_otp_or_pin"):
        parts.append("requests OTP/PIN alongside a link or payment")
    if signals.get("business_domain_mismatch"):
        parts.append("sender domain does not match the business's official domain")
    if signals.get("business_uses_new_domain") and signals.get("asks_payment"):
        parts.append("payment request from a very recently registered domain")
    if signals.get("urgency_language") and signals.get("asks_payment"):
        parts.append("urgency language combined with a payment/fee ask")
    if signals.get("business_high_reports"):
        parts.append("sender has a high recent user-report count")
    return "; ".join(parts) if parts else "multiple corroborating risk signals"


def _call_llm(client, model: str, context: dict) -> dict:
    user_content = (
        SYSTEM_PROMPT
        + "\n\nRoute this message using the context below.\n\n```json\n"
        + json.dumps(context, indent=2, default=str)
        + "\n```"
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = RouteDecision.model_validate_json(raw)
            return parsed.model_dump()
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate" in err_str or "limit" in err_str) and attempt < max_retries - 1:
                time.sleep(5.0 * (attempt + 1))
                continue
            return {
                "action": "digest",
                "message_type": "unknown",
                "reason": f"Fallback decision: LLM call failed ({e}); defaulted to safe low-priority routing.",
                "confidence": 0.3,
                "evidence_message_ids": [],
            }


def _finalize(row: dict, decision: dict, evidence: list[dict]) -> dict:
    action = decision.get("action", "digest")
    if action not in ALLOWED_ACTIONS:
        action = "digest"
    mtype = decision.get("message_type", "unknown")
    if mtype not in ALLOWED_TYPES:
        mtype = "unknown"

    ev_ids = decision.get("evidence_message_ids") or []
    valid_ids = {e["message_id"] for e in evidence}
    seen = set()
    deduped_ids = []
    for e in ev_ids:
        if e in valid_ids and e not in seen:
            seen.add(e)
            deduped_ids.append(e)
    evidence_str = ";".join(deduped_ids) if deduped_ids else "none"

    try:
        conf = float(decision.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    reason = (decision.get("reason") or "").strip() or "No specific reason provided."

    return {
        "message_id": row["message_id"],
        "action": action,
        "message_type": mtype,
        "reason": reason,
        "confidence": round(conf, 2),
        "evidence_message_ids": evidence_str,
    }
