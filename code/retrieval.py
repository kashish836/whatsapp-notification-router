"""
retrieval.py
------------
Finds historical messages relevant to the current incoming message so the
router can (a) cite them as evidence_message_ids and (b) reason about
repetition / past engagement outcomes.

Approach: lightweight TF-IDF cosine similarity (scikit-learn is heavy for a
264-row task, so we roll a tiny bag-of-words scorer with no extra
dependency) restricted to a *scoped* candidate pool:

  1. Same sender_user_id -> same user   (highest priority: direct history)
  2. Same business_id    -> same user
  3. Same group_id                       (group-wide patterns, e.g. spam floods)
  4. Same user, any sender (fallback, general behavioral pattern)

Each candidate is scored by simple token overlap (Jaccard-ish) over the
message_text, boosted if media_type matches. Top-N are returned along with
their message_events outcome (opened/replied/dismissed/reported) so the
LLM can see "this exact kind of message was ignored 3 times before".
"""

from __future__ import annotations
import re
from collections import Counter
import pandas as pd

STOPWORDS = set("""
a an the is are was were to of in on at for and or but with your you our we
i this that it its as be by from up down please pls kindly dear hi hello
""".split())


def tokenize(text: str) -> set[str]:
    if not text:
        return set()
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def score(query_tokens: set[str], cand_tokens: set[str]) -> float:
    if not query_tokens or not cand_tokens:
        return 0.0
    overlap = len(query_tokens & cand_tokens)
    return overlap / (len(query_tokens | cand_tokens) + 1e-9)


def find_evidence(store, message_row: dict, media_text: str = "", top_k: int = 3) -> list[dict]:
    """
    Returns list of dicts: {message_id, created_at, message_text, score, event}
    sorted by relevance, most useful evidence first.
    """
    user_id = message_row.get("user_id", "")
    sender_id = message_row.get("sender_user_id", "")
    business_id = message_row.get("business_id", "")
    group_id = message_row.get("group_id", "")
    query_text = (message_row.get("message_text") or "") + " " + (media_text or "")
    q_tokens = tokenize(query_text)

    pools = []
    if sender_id:
        pools.append(store.sender_history_to_user(user_id, sender_id))
    if business_id:
        pools.append(store.business_history_to_user(user_id, business_id))
    if group_id:
        pools.append(store.group_mates_history(group_id, limit=150))
    pools.append(store.user_history(user_id, limit=150))

    seen_ids = set()
    candidates = []
    for pool in pools:
        for _, row in pool.iterrows():
            mid = row["message_id"]
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            candidates.append(row)

    scored = []
    for row in candidates:
        c_tokens = tokenize(row.get("message_text", ""))
        s = score(q_tokens, c_tokens)
        # same sender / same business gets a relevance floor even with weak text overlap,
        # since *repetition from the same source* is itself the signal
        if row.get("sender_user_id") == sender_id and sender_id:
            s = max(s, 0.15)
        if row.get("business_id") == business_id and business_id:
            s = max(s, 0.15)
        if s <= 0:
            continue
        event = store.event_for(user_id, row["message_id"])
        scored.append({
            "message_id": row["message_id"],
            "created_at": row.get("created_at", ""),
            "message_text": (row.get("message_text", "") or "")[:200],
            "sender_user_id": row.get("sender_user_id", ""),
            "business_id": row.get("business_id", ""),
            "group_id": row.get("group_id", ""),
            "score": round(s, 3),
            "opened": event.get("message_opened", ""),
            "replied": event.get("message_replied", ""),
            "dismissed": event.get("notification_dismissed", ""),
            "muted_after": event.get("muted_after_message", ""),
            "reported": event.get("message_reported", ""),
        })

    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]
