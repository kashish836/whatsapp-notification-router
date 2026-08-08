import os
import time
import random
import string
from collections import defaultdict

from flask import Flask, jsonify, request, send_from_directory

from data_loader import DataStore

app = Flask(__name__)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
store = DataStore(DATASET_DIR)

_groq_client = None
_groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# ── rate limiting state ──
_rate_windows = defaultdict(list)  # ip -> [timestamps]
_daily_count = 0
_daily_reset = time.time()
_MAX_PER_IP = 5
_WINDOW_SECONDS = 10 * 60  # 10 minutes
_MAX_DAILY = 150


def _check_rate_limit(ip):
    global _daily_count, _daily_reset
    now = time.time()

    # reset daily counter at midnight-ish (every 24h)
    if now - _daily_reset > 86400:
        _daily_count = 0
        _daily_reset = now

    # daily cap
    if _daily_count >= _MAX_DAILY:
        return jsonify({"error": "Daily demo quota reached — check back tomorrow, or view the pre-loaded message threads on the left."}), 429

    # per-IP rolling window
    timestamps = _rate_windows[ip]
    cutoff = now - _WINDOW_SECONDS
    _rate_windows[ip] = [t for t in timestamps if t > cutoff]
    if len(_rate_windows[ip]) >= _MAX_PER_IP:
        return jsonify({"error": "Rate limit reached — try again in a few minutes. This demo shares a free API quota with everyone."}), 429

    _rate_windows[ip].append(now)
    _daily_count += 1
    return None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


@app.route("/")
def index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), "index.html")


@app.route("/api/threads")
def api_threads():
    import pandas as pd

    output_path = os.path.join(DATASET_DIR, "output.csv")
    if not os.path.exists(output_path):
        return jsonify([])

    output = pd.read_csv(output_path, dtype=str)
    output = output.set_index("message_id", drop=False)

    messages = store.messages.set_index("message_id", drop=False)

    results = []
    for mid in output.index:
        if mid not in messages.index:
            continue
        msg = messages.loc[mid]
        out = output.loc[mid]
        results.append({
            "message_id": mid,
            "sender_user_id": msg.get("sender_user_id", ""),
            "group_id": msg.get("group_id", ""),
            "business_id": msg.get("business_id", ""),
            "conversation_type": msg.get("conversation_type", ""),
            "message_text": msg.get("message_text", ""),
            "media_type": msg.get("media_type", ""),
            "timestamp": msg.get("created_at", ""),
            "action": out.get("action", ""),
            "message_type": out.get("message_type", ""),
            "reason": out.get("reason", ""),
            "confidence": out.get("confidence", ""),
            "evidence_message_ids": out.get("evidence_message_ids", ""),
        })

    return jsonify(results)


@app.route("/api/directory")
def api_directory():
    import pandas as pd

    result = {"users": {}, "groups": {}, "businesses": {}}

    # users.csv — no name column expected, fall back to raw ID
    users_path = os.path.join(DATASET_DIR, "users.csv")
    if os.path.exists(users_path):
        try:
            df = pd.read_csv(users_path, dtype=str)
            for _, row in df.iterrows():
                uid = row.get("user_id", "")
                if uid:
                    result["users"][uid] = uid
        except Exception:
            pass

    # groups.csv — try group_name, name
    groups_path = os.path.join(DATASET_DIR, "groups.csv")
    if os.path.exists(groups_path):
        try:
            df = pd.read_csv(groups_path, dtype=str)
            name_col = None
            for col in ["group_name", "name", "display_name"]:
                if col in df.columns:
                    name_col = col
                    break
            for _, row in df.iterrows():
                gid = row.get("group_id", "")
                if gid:
                    result["groups"][gid] = row[name_col] if name_col else gid
        except Exception:
            pass

    # business_accounts.csv — try display_name, brand_name, business_name, name
    biz_path = os.path.join(DATASET_DIR, "business_accounts.csv")
    if os.path.exists(biz_path):
        try:
            df = pd.read_csv(biz_path, dtype=str)
            name_col = None
            for col in ["display_name", "brand_name", "business_name", "name"]:
                if col in df.columns:
                    name_col = col
                    break
            for _, row in df.iterrows():
                bid = row.get("business_id", "")
                if bid:
                    result["businesses"][bid] = row[name_col] if name_col else bid
        except Exception:
            pass

    return jsonify(result)


@app.route("/api/route", methods=["POST"])
def api_route():
    from router import route_message

    # rate limit check
    blocked = _check_rate_limit(request.remote_addr)
    if blocked:
        return blocked

    body = request.get_json(silent=True) or {}
    message_text = body.get("message_text", "")
    sender_user_id = body.get("sender_user_id", "")
    business_id = body.get("business_id", "")
    group_id = body.get("group_id", "")
    conversation_type = body.get("conversation_type", "")

    # input validation
    if not message_text or not message_text.strip():
        return jsonify({"error": "message_text is required and cannot be empty."}), 400
    if len(message_text) > 500:
        return jsonify({"error": "message_text must be 500 characters or fewer."}), 400

    if not conversation_type:
        if business_id:
            conversation_type = "business"
        elif group_id:
            conversation_type = "group"
        else:
            conversation_type = "personal"

    user_id = body.get("user_id", "")
    if not user_id:
        if conversation_type == "group" and group_id:
            members = store.group_members[store.group_members["group_id"] == group_id]
            user_id = members.iloc[0]["user_id"] if len(members) else ""
        elif conversation_type == "business" and business_id:
            rels = store.user_business_history[store.user_business_history["business_id"] == business_id]
            user_id = rels.iloc[0]["user_id"] if len(rels) else ""
        else:
            user_id = store.messages.iloc[0]["user_id"] if len(store.messages) else ""

    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    synthetic_id = f"live_{suffix}"

    row = {
        "message_id": synthetic_id,
        "user_id": user_id,
        "conversation_type": conversation_type,
        "group_id": group_id,
        "business_id": business_id,
        "sender_user_id": sender_user_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
        "message_text": message_text,
        "media_type": body.get("media_type", ""),
        "media_id": body.get("media_id", ""),
        "forwarded_count": body.get("forwarded_count", "0"),
    }

    try:
        client = _get_groq_client()
        decision = route_message(client, _groq_model, store, row)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(decision)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
