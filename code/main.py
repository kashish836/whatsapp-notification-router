#!/usr/bin/env python3
"""
main.py  (Groq edition — 100% free)
------------------------------------
Reads dataset/messages.csv, routes every message through the pipeline,
and writes output.csv with the exact required schema.

Setup (one time):
    pip install -r requirements.txt
    export GROQ_API_KEY=your_free_key_from_console.groq.com

Run:
    python main.py                 # full run
    python main.py --limit 10      # quick smoke test on first 10 rows

Resumability: results are checkpointed to code/.checkpoint.json keyed by
message_id. If the run is interrupted (free-tier rate limit, laptop sleep,
Ctrl+C) just re-run the same command — already-solved rows are skipped.

Free-tier pacing: the free Groq tier caps requests per minute (RPM).
--sleep controls the delay between calls; default is tuned to be safe
even at ~30 RPM limits. Raise --sleep if you see 429 errors.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time

from groq import Groq

from data_loader import DataStore
from router import route_message

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), ".checkpoint.json")


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            return json.load(open(CHECKPOINT_PATH))
        except Exception:
            return {}
    return {}


def save_checkpoint(cp: dict):
    json.dump(cp, open(CHECKPOINT_PATH, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "..", "dataset"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only process first N rows (smoke test)")
    ap.add_argument("--model", default=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
    ap.add_argument("--sleep", type=float, default=float(os.environ.get("ROUTER_SLEEP", "4.5")),
                     help="seconds to wait between messages, to stay under free-tier RPM limits")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoint, reprocess everything")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset)
    out_path = args.out or os.path.join(dataset_dir, "output.csv")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: set GROQ_API_KEY in your environment.\n"
              "Get a free key at https://console.groq.com", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)
    store = DataStore(dataset_dir)

    rows = store.messages.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]

    checkpoint = {} if args.no_resume else load_checkpoint()

    total = len(rows)
    todo = [r for r in rows if r["message_id"] not in checkpoint]
    print(f"Routing {total} messages ({len(todo)} remaining, model={args.model}, "
          f"~{args.sleep}s/msg -> ~{len(todo) * args.sleep / 60:.1f} min estimated)...")
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        mid = row["message_id"]
        if mid in checkpoint:
            continue
        try:
            decision = route_message(client, args.model, store, row)
        except Exception as e:
            # last-resort safety net: never let one row kill the whole run
            decision = {
                "message_id": mid,
                "action": "digest",
                "message_type": "unknown",
                "reason": f"Unhandled error during routing ({e}); defaulted to safe low-priority routing.",
                "confidence": 0.3,
                "evidence_message_ids": "none",
            }
        checkpoint[mid] = decision
        save_checkpoint(checkpoint)  # cheap (264 rows max) — save every row so nothing is ever lost
        if i % 5 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  [{i}/{total}] elapsed={elapsed:.0f}s", flush=True)
        time.sleep(args.sleep)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for row in rows:
            d = checkpoint[row["message_id"]]
            w.writerow({k: d.get(k, "") for k in OUTPUT_COLUMNS})

    print(f"Done. Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
