#!/usr/bin/env python3
"""
evaluation/evaluate.py
-----------------------
Runs the routing pipeline on dataset/sample_messages.csv (which has
ground-truth action/message_type/reason/confidence/evidence_message_ids)
and reports accuracy metrics. Use this to tune prompts/rules BEFORE
burning free-tier quota on the full 110-row messages.csv.

Usage:
    export GROQ_API_KEY=...
    python evaluation/evaluate.py
"""

from __future__ import annotations
import csv
import os
import sys
import time

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from groq import Groq  # noqa: E402
from data_loader import DataStore  # noqa: E402
from router import route_message  # noqa: E402

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
SAMPLE_PATH = os.path.join(DATASET_DIR, "sample_messages.csv")


def load_samples():
    with open(SAMPLE_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: set GROQ_API_KEY.", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    sleep_s = float(os.environ.get("ROUTER_SLEEP", "4.5"))

    client = Groq(api_key=api_key)
    store = DataStore(DATASET_DIR)
    samples = load_samples()

    action_correct = 0
    type_correct = 0
    evidence_hits = 0
    evidence_total_gt = 0
    rows_out = []

    for i, gt in enumerate(samples, 1):
        row = {k: gt[k] for k in
               ["message_id", "user_id", "conversation_type", "group_id", "business_id",
                "sender_user_id", "created_at", "message_text", "media_type",
                "media_id", "forwarded_count"]}
        pred = route_message(client, model, store, row)

        a_ok = pred["action"] == gt["action"]
        t_ok = pred["message_type"] == gt["message_type"]
        action_correct += a_ok
        type_correct += t_ok

        gt_ev = set(x for x in gt["evidence_message_ids"].split(";") if x and x != "none")
        pred_ev = set(x for x in pred["evidence_message_ids"].split(";") if x and x != "none")
        if gt_ev:
            evidence_total_gt += 1
            if gt_ev & pred_ev:
                evidence_hits += 1

        flag = "OK " if (a_ok and t_ok) else "ERR"
        print(f"[{flag}] {gt['message_id']:>14} pred=({pred['action']:>6},{pred['message_type']:<14}) "
              f"gt=({gt['action']:>6},{gt['message_type']:<14})")
        rows_out.append({**pred, "gt_action": gt["action"], "gt_message_type": gt["message_type"]})
        time.sleep(sleep_s)

    n = len(samples)
    print("\n--- Evaluation summary ---")
    print(f"rows: {n}")
    print(f"action accuracy:       {action_correct}/{n} = {action_correct / n:.1%}")
    print(f"message_type accuracy: {type_correct}/{n} = {type_correct / n:.1%}")
    if evidence_total_gt:
        print(f"evidence overlap (rows with gt evidence): "
              f"{evidence_hits}/{evidence_total_gt} = {evidence_hits / evidence_total_gt:.1%}")

    out_path = os.path.join(os.path.dirname(__file__), "eval_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nDetailed results written to {out_path}")


if __name__ == "__main__":
    main()
