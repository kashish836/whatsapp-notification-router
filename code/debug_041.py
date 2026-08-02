import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import DataStore
from retrieval import find_evidence

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
SAMPLE_PATH = os.path.join(DATASET_DIR, "sample_messages.csv")

with open(SAMPLE_PATH, newline="", encoding="utf-8") as f:
    samples = list(csv.DictReader(f))

row = next(r for r in samples if r["message_id"] == "sample_msg_041")

print("=== sample_msg_041 ===")
for k, v in row.items():
    print(f"  {k}: {v}")

store = DataStore(DATASET_DIR)
evidence = find_evidence(store, row)

print(f"\nEvidence found: {len(evidence)} items")
for i, e in enumerate(evidence, 1):
    print(f"\n  [{i}]")
    for k, v in e.items():
        print(f"    {k}: {v}")
