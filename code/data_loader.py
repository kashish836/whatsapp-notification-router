"""
data_loader.py
---------------
Loads every dataset/*.csv file and exposes a single function,
`build_context(message_row)`, that assembles ALL relevant structured
context for one incoming message: user profile, group/business info,
sender relationship, and retrieval candidates from message_history.

This keeps main.py / router.py free of pandas plumbing.
"""

from __future__ import annotations
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


class DataStore:
    def __init__(self, dataset_dir: str):
        self.dir = dataset_dir

        def load(name):
            return pd.read_csv(os.path.join(dataset_dir, name), dtype=str, keep_default_na=False)

        self.messages = load("messages.csv")
        self.users = load("users.csv").set_index("user_id", drop=False)
        self.groups = load("groups.csv").set_index("group_id", drop=False)
        self.group_members = load("group_members.csv")
        self.business_accounts = load("business_accounts.csv").set_index("business_id", drop=False)
        self.user_business_history = load("user_business_history.csv")
        self.message_history = load("message_history.csv")
        self.message_events = load("message_events.csv")
        self.images = load("images.csv").set_index("image_id", drop=False)
        self.voice_notes = load("voice_notes.csv").set_index("voice_note_id", drop=False)
        self.daily_summary = load("daily_notification_summary.csv")

        # numeric casts for the columns we actually do math on
        for col in ["messages_opened_30d", "messages_replied_30d",
                    "notifications_dismissed_30d", "messages_reported_30d"]:
            if col in self.users.columns:
                self.users[col] = pd.to_numeric(self.users[col], errors="coerce").fillna(0)

        for col in ["verified", "account_age_days", "messages_sent_30d",
                    "user_reports_30d", "domain_used_by_sender_age_days"]:
            if col in self.business_accounts.columns:
                self.business_accounts[col] = pd.to_numeric(
                    self.business_accounts[col], errors="coerce").fillna(0)

        # fast lookup: (group_id, user_id) -> row
        self.group_members_idx = self.group_members.set_index(["group_id", "user_id"], drop=False)
        # fast lookup: (user_id, business_id) -> row
        self.ubh_idx = self.user_business_history.set_index(["user_id", "business_id"], drop=False)
        # fast lookup: (user_id, message_id) -> event row
        self.events_idx = self.message_events.set_index(["user_id", "message_id"], drop=False)

    # ---------- helpers ----------

    def user_profile(self, user_id: str) -> dict:
        if user_id in self.users.index:
            return self.users.loc[user_id].to_dict()
        return {}

    def group_info(self, group_id: str) -> dict:
        if group_id and group_id in self.groups.index:
            return self.groups.loc[group_id].to_dict()
        return {}

    def group_membership(self, group_id: str, user_id: str) -> dict:
        key = (group_id, user_id)
        if key in self.group_members_idx.index:
            row = self.group_members_idx.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.to_dict()
        return {}

    def business_info(self, business_id: str) -> dict:
        if business_id and business_id in self.business_accounts.index:
            return self.business_accounts.loc[business_id].to_dict()
        return {}

    def user_business_relationship(self, user_id: str, business_id: str) -> dict:
        key = (user_id, business_id)
        if key in self.ubh_idx.index:
            row = self.ubh_idx.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.to_dict()
        return {}

    def event_for(self, user_id: str, message_id: str) -> dict:
        key = (user_id, message_id)
        if key in self.events_idx.index:
            row = self.events_idx.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.to_dict()
        return {}

    def media_path(self, media_type: str, media_id: str) -> Optional[str]:
        if not media_id:
            return None
        if media_type == "image" and media_id in self.images.index:
            return os.path.join(self.dir, self.images.loc[media_id, "file_path"])
        if media_type == "voice" and media_id in self.voice_notes.index:
            return os.path.join(self.dir, self.voice_notes.loc[media_id, "file_path"])
        return None

    def recent_daily_load(self, user_id: str, n_days: int = 7) -> list[dict]:
        rows = self.daily_summary[self.daily_summary["user_id"] == user_id]
        rows = rows.sort_values("date").tail(n_days)
        return rows.to_dict("records")

    def group_mates_history(self, group_id: str, limit: int = 200) -> pd.DataFrame:
        """All historical messages sent in this group (any user), most recent first."""
        if not group_id:
            return self.message_history.iloc[0:0]
        h = self.message_history[self.message_history["group_id"] == group_id]
        return h.sort_values("created_at", ascending=False).head(limit)

    def user_history(self, user_id: str, limit: int = 300) -> pd.DataFrame:
        """All historical messages received by this user, most recent first."""
        h = self.message_history[self.message_history["user_id"] == user_id]
        return h.sort_values("created_at", ascending=False).head(limit)

    def sender_history_to_user(self, user_id: str, sender_user_id: str) -> pd.DataFrame:
        if not sender_user_id:
            return self.message_history.iloc[0:0]
        h = self.message_history[
            (self.message_history["user_id"] == user_id)
            & (self.message_history["sender_user_id"] == sender_user_id)
        ]
        return h.sort_values("created_at", ascending=False)

    def business_history_to_user(self, user_id: str, business_id: str) -> pd.DataFrame:
        if not business_id:
            return self.message_history.iloc[0:0]
        h = self.message_history[
            (self.message_history["user_id"] == user_id)
            & (self.message_history["business_id"] == business_id)
        ]
        return h.sort_values("created_at", ascending=False)
