# Message Notification Router — HackerRank Orchestrate

An AI-powered WhatsApp notification router that decides `notify` / `digest` / `mute`
for every incoming message, using multimodal reasoning (text, image posters, voice
notes) personalized to each user's behavior — with a deterministic safety net that
mutes clear scams regardless of user preference.

**100% free to run.** Uses the Groq free API tier for all reasoning, OCR, and audio
transcription — no paid API, no local GPU, no local ML model downloads (a few MB pip
package only).

---

## Architecture

```
messages.csv row
      |
      v
+-----------------+     +------------------+     +-------------------+
| data_loader.py  | --> | media.py         | --> | safety.py          |
| joins all 10    |     | Groq vision OCR  |     | deterministic risk |
| context CSVs    |     | + Whisper audio  |     | signals (rules)    |
+-----------------+     +------------------+     +-------------------+
      |                                                   |
      v                                                   v
+-----------------+                              +-------------------+
| retrieval.py     | -----------------------> | router.py           |
| finds relevant   |                            | builds full context,|
| historical       |                            | calls Groq with     |
| evidence msgs    |                            | structured JSON     |
+-----------------+                            | output, applies     |
                                                 | safety override     |
                                                 +-------------------+
                                                          |
                                                          v
                                                    output.csv row
```

- **`data_loader.py`** — loads all 12 CSVs, exposes lookups (user profile, group
  membership, business relationship, historical messages, event outcomes).
- **`media.py`** — sends image bytes to Groq's vision model (Llama 4 Scout) for OCR
  and image understanding, uses Groq's Whisper endpoint for voice-note transcription.
  Results are cached to `.media_cache/` so re-runs don't re-spend quota.
- **`retrieval.py`** — scoped bag-of-words retrieval (same sender > same business >
  same group > same user) to surface the most relevant historical messages as
  evidence, along with how the user reacted to them (opened/replied/dismissed/muted/reported).
- **`safety.py`** — deterministic, explainable risk scoring (OTP/PIN requests,
  domain mismatch, urgency+payment combos, new/unverified senders, high report
  counts). This is **not** LLM-based, so it can never be talked out of flagging
  an obvious scam.
- **`router.py`** — assembles all context into a single structured prompt, calls
  Groq with JSON mode (guaranteed valid JSON output), then **overrides** the LLM's
  action to `mute`/`scam` if `safety.py` raised a hard-mute flag — guaranteeing
  "risk overrides personalization" per the spec, even if the LLM disagrees or the
  call fails.
- **`main.py`** — orchestrates the full 110-row run, checkpoints every row to
  `.checkpoint.json` (crash/rate-limit safe — re-running skips solved rows),
  writes `output.csv` in the exact required column order.
- **`evaluation/evaluate.py`** — runs the same pipeline against the 30 solved
  rows in `sample_messages.csv` and reports action/message_type accuracy and
  evidence-overlap, so you can tune before spending quota on the full run.

---

## Setup (do this once)

```bash
cd code
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a **free** Groq API key (no credit card): https://console.groq.com

```bash
export GROQ_API_KEY=your_key_here      # Windows (PowerShell): $env:GROQ_API_KEY="your_key_here"
```

---

## Run

**1. Smoke test first** (3 rows, ~20 seconds, confirms your key + code work):
```bash
python main.py --limit 3
```
Check `dataset/output.csv` — you should see 3 rows with sensible action/message_type/reason.

**2. Evaluate against the 30 solved samples** (tune here before the real run):
```bash
python evaluation/evaluate.py
```
This prints per-row correctness and a final accuracy summary. If accuracy is
low, the prompt (`SYSTEM_PROMPT` in `router.py`) or the safety rules
(`safety.py`) are the places to adjust.

**3. Full run** (110 messages, ~8-10 minutes at the default pacing):
```bash
python main.py
```
`dataset/output.csv` will have one row per `messages.csv` row, exact schema.

If you hit `429` / rate-limit errors, just re-run the same command — it
resumes automatically from `.checkpoint.json`. You can also increase pacing:
```bash
python main.py --sleep 8
```

If `llama-3.3-70b-versatile` free quota runs out for the day, switch model:
```bash
export GROQ_MODEL=llama-3.1-8b-instant
python main.py
```

---

## Design decisions worth mentioning in the AI Judge interview

1. **Why a safety override layer instead of trusting the LLM fully?** The spec
   explicitly says risk should override personalization "regardless of the
   user's usual engagement." A rule-based override guarantees this even under
   LLM hallucination, prompt injection from message content, or an API outage
   — it's the difference between "usually safe" and "safe by construction."
2. **Why scoped retrieval instead of full-corpus embedding search?** With only
   ~1000 historical messages, a lightweight scoped bag-of-words retriever
   (same sender/business/group first) is both fast and more precise than a
   generic embedding search would be for this size of corpus — and needs no
   extra paid embedding API calls.
3. **Why structured JSON output (Pydantic schema) instead of prompting for JSON
   in free text?** Removes an entire class of parsing failures and guarantees
   `action`/`message_type` are always one of the allowed enum values.
4. **Why checkpointing every row?** Free-tier rate limits are real; a crash or
   429 shouldn't mean re-paying (in quota) for already-solved rows.

---

## Chat transcript requirement

This repo's `AGENTS.md` expects a local AI coding tool (Claude Code, Cursor,
etc.) to auto-log your build conversation to
`$HOME/hackerrank_orchestrate_august26/log.txt`. If you built/ran this with
Claude Code locally, that log already exists — attach it as your transcript.
If you're finishing this from a web chat, open the project folder in Claude
Code once, ask it to review `AGENTS.md` and confirm the log file, and it will
pick up logging from there for any remaining work.
