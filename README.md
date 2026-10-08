# Mini-Jev: Support Inbox Triage

A lightweight, deterministic support ticket triage tool powered by zero-shot Natural Language Inference (NLI) using `deberta-v3-base-zeroshot-v2.0`.

## Features

- **Department Routing**: Classifies messages into candidate departments (`billing`, `technical problem`, `sales question`, `thank you`).
- **Human-in-the-Loop Threshold**: Automatically routes confident predictions; flags lower-confidence messages for human escalation.
- **Urgency Scoring**: Computes an expected priority score between 1.0 and 3.0.
- **Customer Anger / Frustration Metric**: Gauges customer frustration level.

## Setup

```bash
# Set up environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
