import json
import re
from utils import call_llm


def _parse_json(text):
    if not text:
        return None
    t = text.strip()

    try:
        return json.loads(t)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", t)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def score_answers(ans1, ans2, context, question):

    prompt = f"""
You are a scoring AI.

Score both answers from 1 to 10 on:
- C (Correctness)
- CL (Clarity)
- CF (Confidence)

Return ONLY valid JSON with this exact schema:
{{
  "Answer1": {{"C": 0, "CL": 0, "CF": 0}},
  "Answer2": {{"C": 0, "CL": 0, "CF": 0}}
}}

Question:
{question}

Context:
{context}

Answer1:
{ans1}

Answer2:
{ans2}
"""

    raw = call_llm(prompt, max_length=160, temperature=0.0)
    parsed = _parse_json(raw)

    if parsed is None:
        return {
            "Answer1": {"C": 0, "CL": 0, "CF": 0},
            "Answer2": {"C": 0, "CL": 0, "CF": 0},
            "error": "Could not parse scoring output",
            "raw": raw,
        }

    return parsed