import json
import re
from utils import call_llm


# -------------------------------
# SAFE JSON PARSER
# -------------------------------
def _extract_json(text):
    if not text:
        return None

    text = text.strip()

    # Try direct parsing
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract JSON block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


# -------------------------------
# ANSWER AGENT
# -------------------------------
def answer_agent(context, question, style="logical"):

    style_rules = {
        "logical": (
            "Style: precise, structured, technical.\n"
            "Explain step-by-step if needed.\n"
            "Focus on correctness and completeness."
        ),
        "creative": (
            "Style: intuitive and engaging.\n"
            "Use ONE analogy or example if useful.\n"
            "Keep explanation accurate but easy to understand."
        ),
    }

    selected_style = style_rules.get(style, style_rules["logical"])

    prompt = f"""
You are a high-quality AI assistant.

{selected_style}

STRICT RULES:
- Answer ONLY the question.
- Use context ONLY if it is relevant and helpful.
- If context is empty or weak, rely on your own knowledge.
- Use your internal knowledge to provide accurate and complete answers.
- Do NOT say "based on context" explicitly.
- Do NOT repeat sentences.
- Be concise but COMPLETE.
- If question asks multiple parts → answer ALL parts.
- Use clean formatting when needed (bullets/code).
- Ensure no part is cut off.

Question:
{question}

Context:
{context}

Output:
Return ONLY the final answer.
"""

    return call_llm(
        prompt,
        max_length=400,   # 🔥 increased to avoid cut answers
        temperature=0.6 if style == "creative" else 0.2,
    )


# -------------------------------
# CRITIC AGENT
# -------------------------------
def critic_agent(context, question, answer):

    prompt = f"""
You are a strict expert reviewer.

Evaluate the answer on:
- correctness
- clarity
- relevance
- completeness
- conciseness
- grammar

Return ONLY valid JSON:

{{
  "correctness": "short sentence",
  "clarity": "short sentence",
  "relevance": "short sentence",
  "completeness": "short sentence",
  "conciseness": "short sentence",
  "grammar": "short sentence",
  "action_items": ["item1", "item2", "item3"]
}}

Rules:
- Be strict
- STRICT RULE:
- You MUST return ONLY valid JSON.
- Do NOT write explanations.
- Do NOT add extra text.
- If unsure, still return best-effort JSON.
- No markdown

Question:
{question}

Context:
{context}

Answer:
{answer}
"""

    raw = call_llm(prompt, max_length=300, temperature=0.1)

    parsed = _extract_json(raw)

    # fallback if parsing fails
    if parsed is None:
        return {
            "correctness": "Answer may be partially correct but needs verification.",
            "clarity": "Clarity could be improved.",
            "relevance": "Mostly relevant to the question.",
            "completeness": "Some details may be missing.",
            "conciseness": "Could be more concise.",
            "grammar": "Minor grammar issues possible.",
            "action_items": [
                "Improve correctness",
                "Add missing details",
                "Make explanation clearer"
            ],
        }

    return parsed


# -------------------------------
# REFINER AGENT
# -------------------------------
def refiner_agent(context, question, answer, feedback):

    feedback_text = json.dumps(feedback)

    prompt = f"""
You are an expert editor.

Task:
Improve the answer using feedback.

STRICT RULES:
- Do NOT copy original sentences.
- You MUST rewrite the answer, not lightly edit it.
- Fix all weaknesses from feedback.
- MUST improve correctness OR completeness.
- Improve depth OR clarity OR structure.
- Add missing details if needed.
- Keep it concise but COMPLETE.
- Answer ALL parts of the question.
- Ensure answer is NOT cut off.

Question:
{question}

Context:
{context}

Original Answer:
{answer}

Feedback:
{feedback_text}

Output:
Return ONLY improved answer.
"""

    return call_llm(
        prompt,
        max_length=450,   # 🔥 increased for full answers
        temperature=0.2
    )


# -------------------------------
# JUDGE AGENT (IMPROVED)
# -------------------------------
def judge_agent(ans1, ans2, context, question, previous_winner=None):

    prompt = f"""
You are a strict evaluator.

Compare Answer_1 and Answer_2 based on:
- correctness (MOST IMPORTANT)
- completeness
- clarity

Rules:
- Prefer more COMPLETE answer if correctness is equal
- Ignore style differences unless clarity is affected
- Prefer correctness over style.
- If one answer is more factually accurate, choose it even if less creative.
- Do NOT be random

If the winner changed from previous_winner, explain why.

Return ONLY JSON:

{{
  "winner": "ANSWER_1" or "ANSWER_2",
    "reason": "short reason",
    "changed": true or false,
    "change_reason": "short reason if changed else empty string"
}}

Question:
{question}

Previous winner:
{previous_winner}

Context:
{context}

Answer_1:
{ans1}

Answer_2:
{ans2}
"""

    result = call_llm(prompt, max_length=80, temperature=0.0)

    parsed = _extract_json(result)

    if parsed and "winner" in parsed:
        winner = parsed["winner"].upper()

        print(f"Reason: {parsed.get('reason', 'N/A')}")
        if parsed.get("changed"):
            print(f"Change reason: {parsed.get('change_reason', 'N/A')}")

        if winner == "ANSWER_1":
            return "Answer1"
        elif winner == "ANSWER_2":
            return "Answer2"

    # fallback
    if "2" in result:
        return "Answer2"
    return "Answer1"