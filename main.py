import json
import os
import re
import sys
from datetime import datetime
import argparse

from agents import answer_agent, critic_agent, refiner_agent, judge_agent
from cv_tools import run_cv_mode
from rag import retrieve_context
from scoring import score_answers
from utils import call_llm





ROUNDS = 3

MAX_ANSWER_CHARS = 1000

MAX_ANSWER_SENTENCES = 6

META_PREFIX_PATTERNS = [
    r"^\s*here is (the )?(rewritten|improved|refined) answer\s*[:\-]?\s*",
    r"^\s*(rewritten|improved|refined) answer\s*[:\-]?\s*",
    r"^\s*answer\s*[:\-]\s*",
]

DEFAULT_LOG_DIR = "logs"


def _count_sentences(text):
    return len(re.findall(r"[^.!?]+[.!?]", text))


def _truncate_sentences(text,max_sentences=MAX_ANSWER_SENTENCES):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()


def _strip_meta_prefix(text):
    if not text:
        return text

    cleaned = text.strip()

    # Remove markdown headings and labels often added by LLMs before the real answer.
    cleaned = re.sub(r"^\s*#{1,6}\s*.*$", "", cleaned, flags=re.MULTILINE).strip()

    # Remove one-line prefaces like "Here is the rewritten answer:".
    for pattern in META_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # If there is a leading label line before a blank line, keep content after it.
    cleaned = re.sub(
        r"^\s*(here is[^\n]*|rewritten answer[^\n]*|improved answer[^\n]*)\n+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    return cleaned



def _needs_normalization(text):

    if not text:

        return False
    
    lowered = text.lower().strip()

    meta_markers = [

        "here is",
        "i have rewritten",
        "note:",
        "rewritten answer",

    ]


    has_meta = any(marker in lowered for marker in meta_markers)
    too_long = len(text) > MAX_ANSWER_CHARS
    too_many_sentences = _count_sentences(text) > MAX_ANSWER_SENTENCES

    return has_meta or too_long or too_many_sentences



def _normalize_answer(text, question):

    if not text or text.startswith("LLM_ERROR:"):
        return text
    
    cleaned = _strip_meta_prefix(text)

    if not _needs_normalization(cleaned):
        return cleaned
    

    prompt = f"""
    Rewrite the answer to make it complete and concise.


Rules:
- Keep the core meaning and factual content.
- Remove meta commentary, notes, and self-references.
- Do not include markdown headings unless user asked.
- Maximum {MAX_ANSWER_SENTENCES} sentences.
- Maximum {MAX_ANSWER_CHARS} characters.
- Use grammatically correct English.



Question:
{question}

Answer:
{cleaned}

Return ONLY the cleaned answer.

"""
    
    rewritten = call_llm(prompt, max_length=280, temperature=0.0).strip()

    if rewritten and not rewritten.startswith("LLM_ERROR:"):
        cleaned = _strip_meta_prefix(rewritten)

    cleaned = _truncate_sentences(cleaned, max_sentences=MAX_ANSWER_SENTENCES)

    if len(cleaned) > MAX_ANSWER_CHARS:
        cleaned = cleaned[:MAX_ANSWER_CHARS].rstrip()

    return cleaned


def _build_log_path(question, log_dir=DEFAULT_LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", question).strip("_").lower()
    slug = (slug[:40] or "debate").strip("_")
    return os.path.join(log_dir, f"debate_{timestamp}_{slug}.json")


def _write_debate_log(payload, log_dir=DEFAULT_LOG_DIR):
    log_path = _build_log_path(payload.get("question", "debate"), log_dir=log_dir)
    with open(log_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=True)
    return log_path


def _run_cv_cli(argv):
    parser = argparse.ArgumentParser(prog="main.py cv", description="Run CV features on an image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument(
        "--mode",
        choices=["ocr", "qa", "classify", "chart", "describe"],
        default="describe",
        help="CV action to perform",
    )
    parser.add_argument("--question", default="", help="Question for image QA or chart analysis")
    parser.add_argument("--model", default="", help="Optional Ollama vision model override")
    args = parser.parse_args(argv)

    result = run_cv_mode(
        args.image,
        mode=args.mode,
        question=args.question or None,
        model=args.model or None,
    )

    print(result)



def run_system(question):

    context = retrieve_context(question)

    # Step 1: Initialize answers
    ans_logical = answer_agent(context, question, style="logical")
    ans_creative = answer_agent(context, question, style="creative")

    ans_logical = _normalize_answer(ans_logical,question)

    ans_creative = _normalize_answer(ans_creative,question)

    print("\nInitial Answers:")
    print("Logical Agent:", ans_logical)
    print("Creative Agent:", ans_creative)

    initial_logical = ans_logical
    initial_creative = ans_creative

    previous_winner = None
    round_logs = []

    # Multi-round iterative debate
    for round_num in range(1, ROUNDS + 1):
        print(f"\n=== ROUND {round_num} ===")

        # Judge selects best answer
        winner = judge_agent(ans_logical, ans_creative, context, question, previous_winner)
        print("\nJudge decision:", winner)
        previous_winner = winner

        # Critic reviews both answers for fairness
        feedback_logical = critic_agent(context, question, ans_logical)
        feedback_creative = critic_agent(context, question, ans_creative)

        print("\nCritic Feedback Logical:", feedback_logical)
        print("Critic Feedback Creative:", feedback_creative)

        # Refine both answers
        ans_logical = refiner_agent(context, question, ans_logical, feedback_logical)
        ans_creative = refiner_agent(context, question, ans_creative, feedback_creative)


         
        ans_logical = _normalize_answer(ans_logical, question)
        ans_creative = _normalize_answer(ans_creative, question)


        print("\nRefined Logical:", ans_logical)
        print("Refined Creative:", ans_creative)

        # Score both answers
        scores = score_answers(ans_logical, ans_creative, context, question)
        print("\nScores:", scores)

        round_logs.append(
            {
                "round": round_num,
                "judge_decision": winner,
                "feedback": {
                    "logical": feedback_logical,
                    "creative": feedback_creative,
                },
                "refined_answers": {
                    "logical": ans_logical,
                    "creative": ans_creative,
                },
                "scores": scores,
            }
        )

    print("\n=== FINAL WINNER ===")
    final_winner = judge_agent(ans_logical, ans_creative, context, question, previous_winner)
    print(final_winner)

    log_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "context": context,
        "initial_answers": {
            "logical": initial_logical,
            "creative": initial_creative,
        },
        "rounds": round_logs,
        "final_winner": final_winner,
    }
    log_path = _write_debate_log(log_payload)
    print(f"Debate log saved to: {log_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"cv", "--cv"}:
        _run_cv_cli(sys.argv[2:])
        raise SystemExit(0)
    elif len(sys.argv) > 1:
        q = " ".join(sys.argv[1:]).strip()
    else:
        q = input("Ask a question: ").strip()

    if not q:
        raise SystemExit("Question cannot be empty.")

    run_system(q)





