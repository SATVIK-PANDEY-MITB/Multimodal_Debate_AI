from agents import answer_agent, critic_agent, refiner_agent, judge_agent
from rag import retrieve_context
from scoring import score_answers
from utils import call_llm





ROUNDS = 3

max_answer_chars = 1000

max_answer_sentences = 6


def _count_sentences(text):
    return len(re.findall(r"[^.!?]+[.!?]", text))


def _truncate_sentences(text,max_sentences=MAX_ANSWER_SENTENCES):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()



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
    too_long = len(text) > max_answer_chars
    too_many_sentences = _count_sentences(text) > max_answer_sentences

    return has_meta or too_long or too_many_sentences



def _normalize_answer(text, question):

    if not text or text.startswith("LLM_ERROR:"):
        return text
    
    cleaned = text.strip()

    if not _needs_normalization(cleaned):
        return cleaned
    

    prompt = f"""
    Rewrite the answer to make it compliment and concise.


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
        
        cleaned = rewritten

    cleaned = _truncate_sentences(cleaned, max_sentences=MAX_ANSWER_SENTENCES)

    if len(cleaned) > max_answer_chars:
        cleaned = cleaned[:MAX_ANSWER_CHARS].rstrip()

    return cleaned



def run_system(question):

    context = retrieve_context(question)

    # Step 1: Initialize answers
    ans_logical = answer_agent(context, question, style="logical")
    ans_creative = answer_agent(context, question, style="creative")

    ans_logical = normalize_answer(ans_logical,question)

    ans_creative = normalize_answer(ans_creative,question)

    print("\nInitial Answers:")
    print("Logical Agent:", ans_logical)
    print("Creative Agent:", ans_creative)

    previous_winner = None

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

    print("\n=== FINAL WINNER ===")
    final_winner = judge_agent(ans_logical, ans_creative, context, question, previous_winner)
    print(final_winner)


if __name__ == "__main__":

    q = input("Ask a question: ")
    run_system(q)





