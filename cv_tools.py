import os
from pathlib import Path

from utils import call_vision_llm, OLLAMA_VISION_MODEL
from utils import call_llm


VALID_CV_MODES = {"ocr", "qa", "classify", "chart", "describe"}


def _fallback_text(message):
    return message


def _vision_call(prompt, image_path, model=None, max_length=350, temperature=0.2):
    return call_vision_llm(
        prompt,
        _require_image(image_path),
        model=model or OLLAMA_VISION_MODEL,
        max_length=max_length,
        temperature=temperature,
    )


def _require_image(image_path):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {image_path}")
    return str(path)


def extract_text_from_image(image_path, model=None):
    prompt = (
        "You are an OCR engine. Extract all visible text from the image verbatim. "
        "Preserve line breaks, punctuation, numbers, and bullet points. "
        "If the image has no readable text, reply with NO_TEXT_FOUND. "
        "Return only the extracted text."
    )
    result = _vision_call(prompt, image_path, model=model, max_length=800, temperature=0.0)
    if result and not result.startswith("LLM_ERROR:"):
        return result
    return _fallback_text("NO_TEXT_FOUND")


def answer_image_question(image_path, question, model=None):
    image_description = describe_image(image_path, model=model)
    if image_description and not image_description.startswith("LLM_ERROR:"):
        answer_prompt = (
            "Answer the user's question using the image description below. "
            "If the description is insufficient, say that clearly. "
            f"Question: {question}\n\nImage description:\n{image_description}"
        )
        answer = call_llm(answer_prompt, max_length=220, temperature=0.2)
        if answer and not answer.startswith("LLM_ERROR:"):
            return answer

    return "No answer was returned by the vision model. Try a PNG or JPG image, or retry with a smaller/lower-compression file."


def classify_image(image_path, model=None):
    prompt = (
        "Classify the image into one of these categories: photo, screenshot, document, chart, diagram, collage, other. "
        "Explain the choice in one or two short sentences and mention the most important visible cues. "
        "Return a compact answer."
    )
    result = _vision_call(prompt, image_path, model=model, max_length=250, temperature=0.1)
    if result and not result.startswith("LLM_ERROR:"):
        return result

    description = describe_image(image_path, model=model)
    if description and not description.startswith("LLM_ERROR:"):
        classifier_prompt = (
            "Classify the image based on this description. "
            "Choose one label from: photo, screenshot, document, chart, diagram, collage, other. "
            "Explain the label in one short sentence.\n\n"
            f"Description:\n{description}"
        )
        fallback = call_llm(classifier_prompt, max_length=180, temperature=0.0)
        if fallback and not fallback.startswith("LLM_ERROR:"):
            return fallback

    return _fallback_text("Could not classify the image. Try a PNG or JPG image.")


def understand_chart(image_path, question=None, model=None):
    prompt = (
        "You are analyzing a chart or graph. Summarize the chart, identify visible axes, trends, peaks, dips, "
        "and notable values. If a question is provided, answer it directly. "
    )
    if question:
        prompt += f"Question: {question} "
    prompt += "Return a concise but useful analysis."
    result = _vision_call(prompt, image_path, model=model, max_length=500, temperature=0.2)
    if result and not result.startswith("LLM_ERROR:"):
        return result
    return _fallback_text("Could not analyze the chart. Try a PNG or JPG image.")


def describe_image(image_path, model=None):
    prompt = (
        "Describe this image in a clear, concise way. Mention objects, layout, text, and any obvious context. "
        "If it looks like a chart or screenshot, mention that too."
    )
    result = _vision_call(prompt, image_path, model=model, max_length=350, temperature=0.2)
    if result and not result.startswith("LLM_ERROR:"):
        return result
    return _fallback_text("Could not describe the image. Try a PNG or JPG image.")


def run_cv_mode(image_path, mode="describe", question=None, model=None):
    mode = (mode or "describe").lower().strip()
    if mode not in VALID_CV_MODES:
        raise ValueError(f"Unsupported CV mode: {mode}")

    suffix = Path(image_path).suffix.lower()
    if suffix == ".avif":
        return (
            "AVIF images are not reliable with the current Ollama vision pipeline. "
            "Please convert the file to PNG or JPG and try again."
        )

    if mode == "ocr":
        return extract_text_from_image(image_path, model=model)
    if mode == "qa":
        result = answer_image_question(image_path, question or "What is in this image?", model=model)
        if not result or result.startswith("LLM_ERROR:"):
            return (
                "No answer was returned by the vision model. "
                "Try a PNG or JPG image, or retry with a smaller/lower-compression file."
            )
        return result
    if mode == "classify":
        return classify_image(image_path, model=model)
    if mode == "chart":
        return understand_chart(image_path, question=question, model=model)
    return describe_image(image_path, model=model)
