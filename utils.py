import base64
import os

import requests

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate")
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "60"))
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "moondream")


def _encode_image_file(image_path):
    with open(image_path, "rb") as file_handle:
        return base64.b64encode(file_handle.read()).decode("utf-8")


def call_llm(prompt, max_length=220, temperature=0.2, timeout=OLLAMA_REQUEST_TIMEOUT):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": max_length,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        return f"LLM_ERROR: request_failed: {e}"
    except ValueError:
        return "LLM_ERROR: invalid_json_response"


def call_vision_llm(
    prompt,
    image_paths,
    model=None,
    max_length=220,
    temperature=0.2,
    timeout=OLLAMA_REQUEST_TIMEOUT,
):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    encoded_images = []
    for image_path in image_paths:
        encoded_images.append(_encode_image_file(image_path))

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model or OLLAMA_VISION_MODEL,
                "prompt": prompt,
                "images": encoded_images,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": max_length,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        return f"LLM_ERROR: request_failed: {e}"
    except ValueError:
        return "LLM_ERROR: invalid_json_response"
   
